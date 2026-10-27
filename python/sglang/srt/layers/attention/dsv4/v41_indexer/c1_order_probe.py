"""Default-off C1/C2 index-order probe. Graph-safe D2D into a ring; D2H after each request."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch

LAYERS = (2, 8, 14, 20)
LAYER_INDEX = {2: 0, 8: 1, 14: 2, 20: 3}
MAX_DECODE_STEPS = 16
MAX_K = 512
MAX_SCORE = 16384
MAX_Q_BYTES = 65536
MAX_O_BYTES = 65536

DISPATCH_SELECT_PAGED_V2_FINISH = 0
DISPATCH_FULL_TOPK_DEEPGEMM = 1
DISPATCH_FULL_TOPK_TORCH = 2
DISPATCH_WRITE_DECODE = 3
DISPATCH_SKIPPED = 4
DISPATCH_NAMES = {
    0: "select_decode_paged_v2_finish",
    1: "full_topk_deep_gemm_paged",
    2: "full_topk_torch",
    3: "write_decode",
    4: "skipped",
}

_NMETA = 12
_lock = threading.Lock()
_state = None


def probe_dir() -> Optional[str]:
    raw = os.environ.get("SGLANG_C1_ORDER_PROBE_DIR", "").strip()
    return raw or None


def active() -> bool:
    return probe_dir() is not None and _state is not None


def maybe_init(*, device: torch.device, max_bs: int, max_k: int = MAX_K) -> None:
    if probe_dir() is None:
        return
    global _state
    with _lock:
        if _state is not None:
            return
        _state = _ProbeState(device=device, max_bs=max_bs, max_k=max_k)


class _ProbeState:
    def __init__(self, *, device: torch.device, max_bs: int, max_k: int):
        self.device = device
        self.max_bs = max(int(max_bs), 1)
        self.max_k = int(max_k)
        n_steps = MAX_DECODE_STEPS
        n_layers = len(LAYERS)
        self.n_rows = n_steps * n_layers
        self.slot = torch.zeros((1,), dtype=torch.int64, device=device)
        self.cap = torch.tensor([MAX_DECODE_STEPS - 1], dtype=torch.int64, device=device)
        self.layer_offsets = torch.arange(len(LAYERS), dtype=torch.int64, device=device)
        self.raw = torch.full((self.n_rows, self.max_k), -1, dtype=torch.int32, device=device)
        self.page = torch.full((self.n_rows, self.max_k), -1, dtype=torch.int32, device=device)
        self.score = torch.zeros((self.n_rows, MAX_SCORE), dtype=torch.float32, device=device)
        self.q_bytes = torch.zeros((self.n_rows, MAX_Q_BYTES), dtype=torch.uint8, device=device)
        self.o_bytes = torch.zeros((self.n_rows, MAX_O_BYTES), dtype=torch.uint8, device=device)
        self.meta = torch.zeros((self.n_rows, _NMETA), dtype=torch.int32, device=device)
        self.raw_scratch = torch.full((self.max_bs, self.max_k), -1, dtype=torch.int32, device=device)
        self.i32_tmp = torch.zeros((1,), dtype=torch.int32, device=device)
        self.raw_tmp = torch.full((1, self.max_k), -1, dtype=torch.int32, device=device)
        self.page_tmp = torch.full((1, self.max_k), -1, dtype=torch.int32, device=device)
        self.score_tmp = torch.zeros((1, MAX_SCORE), dtype=torch.float32, device=device)
        self.q_tmp = torch.zeros((1, MAX_Q_BYTES), dtype=torch.uint8, device=device)
        self.o_tmp = torch.zeros((1, MAX_O_BYTES), dtype=torch.uint8, device=device)
        self.meta_tmp = torch.zeros((1, _NMETA), dtype=torch.int32, device=device)
        self.o_n = torch.zeros((self.n_rows,), dtype=torch.int32, device=device)
        self.q_n = torch.zeros((self.n_rows,), dtype=torch.int32, device=device)
        self.attn_flag = torch.zeros((self.n_rows,), dtype=torch.int32, device=device)
        self.host_decode_steps = 0
        self.req_seq = 0
        self.dump_count = 0

    def reset_request(self) -> None:
        self.slot.zero_()
        self.raw.fill_(-1)
        self.page.fill_(-1)
        self.score.zero_()
        self.q_bytes.zero_()
        self.o_bytes.zero_()
        self.meta.zero_()
        self.raw_scratch.fill_(-1)
        self.o_n.zero_()
        self.q_n.zero_()
        self.attn_flag.zero_()
        self.host_decode_steps = 0


def reset_after_warmup() -> None:
    if _state is None:
        return
    _state.reset_request()
    _state.req_seq = 0
    _state.dump_count = 0


def raw_scratch() -> Optional[torch.Tensor]:
    if _state is None:
        return None
    return _state.raw_scratch


def _row_index(layer_index: int) -> torch.Tensor:
    step = torch.minimum(_state.slot, _state.cap)
    return step * len(LAYERS) + int(layer_index)


def _fill_i32_row(dst_1xk: torch.Tensor, src: torch.Tensor, fill: int = -1) -> None:
    dst_1xk.fill_(fill)
    flat = src.reshape(-1).to(torch.int32)
    n = min(flat.shape[0], dst_1xk.shape[-1])
    if n:
        dst_1xk[0, :n].copy_(flat[:n])


def _fill_bytes(dst_1xn: torch.Tensor, src: torch.Tensor) -> int:
    dst_1xn.zero_()
    view = src.contiguous().view(torch.uint8).reshape(-1)
    n = min(view.shape[0], dst_1xn.shape[-1])
    if n:
        dst_1xn[0, :n].copy_(view[:n])
    return n


def _copy_i32_at(dst: torch.Tensor, row: torch.Tensor, value: int) -> None:
    _state.i32_tmp[0] = int(value)
    dst.index_copy_(0, row, _state.i32_tmp)


def store_indexer(
    *,
    layer_id: int,
    compress_ratio: int,
    compressed_seq_len: torch.Tensor,
    topk_length: int,
    use_topk_v2: bool,
    dispatch_id: int,
    raw: torch.Tensor,
    page: torch.Tensor,
    scores: Optional[torch.Tensor],
    layer_input: Optional[torch.Tensor],
) -> None:
    if _state is None or layer_id not in LAYER_INDEX:
        return
    li = LAYER_INDEX[layer_id]
    row = _row_index(li)
    raw_src = raw[0] if raw.dim() > 1 else raw
    page_src = page[0] if page.dim() > 1 else page
    _fill_i32_row(_state.raw_tmp, raw_src)
    _fill_i32_row(_state.page_tmp, page_src)
    _state.raw.index_copy_(0, row, _state.raw_tmp)
    _state.page.index_copy_(0, row, _state.page_tmp)
    score_width = 0
    if scores is not None:
        s = scores[0] if scores.dim() > 1 else scores
        s = s.reshape(-1).to(torch.float32)
        _state.score_tmp.zero_()
        n = min(s.shape[0], MAX_SCORE)
        if n:
            _state.score_tmp[0, :n].copy_(s[:n])
            score_width = n
        _state.score.index_copy_(0, row, _state.score_tmp)
    q_nbytes = 0
    if layer_input is not None:
        q_nbytes = _fill_bytes(_state.q_tmp, layer_input)
        _state.q_bytes.index_copy_(0, row, _state.q_tmp)
        _copy_i32_at(_state.q_n, row, q_nbytes)
    _state.meta_tmp.zero_()
    _state.meta_tmp[0, 0] = int(layer_id)
    _state.meta_tmp[0, 1] = int(compress_ratio)
    if isinstance(compressed_seq_len, torch.Tensor):
        _state.meta_tmp[0, 2] = compressed_seq_len.reshape(-1)[0].to(torch.int32)
    else:
        _state.meta_tmp[0, 2] = int(compressed_seq_len)
    _state.meta_tmp[0, 3] = int(topk_length)
    _state.meta_tmp[0, 4] = 1 if use_topk_v2 else 0
    _state.meta_tmp[0, 5] = int(dispatch_id)
    _state.meta_tmp[0, 7] = int(score_width)
    _state.meta_tmp[0, 8] = int(q_nbytes)
    _state.meta_tmp[0, 10] = int(_state.max_k)
    _state.meta_tmp[0, 11] = 1
    _state.meta.index_copy_(0, row, _state.meta_tmp)


def store_attn(
    *,
    layer_id: int,
    q: Optional[torch.Tensor],
    o: Optional[torch.Tensor],
    extra_indices: Optional[torch.Tensor],
) -> None:
    if _state is None or layer_id not in LAYER_INDEX:
        return
    li = LAYER_INDEX[layer_id]
    row = _row_index(li)
    if extra_indices is not None:
        page_src = extra_indices.reshape(-1, extra_indices.shape[-1])[0]
        _fill_i32_row(_state.page_tmp, page_src)
        _state.page.index_copy_(0, row, _state.page_tmp)
    if o is not None:
        o_nbytes = _fill_bytes(_state.o_tmp, o)
        _state.o_bytes.index_copy_(0, row, _state.o_tmp)
        _copy_i32_at(_state.o_n, row, o_nbytes)
    if q is not None:
        q_nbytes = _fill_bytes(_state.q_tmp, q)
        _state.q_bytes.index_copy_(0, row, _state.q_tmp)
        _copy_i32_at(_state.q_n, row, q_nbytes)
    _copy_i32_at(_state.attn_flag, row, 2)
    if li == len(LAYERS) - 1:
        _state.slot.add_(1)


def _sha256_bytes(buf: bytes) -> str:
    return hashlib.sha256(buf).hexdigest()


def _tensor_hash_cpu(arr: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(arr).tobytes())


def _pair_hashes(raw: np.ndarray, page: np.ndarray, topk_length: int) -> tuple[str, str, list, list]:
    n = min(int(topk_length), int(raw.shape[0]), int(page.shape[0]))
    r = raw[:n].astype(np.int32, copy=False)
    p = page[:n].astype(np.int32, copy=False)
    valid = (r >= 0) & (p >= 0)
    r = np.ascontiguousarray(r[valid])
    p = np.ascontiguousarray(p[valid])
    pairs = np.empty((r.size, 2), dtype=np.int32)
    pairs[:, 0] = r
    pairs[:, 1] = p
    order_h = _tensor_hash_cpu(pairs)
    if pairs.size:
        order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        set_h = _tensor_hash_cpu(np.ascontiguousarray(pairs[order]))
    else:
        set_h = _sha256_bytes(b"")
    return order_h, set_h, r.tolist(), p.tolist()


def harvest_after_decode(forward_batch) -> None:
    """Host-side: count real DECODE replays on DP5; D2H after 15 steps."""
    if _state is None:
        return
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None:
        return
    is_idle = bool(getattr(mode, "is_idle", lambda: False)())
    is_decode = bool(getattr(mode, "is_decode", lambda: False)())
    if is_idle or not is_decode:
        return
    bs = int(getattr(forward_batch, "batch_size", 0) or 0)
    raw_bs = getattr(forward_batch, "raw_batch_size", None)
    if raw_bs is None:
        reqs = getattr(forward_batch, "reqs", None)
        raw_bs = len(reqs) if reqs is not None else bs
    raw_bs = int(raw_bs or 0)
    if raw_bs <= 0 and bs <= 0:
        return
    try:
        from sglang.srt.runtime_context import get_parallel

        ps = get_parallel()
        dp = int(getattr(ps, "attn_dp_rank", getattr(ps, "dp_rank", -1)))
    except Exception:
        dp = -1
    if dp != 5:
        return
    _state.host_decode_steps += 1
    d = probe_dir()
    if d and _state.host_decode_steps <= 3:
        try:
            with open(d + "/harvest_tick.jsonl", "a") as fh:
                fh.write(
                    json.dumps(
                        {
                            "host_decode_steps": _state.host_decode_steps,
                            "slot": int(_state.slot.item()),
                            "bs": bs,
                            "raw_bs": raw_bs,
                            "dp": dp,
                            "rid": getattr(getattr(forward_batch, "reqs", [None])[0], "rid", None)
                            if getattr(forward_batch, "reqs", None)
                            else None,
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
    if _state.host_decode_steps < 15:
        return
    _dump_request(forward_batch)
    _state.reset_request()
    _state.req_seq += 1


def _dump_request(forward_batch) -> None:
    d = probe_dir()
    if not d or _state is None:
        return
    torch.cuda.synchronize()
    n_steps = int(min(int(_state.slot.item()), MAX_DECODE_STEPS))
    if n_steps <= 0:
        n_steps = _state.host_decode_steps
    n_steps = min(n_steps, MAX_DECODE_STEPS)
    n_rows = n_steps * len(LAYERS)
    raw = _state.raw[:n_rows].detach().cpu().numpy().reshape(n_steps, len(LAYERS), -1)
    page = _state.page[:n_rows].detach().cpu().numpy().reshape(n_steps, len(LAYERS), -1)
    score = _state.score[:n_rows].detach().cpu().numpy().reshape(n_steps, len(LAYERS), -1)
    q_bytes = _state.q_bytes[:n_rows].detach().cpu().numpy().reshape(n_steps, len(LAYERS), -1)
    o_bytes = _state.o_bytes[:n_rows].detach().cpu().numpy().reshape(n_steps, len(LAYERS), -1)
    meta = _state.meta[:n_rows].detach().cpu().numpy().reshape(n_steps, len(LAYERS), -1)
    o_n_arr = _state.o_n[:n_rows].detach().cpu().numpy().reshape(n_steps, len(LAYERS))
    q_n_arr = _state.q_n[:n_rows].detach().cpu().numpy().reshape(n_steps, len(LAYERS))
    attn_flag = _state.attn_flag[:n_rows].detach().cpu().numpy().reshape(n_steps, len(LAYERS))
    rid = None
    reqs = getattr(forward_batch, "reqs", None)
    if reqs:
        rid = getattr(reqs[0], "rid", None)
    rids = getattr(forward_batch, "rids", None)
    if rid is None and rids:
        rid = rids[0]
    out_dir = Path(d)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / "records.jsonl"
    rows = []
    for step in range(n_steps):
        for li, layer_id in enumerate(LAYERS):
            m = meta[step, li].astype(np.int32, copy=False)
            flags = int(m[11]) | int(attn_flag[step, li])
            if flags == 0:
                continue
            topk_length = int(m[3]) or _state.max_k
            csl = int(m[2])
            order_h, set_h, raw_list, page_list = _pair_hashes(
                raw[step, li], page[step, li], topk_length
            )
            score_width = int(m[7])
            if score_width > 0:
                use_n = min(score_width, MAX_SCORE)
                if csl > 0:
                    use_n = min(use_n, csl)
                score_h = _tensor_hash_cpu(score[step, li, :use_n])
            else:
                score_h = None
            q_n = int(q_n_arr[step, li] or m[8])
            o_n = int(o_n_arr[step, li] or m[9])
            layer_in_h = _sha256_bytes(bytes(q_bytes[step, li, :q_n].tobytes())) if q_n else None
            sparse_h = _sha256_bytes(bytes(o_bytes[step, li, :o_n].tobytes())) if o_n else None
            raw_valid = raw[step, li][:topk_length]
            raw_valid = raw_valid[raw_valid >= 0]
            raw_order_h = _tensor_hash_cpu(raw[step, li][:topk_length].astype(np.int32))
            raw_set_h = _tensor_hash_cpu(np.sort(raw_valid.astype(np.int32))) if raw_valid.size else _sha256_bytes(b"")
            rec = {
                "req_seq": _state.req_seq,
                "rid": rid,
                "decode_step": step,
                "layer_id": int(m[0]) or layer_id,
                "compress_ratio": int(m[1]),
                "compressed_seq_len": csl,
                "topk_length": topk_length,
                "actual_dispatch": DISPATCH_NAMES.get(int(m[5]), str(int(m[5]))),
                "metadata.use_topk_v2": bool(int(m[4])),
                "layer_input_hash": layer_in_h,
                "score_hash": score_h,
                "selected_set_hash": set_h,
                "emitted_order_hash": order_h,
                "raw_topk_order_hash": raw_order_h,
                "raw_topk_set_hash": raw_set_h,
                "raw_indices": raw_list,
                "physical_page_indices": page_list,
                "sparse_out_hash": sparse_h,
                "n_valid": len(raw_list),
                "flags": flags,
            }
            rows.append(rec)
    with rec_path.open("a") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
    summary = {
        "req_seq": _state.req_seq,
        "rid": rid,
        "n_steps": n_steps,
        "n_records": len(rows),
        "at": datetime.now(timezone.utc).isoformat(),
        "dp_rank": 5,
    }
    with (out_dir / "requests.jsonl").open("a") as fh:
        fh.write(json.dumps(summary) + "\n")
    _state.dump_count += 1
