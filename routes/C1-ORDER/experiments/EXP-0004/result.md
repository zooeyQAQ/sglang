## Verdict

pass

## Metrics

N511 requests: 30 serial, pid 122554, all HTTP 200, cached_tokens=0, completion 16.
Token-4: 201 × 9, 43051 × 21. 8 distinct output_ids.
Probe records: 1920 = 30 × 16 decode slots × 4 layers. Harvest rank tp_rank=0, attn_dp_size=1, dp_rank=0.

N8193 requests: 2 serial, same pid, both HTTP 200, cached_tokens=0. Stopped on the C2 same-score same-set different-order stop condition.

Launch: TP8 EP4, no DP attention, `--mem-fraction-static 0.80`, Engram host table `per_rank` (layer 1 and 14, 12085 MiB/rank, 100% huge pages). GPU used ~60 GB / 72 GB after boot, `available_gpu_mem=12.26 GB`, `max_total_num_tokens=11422464`.

## Observation

Live L20 decode writer is `full_topk_deep_gemm_paged`. Ring leftover: request 0 step 2 is empty; first real N511 decode is step 2, `compressed_seq_len=513`, `n_valid=512`, 29 requests.

N511 L20 step 2 (`n_valid=512`):

- 1 score hash `d5a269f5698a…`
- 1 selected-set hash `25a2864690b5…`
- 20 emitted-order hashes
- 20 sparse_out hashes
- set_hash != order_hash on every filled request

Token-4 201 vs 43051 is later than that hidden-state split. L2/L8/L14 at N511 still `n_valid=0` (ratio-2 below a real Top-K screen on this graph variant).

N8193 C2 above K is observed. L2 first real decode (step 1, `csl=4097`, `n_valid=512`, `full_topk_deep_gemm_paged`):

| req | selected_set | emitted_order | score | sparse_out |
| --- | --- | --- | --- | --- |
| 0 | 141440e3b7f3 | 04502670b6ea | d1972d3a0a64 | 3e5fe0ec35ea |
| 1 | 141440e3b7f3 | de9ab6b57366 | d1972d3a0a64 | ccbfb0907f9f |

Same score, same set, two orders, two sparse_out. The same pattern holds on L2 steps 1–15. L8/L14/L20 at N8193 already differ in score at step 1 (downstream of the L2 order fork).

C2 causal AA/AB/BA/BB kernel replay was not run in this EXP. Tensors are in `probe/records.jsonl`.

mf=0.75 without host table refused (KV floor 0.860). mf=0.88 without host table refused (SWA 1.97 GB vs 1.42 GB). Host table + mf=0.80 booted.

## Conclusion

The EXP-0003 N511 L20 finding reproduces on TP8 DP1: same scores and the same selected set are emitted in many physical orders, and sparse_out follows the order. It is not a DP8 attn_tp=1 artifact.

N8193 runs on this launch. Ratio-2 L2, once `compressed_seq_len=4097 > K=512`, shows the same same-score same-set different-order pattern. Ratio-2 sources are not immune. No sort fix was applied. `allow_fused_implementation` stays false.

## Key findings

- N511 L20 first real decode: 1 score, 1 set, 20 orders, 20 sparse_out (29 filled requests).
- N8193 L2 first real decode: 1 score, 1 set, 2 orders, 2 sparse_out; C2 order fork observed.
- N8193 no OOM under TP8 + Engram host table + mf=0.80.
- No fused sort. Harvest dumped tp_rank==0.
