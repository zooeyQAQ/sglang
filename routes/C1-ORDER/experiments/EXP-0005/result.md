## Verdict

pass

First-round acceptance only: Stage 0 identity, historical C1 faithful replay, FP64 self-test, and FlashInfer vs Torch FP32 vs FP64 same-input metrics. V1 CUDA, permutation matrix, current-tree C1/C2 capture, and graph-100 causal matrix are `not_run`.

## Metrics

Fixture: pd_transfer EXP-0025 L20 C1 pair, req0000 vs req0002, decode step 1, H=64, SWA 128, extra 512, softmax_scale=0.04419417382415922, head_dim_v=512.
SHA `54e84fa9f3`. FlashInfer 0.7.1. Worktree `flash_mla_sm120.py`. GPU0, CUDA_VISIBLE_DEVICES=0.

Faithful replay on this module: AA/BB bitwise match production `op_sparse_out` for fill 0x00 and 0xA5. FlashInfer fixed-order 20 repeats bitwise stable. Classification `order_causal_replay_pass`.

FP64 oracle self-test: Q=0 no-sink / Q=0 sink=0 / padding change; scaled max error 5e-15, `pass_1e9=true`. M_valid=640.

S_order (A vs B, extra order only):

| Variant | max_abs | rel_l2 | bitwise equal |
| --- | ---:| ---:| --- |
| production / FlashInfer | 0.0078125 | 0.002325 | no |
| Torch FP32 then BF16 | 0.001953125 | 4.09e-5 | no |
| Torch FP32 (pre-cast) | 9.54e-7 | 1.73e-7 | no |
| FP64 then BF16 | 0 | 0 | yes |
| FP64 | 0 | 0 | (values match; shape/view) |

E_ref vs independent FP64 cast to BF16, request A:

| Variant | max_abs | rel_l2 |
| --- | ---:| ---:|
| FlashInfer | 0.03125 | 0.00962 |
| Torch FP32 then BF16 | 0.001953125 | 4.21e-5 |
| Torch FP32 vs FP64 | 2.38e-6 | 3.37e-7 |

FlashInfer vs production: bitwise equal.

## Observation

`merge.cuh` reads `const bf16* mid_out`. `fp8_decode/decode.cuh` writes split partials with `__floats2bfloat162_rn` from FP32 accumulators. Python allocates `mid_out` as BF16.

The existing Torch fallback dequants to BF16, then runs QK / softmax+sink / PV in FP32 (TF32 off). Independent FP64 decodes FP8×e8m0 and rope BF16 to FP64 and is permutation-invariant on this pair after the final BF16 cast.

Production extra error versus FP64 is real. Full FP32 compute (Torch) cuts S_order max_abs 4× (0.0078125 → 0.001953125) and rel_l2 ~57×. That path changes QK, softmax, and PV together, so it does not isolate BF16 split-K partials (V1) or FP8 PV (V2).

## Conclusion

Historical C1 fixtures bitwise-restore on `54e84fa9f3` FlashInfer. The production kernel is order-sensitive at 0.0078125 max_abs; the FP64 oracle is not. Torch FP32 improves both error-to-reference and order sensitivity. V1 CUDA (FP32 `mid_out` only) is justified next and remains `not_run` in this first round.

## Key findings

- Stage 0: FlashInfer 0.7.1, DSV4 decode dispatch, `mid_out` BF16 in producer and merge.
- Historical C1 replay faithful; no recapture required for this fixture.
- Production extra error vs FP64 confirmed (max_abs 0.03125, S_order 0.0078125).
- V1/V2/V3 CUDA `not_run`. Current-tree C2 capture `not_run`.
