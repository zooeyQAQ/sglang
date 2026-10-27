## Verdict

pass

## Metrics

N511 requests: 3 serial, pid 107232, all HTTP 200, dp_rank=5, cached_tokens=0, completion 16.
Token-4: 15, 15, 32.
Probe records: 192 = 3 × 16 decode slots × 4 layers.

N8193 requests: 0 completed. First request on pid 107232 and a cold retry on pid 110801 both died in SM120 FlashMLA prefill `new_empty` 512 MiB OOM (GPU5, leftover ~0.35–1.3 GiB under frozen `--mem-fraction-static 0.97`).

## Observation

Live L20 decode on this CUDA-graph variant is `full_topk_deep_gemm_paged` (`topk_transform_paged_from_metadata`), not `select_decode`+`finish_paged_indexer_topk`. Ring slots 0–1 are empty leftover; first real decode is slot 2 with `compressed_seq_len=513` and 512 valid pairs.

L20 slot 2 (first real decode, before token-4):

| req | token-4 | selected_set | emitted_order | score | sparse_out |
| --- | --- | --- | --- | --- | --- |
| 0 | 15 | 53308bb0faf6 | 058c1848839f | 80a07c1bea9f | 43ac0c2e5bcf |
| 1 | 15 | 53308bb0faf6 | 0295e13acd74 | 80a07c1bea9f | a7b56ebc623d |
| 2 | 32 | 53308bb0faf6 | 238084bd5f35 | 80a07c1bea9f | 0aeb5fc5d1f5 |

Same score hash, same set hash, three different emitted-order hashes, three different sparse_out hashes. set_hash != order_hash.

L20 slot 4 (`csl=515`, the decode that emits token-4): req0/req1 still share set+score and differ in order+sparse_out; req2 (token 32) has a different set and score.

L2/L8/L14: 48 records each, all `n_valid=0`. No C2 top-k pairs were written. This graph variant does not exercise ratio-2 as a real Top-K at N511. Later C2 sparse_out hashes split only after L20 has already forked.

N8193 C2 order change: not observed (OOM). That is not a claim that C2 cannot fork.

## Conclusion

On this SM120 candidate-enabled tree, the earliest measured N511 difference is at layer 20, the only ratio-1 source, once its compressed length is above K=512. The live writer is full Top-K paged transform. Same scores and the same selected set are emitted in different physical orders, and sparse_out follows the order. Ratio-2 sources at layers 2/8/14 stayed below a real Top-K screen in this N511 graph variant. N8193, which would put C2 above K, did not run on the frozen EXP-0002 launch (prefill temp OOM). A later EXP with more KV/prefill headroom is required before a C2-only vs universal Top-K contract can be decided.

In the candidate-enabled configuration, layer 20 is also the candidate source. The captured decode graph used the full-topk path rather than dense-blocks `publish_decode`, so this trace is a ratio-1 boundary measurement on the graph variant that actually served the requests.

## Key findings

- N511 L20 first real decode: same score, same selected set, three emitted orders, three sparse_out tensors.
- Token-4 15 vs 32 is later than that hidden-state split.
- N511 C2 top-k pairs were not recorded (`n_valid=0`); not a C2 immunity result.
- N8193 C2 order fork: not observed (OOM on frozen 0.97 launch).
- No sort fix was applied. `allow_fused_implementation` stays false until C2 above K is measured.
