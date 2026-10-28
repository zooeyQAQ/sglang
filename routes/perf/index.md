# perf

- repo: `zooeyQAQ/sglang`
- code_branch: `dev/perf`
- base_commit: `faa56e2b5c5c5dee60eb5dd7b6251944a21a6104`
- worktree: `/data/sgl_pr/worktrees/perf`
- status: active

## 当前结论

EXP-0001 **invalid**（冻结 mf=0.80 + Engram GPU，KV 地板）。
EXP-0002 **invalid**（host table / CPU Engram 越过 KV，随后 `dsv4_attn_backend=flashmla` 需要镜像没有的 `flash_mla` 包）。无计时与 profiler 证据。

## 未解决问题

本 0.5.22 镜像缺少 `sgl-flash-mla`。当前 main 的 dsv4 路径在 SM120 上默认 `flashmla` 并硬导入该模块。A/B/C/D 热点未知。

## 下一步

需要带 `sgl-flash-mla` 的镜像/wheel，或改用仍走 FlashInfer、不导入 `flash_mla` 的代码基点。不要在 EXP-0002 上装包补跑。

## 实验表

| 编号 | 终态 | 一句话结论 | 记录 |
| --- | --- | --- | --- |
| EXP-0001 | invalid | 冻结 mf=0.80 被原生 KV 地板拒绝（min 0.8731） | [result](experiments/EXP-0001/result.md) |
| EXP-0002 | invalid | host table 越过 KV 后缺 `flash_mla` 模块 | [result](experiments/EXP-0002/result.md) |
