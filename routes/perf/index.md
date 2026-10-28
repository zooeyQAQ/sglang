# perf

- repo: `zooeyQAQ/sglang`
- code_branch: `dev/perf`
- base_commit: `faa56e2b5c5c5dee60eb5dd7b6251944a21a6104`
- worktree: `/data/sgl_pr/worktrees/perf`
- status: active

## 当前结论

EXP-0001 **invalid**。冻结 `mem_fraction_static=0.80` 在该 SHA 原生 Engram GPU placement 下无法为 KV 留出空间（运行时 min viable 0.8731）。未改 mf、未开 host table，无计时与 profiler 证据。

## 未解决问题

72GB SM120 上该 SHA 的可启动 serving 信封（mf 或 Engram host-table）尚未在本路线测过。A/B/C/D 热点未知。候选进入修改阶段前需重新核对当时 upstream main 和相邻 issue/PR。

## 下一步

执行 [EXP-0002](experiments/EXP-0002/task.yaml)：TP8×EP4、Engram host table、decode CUDA graph、无 DSpark。

## 实验表

| 编号 | 终态 | 一句话结论 | 记录 |
| --- | --- | --- | --- |
| EXP-0001 | invalid | 冻结 mf=0.80 被原生 KV 地板拒绝（min 0.8731） | [result](experiments/EXP-0001/result.md) |
