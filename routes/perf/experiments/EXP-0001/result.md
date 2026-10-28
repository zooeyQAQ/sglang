# perf/EXP-0001

状态：invalid
计划：`d6e3a243de880df22194457397d382906773ff0f` `routes/perf/experiments/EXP-0001/task.yaml`
辅助脚本：`1bd1adaea05d43c8bb5bfc199997cf924b7e0fab` `routes/perf/experiments/EXP-0001/scripts/`
实际条件：`ml-pro5000-ser237.qgz02` 容器 `zooey_sglang0_5_22_dsv41_8005`；8× RTX PRO 5000 72GB SM120 driver 580.126.09；worktree `/data/sgl_pr/worktrees/perf` SHA `faa56e2b5c5c5dee60eb5dd7b6251944a21a6104` 干净；`PYTHONPATH` 指向该树 python；`/opt/sglang/bin/python3` torch 2.14.1+cu130、sgl_kernel 0.4.9、FlashInfer 0.7.1。冻结启动：TP8/EP8、`mem_fraction_static=0.80`、page 256、chunk 8192、max_running 8、radix 关闭、`127.0.0.1:30000`。原生解析：attention `dsv4`、MoE `flashinfer_mxfp4`、KV `fp8_e4m3`、Engram `placement=gpu`、decode CUDA graph `full` max_bs=8、prefill graph 关闭、decoder SWA bounded replay True。未改 mf、未开 host table、未装包。
证据：host `10.207.193.166:8005`；run 根目录 `/data/sgl_pr/runs/perf/EXP-0001/`；关键文件 `serve/serve.log` sha256 `d63fc83b1abb6235e7068a96a8e4e2ab66cd578363fed92668f97d384c60ead0`、`serve/launch_cmd.txt`、`identity/identity.json`；本机副本 `/Users/didi/Documents/pro5k/profiling/pro5000/DSv4.1-flash/pr/EXP-0001/`（无 profiler trace）。

## 执行与结果

| Case / Run | 代码 SHA | 实际配置或命令引用 | 关键结果 | 有效性/偏差 |
|---|---|---|---|---|
| serve / boot | faa56e2b5c5c5dee60eb5dd7b6251944a21a6104 | `serve/launch_cmd.txt`；pid 146751；04:03:23Z 启动 | 04:05:27Z 全部 TP rank 在 KV 配置阶段 `ValueError`：`--mem-fraction-static=0.8` 加载权重后无 KV 空间，要求高于 0.874（minimum viable 0.8731）。随后 sigquit。HTTP 从未 ready。 | invalid：冻结 mf=0.80 无法启动。按停止条件未改 mf / backend / SHA。 |
| A–D timing | — | 未运行 | 无 | 服务未 ready |
| A–D profile | — | 未运行 | 无 trace | 服务未 ready |

## 结论与局限

- 证明了什么：在本机 8×72GB SM120、该 SHA 原生 Engram GPU placement 下，冻结 `mem_fraction_static=0.80` 不能为 KV 留出空间（运行时声明 min viable 0.8731）。这是启动合同失败，不是性能数字。
- 尚未证明什么：A/B/C/D 吞吐、TPOT、8-rank prefill/decode 热点均未采集。包版本（sgl_kernel 0.4.9 / FlashInfer 0.7.1 vs pyproject 0.4.10 / 0.7.2）未单独作为失败判据。
- 偏差与停止：停止条件「OOM/崩溃，不得临场改配置」。orchestrator 按 30 分钟 ready 超时退出（`status.json` `serve_not_ready_30min`）；实际崩溃发生在启动后约 2 分钟。未占用其他路线服务。
