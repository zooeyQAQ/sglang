# perf/EXP-0002

状态：invalid
计划：`ab5fa736a3ddcb50e4242a5ca9facd1405f33ac4` `routes/perf/experiments/EXP-0002/task.yaml`
实际条件：`ml-pro5000-ser237.qgz02` 容器 `zooey_sglang0_5_22_dsv41_8005`；worktree `/data/sgl_pr/worktrees/perf` SHA `faa56e2b5c5c5dee60eb5dd7b6251944a21a6104` 干净；TP8×EP4、无 DSpark/DP/PD；`mem_fraction_static=0.75`；Engram host table env 被本 SHA 翻译为 `engram_config.placement=cpu`；decode CUDA graph `full` max_bs=32；prefill graph 关；attention `dsv4` 解析 `dsv4_attn_backend=flashmla`；MoE `flashinfer_mxfp4`；fp8 gemm `flashinfer_cutlass`；`SGLANG_SM120_FLASHMLA_BACKEND=flashinfer` 已导出，未阻止 `import flash_mla`。镜像有 FlashInfer 0.7.1（含 sm120f jit cache），无 `flash_mla` / `sgl-flash-mla` 包。未装包。
证据：host `10.207.193.166:8005`；`/data/sgl_pr/runs/perf/EXP-0002/`；`serve/serve.log` sha256 `4069695b4327799cd11bc776d086b4f0c5488ab7edecf1b0b1c5e8b17c8c02ac`；本机副本 `/Users/didi/Documents/pro5k/profiling/pro5000/DSv4.1-flash/pr/EXP-0002/`。无 profiler trace。

## 执行与结果

| Case / Run | 代码 SHA | 实际配置或命令引用 | 关键结果 | 有效性/偏差 |
|---|---|---|---|---|
| serve / boot | faa56e2b5c5c5dee60eb5dd7b6251944a21a6104 | `serve/launch_cmd.txt`；pid 153870；05:19:44Z | 05:19:52Z Engram `placement=cpu`。权重与 KV 配置走完。05:33:58Z 全部 rank 在 `deepseek_v4_backend` 初始化 `import flash_mla` → `ModuleNotFoundError`。sigquit。 | invalid：缺失 ABI。按停止条件未 pip 安装。 |
| A–D timing / profile | — | 未运行 | 无 | 服务未 ready |

## 结论与局限

- 证明了什么：在本镜像上，该 SHA 的 host-table/CPU Engram + mf=0.75 能越过 EXP-0001 的 KV 地板。随后 dsv4 默认 `dsv4_attn_backend=flashmla` 需要 `sgl-flash-mla` wheel（`flash_mla` 模块），镜像没有。`SGLANG_SM120_FLASHMLA_BACKEND=flashinfer` 只作用于已导入的 SM120 路径，挡不住顶层 `import flash_mla`。
- 尚未证明什么：A/B/C/D 计时和 8-rank profiler 未采集。decode CUDA graph 未实际 capture。
- 偏差与停止：停止条件「缺失 API/ABI、不得装包」。`status.json` `serve_pid_died_before_ready`。EXP-0001 无效 run 目录已按人要求删除；labbook EXP-0001 结果保留。
