# EXP-0005 建议计划：SM120 sparse attention 的排列敏感性与数值误差归因

日期：2026-10-27  
状态：planned；本文仅为实验合同，尚未执行  
路线：C1-ORDER  
建议实验号：EXP-0005；执行前检查是否已占用，已占用则使用下一空号。  
目的：判断真实 C1/C2 输入上的输出差异，主要来自 BF16 split-K 中间舍入、分块 FP8 PV 量化，还是索引/布局/同步错误，并评估从 attention 算子内部改进的可行性。

## 1. 本轮要回答什么

1. 在固定 Q、sink、SWA、extra KV、有效长度和选中集合后，仅改变 extra indices 排列，能否重现现场输出差异？
2. 同一组已经量化存储的 KV，以 FP64 计算 attention 时，当前 kernel 偏离参考值多少？排列变化造成的误差占多少？
3. 仅提升 split-K 中间输出精度，排列敏感性是否明显下降？
4. 保持 QK/softmax 不变，仅替换 NoPE PV 中的 FP8 权重量化/乘加，是否明显改善？
5. 改善数值精度后，结果是“更准确但仍随排列变化”，还是“本批排列已逐位一致”？两者分别报告。

本轮不把 greedy token 翻转等同于算子算错，也不预先要求随机排列必须 bitwise equal。正常浮点归约也可能受排列影响。

## 2. 已知事实与待证假设

### 2.1 已知事实

| 数据 | 已有结论 | 本轮用途 |
| --- | --- | --- |
| pd_transfer EXP-0026，旧 C1 请求对 | 布局忠实的 AA/AB/BA/BB 回放成立，排序后 AS=BS | 优先复用离线数据、工具和严格布局校验 |
| C1-ORDER EXP-0003，DP8 N511 | L20 同 score、同集合、不同排列，sparse_out 不同 | 既有现象，不重复大规模发请求 |
| C1-ORDER EXP-0004，TP8 DP1 N511 | 29 个有效观察出现 20 种排列及 20 种 sparse_out | 当前树 C1 现象 |
| C1-ORDER EXP-0004，TP8 DP1 N8193 | L2 C2，csl=4097，512 个有效项；同 score/集合，不同排列和输出 | 当前树 C2 现象；完整因果回放待补 |

EXP-0004 的 records.jsonl 只有索引列表、元信息和摘要，不等同于完整 Q/KV/scale fixture。缺少完整张量时，不能仅凭它进行因果 replay。

### 2.2 假设

| 假设 | 检验方式 | 不足以证明它的现象 |
| --- | --- | --- |
| H1：BF16 中间结果舍入放大排列差异 | 同一生产计算，仅把 partial output 的保存和读取改成 FP32 | 只比较最终 BF16 输出不同 |
| H2：按候选块重新量化 PV 权重导致明显敏感性 | 保持 QK、online softmax 和分组不变，替换 NoPE PV 精度；比较块内/跨块排列 | 跨块打乱误差较大本身，只是线索 |
| H3：主要是浮点归约/最终 BF16 舍入 | 高精度 PV + FP32 partial 后残差很小，且接近参考 | 默认实现通过较宽 allclose |
| H4：mask、scale、索引或同步存在实现缺陷 | padding/未读字节填充测试、固定顺序重复、参考值及安全检查 | 任意排列下产生几个不同 BF16 数值 |

在核对过的 FlashInfer v0.7.1 源码里，decode partial output 为 BF16，NoPE PV 会折叠 KV scale 并按候选 tile 量化 FP8 权重。现场运行的模块是否对应这份源码，必须在 Stage 0 核实。

## 3. 工作范围与代码身份

沿用用户指定的环境：

- 166 的 8005 容器：zooey_sglang0_5_22_dsv41_8005。
- SGLang worktree：/data/sgl_pr/worktrees/c1-index-order。
- 分支：dev/c1-index-order。
- 当前已知实验提交：54e84fa9f38fbea162af599222370961a1d2c7d7。
- v0.5.22 基点：baa9befa381511cd4d56f83980951e5a45fffc6d。
- prerequisite：上游 PR #42371，2f2173c1191ad4da45fcd3046d025c6ba3707a2b；本地 cherry-pick 6ae2f5c2ef8557fdf697c36f21a07ef39cf6e0ae。
- 旧 C1 replay：0de2000c3ba9503140ab31a3cc9e90a64c4323c6。

执行前只读检查实际 HEAD、工作区和 GPU 占用。若存在用户未提交修改，保留并记录；不得自行清理。旧 EXP tag/result 保持，结论更正使用追加记录。

SGLang 中只加入必要的采集和测试代码。FlashInfer 诊断变体使用独立源码副本/独立 JIT 模块名称与缓存；不覆盖系统安装，不全局 pip upgrade，不让 baseline 误加载修改后的 kernel。每个变体的源码、编译参数和产物身份可追溯。

本轮不改 Top-K 算法，不加生产 canonical sort，不把诊断高精度分支设为默认。

## 4. Stage 0：身份、源码与资源预检

### 操作

1. 记录 SGLang HEAD、父提交、diff、导入路径及对应文件。
2. 使用白名单收集 Python、Torch、CUDA runtime、driver、nvcc、sgl-kernel、FlashInfer、DeepGEMM、GPU 型号/SM。
3. 记录 FlashInfer 实际导入文件、实际加载的 JIT/AOT 模块、源码来源与本地修改。版本号 0.7.1 不能替代模块身份。
4. 定位真实调用：SGLang wrapper、FlashInfer dispatcher、decode kernel 和 merge kernel。记录真正的 KV model/layout 类型，不能仅凭模型名 DeepSeek-V4.1 选择 FlashInfer 的 DSV4_1 新格式。
5. 记录实际 Q shape/dtype、真实头数与 padded/scratch 头数、page size/stride、有效长度、BI/tile 宽度、chunks_per_block、allocated/active splits、执行 stream 和 graph key。
6. 记录启动前后实际环境选择，特别是 metadata.use_topk_v2；shell export 不代表最终 dispatch。
7. 只复用空闲 GPU。模型采集最多使用一个 standalone 服务；离线回放使用一张空闲卡即可。

### 验收

- 能明确指出当前模块中，哪一处发生 FP8 PV 量化、哪一处写 BF16 partial、哪一处合并。
- 运行模块与所审计源码对应，或已明确差异并据现场源码修订以下变体；不得套用不兼容 ABI。
- 不需要从“最新 main”迁移其他优化才能继续。
- 若代码身份、布局或加载模块无法确认，分类 identity_or_dispatch_unresolved，停止 kernel 改动。

## 5. Stage 1：准备真实 fixture，优先复用，必要时只采一次

### 5.1 数据优先级

1. 检查旧 EXP-0025/0026 完整 .pt/.json 是否仍可读。
2. 检查 EXP-0004 归档是否另有完整张量；只有 records.jsonl 不够。
3. 若旧 C1 fixture 在当前生产模块上不能逐位还原，标记 historical_replay_mismatch，改采当前 C1；不得调宽容差混过去。
4. C2 缺完整数据时，在 EXP-0004 可启动配置上补采 L2。

新采集启动沿用 EXP-0004：

- PP1 / TP8 / attention DP1 / EP4。
- mf=0.80；Engram host table=on，layout=per_rank。
- FlashInfer SM120 attention；FP8 KV，page size 256；MoE/GEMM 后端沿用。
- decode graph on、max bs 8；prefill graph off；radix off；overlap off；DSpark off。
- 保留已归档的全部有效启动 env，包括 SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0。
- 启动 CLI 的 chunked-prefill-size 为 65536；记录其实际解析值，不沿用 DP8 时的 8192 假设。
- 不混入 PP2、fake-PD 或 SWA bounded replay。

### 5.2 采集预算与停止

- 一个正式 cold start。
- N511 最多 10 次，N8193 最多 6 次；均固定原始 input_ids、temperature=0、16 outputs、ignore_eos=true，串行、cached_tokens=0。
- 取得一个合格当前 C1 对、一个合格当前 C2 对即停止；若已有合格 C1 fixture，只采 C2。
- 不要求输出 token 等于历史 15/32 或 201/43051。
- 非预先核实的启动失败不自动重启；模型 OOM/assert 后停止并归档。若仍沿用旧实验的一次 RPC 端口重拉约定，必须在本次启动前明确记录其适用条件，不能临场追加。
- 上限内未取得合格对，分类 capture_inconclusive；不无限加请求。

### 5.3 精确采集位置

- C1：N511，L20，优先 csl=513 附近；以实际 positions/csl 和 consumed prefix 关联。
- C2：N8193，L2，优先 csl=4097；比较处两请求 consumed prefix 必须相同。
- 主采 TP0。此次只对该局部算子建立因果证据，单个 rank 阳性足够；不声称覆盖所有 rank 或全模型首差。
- 记录实际 DECODE row、graph variant、request/trial、TP/DP rank、位置和本次采集序号。不能把 ring 下标直接当 decode step。

### 5.4 每个 fixture 必备内容

| 类别 | 内容 |
| --- | --- |
| 查询 | 原始 Q 全值、dtype/shape/stride、softmax_scale、head_dim_v、attn_sink |
| SWA | kernel 实际 indices、有效长度、实际读到的完整 KV 页或逐行 data+scale、布局和页 stride |
| C1/C2 | raw logical indices、physical indices、有效长度、原始排列、实际读取的完整页或 data+scale |
| 输出 | 生产 sparse_out 原始张量；可取得时保存 LSE、partial 输出及 split 计划 |
| 身份 | 输入 token fixture、已消费生成前缀、positions、layer/ratio、rank、graph key、实际内核与源码身份 |
| 完整性 | 必备字段检查、shape/边界检查、原始文件校验信息 |

raw indexer scores 与 attention 的 QK logits 是两回事。前者相同不能替代 Q、attention cache payload 和 sink 相同。

### 5.5 采集实现注意事项

- 预分配 GPU ring；图内只 D2D copy，请求结束后 D2H。
- layer_input、attention Q、sparse_out 使用独立缓冲。修复旧探针中 Q 覆盖 layer_input 的命名/存储问题。
- capture/warmup 完成后，在合法边界清空计数与槽位；禁止遗留前两槽再靠推测对齐。
- ring 超限必须显式标 invalid，不能 clamp 后重复覆盖最后槽。
- 数据副本须在其生产 stream 上、读写完成的正确位置取得；跨 stream 必须有对应 event 依赖。
- 原始 tensor 的最后一次写入与快照 copy 的先后必须可审计。不得为了方便在算子前后插同步并把新调度当作原服务测量。
- 不只保存摘要；需能离线重新构造实际算子调用。
- 文件名包含 host/rank/trial/layer/position，避免跨请求覆盖。

## 6. Stage 2：布局忠实 replay 与 C2 因果闭环

### 6.1 合格请求对门槛

比较 A/B：

- Q、sink、scale、SWA 选择与有效 payload 完全相同。
- extra 的有效 (raw, physical) 多重集合相同；按 raw 对齐后，data/scale 相同。
- extra order 不同，生产 sparse_out 不同。
- 重复项不得被静默去重，padding 不算有效项。
- 如果物理映射不同，不直接将 B 的物理列表放入 A 的 cache。优先另找符合条件的对；若用逻辑排列映射到 A，只可标为额外的单输入敏感性实验，不能直接声称 AB 等于生产 B。

### 6.2 重建

- 保留实际 physical slot、页大小、页 stride、视图 stride、dtype 和 cache format。
- 不把 C1 的 149760-byte 页 stride 套给 C2；按现场布局决定。
- V4 的 data/scale 是分平面存储；不能把每个 token 当作连续 584 B。
- 重建后，用独立地址计算回读全部有效 data/scale，与快照逐字节比较。
- 仅需保存被读页，但重放 buffer 必须容纳原 slot。若原 slot 导致单卡容量超预算，先报告；不能无门槛地压紧 slot。
- 估算所需字节并预留至少 4 GiB。离线无需完整模型和全部 KV 池。

### 6.3 忠实性与未读字节测试

对未观测区域分别填充 0x00 与 0xA5，再植入所有实际有效 payload：

- A 原排列必须逐位匹配生产 A。
- B 原排列必须逐位匹配生产 B。
- 两种填充下均成立，输出有限。
- 若填充改变输出，先查遗漏读取、mask 或布局；此时不做正常数值误差归因。

### 6.4 因果矩阵

| Case | 基础输入 | extra 排列 | 预期 |
| --- | --- | --- | --- |
| AA | A | A | 逐位等于生产 A |
| AB | A | B | 在全部其他输入相同的前提下，逐位等于生产 B |
| BB | B | B | 逐位等于生产 B |
| BA | B | A | 逐位等于生产 A |
| AS / BS | A / B | 相同的 raw-position canonical 顺序 | 彼此逐位相同 |

每 case：eager 100 次；一个 graph 静态 indices buffer，通过 copy_ 更新后 replay 100 次。单一固定排列的重复输出必须稳定。

### 验收与分类

- 忠实性和交换全部满足：order_causal_replay_pass。
- 忠实性成立、换序有影响，但不能交换生产 A/B：order_sensitive_other_inputs_unresolved。
- 固定排列自身不稳定：fixed_order_nondeterministic；转查同步、读写或未初始化数据。
- 原样不匹配：replay_not_faithful。
- 门槛未过，不进入精度变体比较。

## 7. Stage 3：独立 FP64 参考

### 7.1 参考目标

参考输入是同一份已量化缓存解码得到的 KV，不使用原 checkpoint BF16 权重或“量化前 latent”。这样不会把缓存存储量化误差误算为 attention kernel 的额外损失。

为每个 Q head 计算：

1. 按正确布局解码 SWA 与 extra 的 K/V 为 FP64。
2. 依据实际长度和 -1 mask 取有效项。
3. 计算 z_i = softmax_scale * dot(Q, K_i)。
4. 用稳定 softmax 计算分母；若有 sink，将 exp(sink_h) 加入分母，sink 不贡献 V。
5. 计算 weighted sum。K/V 的选择、head_dim_v 和 RoPE/NoPE 内容按真实模型格式处理。
6. 同时保留 FP64 输出以及最后一步转成 BF16 的输出。

SWA 和 extra 是两个来源的 attention 项，不能因 logical position 重叠而合并去重。保留每个来源原本的重复次数。

### 7.2 独立性

- 优先 CPU FP64 reference，单个算子的规模足够小。
- layout 解码与生产 helper 独立实现并互相核验，避免相同错误自证。
- 固定 canonical 顺序只用于 reference 汇总和一个诊断对照，不接入生产路径。
- 为量化 PV 的局部归因，另保存生产 QK/softmax 的 shadow trace；不要把 ideal FP64 reference 中包含的 Q 重新量化误差，全归给 PV。
- internal LSE 可能为 log2，reference 默认为自然对数。先按模块约定换算再比较。
- mask 引入的 -inf 合法；验收要求有效输出无 NaN/Inf，不能把所有中间 -inf 当错误。

### 7.3 参考自检与负控制

- Q=0、无 sink：输出应为有效 V 的平均值。
- Q=0、sink=0：M 个有效项的均值乘以 M/(M+1)。
- 常量 V、不同合法排列：符合解析式。
- masked padding 内容变化，不改变参考输出。
- 改一个具有非零权重的有效 V，reference 必须发生可检测变化。
- MLA 的 K/V 常由同一份缓存 latent 提供。若改动真实缓存字节会同时改变 K 和 V，reference 必须同步重算 QK 与 PV；不能把这种负控制描述成“仅改 V”。仅改 V 的验证限于显式分开的 reference/shadow 输入。
- 改 sink，分母及输出按定义变化。
- 对同一数据按 A/B/逆序执行 FP64 汇总，参考排列差异须远小于生产差异。

参考自检门槛：解析式 FP64 检查使用 scaled max error <= 1e-9、relative L2 <= 1e-10。近零参考使用绝对误差。若输入病态导致未通过，记录原因并提高参考汇总精度/方法，不放宽生产结论。

## 8. Stage 4：排列矩阵

只改变 extra 列表，SWA 列表固定；raw/physical pair 成对移动，KV 不搬动，有效长度和 padding 位置保持。

冻结随机种子 4102，先生成并保存全部 permutation，再跑数值测试。

| 家族 | 最多数量 | 变化 | 目的 |
| --- | ---:| --- | --- |
| 现场与基准 | 4 | A、B、canonical、逆序 | 连接实际差异 |
| tile 内打乱 | 8 | 每个实际 BI tile 的成员集合不变 | 检查 tile 内归约、MMA 顺序 |
| split 内 tile 换位 | 4 | 每个 split 成员不变，只改变其中 tile 遍历 | 检查 online softmax/rescale |
| extra-only 整 split 换位 | 4 | split 内成员与顺序不变，只交换等长 split | 检查 split 合并的顺序影响 |
| 跨 tile 完全打乱 | 16 | 任意有效 extra pair 重新排列 | 同时改变 tile 成员和分块量化尺度 |

总数最多 36 个；去重后按实际数量报告。

先读取真实 BI、chunks_per_block 和 tile-to-split 映射。64 项 tile 不一定等于一个 split。某家族在当前计划下不存在（例如 CPB=1 没有 split 内 tile 换位）则记 N/A，不构造假对照。

同一 fixture/变体冻结 split 计划；不让 autotune 或 CPB 在排列之间变化。额外测试不同 CPB 只能作为独立轴，不混进主要比较。

每个排列、每个可用 variant：eager 100 次，graph 100 次，检查固定排列稳定性。C1 与 C2 分别汇总，不混为一个总体数字。

“跨 tile 差异大于 tile 内差异”仅为分组效应证据，必须结合下一节定点变体才能归因量化。

## 9. Stage 5：2×2 算子精度对照

使用如下四个版本，V0 始终保留生产实现：

| Variant | NoPE PV | split-K partial 输出 | 改变范围 |
| --- | --- | --- | --- |
| V0 | 原生产 FP8 权重量化/FP8 MMA | BF16 | baseline |
| V1 | 与 V0 相同 | FP32 | 只消除 partial 保存的 BF16 舍入 |
| V2 | 使用同一 FP32 softmax 权重，解码 V 后用高精度 PV 诊断路径 | BF16 | 只替换 NoPE PV |
| V3 | 与 V2 相同 | FP32 | 同时移除以上两项低精度因素 |

### 9.1 分步开发

1. 先完成 V0、reference、排列矩阵和 V1。
2. V1 归因不充分时，再做 V2/V3。V2 先用保存的生产 QK/P 做 shadow PV 计算；有明显信号再实现可回放的诊断 kernel。
3. 若 V2/V3 未实现或只有 shadow，明确标 not_run / shadow_only；不能声称已经定位 PV 根因或可以报性能。
4. 无论 V1 结果如何，要声称“FP8 PV 不是贡献者”仍需 V2/V3 直接对照；不能靠排除法。

### 9.2 V1 注意事项

- 生产 kernel 累加逻辑不变，改变的是 partial 写出/保存/读取精度。
- producer 指针、FFI 检查、scratch 分配、stride/字节数、merge 读取必须一起改。
- 仅将 Python tensor dtype 换为 FP32 而 C++ 仍当 bf16 指针读写，是无效且危险的对照。
- mid_lse 保持原精度；split 数、CPB、sink、最终 BF16 cast 不变。
- NoPE 和 RoPE 两段 partial 都必须一致处理。
- H=8 专用与 H=64/padded 头数的 scratch stride 分别校验。

### 9.3 V2/V3 注意事项

- QK、Q 的量化、softmax_scale、running max/sum、mask、split 计划保持原样。
- 使用同一 w_pre 与同一 V 字节/scale，只替换 NoPE 的 PV 近似；不能顺手改变 softmax 分母。
- 保存 QK/running max/sum，验证 V0/V2 与 V1/V3 在这些边界逐位相同。
- 增加中间值导出的 V0 版本必须先逐位匹配未加导出的生产 V0；若导出改变了计算或时序结果，该 shadow 不能直接用于归因。
- 原 RoPE PV 暂保持不变，分别报告 NoPE/RoPE 的输出误差；如果残差主要在 RoPE，再开明确的独立轴。
- 使用 FP32 PV shadow 或 scalar 诊断路径是允许的。它不代表可部署的快路径，也不用于声称吞吐收益。
- 不直接套用要求另一种 KV format 的新 BF16 API，不把 FP32 Q 传入只接受 BF16 指针的接口。

### 9.4 固定随机性与安全

所有 variants 使用相同输入和排列；预热/编译在测量前完成。对原始 input buffer 做前后比对，确认未被 kernel 修改。若出现非法访问、guard 被改写、固定顺序不稳定，停止数值归因并转安全/同步检查。

## 10. 指标与预先冻结的解释门槛

令 R 为固定 canonical FP64 参考，O_p 为排列 p 的输出。分别在最终 BF16 输出和能导出的最终 cast 前 FP32 值上计算：

- E_ref(p) = norm(O_p - R) / max(norm(R), 1e-12)。
- S_order = max over p,q norm(O_p - O_q) / max(norm(R), 1e-12)。
- max_abs：对参考以及排列间两种差异分别报告。
- 每 head、全张量、NoPE/RoPE 分段的 relative L2。
- 与 R 转 BF16 后不相同的元素比例；可附正确处理符号与指数的 BF16 ULP 距离，不能把 0.0078125 当作全值域固定 ULP。
- 固定排列的不同输出数量、跨排列不同输出数量。
- 参考误差的 median、p95、max；不只挑最好的排列。

所有“明显改善”阈值在看到 variant 结果前固定。以下是本实验工程门槛，不是 SGLang/FlashInfer 的官方容差：

| 结论 | 本实验判定 |
| --- | --- |
| 某机制有明显排列敏感性贡献 | 对该机制做唯一改变后 S_order 至少下降 4 倍，并且其他阶段一致性检查通过 |
| 值得继续优化的数值候选 | S_order 至少下降 4 倍；p95 E_ref 至少下降 2 倍；各排列 E_ref 没有超过 baseline 对应值 10% 的显著回退 |
| 只改善重复性敏感度 | S_order 下降，但 E_ref 未明显下降；准确命名，不叫整体精度修复 |
| 只提高准确度 | E_ref 下降但最终 BF16 仍有多个输出；不叫确定性修复 |
| 本批排列逐位稳定 | 全部已测试排列输出相同，固定排列的 eager/graph 100 次一致；只对本批 fixture/shape 成立 |
| 未见明显改善 | 不满足前述效果门槛，如实记录；不能放宽阈值“做成 pass” |

若 baseline S_order 或 E_ref 已接近参考数值噪声，比例门槛无意义，改报绝对差异与上界，分类 contribution_unresolved。

还应按现场对应版本的原有 correctness tests 判定是否违反现有契约。核对的 v0.7.1 DSv4 测试中存在 atol=5e-2、rtol=5e-2；必须运行实际适用的 DSv4/dual-cache 用例，不能拿其他模型的阈值套过来。通过已有容差不等于逐位稳定；超出自行设定的 1 ULP 也不自动证明上游 bug。

## 11. 性能与 graph 验证

只在忠实性、reference 和数值归因通过后运行；慢的诊断版本也可完成归因，不因性能差而抹掉数值结论。

- 先测真实 fixture 的 B=1 和真实 H。需要时补 H=8/H=64 的合法独立输入。
- B=8/32 作为补充 microbench，明确是构造 batch，不视为真实 serving 吞吐。
- correctness/trace 与 timing 分开；timing 关闭影子导出、日志和 host transfer。
- JIT 完成、预热至少 20 次；每版本至少 5 组，每组 200 次 graph replay，交错 V0/候选测量。
- 报告 p50/p95、显存 scratch 字节、graph capture/replay 是否成功、是否出现新增生产 launch。
- 控制变量包括 L2 warm/cold 口径、输入缓冲、时钟/温度状态。主结果注明 warm replay；不擅自修改 GPU clocks。
- copy_ 更新 indices 在正确性检查中必须真实执行；性能测量如不包含 copy_，baseline/候选口径一致且明确说明。
- 预分配所有 buffers；检查 graph 内分配行为与 allocated/reserved/peak。仅 memory_allocated 净增为零不能单独证明没有临时分配。

生产优化候选的暂定性能门槛：在真实 B=1 fixture 上，graph p50 和 p95 均不比 V0 慢超过 5%，数值门槛也通过。这个门槛只决定是否继续产品化；不限制诊断版本，也不是社区官方要求。

## 12. 必做负控制与常见陷阱

1. 同一排列重复 100 次：不稳定先查 kernel 状态/同步，不能直接叫舍入。
2. 未读字节填充 0x00/0xA5：原始 replay 都应忠实。
3. padding 变化：有效集合、长度不变，输出不得变化；不能把 padding 随机移动到有效前缀后仍声称同一输入。
4. 有效 payload 改动：reference 能检测；选择足够可见且不溢出的改动验证生产路径确实读取该行。
5. sink 语义：只增加分母；不要把 sink 当额外非零 V。
6. C1/C2 cache 的物理布局不假定相同，data 和 scale 不按“连续 584 B”读取。
7. 不去重 SWA/extra，也不去掉真实重复项。
8. 不能根据 kernel 名叫 DeepGEMM/FlashMLA 就推断实际内部精度。
9. L20 候选 source 属性不等于当前 graph 执行 publish/consume。
10. tile 换位、split 换位、CPB 改动是不同实验轴。
11. BF16 输出相同不表示内部 FP32 一致；同样，FP32 误差明显降低也不保证 BF16 最终完全一致。
12. C2 L2 的变化之后，L8/L14/L20 score 分叉不能当作三个独立根因。
13. 当前 TP0 阳性可以成立局部因果；没有覆盖全部 rank 时不做全 rank 阴性保证。
14. 新诊断 kernel 成功不等于默认 FlashInfer 已改善；核对模块名称和产物，防止 baseline/variant 串用。
15. 不以 cosine 接近 1 或某次 greedy token 相同代替数值与重复性验收。

## 13. 总体验收与退出条件

| 门槛 | 必须满足 | 未满足时 |
| --- | --- | --- |
| G0 身份/dispatch | 现场模块、布局、源码和运行计划明确 | identity_or_dispatch_unresolved |
| G1 完整采集/忠实回放 | Q/sink/KV/scale/index 完整，A/B 原样逐位匹配，填充测试通过 | capture_inconclusive / replay_not_faithful |
| G2 单变量交换 | 其他输入相同，AA/AB/BA/BB 与生产交换成立，AS=BS | order_sensitive_other_inputs_unresolved |
| G3 reference | 独立 FP64 自检/负控制通过 | oracle_invalid |
| G4 数值归因 | 至少 V0/V1 完成；若归因 FP8 PV，则 V2/V3 或等价直接证据完成 | contribution_unresolved / pv_not_run |
| G5 graph | 固定排列 eager/graph 一致，变体输入不被污染，无错误读取 | fixed_order_nondeterministic / graph_mismatch |
| G6 继续产品化 | 数值候选门槛成立，性能与显存成本明确 | correct_but_slow / precision_only / sensitivity_only |

允许的最终结果：

- bf16_partial_order_sensitivity_identified。
- fp8_pv_grouping_sensitivity_identified。
- mixed_precision_contributions_identified。
- layout_mask_or_sync_defect_suspected（先附具体反例）。
- precision_improved_not_bitwise_stable。
- permutation_stable_on_tested_fixtures。
- contribution_unresolved。
- capture_inconclusive / replay_not_faithful / oracle_invalid。

按 fixture 分别给出 C1、C2 结论。允许“C1 已完成，C2 缺数据”的 partial 结果，不包装成两者全部闭环。

遇到未授权进程占用、OOM/device assert、图形状变化、输入对齐不合格、原样回放失败，停止受影响阶段。已发生 device assert 的进程退出后再进行后续合法工作，不在同一 CUDA context 吞异常继续跑。

本轮不启动 full-model 生产修复 A/B，不跑 GSM8K，不开 fused Top-K PR。若得到具体、有效的算子候选，下一实验再做关闭探针的服务级重复性和性能/质量验证。

## 14. 交付目录与结果表

建议归档根：/data/sgl_pr/runs/C1-ORDER/EXP-0005/。实际编号按未占用编号确定。

~~~text
result.md
identity.json
source_map.md
capture/
  launch.sh
  effective_config.json
  requests.jsonl
  c1/
  c2/
fixtures/
  manifest.json
  c1_A.pt
  c1_B.pt
  c2_A.pt
  c2_B.pt
replay/
  layout_readback.json
  faithful_replay.json
  causal_matrix.json
reference/
  oracle_selftest.json
  oracle_outputs.pt
permutations/
  permutations.json
variants/
  V0/
  V1/
  V2/
  V3/
metrics/
  numerical.csv
  stability.json
  intermediate_checks.json
  performance.csv
patches/
  sglang_capture.patch
  flashinfer_diagnostic.patch
~~~

未产生的文件标 not_run，不创建空文件假装完成。

最终 result.md 至少包含下表：

| Fixture | Variant | 排列数量 | 固定排列稳定 | S_order | p95 E_ref | max_abs | 不同 BF16 输出数 | graph p50/p95 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | V0 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C1 | V1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C2 | V0 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C2 | V1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

V2/V3 执行后增加行；否则填 not_run。附上 AA/AB/BA/BB、reference 负控制、NoPE/RoPE 分段结果、实际头数/CPB/split 计划、编译身份和完整失败记录。

冻结 code commit 后创建新的 EXP tag；不移动旧 tag。公共结果摘要用表格说明数值，不堆砌校验摘要；完整性校验信息留在附件。

## 15. 执行顺序与资源边界

1. Stage 0：只读身份/源码核对。
2. Stage 1：读现有 fixtures；不足时最多一次正式 TP8 服务补采。
3. 停采集服务并确认归属，然后单 GPU 执行忠实 replay 与 reference。
4. V0 排列矩阵，V1 FP32 partial。
5. 需要归因 PV 时补 V2/V3；每个未完成项如实记录。
6. 达到数值门槛后才测 microbench。
7. 归档、提交诊断代码/patch 身份和结果，报告最终分类。

每阶段过门槛再进入下一阶段。固定的工作范围不因“继续跑完”而扩展到修改其他服务、升级系统包、改生产后端或提交外部 Issue/PR。

## 16. 源码依据

以下为 Planner 已核对的源码入口；现场模块身份优先于远端同版本标签：

- 当前 SGLang wrapper：54e84fa9f3，python/sglang/kernels/ops/attention/flash_mla_sm120.py。
- 当前 probe：54e84fa9f3，python/sglang/srt/layers/attention/dsv4/v41_indexer/c1_order_probe.py。
- 历史忠实 C1 replay：0de2000c3ba9503140ab31a3cc9e90a64c4323c6，test/registered/kernels/ops/attention/test_dsv41_sm120_c1_order_replay.py。
- FlashInfer v0.7.1：include/flashinfer/attention/sparse_mla_sm120/kernels/fp8_decode/decode.cuh，FP8 PV 量化与 partial 保存。
- FlashInfer v0.7.1：include/flashinfer/attention/sparse_mla_sm120/execution/merge.cuh，BF16 partial 读取及 FP32 合并。
- FlashInfer v0.7.1：tests/attention/test_sparse_mla_sm120.py，实际适用的 DSv4、dual-cache、长度/mask 及 graph 回归用例。

本计划的效果比例和性能门槛由本轮实验预先设定，不能表述为上游官方标准。

