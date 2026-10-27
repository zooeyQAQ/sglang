# EXP-0005 后续：只提升 split-K partial 精度的 V1 对照

日期：2026-10-27  
状态：待执行；本文不代表 V1 已实现或通过。  
归属：继续 EXP-0005，不移动已有冻结 tag，不改写第一轮结果。

## 1. 为什么继续 V1

第一轮结果来自 labbook 提交 `032b49ba99`，代码仍为 `54e84fa9f38fbea162af599222370961a1d2c7d7`。输入是历史 EXP-0025 L20 C1、req0000/req0002、step 1：H=64、SWA=128、extra=512、head_dim_v=512。

| 同输入对照 | 已观察结果 |
| --- | --- |
| 原样 replay | AA/BB 对生产 sparse_out 逐位一致；未观测区域填 0x00、0xA5 均通过 |
| 固定排列 | FlashInfer 重复 20 次逐位稳定 |
| 独立 FP64 | 已执行的解析自检通过；A/B 转 BF16 后相同 |
| FlashInfer 的排列差 | max_abs=0.0078125，rel_l2=0.002325 |
| Torch FP32→BF16 的排列差 | max_abs=0.001953125，rel_l2=4.09e-5 |
| 请求 A 对 FP64→BF16 的误差 | FlashInfer max_abs=0.03125；Torch FP32→BF16 为 0.001953125 |

这些结果支持继续做精度隔离。Torch 同时替换 QK、softmax、PV，尚不能定位哪一项贡献最大。FP64 在这两个排列上相同，也不构成任意输入、任意排列均逐位一致的保证。

本次只回答：保留生产 FP8 PV 等计算，只取消 split partial 写入 BF16 的舍入，能改善多少？

## 2. 本次范围

- 使用现有忠实 H=64 C1 fixture，单张空闲 GPU 离线执行。
- 不启动模型服务，不补采当前 H=8 C1/C2；这些仍留待后续。
- 不改 Top-K、indices 顺序、KV 存储格式，不加入 canonical sort。
- 不改 Q 的处理、QK、online softmax、NoPE FP8 PV、RoPE PV、sink 和最终 BF16 输出。
- 不做 V2/V3，不跑 GSM8K，不进行服务级性能或 token 重复性测试。
- 不覆盖已安装 FlashInfer；使用独立源码副本、模块名和编译缓存，保留原生产 V0 可调用。

本次原始计划中的其他阶段仍适用；本文件明确后续执行顺序和范围。

## 3. V0 / V1 定义

| 项目 | V0 | V1 |
| --- | --- | --- |
| decode 内计算 | 原生产实现 | 相同 |
| split partial 写出 | FP32 值转 BF16 保存 | 同一位置的 FP32 值直接保存 |
| mid_out 存储 | BF16 | FP32 |
| merge 读入 | BF16 转 FP32 | 直接读 FP32 |
| merge 权重与累加 | 原生产 FP32 逻辑 | 相同 |
| mid_lse | 原生产 FP32 | 相同 |
| 最终输出 | BF16 | BF16 |

V1 保存的仍是原生产归一化 split partial，例如 `acc_nope * inv_g` 和 `acc_rope * inv_g`；不能顺便改为未归一化分子或改变 merge 公式。

### 需要一致修改的接口

以现场已核实的 FlashInfer 模块为准，追踪并同时修改：

1. Python scratch 分配与 dtype 检查。
2. FFI/C++ 参数检查、pointer cast、所有实际调用的 launch 签名。
3. decode producer 的 mid_out 指针及 NoPE/RoPE 两段写出。
4. scratch 的元素 stride、字节预算、有效 head/split 范围和对齐。
5. merge 的 mid_out 指针和向量 load。
6. JIT 编译入口、缓存 key 与模块身份，确保 V0/V1 不会串用。

参考入口：`fp8_decode/decode.cuh`、`execution/merge.cuh`、`execution/decode_launch.cuh`、FFI binding 和 Python API。远端标签不能替代本机实际源码身份。

原 merge 的 `uint4` load 一次含 8 个 BF16。改为 FP32 后，同样 16 字节只能含 4 个 float；必须重算向量步长和覆盖范围，不能只换类型。producer 中原 `__nv_bfloat162` 写出也必须替换。最终 output 的 BF16 store 保持原样。

记录真实 `BI`、`chunks_per_block`、active/allocated splits、scratch head 数及各维 stride，冻结调度。不要因为 dtype 改变重新 autotune 出另一套 split 计划。头数 padding 与有效 heads 分开处理，不读取未写入的 padding。

FP32 mid_out 在元素数不变时占用 V0 两倍字节；检查分配和读写边界。若实际走了另一条未修改的 producer，不能把它报成 V1。

## 4. 执行顺序

### A. 先验证独立编译的 baseline

独立模块的 V0，在 A/B 两个排列、0x00/0xA5 两种填充下，必须逐位匹配原生产 V0 及已有生产输出。记录编译配置与实际加载模块。

若仅重新编译 V0 就改变结果，先处理模块、编译或 dispatch 差异；不直接把后面的差异归因于 mid_out dtype。

### B. 验证 V1 的接口和隔离

先只跑 A/B：

- 输入前后未被修改，无非法访问或 guard 区损坏；有效输出有限。
- mid_out 实际为 FP32，producer 和 merge 使用同一 layout；mid_lse 仍为 FP32。
- 实际 launch、split 计划、有效 mask 和 sink 与 V0 一致。
- 固定排列的 mid_lse 与 V0 逐位相同；V1 partial 转 BF16 后应与 V0 的有效 partial 一致。

增加一个诊断 round-back 对照：V1 的 FP32 partial 在 merge 读入后，先按原舍入规则转 BF16 再回 FP32，然后执行原 merge。它应逐位复现 V0 输出。该对照只用于隔离，不用于性能。

如果上述中间值或 round-back 不一致，检查编译器变化、stride、漏改调用链及计算路径。必要时做共享 producer 的双写诊断，在同一次计算中同时保存舍入前 FP32 和原 BF16 partial；双写版本也须先确认 BF16 支路逐位匹配 V0。隔离未通过时，结论是 `variant_isolation_unresolved`，不能报告“只改 partial 的效果”。

如果无法在本轮取得中间值，仍可记录 V1 输出，但明确隔离证据不足，不能把源代码意图当作已验证事实。

### C. 稳定性与因果 replay

补齐 AA/AB/BA/BB 和 AS/BS，保持相同输入门槛与物理 slot，不紧缩重排 cache。

- V0 交换排列应复现对应生产 A/B；V1 没有生产目标输出，不要求匹配 V0。
- 各变体的 AS/BS 使用完全相同输入与 canonical 顺序，须逐位一致。
- A/B 两种填充均应不影响对应输出。
- 每个固定 case，eager 100 次、graph 100 次逐位稳定。
- graph 使用静态 indices buffer；在正确 stream 上 copy_ 更新后 replay，确认实际读取新排列。
- 出现固定排列不稳定、非法访问或填充依赖时，暂停数值归因，检查布局、mask 或同步。

### D. 再完成排列矩阵

沿用原 plan.md 的最多 36 个排列，冻结 seed=4102；先保存排列再测试。只换 extra 有效 pair 的顺序，SWA、padding、长度和 KV 不动。

| 家族 | 最多数量 |
| --- | ---: |
| A、B、canonical、逆序 | 4 |
| 实际 tile 内打乱 | 8 |
| split 内 tile 换位 | 4 |
| extra-only 整 split 换位 | 4 |
| 跨 tile 完全打乱 | 16 |

根据实际 BI/CPB/split 映射判断家族是否存在；不适用记 N/A，重复排列去重。每个排列在 V0/V1 上 eager、graph 各 100 次固定输入稳定性检查。

FP64 reference 在开始后续测试前复用并核验。第一轮只完成了部分参考自检；原计划中尚未执行的解析式、sink 和有效 payload 负控制须明确补做或标未完成，不能扩大 `pass_1e9` 的覆盖范围。

## 5. 指标与结果解释

使用同一个 canonical FP64 输出 R，且 R 的输入仍为同一份已量化 KV 解码后的值。

- `S_order`：排列间最大 L2 距离 / max(norm(R), 1e-12)，另报排列间 max_abs。
- `E_ref`：对 FP64 R 的 relative L2 与 max_abs，各排列汇总 median/p95/max。
- 另报最终 BF16 输出对 R 转 BF16 的误差，以便与第一轮表格对应。不要把两种参考混成一个指标。
- NoPE/RoPE 分段误差、每 head 误差、不同 BF16 输出数量、固定排列稳定性。
- 有导出 cast 前最终值时单独报告；没有则记 unavailable，不能由 BF16 输出反推。

原计划的工程门槛继续使用，不因看到 V1 结果而调整：

| 结果 | 可以作出的结论 |
| --- | --- |
| 隔离通过，S_order 至少降低 4 倍 | BF16 partial 舍入对本 fixture 的排列敏感性有明显贡献 |
| 再满足 p95 E_ref 至少降低 2 倍，且无超过原门槛的误差回退 | 值得继续优化的数值候选 |
| 改善不足 4 倍 | 报实际变化；未达到预设明显贡献门槛，不等于零贡献 |
| 准确度改善但 BF16 仍有多种输出 | 精度改善；尚未得到排列逐位稳定 |
| V1 基本无改善 | partial 精度单项未见明显收益；不能据此认定 FP8 PV 是根因 |
| 中间值/round-back 隔离失败 | V1 归因未成立，先排查控制变量 |

这轮只有 H=64 历史 C1 fixture。即使通过，也不外推为当前 H=8 或 C2 已修复，不声称一般排列不变性。

## 6. 性能何时测

完成隔离和数值检查后，才做真实 B=1、H=64 fixture 的 V0/V1 graph microbench。沿用原计划：至少 20 次预热，5 组、每组 200 次 replay，交错测量，报告 p50/p95、scratch 字节和 launch 数。关闭中间值导出与日志，不包含 host copy。

诊断 V1 可以慢；性能差不抹掉归因结论。原先 5% 的生产候选门槛仅决定后续产品化。本轮不自动增加 synthetic batch 或重启模型做性能测试。

## 7. 交付和验收

保留现有第一轮 result.md、tag 和归档。新增 V1 结果、独立 FlashInfer patch/模块加载说明、数值 CSV、排列与稳定性结果；result.md 追加后续摘要，标明第一轮与 V1 各自范围。

| 必须交付 | 验收 |
| --- | --- |
| V0 独立编译复现 | A/B、两种填充，逐位匹配生产 |
| V1 完整接口 | Python/FFI/producer/scratch/merge 的 FP32 链路可核对 |
| 单变量隔离 | LSE、partial、round-back 检查通过，或明确未成立 |
| 正确性与稳定性 | 固定排列 eager/graph 各 100 次稳定，无输入污染或越界 |
| 数值对照 | V0/V1 同排列矩阵，报告 S_order 与 E_ref，不只报最好一次 |
| 结果范围 | 明确 H=64 C1；H=8/C2/V2/V3 仍为 not_run |

至少给出以下结果表：

| Variant | 排列数 | S_order max_abs / rel_l2 | p95 E_ref | 最差 E_ref | BF16 输出数 | eager/graph 固定排列稳定 | graph p50/p95 |
| --- | ---: | --- | --- | --- | ---: | --- | --- |
| V0 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| V1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

冻结新代码和 FlashInfer patch 身份后使用新的后续 tag，例如 `exp/C1-ORDER/EXP-0005-v1`；不得移动 `exp/C1-ORDER/EXP-0005`。

本次验收是完成一个有效、可解释的 V0/V1 对照。V1 无明显收益也属于有效实验结果；随后再依据数据决定是否做 PV 的 V2/V3。
