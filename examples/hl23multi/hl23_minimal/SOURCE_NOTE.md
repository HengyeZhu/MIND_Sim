# HL23 Minimal Regression Source Note

当前仓内 HL23 主回归资产收口在 `examples/hl23multi/hl23_minimal/`。

这里保留的是 HL23 精度/性能验证真正需要的最小集合：

1. morphology SWC
2. baseline NEURON+CoreNEURON 所需 template 与 biophys HOC
3. baseline/export helper：`export_via_neuron.py`
4. MIND_Sim 内置 NMODL 编译所需 HL23 MOD

这里明确不保留：

1. `x86_64/` 预编译产物
2. NeuroML 中间产物
3. 可视化/转换辅助脚本
4. 与主回归无关的历史实验文件

当前测试策略：

1. MIND_Sim 侧完全通过 Python API 构建 microcircuit，不加载 HOC。
2. MIND_Sim 侧直接加载 `mod/` 目录，由内置 NMODL 源码生成机制元数据和机制实现。
3. baseline 侧使用 NEURON+CoreNEURON 和 HOC template/biophys，作为精度对照。
4. compare 对比 MIND_Sim 输出与 baseline 输出。

因此，HL23 主回归资产只依赖本目录及其 `mod/`。
