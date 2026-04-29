# HL23 Minimal Regression Assets

这个目录现在是 HL23 主回归的正式仓内资产。
HL23 相关脚本和测试统一从这里取资产。

当前包含：

1. 四个 morphology SWC：
   - `HL23PYR.swc`
   - `HL23SST.swc`
   - `HL23PV.swc`
   - `HL23VIP.swc`
2. baseline/export 需要的模板与 biophys HOC：
   - `NeuronTemplate.hoc`
   - `NeuronTemplate_full.hoc`
   - `biophys_HL23*.hoc`
3. baseline/export helper：
   - `export_via_neuron.py`
4. baseline NEURON 侧需要编译的 MOD：
   - `mod/*.mod`

测试运行时会把本目录复制到临时目录，并在临时目录内编译 `mod/`，因此：

1. 仓内正式资产目录不需要保留 `x86_64/`
2. 不应把运行产物提交回本目录

当前主要消费者：

1. `examples/hl23multi/run_mind_sim.py`
2. `examples/hl23multi/run_nrn_coreneuron.py`
