#!/usr/bin/env python3
"""
使用 NEURON 的 Import3d 从 SWC 构建细胞，并导出：
- 一个可复用的 HOC 模板（包含几何/拓扑/SectionList/geom_nseg/biophys）
- 可选：NeuroML2（仅形态）

说明：
- 生成 HOC 的过程中会先实例化一个单细胞，再从 NEURON 对象中提取形态信息；
  这样导出的模板可用于后续多细胞重复实例化（避免 Import3d 逐次解析开销）。
- 可通过 --template-hoc 选择不同的 NeuronTemplate 版本：
  - NeuronTemplate.hoc: 默认版本（可能包含 delete_axon_BPO 等简化逻辑）
  - NeuronTemplate_full.hoc: 保留完整轴突树的版本（用于性能/精度对比）
"""

import argparse
import math
import shutil
import tempfile
import sys
import subprocess
from pathlib import Path
from neuron import h

# 加载标准库
h.load_file("stdrun.hoc")
h.load_file("import3d.hoc")

def _compile_and_load_mechs(base_dir: Path) -> Path | None:
    """Ensure x86_64/libnrnmech.so exists under base_dir and load it.
    Returns the loaded library path or None.
    """
    candidates = [
        base_dir / "x86_64/.libs/libnrnmech.so",
        base_dir / "x86_64/libnrnmech.so",
    ]
    # Try existing
    for lib in candidates:
        if lib.exists():
            try:
                h.nrn_load_dll(str(lib))
                print(f"🔌 已加载机制库: {lib}")
                return lib
            except Exception as e:
                print(f"⚠️ 加载机制库失败: {lib} -> {e}")
    # Build in place
    mod_dir = base_dir / "mod"
    if mod_dir.exists():
        print(f"🛠️  正在编译机制: {mod_dir}")
        try:
            subprocess.run(["nrnivmodl", str(mod_dir)], cwd=str(base_dir), check=True)
            for lib in candidates:
                if lib.exists():
                    try:
                        h.nrn_load_dll(str(lib))
                        print(f"🔌 已加载机制库: {lib}")
                        return lib
                    except Exception as e:
                        print(f"⚠️ 加载机制库失败: {lib} -> {e}")
        except Exception as e:
            print(f"⚠️ 编译机制失败: {e}")
    print("ℹ️ 未找到可加载的机制库，且编译未成功")
    return None

def _rebuild_sectionlists(cell) -> None:
    """Rebuild SectionList objects to avoid stale references after delete_axon."""
    cell.all = h.SectionList()
    cell.somatic = h.SectionList()
    cell.apical = h.SectionList()
    cell.basal = h.SectionList()
    cell.axonal = h.SectionList()

    def base_name(sec) -> str:
        name = sec.name()
        if "." in name:
            name = name.split(".")[-1]
        return name.split("[", 1)[0]

    cell_prefix = cell.hname() + "."
    for sec in h.allsec():
        # Match by name prefix to avoid brittle HocObject identity checks.
        try:
            if not sec.name().startswith(cell_prefix):
                continue
        except Exception:
            continue

        base = base_name(sec)
        if base == "soma":
            cell.somatic.append(sec)
            cell.all.append(sec)
        elif base == "dend":
            cell.basal.append(sec)
            cell.all.append(sec)
        elif base == "apic":
            cell.apical.append(sec)
            cell.all.append(sec)
        elif base == "axon":
            cell.axonal.append(sec)
            cell.all.append(sec)
        # Intentionally exclude myelin from lists (matches NeuronTemplate.hoc delete_axon).

def export_hoc_from_neuron(
    *,
    cell_name: str,
    swc_file: Path,
    biophys_file: Path,
    output_hoc: Path,
    template_hoc: Path,
):
    """
    使用 NEURON 机制导出 HOC 模板。
    
    参数:
        swc_file: SWC 形态文件路径
        biophys_file: 生物物理参数文件路径
        output_hoc: 输出 HOC 文件路径
    """
    print(f"📖 加载 NeuronTemplate: {template_hoc.name}")
    h.xopen(str(template_hoc))
    
    print(f"🧬 实例化细胞: {swc_file.name}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        morph_dir = Path(tmp_dir) / "morphologies"
        morph_dir.mkdir(parents=True, exist_ok=True)
        swc_link = morph_dir / swc_file.name
        try:
            swc_link.symlink_to(swc_file)
        except Exception:
            shutil.copyfile(swc_file, swc_link)
        cell = h.NeuronTemplate(str(swc_link))
    _rebuild_sectionlists(cell)
    
    # Ensure mechanisms are available before applying biophys
    _compile_and_load_mechs(Path(__file__).parent)

    print(f"⚡ 应用生物物理参数...")
    h.xopen(str(biophys_file))
    biophys_proc = f"biophys_{cell_name}"
    if not hasattr(h, biophys_proc):
        raise RuntimeError(
            f"biophys proc not found: {biophys_proc} (loaded from {biophys_file}); "
            f"hint: ensure the hoc defines `proc {biophys_proc}()` and call it as {biophys_proc}(cell)"
        )
    getattr(h, biophys_proc)(cell)

    def _base_name(sec) -> str:
        name = sec.name()
        if "." in name:
            name = name.split(".")[-1]
        return name.split("[", 1)[0]

    cell_prefix = cell.hname() + "."
    export_sections = [sec for sec in h.allsec() if sec.name().startswith(cell_prefix)]
    export_names = {sec.name() for sec in export_sections}
    n_myelin = sum(1 for sec in export_sections if _base_name(sec) == "myelin")
    
    # 统计信息
    n_sections = sum(1 for _ in cell.all)
    n_soma = sum(1 for _ in cell.somatic)
    n_axon = sum(1 for _ in cell.axonal)
    n_apical = sum(1 for _ in cell.apical)
    n_basal = sum(1 for _ in cell.basal)
    
    print(f"\n📊 细胞统计:")
    print(f"  总节段数: {n_sections}")
    print(f"  soma: {n_soma}, axon: {n_axon}, apical: {n_apical}, basal: {n_basal}")
    
    # 生成 HOC 模板
    print(f"\n✍️  生成 HOC 模板: {output_hoc.name}")
    
    hoc_content = []
    hoc_content.append(f"begintemplate {cell_name}\n")
    hoc_content.append("\n// 公共声明")
    hoc_content.append("public init, biophys, geom_nseg")
    
    # 只声明实际存在的节段类型
    public_secs = []
    if n_soma > 0:
        public_secs.append("soma")
    if n_basal > 0:
        public_secs.append("dend")
    if n_apical > 0:
        public_secs.append("apic")
    if n_axon > 0:
        public_secs.append("axon")
    if n_myelin > 0:
        public_secs.append("myelin")
    
    if public_secs:
        hoc_content.append(f"public {', '.join(public_secs)}")
    
    hoc_content.append("public all, somatic, apical, basal, axonal")
    hoc_content.append("public nSecAll, nSecSoma, nSecApical, nSecBasal, nSecAxonal\n")
    
    hoc_content.append("\n// 对象引用")
    hoc_content.append("objref all, somatic, apical, basal, axonal, sref, this")
    hoc_content.append("strdef tstr\n")
    
    # 统计每种类型的节段数
    soma_count = sum(1 for sec in cell.somatic)
    dend_count = sum(1 for sec in cell.basal)
    apic_count = sum(1 for sec in cell.apical)
    axon_count = sum(1 for sec in cell.axonal)
    
    hoc_content.append(f"\n// 创建节段")
    create_parts = []
    if soma_count > 0:
        create_parts.append(f"soma[{soma_count}]")
    if dend_count > 0:
        create_parts.append(f"dend[{dend_count}]")
    if apic_count > 0:
        create_parts.append(f"apic[{apic_count}]")
    if axon_count > 0:
        create_parts.append(f"axon[{axon_count}]")
    if n_myelin > 0:
        create_parts.append(f"myelin[{n_myelin}]")
    
    hoc_content.append(f"create {', '.join(create_parts)}\n")
    
    # init 过程
    hoc_content.append("\nproc init() {")
    hoc_content.append("    all = new SectionList()")
    hoc_content.append("    somatic = new SectionList()")
    hoc_content.append("    basal = new SectionList()")
    hoc_content.append("    apical = new SectionList()")
    hoc_content.append("    axonal = new SectionList()")
    hoc_content.append("    ")
    hoc_content.append("    define_geometry()")
    hoc_content.append("    build_connectivity()")
    hoc_content.append("    build_lists()")
    hoc_content.append("    geom_nseg()")
    hoc_content.append("    biophys()")
    hoc_content.append("}\n")
    
    # 生成 define_geometry() - 分块处理
    hoc_content.append("\nproc define_geometry() {")
    
    # 收集所有节段及其几何信息
    # 如果某节段没有足够的 3D 点（n3d < 2），则回退为 L/diam 赋值导出
    sections_data = []
    for sec in cell.all:
        sec_name = sec.name().split('.')[-1]  # 获取节段名称
        n3d = sec.n3d()
        if n3d >= 2:
            pts = []
            for i in range(n3d):
                pts.append((sec.x3d(i), sec.y3d(i), sec.z3d(i), sec.diam3d(i)))
            sections_data.append({
                "name": sec_name,
                "mode": "pt3d",
                "pts": pts,
            })
        else:
            # 没有 3D 形态信息：导出为标量几何
            # 读取 L 与两端直径（如相等则用标量直径）
            try:
                L = float(sec.L)
            except Exception:
                L = None
            # 通过 Segment 访问两端直径，避免需要 push 当前节段
            d0 = None
            d1 = None
            try:
                d0 = float(sec(0.0).diam)
            except Exception:
                pass
            try:
                d1 = float(sec(1.0).diam)
            except Exception:
                d1 = d0
            sections_data.append({
                "name": sec_name,
                "mode": "scalar",
                "L": L,
                "d0": d0,
                "d1": d1,
            })
    
    # 分块大小
    chunk_size = 20
    n_chunks = (len(sections_data) + chunk_size - 1) // chunk_size
    
    # 调用所有分块
    for chunk_idx in range(n_chunks):
        hoc_content.append(f"    geometry_chunk_{chunk_idx}()")
    
    hoc_content.append("}\n")
    
    # 生成各个分块
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(sections_data))
        chunk_sections = sections_data[start_idx:end_idx]
        
        hoc_content.append(f"\nproc geometry_chunk_{chunk_idx}() {{")
        for sec_info in chunk_sections:
            sec_name = sec_info["name"]
            hoc_content.append(f"    {sec_name} {{")
            # 清理 3D 定义，避免重复/污染
            hoc_content.append("        pt3dclear()")
            if sec_info["mode"] == "pt3d":
                for x, y, z, d in sec_info["pts"]:
                    hoc_content.append(f"        pt3dadd({x}, {y}, {z}, {d})")
            else:
                # 标量几何：设置长度与直径
                L = sec_info.get("L")
                d0 = sec_info.get("d0")
                d1 = sec_info.get("d1")
                if L is not None:
                    hoc_content.append(f"        L = {L}")
                # 若两端直径都有效且不同，用区间赋值；否则用标量直径
                if d0 is not None and d1 is not None and abs(d0 - d1) > 1e-9:
                    hoc_content.append(f"        diam(0:1) = {d0}:{d1}")
                elif d0 is not None:
                    hoc_content.append(f"        diam = {d0}")
            hoc_content.append(f"    }}")
        hoc_content.append("}\n")
    
    # 生成 build_connectivity()
    #
    # 重要：connect 语句的顺序会影响 NEURON 内部的 sibling 链表顺序
    # (connect 通常会把 child 挂到 parent->child 的链表头部)，而 CoreNEURON
    # 的 node/section 顺序依赖该顺序（reorder_secorder 的 BFS）。
    #
    # 因此我们不能简单按 `cell.all` 遍历写 connect；必须按 *当前模型* 的 child 顺序
    # (SectionRef.child[0..]) 来生成，并且对每个 parent 的 children 反向输出 connect，
    # 才能在重新加载该 HOC 时重建同样的 child 链表顺序。
    hoc_content.append("\nproc build_connectivity() { localobj sref")

    # Collect root sections in the order they appear in `export_sections`.
    roots = []
    for sec in export_sections:
        sref = h.SectionRef(sec=sec)
        if not sref.has_parent():
            roots.append(sec)
        else:
            parent = sref.parent
            if parent is None or parent.name() not in export_names:
                roots.append(sec)

    from collections import deque

    q = deque(roots)
    seen = set()
    for r in roots:
        seen.add(r.name())

    while q:
        parent_sec = q.popleft()
        sref = h.SectionRef(sec=parent_sec)

        children = []
        for i in range(int(sref.nchild())):
            ch = sref.child[i]
            if ch is None:
                continue
            if ch.name() not in export_names:
                continue
            children.append(ch)

        # Emit connect in reverse child order so that NEURON's prepend semantics
        # reconstruct the original sibling list order.
        parent_name = parent_sec.name().split('.')[-1]
        for ch in reversed(children):
            sec_name = ch.name().split('.')[-1]
            h.pop_section()  # 确保没有当前节段
            ch.push()
            parent_x = h.parent_connection()
            child_x = h.section_orientation()
            h.pop_section()
            hoc_content.append(f"    connect {sec_name}({child_x:.10g}), {parent_name}({parent_x:.10g})")

        for ch in children:
            if ch.name() not in seen:
                seen.add(ch.name())
                q.append(ch)

    hoc_content.append("}\n")
    
    # 生成 build_lists()
    hoc_content.append("\nproc build_lists() {")
    for sec in cell.somatic:
        sec_name = sec.name().split('.')[-1]
        hoc_content.append(f"    {sec_name} {{ somatic.append() all.append() }}")
    for sec in cell.axonal:
        sec_name = sec.name().split('.')[-1]
        hoc_content.append(f"    {sec_name} {{ axonal.append() all.append() }}")
    for sec in cell.apical:
        sec_name = sec.name().split('.')[-1]
        hoc_content.append(f"    {sec_name} {{ apical.append() all.append() }}")
    for sec in cell.basal:
        sec_name = sec.name().split('.')[-1]
        hoc_content.append(f"    {sec_name} {{ basal.append() all.append() }}")
    hoc_content.append("}\n")
    
    # 从 NeuronTemplate.hoc 提取 geom_nseg
    print(f"📋 提取 geom_nseg() 从 NeuronTemplate.hoc...")
    template_text = template_hoc.read_text()
    
    import re
    match = re.search(r'(proc geom_nseg\(\).*?\n\})', template_text, re.DOTALL)
    if match:
        geom_nseg_text = match.group(1)
        hoc_content.append(f"\n{geom_nseg_text}\n")
    else:
        print("⚠️  警告: 无法提取 geom_nseg，使用默认实现")
        hoc_content.append("""
proc geom_nseg() {
    forsec all { nseg = 1 + 2*int(L/40) }
}
""")
    
    # Inject distribute_channels helpers if present (biophys may reference them).
    helper_patterns = [
        r'(func getLongestBranch\(\).*?\n\})',
        r'(proc distribute_channels\(\).*?\n\})',
        r'(func calculate_distribution\(\).*?\n\})',
    ]
    for pattern in helper_patterns:
        helper_match = re.search(pattern, template_text, re.DOTALL)
        if helper_match:
            hoc_content.append(f"\n{helper_match.group(1)}\n")

    # 生成 biophys() - 从输入 biophys hoc 提取并转换
    print(f"📋 转换生物物理参数...")
    biophys_text = biophys_file.read_text()
    
    # 转换 proc biophys_<cell_name>() -> proc biophys()
    # 移除 $o1. 前缀
    import re
    biophys_converted = re.sub(
        rf"\bproc\s+biophys_{re.escape(cell_name)}\s*\(\s*\)",
        "proc biophys()",
        biophys_text,
    )
    biophys_converted = biophys_converted.replace("$o1.", "")
    
    hoc_content.append(f"\n{biophys_converted}")
    
    hoc_content.append(f"\nendtemplate {cell_name}\n")
    
    # 写入文件
    output_hoc.write_text('\n'.join(hoc_content))
    print(f"✅ HOC 模板已生成: {output_hoc}")


def _infer_template_name(hoc_file: Path) -> str:
    try:
        with hoc_file.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if "begintemplate" in line:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        return parts[-1]
    except OSError:
        pass
    return "Cell"


def export_to_neuroml(hoc_file: Path, output_nml: Path):
    """
    导出 HOC 到 NeuroML2 格式。
    
    参数:
        hoc_file: 输入 HOC 模板文件
        output_nml: 输出 NeuroML 文件
    """
    print(f"\n🔄 导出到 NeuroML: {output_nml.name}")
    
    # 创建临时加载器
    template_name = _infer_template_name(hoc_file)
    loader_hoc = hoc_file.parent / f"{hoc_file.stem}_loader.hoc"
    loader_content = f"""
load_file("stdrun.hoc")
load_file("import3d.hoc")
xopen("{hoc_file.name}")
objref cell
cell = new {template_name}()
"""
    loader_hoc.write_text(loader_content)
    
    try:
        try:
            import pyneuroml.neuron  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "pyneuroml is not available; install it or run with --skip-neuroml"
            ) from exc

        pyneuroml.neuron.export_to_neuroml2(  # type: ignore[attr-defined]
            str(loader_hoc),
            str(output_nml),
            includeBiophysicalProperties=False,
            validate=False,
        )
        print(f"✅ NeuroML 已生成: {output_nml}")
    finally:
        loader_hoc.unlink(missing_ok=True)


def main():
    script_dir = Path(__file__).parent

    p = argparse.ArgumentParser(
        description="Export morphology/biophys from NEURON Import3d to a reusable HOC (full morphology supported)"
    )
    p.add_argument(
        "--cell-name",
        default=None,
        help="Output template name (default: SWC stem, e.g. HL23PV / HL23VIP)",
    )
    p.add_argument("--swc-file", default=str(script_dir / "HL23PV.swc"), help="Input SWC file")
    p.add_argument("--biophys-file",
                   default=str(script_dir / "biophys_HL23PV.hoc"),
                   help="Input biophys hoc (expects proc biophys_<cell_name>(cell))")
    p.add_argument("--template-hoc",
                   default=str(script_dir / "NeuronTemplate.hoc"),
                   help="NeuronTemplate hoc to use for Import3d instantiation")
    p.add_argument("--output-hoc", default=str(script_dir / "HL23PV.hoc"), help="Output HOC template path")
    p.add_argument("--output-nml",
                   default=str(script_dir / "HL23PV.morph.cell.nml"),
                   help="Output NeuroML2 path (optional)")
    p.add_argument("--skip-neuroml", action="store_true", help="Skip NeuroML export (no pyneuroml dependency)")
    args = p.parse_args()

    swc_file = Path(args.swc_file).expanduser().resolve()
    biophys_file = Path(args.biophys_file).expanduser().resolve()
    template_hoc = Path(args.template_hoc).expanduser().resolve()
    output_hoc = Path(args.output_hoc).expanduser().resolve()
    output_nml = Path(args.output_nml).expanduser().resolve()

    cell_name = str(args.cell_name) if args.cell_name else swc_file.stem
    if not cell_name:
        raise SystemExit("error: empty cell_name (pass --cell-name or use a non-empty SWC stem)")
    
    # 检查输入文件
    if not swc_file.exists():
        print(f"❌ 错误: 找不到 {swc_file}")
        sys.exit(1)
    if not biophys_file.exists():
        print(f"❌ 错误: 找不到 {biophys_file}")
        sys.exit(1)
    if not template_hoc.exists():
        print(f"❌ 错误: 找不到 {template_hoc}")
        sys.exit(1)
    
    print("=" * 60)
    print("🧠 NEURON 形态导出工具")
    print("=" * 60)
    
    # 第一步: 导出 HOC
    export_hoc_from_neuron(
        cell_name=cell_name,
        swc_file=swc_file,
        biophys_file=biophys_file,
        output_hoc=output_hoc,
        template_hoc=template_hoc,
    )

    # 第二步: 导出 NeuroML（可选）
    if not args.skip_neuroml:
        export_to_neuroml(output_hoc, output_nml)
    else:
        print("ℹ️  skip_neuroml=1: skipping NeuroML export")
    
    print("\n" + "=" * 60)
    print("✅ 转换完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
