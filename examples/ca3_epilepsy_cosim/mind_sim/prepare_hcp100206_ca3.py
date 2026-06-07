#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np


def file_status(path: Path) -> dict[str, object]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "is_symlink": path.is_symlink(),
        "size_bytes": path.stat().st_size if exists and path.is_file() else None,
    }


def command_status(name: str) -> dict[str, object]:
    found = shutil.which(name)
    return {"command": name, "available": found is not None, "path": found}


def relative_symlink(source: Path, link: Path, *, replace: bool) -> None:
    if not source.exists():
        raise SystemExit(f"missing input: {source}")
    if link.exists() or link.is_symlink():
        if not replace:
            return
        link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(source, start=link.parent)
    link.symlink_to(target)


def require_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise SystemExit(
            "This script requires nibabel. Install it in the data-prep environment, "
            "for example: conda install -c conda-forge nibabel"
        ) from exc
    return nib


def load_int_volume(nib, path: Path) -> tuple[object, np.ndarray]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if not np.issubdtype(data.dtype, np.integer):
        data = np.rint(data).astype(np.int32)
    else:
        data = data.astype(np.int32, copy=False)
    return img, data


def read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = "\t" if "\t" in sample else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise SystemExit(f"{path} has no header row")
        rows = [{k: (v or "").strip() for k, v in row.items()} for row in reader]
    return [name.strip() for name in reader.fieldnames], rows


def parse_index_name_table(path: Path) -> dict[int, str]:
    fieldnames, rows = read_table(path)
    index_key = next((k for k in fieldnames if k.lower() in {"index", "label", "id", "value"}), None)
    name_key = next((k for k in fieldnames if k.lower() in {"name", "label_name", "abbreviation", "abbr"}), None)
    if index_key is None or name_key is None:
        raise SystemExit(f"{path} must contain index/name columns")

    labels: dict[int, str] = {}
    for row in rows:
        if not row.get(index_key):
            continue
        try:
            index = int(float(row[index_key]))
        except ValueError as exc:
            raise SystemExit(f"{path} has a non-integer label index: {row[index_key]!r}") from exc
        name = row.get(name_key, "").strip()
        if not name:
            raise SystemExit(f"{path} has an empty name for label {index}")
        labels[index] = name
    if not labels:
        raise SystemExit(f"{path} contains no label mappings")
    return labels


def parse_hippunfold_label_table(path: Path, observed_values: list[int]) -> dict[int, str]:
    fieldnames, rows = read_table(path)
    lower_fields = {name.lower(): name for name in fieldnames}
    if {"index", "name"} <= set(lower_fields) or {"index", "abbreviation"} <= set(lower_fields):
        return parse_index_name_table(path)

    metadata_columns = {"subject", "hemi", "session", "space", "atlas"}
    label_columns = [name for name in fieldnames if name.lower() not in metadata_columns]
    active_label_columns: list[str] = []
    for name in label_columns:
        active = False
        for row in rows:
            try:
                active = active or float(row.get(name, "0") or "0") > 0.0
            except ValueError:
                active = True
        if active:
            active_label_columns.append(name)
    label_columns = active_label_columns
    if len(label_columns) != len(observed_values):
        raise SystemExit(
            f"{path} looks like a HippUnfold volumes.tsv, but it has {len(label_columns)} label columns "
            f"and the dseg has {len(observed_values)} non-zero values. Pass an index/name label TSV instead."
        )
    if not rows:
        raise SystemExit(f"{path} contains no rows")
    return {value: normalize_subfield_name(name) for value, name in zip(observed_values, label_columns)}


def normalize_subfield_name(name: str) -> str:
    text = name.strip().replace("_", "-").replace(" ", "-")
    text = re.sub(r"^(left|right|lh|rh|hemi-[lr]|[lr])-", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bca([1-4])\b", lambda m: f"CA{m.group(1)}", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdg\b", "DG", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsrlm\b", "SRLM", text, flags=re.IGNORECASE)
    return text


def side_name(side: str, name: str) -> str:
    return f"{side}-{normalize_subfield_name(name)}"


def is_ca3_name(name: str, pattern: re.Pattern[str]) -> bool:
    return pattern.search(name) is not None


def append_label(
    out: np.ndarray,
    mask: np.ndarray,
    labels: list[dict[str, object]],
    *,
    name: str,
    source: str,
    role: str,
) -> None:
    if not np.any(mask):
        return
    index = len(labels) + 1
    out[mask] = index
    labels.append(
        {
            "index": index,
            "name": name,
            "source": source,
            "role": role,
            "voxel_count": int(np.count_nonzero(mask)),
        }
    )


def append_base_labels(
    out: np.ndarray,
    labels: list[dict[str, object]],
    base: np.ndarray,
    base_names: dict[int, str],
    skip_base_values: set[int],
    exclude_mask: np.ndarray,
) -> None:
    base_values = sorted(int(x) for x in np.unique(base) if int(x) not in skip_base_values)
    missing = [value for value in base_values if value not in base_names]
    if missing:
        raise SystemExit(f"base label table is missing labels required by the base parcellation: {missing}")
    for value in base_values:
        append_label(
            out,
            (base == value) & ~exclude_mask,
            labels,
            name=base_names[value],
            source=f"base:{value}",
            role="whole_brain_base",
        )


def append_hippunfold_subfields(
    out: np.ndarray,
    labels: list[dict[str, object]],
    *,
    side: str,
    dseg: np.ndarray,
    label_names: dict[int, str],
    ca3_pattern: re.Pattern[str],
) -> int:
    observed_values = sorted(int(x) for x in np.unique(dseg) if int(x) != 0)
    missing = [value for value in observed_values if value not in label_names]
    if missing:
        raise SystemExit(f"HippUnfold label table is missing dseg values for {side}: {missing}")

    ca3_voxels = 0
    for value in observed_values:
        raw_name = label_names[value]
        name = side_name(side, raw_name)
        mask = dseg == value
        role = "ca3_micro_candidate" if is_ca3_name(raw_name, ca3_pattern) else "hippocampal_subfield"
        if role == "ca3_micro_candidate":
            ca3_voxels += int(np.count_nonzero(mask))
        append_label(
            out,
            mask,
            labels,
            name=name,
            source=f"hippunfold:{side}:{value}",
            role=role,
        )
    return ca3_voxels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HCP 100206 CA3 whole-brain inputs for MIND Sim.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Check HCP, HippUnfold, hybrid atlas, diffusion, and MRtrix inputs.")
    check.add_argument("--hcp-root", type=Path, default=Path("HCP_1200"), help="HCP root path from the examples/ directory.")
    check.add_argument("--subject", default="100206")
    check.add_argument("--bids-root", type=Path, default=Path("ca3_epilepsy_cosim/outputs/hcp_bids"))
    check.add_argument("--hippunfold-root", type=Path, default=Path("ca3_epilepsy_cosim/outputs/hippunfold"))
    check.add_argument("--atlas", default="multihist7")
    check.add_argument("--output", type=Path, default=None)
    check.set_defaults(func=run_check)

    bids = subparsers.add_parser("prepare-bids", help="Create minimal BIDS anat symlinks for HippUnfold.")
    bids.add_argument("--hcp-root", type=Path, default=Path("HCP_1200"), help="HCP root path from the examples/ directory.")
    bids.add_argument("--subject", default="100206")
    bids.add_argument("--output-bids", type=Path, default=Path("ca3_epilepsy_cosim/outputs/hcp_bids"))
    bids.add_argument("--replace", action="store_true")
    bids.set_defaults(func=run_prepare_bids)

    build = subparsers.add_parser(
        "build-parcellation",
        help="Replace base hippocampus labels with HippUnfold T1w-space subfields.",
    )
    add_build_parcellation_args(build)
    build.set_defaults(func=run_build_parcellation)

    connectome = subparsers.add_parser(
        "connectivity-csv",
        help="Convert MRtrix connectome matrices plus ROI labels into one labelled MIND Sim connectivity CSV.",
    )
    connectome.add_argument("--weights-csv", type=Path, required=True)
    connectome.add_argument("--labels-tsv", type=Path, required=True)
    connectome.add_argument("--lengths-csv", type=Path, default=None)
    connectome.add_argument("--output", type=Path, required=True)
    connectome.add_argument("--normalize", choices=("none", "global", "row"), default="global")
    connectome.add_argument("--conduction-speed-mm-per-ms", type=float, default=5.0)
    connectome.add_argument("--default-delay-ms", type=float, default=10.0)
    connectome.set_defaults(func=run_connectivity_csv)
    return parser.parse_args()


def add_build_parcellation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-parcellation", type=Path, required=True, help="Whole-brain aparc+aseg NIfTI in T1w space.")
    parser.add_argument(
        "--base-labels",
        type=Path,
        default=Path("ca3_epilepsy_cosim/data/freesurfer_color_lut.tsv"),
        help="TSV with index/name columns for the base parcellation.",
    )
    parser.add_argument("--left-hippunfold-dseg", type=Path, required=True, help="Left HippUnfold space-T1w desc-subfields dseg.")
    parser.add_argument("--right-hippunfold-dseg", type=Path, required=True, help="Right HippUnfold space-T1w desc-subfields dseg.")
    parser.add_argument("--hippunfold-labels-tsv", type=Path, required=True, help="HippUnfold index/name label TSV or subject volumes.tsv.")
    parser.add_argument("--left-hippocampus-label", type=int, default=17)
    parser.add_argument("--right-hippocampus-label", type=int, default=53)
    parser.add_argument(
        "--hippocampus-policy",
        choices=("hippunfold",),
        default="hippunfold",
        help=(
            "Use HippUnfold as the authoritative hippocampus definition: base "
            "left/right hippocampus labels are removed, non-zero HippUnfold "
            "subfield voxels are inserted, and base hippocampus voxels not "
            "covered by HippUnfold remain background."
        ),
    )
    parser.add_argument("--ca3-name-pattern", default=r"(^|[^A-Za-z0-9])CA3([^A-Za-z0-9]|$)")
    parser.add_argument("--output-volume", type=Path, required=True)
    parser.add_argument("--output-labels", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, default=None)


def run_check(args: argparse.Namespace) -> None:
    subject_root = args.hcp_root / args.subject
    t1w_dir = subject_root / "T1w"
    diffusion_dir = t1w_dir / "Diffusion"
    bids_anat = args.bids_root / f"sub-{args.subject}" / "anat"
    hipp_anat = args.hippunfold_root / f"sub-{args.subject}" / "anat"
    volumes_tsv = hipp_anat / f"sub-{args.subject}_space-cropT1w_desc-subfields_atlas-{args.atlas}_volumes.tsv"
    outputs_root = Path("ca3_epilepsy_cosim/outputs")

    structural = {
        "t1w_acpc_restore": file_status(t1w_dir / "T1w_acpc_dc_restore.nii.gz"),
        "t2w_acpc_restore": file_status(t1w_dir / "T2w_acpc_dc_restore.nii.gz"),
        "aparc_aseg_t1w": file_status(t1w_dir / "aparc+aseg.nii.gz"),
        "wmparc_t1w": file_status(t1w_dir / "wmparc.nii.gz"),
    }
    bids = {
        "dataset_description": file_status(args.bids_root / "dataset_description.json"),
        "t1w": file_status(bids_anat / f"sub-{args.subject}_T1w.nii.gz"),
        "t2w": file_status(bids_anat / f"sub-{args.subject}_T2w.nii.gz"),
    }
    hippunfold = {
        "left_space_t1w_subfields": file_status(
            hipp_anat
            / f"sub-{args.subject}_hemi-L_space-T1w_label-hipp_desc-subfields_atlas-{args.atlas}_dseg.nii.gz"
        ),
        "right_space_t1w_subfields": file_status(
            hipp_anat
            / f"sub-{args.subject}_hemi-R_space-T1w_label-hipp_desc-subfields_atlas-{args.atlas}_dseg.nii.gz"
        ),
        "volumes_tsv": file_status(volumes_tsv),
    }
    hybrid = {
        "volume": file_status(outputs_root / f"{args.subject}_hybrid_ca3.nii.gz"),
        "labels": file_status(outputs_root / f"{args.subject}_hybrid_ca3_labels.tsv"),
        "metadata": file_status(outputs_root / f"{args.subject}_hybrid_ca3_metadata.json"),
        "weights_csv": file_status(outputs_root / f"{args.subject}_hybrid_ca3_weights.csv"),
        "lengths_csv": file_status(outputs_root / f"{args.subject}_hybrid_ca3_lengths.csv"),
        "connectivity_csv": file_status(outputs_root / f"{args.subject}_hybrid_ca3_connectivity.csv"),
    }
    diffusion = {
        "data_nii_gz": file_status(diffusion_dir / "data.nii.gz"),
        "bvals": file_status(diffusion_dir / "bvals"),
        "bvecs": file_status(diffusion_dir / "bvecs"),
        "nodif_brain_mask": file_status(diffusion_dir / "nodif_brain_mask.nii.gz"),
        "grad_dev": file_status(diffusion_dir / "grad_dev.nii.gz"),
    }
    tools = {
        name: command_status(name)
        for name in (
            "hippunfold",
            "mrconvert",
            "dwi2response",
            "dwi2fod",
            "mtnormalise",
            "5ttgen",
            "5tt2gmwmi",
            "tckgen",
            "tcksift2",
            "tck2connectome",
        )
    }

    structural_ready = all(item["exists"] for item in structural.values())
    bids_ready = all(item["exists"] for item in bids.values())
    hippunfold_ready = all(item["exists"] for item in hippunfold.values())
    hybrid_ready = all(item["exists"] for item in hybrid.values())
    diffusion_ready = all(
        item["exists"]
        for key, item in diffusion.items()
        if key in {"data_nii_gz", "bvals", "bvecs", "nodif_brain_mask"}
    )
    mrtrix_ready = all(
        tools[name]["available"]
        for name in (
            "mrconvert",
            "dwi2response",
            "dwi2fod",
            "mtnormalise",
            "5ttgen",
            "5tt2gmwmi",
            "tckgen",
            "tcksift2",
            "tck2connectome",
        )
    )

    report = {
        "subject": args.subject,
        "hcp_root": str(args.hcp_root),
        "bids_root": str(args.bids_root),
        "hippunfold_root": str(args.hippunfold_root),
        "atlas": args.atlas,
        "structural": structural,
        "bids": bids,
        "hippunfold": hippunfold,
        "hybrid_ca3": hybrid,
        "diffusion": diffusion,
        "tools": tools,
        "ready": {
            "structural_inputs": structural_ready,
            "bids_inputs": bids_ready,
            "hippunfold_tool": tools["hippunfold"]["available"],
            "hippunfold_outputs": hippunfold_ready,
            "hybrid_ca3_outputs": hybrid_ready,
            "diffusion_inputs": diffusion_ready,
            "mrtrix_tools": mrtrix_ready,
            "can_prepare_bids": structural_ready,
            "can_run_hippunfold_cli": bids_ready and tools["hippunfold"]["available"],
            "can_build_hybrid_ca3_atlas": structural_ready and hippunfold_ready,
            "can_build_diffusion_connectome": diffusion_ready and mrtrix_ready,
            "has_ready_hybrid_connectivity": hybrid_ready,
        },
    }

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


def run_prepare_bids(args: argparse.Namespace) -> None:
    source_root = args.hcp_root / args.subject / "T1w"
    anat_dir = args.output_bids / f"sub-{args.subject}" / "anat"
    relative_symlink(source_root / "T1w_acpc_dc_restore.nii.gz", anat_dir / f"sub-{args.subject}_T1w.nii.gz", replace=args.replace)
    relative_symlink(source_root / "T2w_acpc_dc_restore.nii.gz", anat_dir / f"sub-{args.subject}_T2w.nii.gz", replace=args.replace)

    description = {
        "Name": "HCP structural symlinks for HippUnfold",
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "GeneratedBy": [{"Name": "prepare_hcp100206_ca3.py prepare-bids"}],
    }
    desc_path = args.output_bids / "dataset_description.json"
    if args.replace or not desc_path.exists():
        desc_path.parent.mkdir(parents=True, exist_ok=True)
        desc_path.write_text(json.dumps(description, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"bids_dir={args.output_bids}")
    print(f"subject=sub-{args.subject}")
    print(f"anat_dir={anat_dir}")


def run_build_parcellation(args: argparse.Namespace) -> None:
    nib = require_nibabel()

    base_img, base = load_int_volume(nib, args.base_parcellation)
    _, left_dseg = load_int_volume(nib, args.left_hippunfold_dseg)
    _, right_dseg = load_int_volume(nib, args.right_hippunfold_dseg)

    base_names = parse_index_name_table(args.base_labels)
    observed_subfield_values = sorted(
        set(int(x) for x in np.unique(left_dseg) if int(x) != 0)
        | set(int(x) for x in np.unique(right_dseg) if int(x) != 0)
    )
    hippunfold_names = parse_hippunfold_label_table(args.hippunfold_labels_tsv, observed_subfield_values)
    ca3_pattern = re.compile(args.ca3_name_pattern, flags=re.IGNORECASE)

    left_hippocampus = base == int(args.left_hippocampus_label)
    right_hippocampus = base == int(args.right_hippocampus_label)
    left_right_overlap = int(np.count_nonzero((left_dseg != 0) & (right_dseg != 0)))
    hippunfold_mask = (left_dseg != 0) | (right_dseg != 0)

    out = np.zeros(base.shape, dtype=np.int32)
    labels: list[dict[str, object]] = []
    skip_base_values = {0, int(args.left_hippocampus_label), int(args.right_hippocampus_label)}
    append_base_labels(out, labels, base, base_names, skip_base_values, hippunfold_mask)
    left_ca3_voxels = append_hippunfold_subfields(
        out,
        labels,
        side="Left",
        dseg=left_dseg,
        label_names=hippunfold_names,
        ca3_pattern=ca3_pattern,
    )
    right_ca3_voxels = append_hippunfold_subfields(
        out,
        labels,
        side="Right",
        dseg=right_dseg,
        label_names=hippunfold_names,
        ca3_pattern=ca3_pattern,
    )
    if left_ca3_voxels == 0:
        raise SystemExit(f"left HippUnfold dseg has no label matching {args.ca3_name_pattern!r}")
    if right_ca3_voxels == 0:
        raise SystemExit(f"right HippUnfold dseg has no label matching {args.ca3_name_pattern!r}")

    output_volume = args.output_volume
    output_labels = args.output_labels
    output_volume.parent.mkdir(parents=True, exist_ok=True)
    output_labels.parent.mkdir(parents=True, exist_ok=True)

    if output_volume.name.endswith(".mgz"):
        out_img = nib.MGHImage(out, base_img.affine, header=base_img.header.copy())
    else:
        out_img = nib.Nifti1Image(out, base_img.affine)
    nib.save(out_img, str(output_volume))

    with output_labels.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "name", "source", "role", "voxel_count"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(labels)

    left_outside_base = int(np.count_nonzero((left_dseg != 0) & ~left_hippocampus))
    right_outside_base = int(np.count_nonzero((right_dseg != 0) & ~right_hippocampus))
    left_base_uncovered = int(np.count_nonzero(left_hippocampus & (left_dseg == 0)))
    right_base_uncovered = int(np.count_nonzero(right_hippocampus & (right_dseg == 0)))
    base_non_hippocampal = (
        (base != 0)
        & (base != int(args.left_hippocampus_label))
        & (base != int(args.right_hippocampus_label))
    )
    base_non_hippocampal_reserved_for_hippunfold = int(np.count_nonzero(hippunfold_mask & base_non_hippocampal))
    metadata = {
        "base_parcellation": str(args.base_parcellation),
        "base_labels": str(args.base_labels),
        "hippocampus_policy": args.hippocampus_policy,
        "hippocampus_policy_description": (
            "HippUnfold-authoritative. Base aparc+aseg left/right hippocampus "
            "labels are excluded from the whole-brain atlas. Non-zero "
            "HippUnfold dseg voxels are reserved before base labels are copied, "
            "so base non-hippocampal ROI cannot also occupy those voxels. Base "
            "hippocampus voxels not covered by HippUnfold remain background."
        ),
        "left_hippunfold_dseg": str(args.left_hippunfold_dseg),
        "right_hippunfold_dseg": str(args.right_hippunfold_dseg),
        "hippunfold_labels_tsv": str(args.hippunfold_labels_tsv),
        "left_right_hippunfold_overlap_voxels": left_right_overlap,
        "base_non_hippocampal_voxels_reserved_for_hippunfold": base_non_hippocampal_reserved_for_hippunfold,
        "left_ca3_voxels": left_ca3_voxels,
        "right_ca3_voxels": right_ca3_voxels,
        "left_hippunfold_voxels_outside_base_hippocampus": left_outside_base,
        "right_hippunfold_voxels_outside_base_hippocampus": right_outside_base,
        "left_base_hippocampus_voxels_not_covered_by_hippunfold": left_base_uncovered,
        "right_base_hippocampus_voxels_not_covered_by_hippunfold": right_base_uncovered,
        "label_count": len(labels),
        "notes": (
            "Base aparc+aseg left/right hippocampus labels are removed and "
            "replaced by HippUnfold hippocampal subfield labels in T1w space. "
            "HippUnfold-labelled voxels are removed from the base ROI masks "
            "before base labels are written, preventing double ownership. No "
            "residual hippocampus ROI is fabricated; base hippocampus voxels "
            "not covered by HippUnfold remain background."
        ),
    }
    if args.output_metadata is not None:
        args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
        args.output_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"output_volume={output_volume}")
    print(f"output_labels={output_labels}")
    print(f"label_count={len(labels)}")
    print(f"left_ca3_voxels={left_ca3_voxels}")
    print(f"right_ca3_voxels={right_ca3_voxels}")


def read_labels(path: Path) -> list[str]:
    labels: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or "name" not in reader.fieldnames:
            raise SystemExit(f"{path} must be a TSV with a 'name' column")
        for row in reader:
            name = (row.get("name") or "").strip()
            if name:
                labels.append(name)
    if not labels:
        raise SystemExit(f"no labels read from {path}")
    if len(set(labels)) != len(labels):
        raise SystemExit(f"{path} contains duplicate ROI names")
    return labels


def read_matrix(path: Path) -> np.ndarray:
    try:
        matrix = np.loadtxt(path, delimiter=",")
    except ValueError:
        matrix = np.loadtxt(path)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise SystemExit(f"{path} must contain a square matrix; got shape {matrix.shape}")
    matrix[~np.isfinite(matrix)] = 0.0
    if np.any(matrix < 0.0):
        raise SystemExit(f"{path} contains negative values")
    np.fill_diagonal(matrix, 0.0)
    return matrix


def normalize_weights(weights: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return weights
    out = weights.astype(float, copy=True)
    if mode == "global":
        peak = float(out.max(initial=0.0))
        if peak > 0.0:
            out /= peak
        return out
    if mode == "row":
        row_sum = out.sum(axis=1, keepdims=True)
        np.divide(out, row_sum, out=out, where=row_sum > 0.0)
        return out
    raise ValueError(mode)


def write_matrix_connectivity(path: Path, labels: list[str], weights: np.ndarray, delays: np.ndarray) -> None:
    if len(set(labels)) != len(labels):
        raise ValueError("labels contain duplicates")
    weights = np.asarray(weights, dtype=float)
    delays = np.asarray(delays, dtype=float)
    expected_shape = (len(labels), len(labels))
    if weights.shape != expected_shape:
        raise ValueError(f"weights shape {weights.shape} does not match {expected_shape}")
    if delays.shape != expected_shape:
        raise ValueError(f"delays shape {delays.shape} does not match {expected_shape}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for section, matrix in (("weights", weights), ("delays_ms", delays)):
            writer.writerow(["matrix", section])
            writer.writerow(["roi", *labels])
            for label, row in zip(labels, matrix):
                writer.writerow([label, *[f"{float(value):.17g}" for value in row]])
            writer.writerow([])


def run_connectivity_csv(args: argparse.Namespace) -> None:
    labels = read_labels(args.labels_tsv)
    weights = normalize_weights(read_matrix(args.weights_csv), args.normalize)
    if weights.shape != (len(labels), len(labels)):
        raise SystemExit(f"weights shape {weights.shape} does not match {len(labels)} labels")

    if args.lengths_csv is not None:
        lengths = read_matrix(args.lengths_csv)
        if lengths.shape != weights.shape:
            raise SystemExit(f"lengths shape {lengths.shape} does not match weights shape {weights.shape}")
        if args.conduction_speed_mm_per_ms <= 0.0:
            raise SystemExit("--conduction-speed-mm-per-ms must be positive")
        delays = np.where(weights > 0.0, lengths / float(args.conduction_speed_mm_per_ms), 0.0)
    else:
        delays = np.where(weights > 0.0, float(args.default_delay_ms), 0.0)

    write_matrix_connectivity(args.output, labels, weights, delays)
    print(f"output={args.output}")
    print(f"labels={len(labels)}")
    print(f"positive_edges={int(np.count_nonzero(weights > 0.0))}")
    print(f"weight_max={float(weights.max(initial=0.0)):.6g}")
    print(f"delay_max_ms={float(delays.max(initial=0.0)):.6g}")


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
