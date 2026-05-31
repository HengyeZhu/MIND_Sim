#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_LEFT_HIPPOCAMPUS = 17
DEFAULT_RIGHT_HIPPOCAMPUS = 53
DEFAULT_CA3_VALUES = "208"


def require_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise SystemExit(
            "This script requires nibabel. Install it in the data-prep environment, "
            "for example: conda install -c conda-forge nibabel"
        ) from exc
    return nib


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for item in str(text).replace(",", " ").split():
        if item:
            values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer label value")
    return values


def parse_lut(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    names: dict[int, str] = {}
    with path.expanduser().open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                label = int(parts[0])
            except ValueError:
                continue
            names[label] = parts[1]
    return names


def load_int_volume(nib, path: Path) -> tuple[object, np.ndarray]:
    img = nib.load(str(path.expanduser()))
    data = np.asanyarray(img.dataobj)
    if not np.issubdtype(data.dtype, np.integer):
        data = np.rint(data).astype(np.int32)
    else:
        data = data.astype(np.int32, copy=False)
    return img, data


def check_same_space(base_img, base_data: np.ndarray, other_img, other_data: np.ndarray, label: str) -> None:
    if other_data.shape != base_data.shape:
        raise SystemExit(
            f"{label} shape {other_data.shape} does not match base parcellation shape {base_data.shape}. "
            "Use the FreeSurfer *.FSvoxelSpace.mgz hippocampal-subfield outputs or resample to the base volume first."
        )
    if not np.allclose(other_img.affine, base_img.affine, rtol=0.0, atol=1e-5):
        raise SystemExit(
            f"{label} affine does not match the base parcellation. "
            "Use the FreeSurfer *.FSvoxelSpace.mgz files in the same subject mri/ space."
        )


def add_label(
    out: np.ndarray,
    mask: np.ndarray,
    labels: list[dict],
    *,
    name: str,
    source_label: str,
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
            "source_label": source_label,
            "role": role,
            "voxel_count": int(np.count_nonzero(mask)),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a contiguous whole-brain parcellation that preserves the FreeSurfer "
            "whole-brain atlas but splits out left/right CA3 from hippocampal-subfield labels."
        )
    )
    parser.add_argument("--base-parcellation", type=Path, required=True, help="Usually SUBJECTS_DIR/<subject>/mri/aparc+aseg.mgz")
    parser.add_argument("--left-hippo-labels", type=Path, required=True, help="Usually lh.hippoAmygLabels-T1.v21.CA.FSvoxelSpace.mgz")
    parser.add_argument("--right-hippo-labels", type=Path, required=True, help="Usually rh.hippoAmygLabels-T1.v21.CA.FSvoxelSpace.mgz")
    parser.add_argument("--lut", type=Path, default=None, help="Optional FreeSurferColorLUT.txt for readable base ROI names")
    parser.add_argument("--left-ca3-values", type=parse_int_list, default=parse_int_list(DEFAULT_CA3_VALUES))
    parser.add_argument("--right-ca3-values", type=parse_int_list, default=parse_int_list(DEFAULT_CA3_VALUES))
    parser.add_argument("--left-hippocampus-label", type=int, default=DEFAULT_LEFT_HIPPOCAMPUS)
    parser.add_argument("--right-hippocampus-label", type=int, default=DEFAULT_RIGHT_HIPPOCAMPUS)
    parser.add_argument("--left-ca3-name", default="Left-CA3")
    parser.add_argument("--right-ca3-name", default="Right-CA3")
    parser.add_argument("--left-residual-name", default="Left-hippocampus-nonCA3")
    parser.add_argument("--right-residual-name", default="Right-hippocampus-nonCA3")
    parser.add_argument("--output-volume", type=Path, required=True)
    parser.add_argument("--output-labels", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nib = require_nibabel()

    base_img, base = load_int_volume(nib, args.base_parcellation)
    left_img, left = load_int_volume(nib, args.left_hippo_labels)
    right_img, right = load_int_volume(nib, args.right_hippo_labels)
    check_same_space(base_img, base, left_img, left, "left hippocampal-subfield labels")
    check_same_space(base_img, base, right_img, right, "right hippocampal-subfield labels")

    lut_names = parse_lut(args.lut)
    left_ca3 = np.isin(left, args.left_ca3_values)
    right_ca3 = np.isin(right, args.right_ca3_values)
    left_hippo = base == int(args.left_hippocampus_label)
    right_hippo = base == int(args.right_hippocampus_label)

    if not np.any(left_ca3):
        raise SystemExit(f"no left CA3 voxels found for values {args.left_ca3_values}")
    if not np.any(right_ca3):
        raise SystemExit(f"no right CA3 voxels found for values {args.right_ca3_values}")

    out = np.zeros(base.shape, dtype=np.int32)
    labels: list[dict] = []

    skip = {0, int(args.left_hippocampus_label), int(args.right_hippocampus_label)}
    for source_label in sorted(int(x) for x in np.unique(base) if int(x) not in skip):
        mask = base == source_label
        add_label(
            out,
            mask,
            labels,
            name=lut_names.get(source_label, f"fs_label_{source_label}"),
            source_label=str(source_label),
            role="whole_brain_base",
        )

    add_label(
        out,
        left_hippo & ~left_ca3,
        labels,
        name=args.left_residual_name,
        source_label=f"{args.left_hippocampus_label}-minus-CA3",
        role="hippocampus_residual",
    )
    add_label(
        out,
        left_ca3,
        labels,
        name=args.left_ca3_name,
        source_label=",".join(str(x) for x in args.left_ca3_values),
        role="ca3_micro_candidate",
    )
    add_label(
        out,
        right_hippo & ~right_ca3,
        labels,
        name=args.right_residual_name,
        source_label=f"{args.right_hippocampus_label}-minus-CA3",
        role="hippocampus_residual",
    )
    add_label(
        out,
        right_ca3,
        labels,
        name=args.right_ca3_name,
        source_label=",".join(str(x) for x in args.right_ca3_values),
        role="ca3_micro_candidate",
    )

    output_volume = args.output_volume.expanduser().resolve()
    output_labels = args.output_labels.expanduser().resolve()
    output_volume.parent.mkdir(parents=True, exist_ok=True)
    output_labels.parent.mkdir(parents=True, exist_ok=True)

    out_img = nib.MGHImage(out, base_img.affine, header=base_img.header.copy()) if output_volume.suffix == ".mgz" else nib.Nifti1Image(out, base_img.affine)
    nib.save(out_img, str(output_volume))

    with output_labels.open("w", encoding="utf-8") as f:
        f.write("index\tname\tsource_label\trole\tvoxel_count\n")
        for row in labels:
            f.write(
                f"{row['index']}\t{row['name']}\t{row['source_label']}\t{row['role']}\t{row['voxel_count']}\n"
            )

    metadata = {
        "base_parcellation": str(args.base_parcellation.expanduser().resolve()),
        "left_hippo_labels": str(args.left_hippo_labels.expanduser().resolve()),
        "right_hippo_labels": str(args.right_hippo_labels.expanduser().resolve()),
        "left_ca3_values": args.left_ca3_values,
        "right_ca3_values": args.right_ca3_values,
        "left_ca3_voxels": int(np.count_nonzero(left_ca3)),
        "right_ca3_voxels": int(np.count_nonzero(right_ca3)),
        "label_count": len(labels),
        "notes": "FreeSurfer hippocampal CA hierarchy includes CA2 in CA3; treat these CA3 labels as CA2/CA3 unless manually refined.",
    }
    if args.output_metadata is not None:
        output_metadata = args.output_metadata.expanduser().resolve()
        output_metadata.parent.mkdir(parents=True, exist_ok=True)
        output_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"output_volume={output_volume}")
    print(f"output_labels={output_labels}")
    print(f"label_count={len(labels)}")
    print(f"left_ca3_voxels={metadata['left_ca3_voxels']}")
    print(f"right_ca3_voxels={metadata['right_ca3_voxels']}")


if __name__ == "__main__":
    main()
