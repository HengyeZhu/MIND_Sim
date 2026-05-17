from __future__ import annotations

import tempfile
from pathlib import Path

import mind_sim as ms


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mapping = root / "region_mapping.txt"
        weights = root / "node_weights.txt"
        mapping.write_text("0 0\n1, 1\n# comment\n2\n", encoding="utf-8")
        weights.write_text("1 2\n3,4\n5\n", encoding="utf-8")

        node_map = ms.NodeToRoiMap.from_file(str(mapping), str(weights))
        require(node_map.node_count == 5, "node_count mismatch")
        require(list(node_map.node_to_roi) == [0, 0, 1, 1, 2], "node_to_roi mismatch")
        require(list(node_map.node_weights) == [1.0, 2.0, 3.0, 4.0, 5.0], "node_weights mismatch")

        default_weights = ms.NodeToRoiMap.from_file(str(mapping))
        require(list(default_weights.node_weights) == [1.0] * 5, "default weights mismatch")


if __name__ == "__main__":
    main()
