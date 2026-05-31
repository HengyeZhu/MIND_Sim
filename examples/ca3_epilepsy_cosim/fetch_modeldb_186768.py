#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO_URL = "https://github.com/ModelDBRepository/186768"
DEFAULT_COMMIT = "e60d5e99eb2a6836485b35cc3f8aae0085bee79f"
HERE = Path(__file__).resolve().parent
TARGET = HERE / "modeldb_186768"


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch ModelDB 186768 into this example directory.")
    parser.add_argument("--commit", default=DEFAULT_COMMIT, help="Git commit or ref to vendor.")
    parser.add_argument("--target", type=Path, default=TARGET, help="Target directory.")
    parser.add_argument("--force", action="store_true", help="Replace an existing target directory.")
    args = parser.parse_args()

    target = args.target.expanduser().resolve()
    if target.exists():
        if not args.force:
            raise SystemExit(f"{target} already exists; pass --force to replace it")
        shutil.rmtree(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="modeldb_186768_") as tmp:
        tmp_path = Path(tmp)
        clone_dir = tmp_path / "repo"
        run(["git", "clone", "--filter=blob:none", REPO_URL, str(clone_dir)])
        run(["git", "checkout", args.commit], cwd=clone_dir)
        archive = tmp_path / "source.tar"
        with archive.open("wb") as fh:
            subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=clone_dir, check=True, stdout=fh)
        target.mkdir(parents=True, exist_ok=True)
        run(["tar", "-xf", str(archive), "-C", str(target)])

    print(f"Fetched ModelDB 186768 into {target}")


if __name__ == "__main__":
    main()
