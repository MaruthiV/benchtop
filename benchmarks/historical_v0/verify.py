import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify(snapshot_dir, working_tree=None):
    manifest = json.loads((snapshot_dir / "manifest.json").read_text())
    archive = snapshot_dir / manifest["archive"]["path"]
    raw = archive.read_bytes()
    if len(raw) != manifest["archive"]["bytes"] or digest(raw) != manifest["archive"]["sha256"]:
        raise ValueError("snapshot archive checksum mismatch")

    expected = manifest["files"]
    seen = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError(f"unexpected archive member: {member.name}")
            with tar.extractfile(member) as source:
                data = source.read()
            record = expected[member.name]
            if len(data) != record["bytes"] or digest(data) != record["sha256"]:
                raise ValueError(f"archived file checksum mismatch: {member.name}")
            seen.add(member.name)
    if seen != set(expected) or len(seen) != manifest["archive"]["file_count"]:
        raise ValueError("archive file inventory mismatch")

    if working_tree is not None:
        for name, record in expected.items():
            path = working_tree / name
            if not path.is_file() or digest(path.read_bytes()) != record["sha256"]:
                raise ValueError(f"working file differs from historical baseline: {name}")
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description="Verify the frozen historical Benchtop baseline.")
    parser.add_argument("--working-tree", action="store_true", help="also compare the current project files")
    args = parser.parse_args()
    snapshot_dir = Path(__file__).resolve().parent
    try:
        count = verify(snapshot_dir, snapshot_dir.parent.parent if args.working_tree else None)
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        parser.exit(1, f"verification failed: {error}\n")
    suffix = " and current project files" if args.working_tree else ""
    print(f"Verified {count} historical files{suffix}.")


if __name__ == "__main__":
    main()
