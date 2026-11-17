import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RUNTIME_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
MANIFEST = "release-manifest.json"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def public_files(root):
    result = {}
    for folder, dirs, files in os.walk(root):
        for name in dirs:
            if name not in RUNTIME_DIRS and (Path(folder) / name).is_symlink():
                raise ValueError("release directories cannot be symlinks")
        dirs[:] = [name for name in dirs if name not in RUNTIME_DIRS]
        for name in files:
            if name == ".git" or name.endswith(".pyc") or name == ".DS_Store":
                continue
            path = Path(folder) / name
            if path.is_symlink():
                raise ValueError("release files cannot be symlinks: " + str(path.relative_to(root)))
            result[str(path.relative_to(root))] = digest(path)
    return result


def main():
    parser = argparse.ArgumentParser(description="Verify and reproduce the release from a disposable copy, without models.")
    parser.add_argument("--skip-classical-replay", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / MANIFEST).read_text())
    if manifest.get("format") != "benchtop_public_release_v1" or manifest.get("protocol_profile") not in {"release_v1", "release_v2"}:
        raise ValueError("unsupported release format or protocol")
    expected = manifest["files"]
    actual = public_files(root)
    actual.pop(MANIFEST)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(name for name in set(actual) & set(expected) if actual[name] != expected[name])
        raise ValueError("release files differ: " + json.dumps({"missing": missing, "extra": extra, "changed": changed}))
    with tempfile.TemporaryDirectory(prefix="benchtop-release-reproduce-") as temporary:
        clean = Path(temporary)
        for name in [MANIFEST, *expected]:
            source = root / name
            if not source.resolve().is_relative_to(root) or Path(name).is_absolute():
                raise ValueError("invalid release path")
            target = clean / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        environment = dict(os.environ)
        environment["BENCHTOP_PROTOCOL"] = manifest["protocol_profile"]
        environment.pop("PYTHONPATH", None)
        command = [sys.executable, "-m", "benchtop.cli", "release", "reproduce", str(clean)]
        if args.skip_classical_replay:
            command.append("--skip-classical-replay")
        subprocess.run(command, cwd=clean, env=environment, check=True)
        subprocess.run([sys.executable, "-m", "publication.verify_supplement"], cwd=clean, env=environment, check=True)


if __name__ == "__main__":
    main()
