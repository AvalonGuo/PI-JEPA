#!/usr/bin/env python3
"""Generic, idempotent uploader for ModelScope model repositories.

Workflow: auth -> ensure repo -> plan (small files first) -> upload (LFS auto)
          -> verify remote sizes -> optional watch for platform-side deletion.

Long uploads must be detached, otherwise the parent shell timeout kills the child:
    setsid nohup python tools/upload_modelscope.py ... > /hy-tmp/logs/ms_upload.log 2>&1 &
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

SMALL_FILE_LIMIT = 64 << 20  # informational: uploads are ordered small -> large
RETRYABLE = ("NetworkError", "ServerError", "RequestTimeoutError", "RateLimitError")


def make_api(token: str):
    """modelscope>=1.40 shim; legacy import is `from modelscope.hub.api import HubApi`."""
    from modelscope_hub.api import HubApi
    return HubApi(token=token)


def resolve_token(args) -> str:
    if args.token:
        return args.token.strip()
    if args.token_file:
        return Path(args.token_file).read_text(encoding="utf-8").strip()
    for key in ("MODELSCOPE_API_TOKEN", "MS_TOKEN"):
        if os.environ.get(key):
            return os.environ[key].strip()
    raise SystemExit("no token: use --token, --token-file or export MS_TOKEN")


def sha256_of(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_repo(api, repo: str, license_name: str, create: bool) -> None:
    from modelscope_hub.errors import AlreadyExistsError

    try:
        if api.repo_exists(repo, "model"):
            print("[repo] {} already exists".format(repo))
            return
    except Exception as exc:  # noqa: BLE001 - auth errors surface here too
        print("[repo] repo_exists failed: {}: {}".format(type(exc).__name__, exc))
    if not create:
        raise SystemExit("[repo] {} missing and --no-create given".format(repo))
    try:
        api.create_repo(repo, "model", visibility="public", license=license_name)
        print("[repo] created {} (public, {})".format(repo, license_name))
    except AlreadyExistsError:
        print("[repo] {} appeared meanwhile, continuing".format(repo))


def plan(files, dest: str):
    """[(local Path, remote path)] sorted ascending by size (small files first)."""
    items = []
    for raw in files:
        local_s, _, rel = raw.partition("::")
        local = Path(local_s).expanduser()
        if not local.is_file():
            raise SystemExit("[plan] not a file: {}".format(local))
        remote = "{}/{}".format(dest.strip("/"), (rel or local.name).strip("/"))
        items.append((local, remote))
    items.sort(key=lambda pair: pair[0].stat().st_size)
    return items


def remote_sizes(api, repo: str) -> dict:
    from modelscope_hub.errors import NotExistError

    try:
        infos = api.list_repo_files(repo, "model", recursive=True)
    except NotExistError:
        return {}
    return {info.path: int(info.size or 0) for info in infos if not info.is_dir}


def upload_one(api, repo: str, local: Path, remote: str, attempts: int = 3) -> bool:
    """upload_file() picks LFS automatically (.ckpt/.bin/... or above the size threshold)."""
    last = None
    for attempt in range(1, attempts + 1):
        try:
            api.upload_file(repo, "model", path_or_fileobj=str(local),
                            path_in_repo=remote,
                            commit_message="Upload {}".format(remote))
            return True
        except Exception as exc:  # noqa: BLE001
            last = exc
            if type(exc).__name__ not in RETRYABLE:
                print("[fail] {}: {}: {}".format(remote, type(exc).__name__, exc))
                return False
            print("[retry {}/{}] {}: {}".format(attempt, attempts, remote, exc))
            time.sleep(5 * attempt)
    print("[fail] {}: {}".format(remote, last))
    return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Upload artifacts to a ModelScope model repo")
    parser.add_argument("--repo", required=True, help="owner/name, e.g. my-org/PI-JEPA")
    parser.add_argument("--dest", required=True, help="remote folder, e.g. metaworld_assembly-pijepa")
    parser.add_argument("--file", action="append", required=True, metavar="LOCAL[::REMOTE_REL]",
                        help="repeatable; REMOTE_REL defaults to the file name")
    parser.add_argument("--token")
    parser.add_argument("--token-file")
    parser.add_argument("--license", default="apache-2.0")
    parser.add_argument("--no-create", action="store_true", help="never create the repo")
    parser.add_argument("--force", action="store_true", help="re-upload even if sizes match")
    parser.add_argument("--sha256", action="store_true", help="print local sha256 before upload")
    parser.add_argument("--watch", type=int, default=0, metavar="SECONDS",
                        help="re-check the remote listing once after N seconds")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    api = make_api(resolve_token(args))
    items = plan(args.file, args.dest)
    print("[plan] {} file(s) -> {}/ ({} total)".format(
        len(items), args.dest.strip("/"),
        ", ".join("{} {:.1f}MiB".format(p.name, p.stat().st_size / 2**20) for p, _ in items)))
    if args.sha256:
        for local, _ in items:
            print("[sha256] {} {}".format(sha256_of(local), local.name))
    if args.dry_run:
        return 0

    ensure_repo(api, args.repo, args.license, create=not args.no_create)
    before = remote_sizes(api, args.repo)
    failed = []
    for local, remote in items:
        size = local.stat().st_size
        if not args.force and before.get(remote) == size:
            print("[skip] {} (remote size matches: {})".format(remote, size))
            continue
        print("[upload] {} -> {}:{} ({:.2f} MiB)".format(local, args.repo, remote, size / 2**20))
        if not upload_one(api, args.repo, local, remote):
            failed.append(remote)

    after = remote_sizes(api, args.repo)
    print("\n[verify] {} status".format(args.repo))
    ok = True
    for local, remote in items:
        want, got = local.stat().st_size, after.get(remote)
        ok = ok and want == got
        print("  {:<55} local={:>12}  remote={:>12}  {}".format(
            remote, want, got if got is not None else "MISSING", "OK" if want == got else "MISMATCH"))

    if args.watch:
        print("[watch] sleeping {}s before the second listing".format(args.watch))
        time.sleep(args.watch)
        late = remote_sizes(api, args.repo)
        dropped = [r for _, r in items if r not in late]
        print("[watch] {}".format("still present" if not dropped else "DELETED: {}".format(dropped)))
        ok = ok and not dropped

    return 0 if ok and not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
