#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUT_DIR" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$1"
SRC_SHA="$(git -C "$ROOT" rev-parse --short HEAD)"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

git -C "$ROOT" archive --format=tar --worktree-attributes HEAD | tar -xf - -C "$OUT_DIR"

git -C "$OUT_DIR" init -b main >/dev/null
git -C "$OUT_DIR" config user.name "Public Mirror"
git -C "$OUT_DIR" config user.email "public-mirror@example.invalid"
git -C "$OUT_DIR" add .
git -C "$OUT_DIR" commit -m "Public mirror from ${SRC_SHA}" >/dev/null

echo "[public-mirror] wrote clean mirror to $OUT_DIR"
echo "[public-mirror] source commit: $SRC_SHA"
