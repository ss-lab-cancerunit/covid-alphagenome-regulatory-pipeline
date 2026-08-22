#!/usr/bin/env python3
"""Fail if the candidate repository contains common Git hygiene problems."""

from __future__ import annotations

import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
excluded = {
    ".git", "input", "results", "logs", "reference", "weights",
    "external", "legacy",
}
archive_suffixes = (".tar", ".tar.gz", ".tgz", ".zip", ".7z", ".rar")
secret_pattern = re.compile(
    r"(github_pat_|ghp_[A-Za-z0-9]{20,}|BEGIN (RSA|OPENSSH) PRIVATE KEY|"
    r"(?:api[_-]?key|password|passwd|secret|token)\s*[=:]\s*[^\s]+)",
    re.IGNORECASE,
)
problems = []

for path in root.rglob("*"):
    rel = path.relative_to(root)
    if any(part in excluded for part in rel.parts):
        continue
    if path.is_dir():
        if path.name == ".git":
            problems.append(f"nested Git directory: {rel}")
        continue
    if path.stat().st_size > 50 * 1024 * 1024:
        problems.append(f"file larger than 50 MiB: {rel}")
    if path.name.endswith(archive_suffixes):
        problems.append(f"archive file: {rel}")
    if rel.as_posix() == "scripts/audit_repository.py":
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue
    if "/rds/general/user/" in text:
        problems.append(f"hard-coded RDS user path: {rel}")
    if secret_pattern.search(text):
        problems.append(f"possible credential: {rel}")

if problems:
    print("Repository audit FAILED:")
    for problem in sorted(set(problems)):
        print(" -", problem)
    sys.exit(1)
print("Repository audit passed")
