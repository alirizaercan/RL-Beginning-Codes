#!/usr/bin/env python3
"""Fix local Qwen tokenizer_config.json for ms-swift compatibility."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Model directory containing tokenizer_config.json")
    parser.add_argument("--backup-suffix", default=".bak", help="Suffix for backup file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    config_path = model_dir / "tokenizer_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"tokenizer_config.json not found: {config_path}")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    extra = data.get("extra_special_tokens")
    if isinstance(extra, dict):
        print("No change needed: extra_special_tokens is already a dict.")
        return 0

    backup_path = config_path.with_name(config_path.name + args.backup_suffix)
    shutil.copy2(config_path, backup_path)
    data["extra_special_tokens"] = {}
    config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Patched: {config_path}")
    print(f"Backup : {backup_path}")
    print("Set extra_special_tokens to {}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
