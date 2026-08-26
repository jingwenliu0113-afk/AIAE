#!/usr/bin/env python3
"""Which adapter is the project model, written down where it can be checked.

``final_eval.json`` decided: ``final_H2``, on lower mean masked loss over the
320 frozen held-out rows. Nothing on disk said so afterwards. The adapter sat
in a returned run directory beside the two arms it was compared against,
distinguishable only by reading a record and knowing which record to read.

This writes the one file that names it, and -- more usefully -- can re-check
that the name still describes what is on disk. Two rules shape it.

**Every path is relative to the repository root.** An absolute path records
the machine it was written on and stops being true the moment the tree moves
or is copied. There is a check that no value in the record starts with ``/``.

**Digests are recomputed, never copied.** ``--verify`` reads the bytes and
hashes them again; a pointer whose digests came from the record that produced
them would agree with itself and say nothing about the files. It also checks
that the manifest beside the weights still describes the same LoRA shape,
which is what ``load_finetuned`` validates against.

It trains nothing, loads no model, reads no test split and publishes nothing.

Usage::

  ./.venv/bin/python scripts/24_project_model.py --write
  ./.venv/bin/python scripts/24_project_model.py --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_ids import (ADAPTER, ADAPTER_REVISION, BASE_MODEL,  # noqa: E402
                           BASE_REVISION, TOKENIZER, TOKENIZER_REVISION)
from src.training.session import sha256_file, write_once_json  # noqa: E402

POINTER = "runs/project_model.json"

#: The files that make an adapter directory loadable, and what each is for.
#: Named rather than listed from disk: a pointer that recorded whatever
#: happened to be in the directory would record a stray file as part of the
#: model.
ADAPTER_FILES = ("adapter_model.safetensors", "brickagain_manifest.json",
                 "adapter_config.json")

MANIFEST_NAME = "brickagain_manifest.json"

LOAD_WITH = ("src.training.lora.load_finetuned(<adapter path>, "
             "device=..., verify_digest=True) -- base, then the published "
             "BrickGPT adapter, then the merge, then this adapter")


def relative(path, *, root: Path) -> str:
    """A repo-relative POSIX path, or a refusal.

    Refused rather than clamped: a target outside the tree cannot be recorded
    relatively at all, and recording it absolutely is the thing this avoids.
    """
    resolved = Path(path).resolve()
    root = Path(root).resolve()
    try:
        rel = resolved.relative_to(root)
    except ValueError:
        raise ValueError(
            f"{Path(path).name!r} resolves outside the repository root, so it "
            "cannot be recorded as a relative path") from None
    return rel.as_posix()


def _digest(path: Path) -> dict:
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def build_record(*, root, adapter_dir, final_eval, training_evidence) -> dict:
    """Assemble the pointer from what is on disk right now."""
    root = Path(root)
    adapter_dir = Path(adapter_dir)
    final_eval = Path(final_eval)
    training_evidence = Path(training_evidence)

    adapter_rel = relative(adapter_dir, root=root)
    eval_rel = relative(final_eval, root=root)
    evidence_rel = relative(training_evidence, root=root)

    files = {}
    for name in ADAPTER_FILES:
        blob = adapter_dir / name
        if not blob.is_file():
            raise ValueError(f"the adapter directory has no {name}")
        files[name] = _digest(blob)

    manifest = json.loads((adapter_dir / MANIFEST_NAME).read_text(
        encoding="utf-8"))
    record = json.loads(final_eval.read_text(encoding="utf-8"))
    chosen = record.get("project_model")
    if not chosen:
        raise ValueError(f"{eval_rel} names no project_model")

    return {
        "kind": "project_model",
        "model": chosen,
        "adapter": {"path": adapter_rel, "files": files},
        "lora": manifest.get("lora"),
        "revisions": {
            "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
            "published_adapter": ADAPTER,
            "published_adapter_revision": ADAPTER_REVISION,
            "tokenizer": TOKENIZER,
            "tokenizer_revision": TOKENIZER_REVISION,
        },
        "selected_by": {
            "record": eval_rel,
            "sha256": sha256_file(final_eval),
            "criterion": (record.get("criterion") or {}).get(
                "primary_criterion"),
            "means": record.get("means"),
        },
        "training_evidence": {"path": evidence_rel,
                              "sha256": sha256_file(training_evidence)},
        "load_with": LOAD_WITH,
        "note": ("A pointer, not a publication. Everything it names lives "
                 "under runs/, which does not leave this machine."),
    }


def verify_problems(body: dict, *, root) -> list[str]:
    """Everything that stops this pointer from still describing the tree.

    Every failing check is reported. A pointer that went stale in three ways
    is worth knowing about in three ways.
    """
    root = Path(root)
    problems: list[str] = []

    def absolute_values(node, trail=""):
        if isinstance(node, dict):
            for k, v in node.items():
                yield from absolute_values(v, f"{trail}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                yield from absolute_values(v, f"{trail}[{i}]")
        elif isinstance(node, str) and (node.startswith("/")
                                        or node.startswith("~")):
            yield trail, node

    for trail, value in absolute_values(body):
        problems.append(f"{trail.lstrip('.')} is an absolute path ({value})")

    adapter = body.get("adapter") or {}
    adapter_rel = adapter.get("path")
    if not isinstance(adapter_rel, str) or not adapter_rel:
        return problems + ["the pointer records no adapter path"]
    adapter_dir = root / adapter_rel
    if not adapter_dir.is_dir():
        return problems + [f"{adapter_rel} is not a directory on this machine"]

    for name, recorded in (adapter.get("files") or {}).items():
        blob = adapter_dir / name
        if not blob.is_file():
            problems.append(f"{adapter_rel}/{name} is missing")
            continue
        got = sha256_file(blob)
        if got != recorded.get("sha256"):
            problems.append(
                f"{adapter_rel}/{name} hashes to {got[:16]}..., not the "
                f"{str(recorded.get('sha256'))[:16]}... recorded")
        if blob.stat().st_size != recorded.get("bytes"):
            problems.append(f"{adapter_rel}/{name} is not the size recorded")
    for name in ADAPTER_FILES:
        if name not in (adapter.get("files") or {}):
            problems.append(f"the pointer records no digest for {name}")

    manifest_path = adapter_dir / MANIFEST_NAME
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except ValueError:
            manifest = {}
            problems.append(f"{adapter_rel}/{MANIFEST_NAME} is unreadable")
        if manifest and manifest.get("lora") != body.get("lora"):
            problems.append(
                f"the manifest beside the weights records lora "
                f"{manifest.get('lora')!r}, not the {body.get('lora')!r} this "
                "pointer records")

    chosen = body.get("model")
    selected = body.get("selected_by") or {}
    eval_rel = selected.get("record")
    if not isinstance(eval_rel, str):
        problems.append("the pointer records no final_eval path")
    else:
        record_path = root / eval_rel
        if not record_path.is_file():
            problems.append(f"{eval_rel} is missing")
        else:
            got = sha256_file(record_path)
            if got != selected.get("sha256"):
                problems.append(
                    f"final_eval {eval_rel} hashes to {got[:16]}..., not the "
                    f"{str(selected.get('sha256'))[:16]}... recorded")
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except ValueError:
                record = {}
                problems.append(f"final_eval {eval_rel} is unreadable")
            if record and record.get("project_model") != chosen:
                problems.append(
                    f"final_eval names {record.get('project_model')!r} as the "
                    f"project model, not {chosen!r}")

    evidence = body.get("training_evidence") or {}
    evidence_rel = evidence.get("path")
    if isinstance(evidence_rel, str):
        path = root / evidence_rel
        if not path.is_file():
            problems.append(f"{evidence_rel} is missing")
        elif sha256_file(path) != evidence.get("sha256"):
            problems.append(f"{evidence_rel} does not match the digest recorded")
    return problems


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Record and verify which adapter is the project model.")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--adapter-dir", metavar="DIR")
    ap.add_argument("--final-eval", metavar="FILE")
    ap.add_argument("--training-evidence", metavar="FILE")
    ap.add_argument("--pointer", metavar="FILE", default=POINTER)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    pointer = ROOT / args.pointer

    if args.write:
        for name, value in (("--adapter-dir", args.adapter_dir),
                            ("--final-eval", args.final_eval),
                            ("--training-evidence", args.training_evidence)):
            if not value:
                print(f"{name} is required with --write", file=sys.stderr)
                return 2
        try:
            body = build_record(root=ROOT, adapter_dir=args.adapter_dir,
                                final_eval=args.final_eval,
                                training_evidence=args.training_evidence)
        except ValueError as exc:
            print(f"refusing to write the pointer: {exc}", file=sys.stderr)
            return 2
        problems = verify_problems(body, root=ROOT)
        if problems:
            print("refusing to write a pointer that does not verify:",
                  file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 2
        write_once_json(pointer, body)
        print(f"wrote {args.pointer}")

    if args.verify or args.write:
        if not pointer.is_file():
            print(f"{args.pointer} does not exist", file=sys.stderr)
            return 2
        body = json.loads(pointer.read_text(encoding="utf-8"))
        problems = verify_problems(body, root=ROOT)
        print(json.dumps(body, indent=2, ensure_ascii=False))
        print(f"\nverify: {'ok' if not problems else 'FAILED'}")
        for problem in problems:
            print(f"  - {problem}")
        return 0 if not problems else 1

    print("nothing to do; pass --write or --verify", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
