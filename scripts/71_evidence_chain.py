#!/usr/bin/env python3
"""Versioned evidence bindings that do not rewrite frozen generations.

There are three deliberately separate contracts here:

* a deterministic ZIP containing the exact 60 bytesets named by the Phase 3C
  gen09 pack evidence, so historical reproducibility no longer means "the
  current working tree must still be identical";
* a live-tree drift declaration, checked independently from that archive;
* a post-score seal that binds the node output seal to the six derived score
  files, report, receipt, frozen verifier, and evaluation pack manifest.

The helpers are additive.  They never edit the gen09 pack evidence, the
BrickNet frozen contract, or any earlier evidence bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]

PHASE3C_DIR = Path("data/phase3c/frozen/gen09")
PACK_EVIDENCE = PHASE3C_DIR / "pack_manifest.json"
EXECUTION_GRANT = PHASE3C_DIR / "execution_authorization.json"
SOURCE_ARCHIVE = PHASE3C_DIR / "source_snapshot_v2.zip"
SOURCE_BINDING = PHASE3C_DIR / "source_binding_v2.json"
# v2 described the tree as it stood when the allowlist last changed. The
# receipt is write-once by design -- `_write_once_json` refuses to replace a
# different one -- so a later change to a declared drifting file gets a new
# receipt rather than an edit to the old one. v2 stays on disk: it recorded
# what was true then, and nothing about it became wrong. v3 followed the
# delivery files; v4 follows reports/figures/figures.json, the manifest CI
# reads instead of redrawing. v2 and v3 both stay.
LIVE_DRIFT = PHASE3C_DIR / "live_drift_v4.json"
POSTSCORE_SEAL = Path("data/reports/bricknet/42_postscore_seal_v1.json")

PARENT_SEAL = Path("artifacts/bricknet/evidence_v5/outputs/seal.json")
FROZEN_VERIFIER = Path(
    "artifacts/bricknet/evidence_v5/frozen_tools/50_bricknet_postscore.py"
)
EVAL_PACK_MANIFEST = Path("artifacts/bricknet/evalpack_frozen/pack_manifest.json")
EXECUTION_MANIFEST = Path("data/bricknet/frozen/eval_execution_manifest.json")

ARMS = (
    "baseline_pt",
    "baseline_sft",
    "inv_lora",
    "inv_lora_constrained",
    "scoped_pt",
    "sft_constrained",
)
POSTSCORE_FILES = tuple(
    [Path(f"data/reports/bricknet/eval/{arm}.json") for arm in ARMS]
    + [
        Path("data/reports/bricknet/eval/seal.json"),
        Path("data/reports/bricknet/42_eval.md"),
        Path("data/reports/bricknet/42_eval_receipt.json"),
    ]
)

# These are the already-published v5 bytes.  A post-score manifest that merely
# signs whatever happens to be on disk would be self-consistent after an
# attacker changed both a file and its row.  The verifier therefore carries
# the independent historical anchors as well as checking the seal's rows.
FROZEN_VERIFIER_SHA256 = "7efa67d21a4cb96eceede923b158a2f7b9a16e26792887626310edf341c40d2b"
EVAL_PACK_MANIFEST_SHA256 = "385f69db4810220af3911bd2a62a2b32b4c94a7ab3e0d69a72838ffb153d0369"
EXECUTION_MANIFEST_SHA256 = "5ec668d634526740502d76a9c28bbebeda0742d327e48d06c36740c5dc8b9a3f"
PARENT_SEAL_SHA256 = "5fc55b28133260e04297ad58222ad10c2cd01c70bb0cae6eb857765b606f59f6"
PARENT_SEAL_DIGEST = "b93cc0d82ec70d93d657a0d51eb9c85c08b731337de9df37b6e8528fed15d689"
POSTSCORE_EXPECTED_SHA256 = {
    "data/reports/bricknet/eval/baseline_pt.json": "7f0cd64d5e9eb9f111ac8f6eed7a1894e6f350514a06b16ac62d7faaf5dfb076",
    "data/reports/bricknet/eval/baseline_sft.json": "f871e1a9c3fdcd91c06ce8233074255668c9cb8e66b06ed8d313d110dac11d62",
    "data/reports/bricknet/eval/inv_lora.json": "86045dd95d01e1db1233edc714a2d60fbb138e102b189f204032f84989e058d1",
    "data/reports/bricknet/eval/inv_lora_constrained.json": "51f380a3e69f09ea0b2f9a308b261b500682196a5c8a4541fd30c0d4549d5051",
    "data/reports/bricknet/eval/scoped_pt.json": "c099ae5f0defbd7046b709f3865d6682b29785208dffab6d9fd7d8f9f7c08f57",
    "data/reports/bricknet/eval/sft_constrained.json": "71d018a5c872d03d862d97f61c9acb5c8df1281e641a2258b2e95a99e6989ed8",
    "data/reports/bricknet/eval/seal.json": PARENT_SEAL_SHA256,
    "data/reports/bricknet/42_eval.md": "9cc61cca6a8a994b6cceb53ed51d7cb4202b8ec9fbd8d27c9a852288d92dca8a",
    "data/reports/bricknet/42_eval_receipt.json": "699bd1bbf113728dc5c0d26c09b84673a94bac30e700eb363d8c2253919a11b1",
}

DRIFT_REASONS = {
    "scripts/17_public_snapshot.py": (
        "The live public-snapshot allowlist has grown since gen09 was packed: "
        "the V1 generation ledger, then the delivery files (MODEL_CARD.md, "
        "Makefile, pyproject.toml, the CI workflow, configs/, notebooks/, "
        "reports/figures/), then reports/figures/figures.json. That file is "
        "not in Phase 3C gen09's 26-file execution closure."
    ),
    "src/eval/visual_stress.py": (
        "The live V1 generation declaration later moved from code to "
        "GENERATIONS.json. This module is not in Phase 3C gen09's 26-file "
        "execution closure."
    ),
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest_of(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_rel(text: str) -> str:
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {text!r}")
    normal = str(path)
    if normal != text or normal.startswith("./"):
        raise ValueError(f"non-canonical relative path: {text!r}")
    return normal


def _write_once_json(path: Path, body: dict) -> None:
    payload = json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"refusing to replace different evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _write_once_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to replace different evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _pack_evidence_problems(evidence: dict, grant: dict) -> list[str]:
    problems: list[str] = []
    manifest = evidence.get("pack_manifest") or {}
    files = manifest.get("files") or {}
    file_map = {key: value.get("sha256") for key, value in files.items()}
    files_digest = sha256_bytes(json.dumps(
        file_map, sort_keys=True, separators=(",", ":")
    ).encode("utf-8"))
    if files_digest != evidence.get("files_digest"):
        problems.append("pack evidence files_digest does not cover its file table")
    pack_digest = digest_of({
        "schema_version": manifest.get("schema_version"),
        "kind": manifest.get("kind"),
        "files_digest": manifest.get("files_digest"),
        "data_digest": manifest.get("data_digest"),
    })
    if pack_digest != evidence.get("pack_digest"):
        problems.append("pack evidence does not recompute its pack_digest")
    if evidence.get("evidence_digest") != digest_of({
        key: value for key, value in evidence.items() if key != "evidence_digest"
    }):
        problems.append("pack evidence does not recompute its evidence_digest")
    if grant.get("pack_digest") != evidence.get("pack_digest"):
        problems.append("authorization names a different pack_digest")
    if grant.get("pack_evidence_digest") != evidence.get("evidence_digest"):
        problems.append("authorization names a different pack_evidence_digest")
    if evidence.get("n_files") != len(files):
        problems.append("pack evidence n_files does not match its table")
    return problems


def _historical_bytes(root: Path, rel: str, expected: str) -> tuple[bytes, dict]:
    live = root / rel
    if live.is_file():
        payload = live.read_bytes()
        if sha256_bytes(payload) == expected:
            return payload, {"kind": "live_exact_at_snapshot_build"}

    history = subprocess.run(
        ["git", "rev-list", "--all", "--", rel],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    for commit in history:
        shown = subprocess.run(
            ["git", "show", f"{commit}:{rel}"],
            cwd=root,
            capture_output=True,
        )
        if shown.returncode == 0 and sha256_bytes(shown.stdout) == expected:
            return shown.stdout, {"kind": "git_object", "commit": commit}
    raise RuntimeError(f"no exact historical bytes found for {rel} ({expected})")


def build_phase3c_snapshot(
    root: Path = ROOT,
    *,
    evidence_path: Path | None = None,
    authorization_path: Path | None = None,
    archive_path: Path | None = None,
    binding_path: Path | None = None,
    drift_path: Path | None = None,
) -> tuple[dict, dict]:
    """Build the deterministic archive and the separate live-drift receipt."""
    evidence_path = evidence_path or root / PACK_EVIDENCE
    grant_path = authorization_path or root / EXECUTION_GRANT
    archive_path = archive_path or root / SOURCE_ARCHIVE
    binding_path = binding_path or root / SOURCE_BINDING
    drift_path = drift_path or root / LIVE_DRIFT
    evidence = _load(evidence_path)
    grant = _load(grant_path)
    problems = _pack_evidence_problems(evidence, grant)
    if problems:
        raise RuntimeError("; ".join(problems))

    table = evidence["pack_manifest"]["files"]
    members: dict[str, bytes] = {}
    origins: dict[str, dict] = {}
    for rel, entry in sorted(table.items()):
        _safe_rel(rel)
        payload, origin = _historical_bytes(root, rel, entry["sha256"])
        if len(payload) != entry["bytes"]:
            raise RuntimeError(f"historical byte count differs for {rel}")
        members[rel] = payload
        origins[rel] = origin

    from io import BytesIO

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for rel in sorted(members):
            info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, members[rel])
    archive_bytes = buffer.getvalue()
    _write_once_bytes(archive_path, archive_bytes)

    binding = {
        "kind": "brickagain.phase3c_historical_source_snapshot",
        "schema_version": 2,
        "generation": evidence["generation"],
        "authority": (
            "Exact member bytes in source_snapshot_v2.zip. Git origins describe "
            "how missing live bytes were recovered; they are not substitutes for "
            "the archive's per-member hashes."
        ),
        "frozen_pack_evidence": {
            "path": str(PACK_EVIDENCE),
            "sha256": sha256_file(evidence_path),
            "evidence_digest": evidence["evidence_digest"],
            "pack_digest": evidence["pack_digest"],
            "files_digest": evidence["files_digest"],
        },
        "archive": {
            "path": str(SOURCE_ARCHIVE),
            "sha256": sha256_bytes(archive_bytes),
            "bytes": len(archive_bytes),
            "compression": "stored",
            "member_count": len(members),
        },
        "members": {
            rel: {
                "sha256": table[rel]["sha256"],
                "bytes": table[rel]["bytes"],
                "origin": origins[rel],
            }
            for rel in sorted(members)
        },
    }
    binding["binding_digest"] = digest_of(binding)
    _write_once_json(binding_path, binding)

    drift = make_live_drift(root, binding)
    _write_once_json(drift_path, drift)
    return binding, drift


def historical_snapshot_problems(
    binding: dict,
    archive_path: Path,
    *,
    root: Path = ROOT,
    evidence_path: Path | None = None,
    authorization_path: Path | None = None,
) -> list[str]:
    """Verify history using the snapshot, never the mutable live source tree."""
    problems: list[str] = []
    if binding.get("binding_digest") != digest_of({
        key: value for key, value in binding.items() if key != "binding_digest"
    }):
        problems.append("source binding digest differs")
    if binding.get("kind") != "brickagain.phase3c_historical_source_snapshot":
        problems.append("source binding kind differs")
    if binding.get("schema_version") != 2:
        problems.append("source binding schema differs")

    evidence_path = evidence_path or root / PACK_EVIDENCE
    grant_path = authorization_path or root / EXECUTION_GRANT
    try:
        evidence = _load(evidence_path)
        grant = _load(grant_path)
    except (OSError, ValueError) as exc:
        return problems + [f"cannot read frozen pack evidence: {exc}"]
    problems.extend(_pack_evidence_problems(evidence, grant))
    frozen = binding.get("frozen_pack_evidence") or {}
    if frozen.get("sha256") != sha256_file(evidence_path):
        problems.append("source binding names a different pack-evidence file")
    for key in ("evidence_digest", "pack_digest", "files_digest"):
        if frozen.get(key) != evidence.get(key):
            problems.append(f"source binding {key} differs from pack evidence")

    if not archive_path.is_file():
        return problems + ["historical source archive is missing"]
    archive_meta = binding.get("archive") or {}
    if archive_meta.get("sha256") != sha256_file(archive_path):
        problems.append("historical source archive hash differs")
    if archive_meta.get("bytes") != archive_path.stat().st_size:
        problems.append("historical source archive byte count differs")

    pack_files = evidence.get("pack_manifest", {}).get("files", {})
    members = binding.get("members") or {}
    if set(members) != set(pack_files):
        problems.append("source binding member set differs from pack evidence")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                problems.append("historical source archive has duplicate members")
            if set(names) != set(pack_files):
                problems.append("historical source archive member set differs")
            if archive_meta.get("member_count") != len(names):
                problems.append("historical source archive member_count differs")
            for rel in sorted(set(names) & set(pack_files)):
                try:
                    _safe_rel(rel)
                except ValueError as exc:
                    problems.append(str(exc))
                    continue
                payload = archive.read(rel)
                expected = pack_files[rel]
                if len(payload) != expected.get("bytes"):
                    problems.append(f"historical member byte count differs: {rel}")
                if sha256_bytes(payload) != expected.get("sha256"):
                    problems.append(f"historical member hash differs: {rel}")
                bound = members.get(rel) or {}
                if (bound.get("bytes"), bound.get("sha256")) != (
                    expected.get("bytes"), expected.get("sha256")
                ):
                    problems.append(f"source binding row differs: {rel}")
    except (OSError, zipfile.BadZipFile) as exc:
        problems.append(f"historical source archive is unreadable: {exc}")
    return problems


def make_live_drift(root: Path, binding: dict) -> dict:
    differing: list[dict] = []
    for rel, expected in sorted((binding.get("members") or {}).items()):
        path = root / rel
        if not path.is_file():
            differing.append({
                "path": rel,
                "status": "missing",
                "expected_sha256": expected["sha256"],
                "expected_bytes": expected["bytes"],
                "reason": DRIFT_REASONS.get(rel, "No migration reason has been recorded."),
            })
            continue
        actual = path.read_bytes()
        actual_sha = sha256_bytes(actual)
        if actual_sha != expected["sha256"] or len(actual) != expected["bytes"]:
            differing.append({
                "path": rel,
                "status": "different",
                "expected_sha256": expected["sha256"],
                "expected_bytes": expected["bytes"],
                "actual_sha256": actual_sha,
                "actual_bytes": len(actual),
                "reason": DRIFT_REASONS.get(rel, "No migration reason has been recorded."),
            })
    result = {
        "kind": "brickagain.phase3c_live_tree_drift",
        "schema_version": 2,
        "generation": binding.get("generation"),
        "historical_binding_digest": binding.get("binding_digest"),
        "rule": (
            "This receipt describes the current tree only. Drift does not alter "
            "or invalidate the separately verified historical source snapshot."
        ),
        "checked_files": len(binding.get("members") or {}),
        "differing_count": len(differing),
        "differing": differing,
    }
    result["drift_digest"] = digest_of(result)
    return result


def live_drift_problems(root: Path, binding: dict, declared: dict) -> list[str]:
    problems: list[str] = []
    if declared.get("drift_digest") != digest_of({
        key: value for key, value in declared.items() if key != "drift_digest"
    }):
        problems.append("live drift digest differs")
    actual = make_live_drift(root, binding)
    if declared != actual:
        problems.append("live drift declaration does not match the current tree")
    unreasoned = [
        row.get("path") for row in declared.get("differing", [])
        if row.get("reason") == "No migration reason has been recorded."
    ]
    if unreasoned:
        problems.append(f"live drift lacks reasons: {', '.join(unreasoned)}")
    return problems


def append_only_problems(
    frozen_snapshot: bytes, live_contract: bytes, frozen_sha256: str
) -> list[str]:
    """Reject edits to frozen bytes; permit only an unchanged prefix plus suffix."""
    problems: list[str] = []
    if sha256_bytes(frozen_snapshot) != frozen_sha256:
        problems.append("frozen contract snapshot digest differs")
    if not live_contract.startswith(frozen_snapshot):
        problems.append("live contract does not retain the frozen byte prefix")
    return problems


def append_only_record(frozen_path: Path, live_path: Path) -> dict:
    frozen = frozen_path.read_bytes()
    live = live_path.read_bytes()
    body = {
        "kind": "brickagain.append_only_contract_binding",
        "schema_version": 1,
        "frozen": {
            "path": str(frozen_path),
            "sha256": sha256_bytes(frozen),
            "bytes": len(frozen),
        },
        "live": {
            "path": str(live_path),
            "sha256": sha256_bytes(live),
            "bytes": len(live),
        },
        "appended_bytes": len(live) - len(frozen),
        "problems": append_only_problems(frozen, live, sha256_bytes(frozen)),
    }
    body["binding_digest"] = digest_of(body)
    return body


def _file_record(root: Path, path: Path) -> dict:
    absolute = root / path
    if not absolute.is_file():
        raise RuntimeError(f"required evidence file is missing: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(absolute),
        "bytes": absolute.stat().st_size,
    }


def build_postscore_seal(root: Path = ROOT, output: Path | None = None) -> dict:
    output = output or root / POSTSCORE_SEAL
    parent = _load(root / PARENT_SEAL)
    receipt = _load(root / "data/reports/bricknet/42_eval_receipt.json")
    pack = _load(root / EVAL_PACK_MANIFEST)
    execution = _load(root / EXECUTION_MANIFEST)
    body = {
        "kind": "bricknet_postscore_seal",
        "schema_version": 1,
        "contract": (
            "Post-score artifacts are a distinct layer. This seal binds their "
            "exact bytes to the immutable node-output seal and scoring inputs."
        ),
        "parent_output_seal": dict(
            _file_record(root, PARENT_SEAL), seal_digest=parent.get("seal_digest")
        ),
        "frozen_verifier": _file_record(root, FROZEN_VERIFIER),
        "eval_pack_manifest": dict(
            _file_record(root, EVAL_PACK_MANIFEST),
            kind=pack.get("kind"),
            file_count=pack.get("file_count"),
        ),
        "execution_manifest": dict(
            _file_record(root, EXECUTION_MANIFEST),
            manifest_digest=execution.get("manifest_digest"),
        ),
        "scoring_environment": receipt.get("verified_on"),
        "files": [_file_record(root, path) for path in POSTSCORE_FILES],
        "expected_arms": list(ARMS),
        "file_count": len(POSTSCORE_FILES),
    }
    body["postscore_digest"] = digest_of(body)
    _write_once_json(output, body)
    return body


def postscore_seal_problems(seal: dict, root: Path = ROOT) -> list[str]:
    problems: list[str] = []
    if seal.get("postscore_digest") != digest_of({
        key: value for key, value in seal.items() if key != "postscore_digest"
    }):
        problems.append("postscore seal digest differs")
    if seal.get("kind") != "bricknet_postscore_seal" or seal.get("schema_version") != 1:
        problems.append("postscore seal kind or schema differs")
    rows = seal.get("files") or []
    expected_paths = {str(path) for path in POSTSCORE_FILES}
    actual_paths = {row.get("path") for row in rows}
    if actual_paths != expected_paths or len(rows) != len(actual_paths):
        problems.append("postscore file set differs")
    if seal.get("file_count") != len(POSTSCORE_FILES):
        problems.append("postscore file_count differs")
    if seal.get("expected_arms") != list(ARMS):
        problems.append("postscore expected arms differ")

    def check_row(row: dict, label: str) -> None:
        try:
            rel = _safe_rel(row.get("path", ""))
        except (TypeError, ValueError) as exc:
            problems.append(f"unsafe {label} path: {exc}")
            return
        path = root / rel
        if not path.is_file():
            problems.append(f"{label} file is missing: {rel}")
            return
        if row.get("bytes") != path.stat().st_size:
            problems.append(f"{label} byte count differs: {rel}")
        if row.get("sha256") != sha256_file(path):
            problems.append(f"{label} hash differs: {rel}")

    for row in rows:
        check_row(row, "postscore")
    for key in ("parent_output_seal", "frozen_verifier", "eval_pack_manifest", "execution_manifest"):
        check_row(seal.get(key) or {}, key.replace("_", " "))

    anchored = {
        str(PARENT_SEAL): PARENT_SEAL_SHA256,
        str(FROZEN_VERIFIER): FROZEN_VERIFIER_SHA256,
        str(EVAL_PACK_MANIFEST): EVAL_PACK_MANIFEST_SHA256,
        str(EXECUTION_MANIFEST): EXECUTION_MANIFEST_SHA256,
        **POSTSCORE_EXPECTED_SHA256,
    }
    for rel, expected_sha in anchored.items():
        path = root / rel
        if not path.is_file() or sha256_file(path) != expected_sha:
            problems.append(f"historical postscore anchor differs: {rel}")

    try:
        parent = _load(root / PARENT_SEAL)
        parent_digest = digest_of({
            key: value for key, value in parent.items() if key != "seal_digest"
        })
        if parent_digest != parent.get("seal_digest"):
            problems.append("parent output seal does not recompute")
        if parent.get("seal_digest") != PARENT_SEAL_DIGEST:
            problems.append("parent output seal historical digest differs")
        if (seal.get("parent_output_seal") or {}).get("seal_digest") != parent.get("seal_digest"):
            problems.append("postscore seal names a different parent seal digest")
        report_seal = root / "data/reports/bricknet/eval/seal.json"
        if report_seal.is_file() and report_seal.read_bytes() != (root / PARENT_SEAL).read_bytes():
            problems.append("report output seal differs from parent output seal")
        receipt = _load(root / "data/reports/bricknet/42_eval_receipt.json")
        receipt_digest = digest_of({
            key: value for key, value in receipt.items() if key != "receipt_digest"
        })
        if receipt_digest != receipt.get("receipt_digest"):
            problems.append("postscore receipt does not recompute")
        if receipt.get("seal_digest") != parent.get("seal_digest"):
            problems.append("postscore receipt names a different parent seal")
        if seal.get("scoring_environment") != receipt.get("verified_on"):
            problems.append("postscore scoring environment differs from receipt")
        execution = _load(root / EXECUTION_MANIFEST)
        named_manifest = (seal.get("execution_manifest") or {}).get("manifest_digest")
        if named_manifest != execution.get("manifest_digest"):
            problems.append("postscore seal names a different execution manifest")
        for arm in ARMS:
            score = _load(root / f"data/reports/bricknet/eval/{arm}.json")
            if score.get("arm") != arm:
                problems.append(f"score arm differs: {arm}")
            if score.get("manifest_digest") != execution.get("manifest_digest"):
                problems.append(f"score manifest differs: {arm}")
    except (OSError, ValueError) as exc:
        problems.append(f"cannot inspect postscore structure: {exc}")
    return problems


def verify_phase3c(root: Path = ROOT) -> list[str]:
    binding = _load(root / SOURCE_BINDING)
    declared = _load(root / LIVE_DRIFT)
    problems = historical_snapshot_problems(binding, root / SOURCE_ARCHIVE, root=root)
    problems.extend(live_drift_problems(root, binding, declared))
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("build-phase3c", "verify-phase3c", "build-postscore", "verify-postscore", "verify-all"),
    )
    args = parser.parse_args(argv)
    if args.command == "build-phase3c":
        binding, drift = build_phase3c_snapshot()
        print(f"historical snapshot: {binding['archive']['member_count']} files")
        print(f"live drift: {drift['differing_count']} files")
        return 0
    if args.command == "build-postscore":
        seal = build_postscore_seal()
        print(f"postscore seal: {seal['file_count']} files")
        return 0
    problems: list[str] = []
    if args.command in ("verify-phase3c", "verify-all"):
        problems.extend(verify_phase3c())
    if args.command in ("verify-postscore", "verify-all"):
        problems.extend(postscore_seal_problems(_load(ROOT / POSTSCORE_SEAL)))
    if problems:
        print("REFUSED")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
