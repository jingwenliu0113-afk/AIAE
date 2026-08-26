"""The project-model pointer: one file that says which adapter is the model.

``final_eval.json`` decided. Nothing on disk said so afterwards -- the
adapter sat in a returned run directory beside two others, distinguishable
only by reading a record and knowing which record to read. This is the file
that names it.

A pointer is only worth having if it can be checked, so almost everything
here is about ``--verify``: the paths still resolve, every digest still
recomputes, and the manifest beside the weights still describes the same LoRA
shape. A pointer that cannot be re-verified is a note.

Two rules it must not break. **No absolute paths** -- this file records where
things are inside the repository, and an absolute path both leaks the machine
it was written on and stops being true the moment the tree moves. And **it
verifies rather than trusts**: the digests are recomputed from the bytes, not
copied from the record that produced them.

Nothing here loads a model, opens the dataset or touches a device.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "24_project_model.py"


def names_used(path: Path) -> set:
    used = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.alias):
            used.add(node.asname or node.name.split(".")[-1])
    return used


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("project_model", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def tree(tmp_path):
    """A miniature of the real layout: an adapter, an eval record, evidence."""
    from src.training.session import sha256_file

    adapter = tmp_path / "runs" / "returns" / "final_H2" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "brickagain_manifest.json").write_text(json.dumps(
        {"lora": {"r": 32, "alpha": 16,
                  "target_modules": ["q_proj", "v_proj"]},
         "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors")}),
        encoding="utf-8")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    evidence = adapter.parent / "final_H2_evidence.json"
    evidence.write_text(json.dumps({
        "arm": "H2", "run": "final_H2",
        "adapter": {"sha256": sha256_file(
            adapter / "adapter_model.safetensors")}}), encoding="utf-8")
    record = tmp_path / "runs" / "returns" / "final_eval.json"
    record.write_text(json.dumps({
        "kind": "final_eval", "project_model": "final_H2",
        "means": {"existing_H2": 0.3, "final_H2": 0.2},
        "criterion": {"primary_criterion": "lower wins"}}), encoding="utf-8")
    return {"root": tmp_path, "adapter": adapter, "evidence": evidence,
            "record": record}


def build(mod, tree, **over):
    kw = {"root": tree["root"],
          "adapter_dir": tree["adapter"],
          "final_eval": tree["record"],
          "training_evidence": tree["evidence"]}
    kw.update(over)
    return mod.build_record(**kw)


class TestThePointerNamesTheModel:

    def test_it_records_the_model_and_where_it_came_from(self, mod, tree):
        body = build(mod, tree)
        assert body["kind"] == "project_model"
        assert body["model"] == "final_H2"
        assert body["selected_by"]["record"].endswith("final_eval.json")
        assert body["selected_by"]["means"] == {"existing_H2": 0.3,
                                                "final_H2": 0.2}

    def test_it_records_the_three_digests(self, mod, tree):
        from src.training.session import sha256_file

        body = build(mod, tree)
        assert body["adapter"]["files"]["adapter_model.safetensors"]["sha256"] \
            == sha256_file(tree["adapter"] / "adapter_model.safetensors")
        assert body["adapter"]["files"]["brickagain_manifest.json"]["sha256"] \
            == sha256_file(tree["adapter"] / "brickagain_manifest.json")
        assert body["selected_by"]["sha256"] == sha256_file(tree["record"])

    def test_it_records_the_model_revisions(self, mod, tree):
        from src.model_ids import (ADAPTER, ADAPTER_REVISION, BASE_MODEL,
                                   BASE_REVISION, TOKENIZER_REVISION)

        r = build(mod, tree)["revisions"]
        assert r["base_model"] == BASE_MODEL
        assert r["base_revision"] == BASE_REVISION
        assert r["published_adapter"] == ADAPTER
        assert r["published_adapter_revision"] == ADAPTER_REVISION
        assert r["tokenizer_revision"] == TOKENIZER_REVISION

    def test_it_records_the_lora_shape_from_the_manifest_on_disk(self, mod,
                                                                 tree):
        body = build(mod, tree)
        assert body["lora"] == {"r": 32, "alpha": 16,
                                "target_modules": ["q_proj", "v_proj"]}

    def test_it_says_how_to_load_it(self, mod, tree):
        body = build(mod, tree)
        assert "load_finetuned" in body["load_with"]
        assert "verify_digest" in body["load_with"]


class TestEveryPathIsRelative:

    def test_no_value_anywhere_is_an_absolute_path(self, mod, tree):
        body = build(mod, tree)

        def walk(node, trail="") :
            if isinstance(node, dict):
                for k, v in node.items():
                    yield from walk(v, f"{trail}.{k}")
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    yield from walk(v, f"{trail}[{i}]")
            elif isinstance(node, str):
                yield trail, node

        for trail, value in walk(body):
            assert not value.startswith("/"), (trail, value)
            assert not value.startswith("~"), (trail, value)
            assert "\\\\" not in value, (trail, value)

    def test_the_paths_are_relative_to_the_repository_root(self, mod, tree):
        body = build(mod, tree)
        for rel in (body["adapter"]["path"], body["selected_by"]["record"],
                    body["training_evidence"]["path"]):
            assert not Path(rel).is_absolute(), rel
            assert (tree["root"] / rel).exists(), rel

    def test_a_target_outside_the_root_is_refused(self, mod, tree, tmp_path):
        outside = tmp_path.parent / "elsewhere"
        outside.mkdir(exist_ok=True)
        with pytest.raises(ValueError):
            build(mod, tree, adapter_dir=outside)

    def test_it_leaks_no_identifier(self, mod, tree):
        from src.training.longrun import leaked_identifiers

        assert leaked_identifiers(json.dumps(build(mod, tree))) == []
        assert leaked_identifiers(SCRIPT.read_text(encoding="utf-8")) == []


class TestVerifyRecomputesRatherThanTrusts:

    def test_an_untouched_tree_verifies(self, mod, tree):
        assert mod.verify_problems(build(mod, tree), root=tree["root"]) == []

    def test_a_changed_adapter_is_caught(self, mod, tree):
        body = build(mod, tree)
        (tree["adapter"] / "adapter_model.safetensors").write_bytes(b"other")
        problems = mod.verify_problems(body, root=tree["root"])
        assert any("adapter_model.safetensors" in p for p in problems), problems

    def test_a_changed_manifest_is_caught(self, mod, tree):
        body = build(mod, tree)
        (tree["adapter"] / "brickagain_manifest.json").write_text("{}")
        problems = mod.verify_problems(body, root=tree["root"])
        assert problems

    def test_a_changed_eval_record_is_caught(self, mod, tree):
        body = build(mod, tree)
        tree["record"].write_text(json.dumps({"kind": "final_eval",
                                              "project_model": "final_H2"}))
        problems = mod.verify_problems(body, root=tree["root"])
        assert any("final_eval" in p for p in problems), problems

    def test_a_missing_file_is_caught(self, mod, tree):
        body = build(mod, tree)
        (tree["adapter"] / "adapter_model.safetensors").unlink()
        assert mod.verify_problems(body, root=tree["root"])

    def test_a_manifest_describing_another_shape_is_caught(self, mod, tree):
        body = build(mod, tree)
        body["lora"]["r"] = 16
        problems = mod.verify_problems(body, root=tree["root"])
        assert any("lora" in p for p in problems), problems

    def test_a_record_that_no_longer_names_this_model_is_caught(self, mod,
                                                                tree):
        body = build(mod, tree)
        body["model"] = "existing_H2"
        assert mod.verify_problems(body, root=tree["root"])

    def test_every_failure_is_reported_not_only_the_first(self, mod, tree):
        body = build(mod, tree)
        (tree["adapter"] / "adapter_model.safetensors").write_bytes(b"a")
        (tree["adapter"] / "brickagain_manifest.json").write_text("{}")
        assert len(mod.verify_problems(body, root=tree["root"])) >= 2


class TestItChangesNothing:

    def test_it_trains_nothing_and_loads_no_model(self):
        used = names_used(SCRIPT)
        for forbidden in ("backward", "AdamW", "load_finetuned", "build_model",
                          "run_gate", "run_final", "from_pretrained",
                          "save_pretrained", "no_grad"):
            assert forbidden not in used, forbidden

    def test_the_test_split_is_unreachable(self):
        source = SCRIPT.read_text(encoding="utf-8")
        assert "instruct_inv_test" not in source
        assert "_test.jsonl" not in source

    def test_nothing_is_discovered(self):
        used = names_used(SCRIPT)
        for forbidden in ("glob", "rglob", "listdir", "scandir", "walk",
                          "st_mtime"):
            assert forbidden not in used, forbidden

    def test_it_publishes_nothing(self):
        used = names_used(SCRIPT)
        for forbidden in ("push", "publish", "upload", "urlopen", "requests",
                          "Popen"):
            assert forbidden not in used, forbidden

    def test_the_pointer_is_written_once(self):
        assert "write_once_json" in SCRIPT.read_text(encoding="utf-8")

    def test_it_stays_out_of_the_gpu_pack(self):
        from src.training import pack

        assert pack.classify("scripts/24_project_model.py")[0] == "exclude"
        assert pack.classify("tests/test_project_model.py")[0] == "exclude"
        included = {e["path"] for e in pack.manifest(ROOT)["include"]}
        assert "scripts/24_project_model.py" not in included

    def test_the_pointer_itself_is_not_public(self):
        """It lives under runs/, which never leaves this machine."""
        from src.training import pack

        assert pack.classify("runs/project_model.json")[0] == "exclude"
