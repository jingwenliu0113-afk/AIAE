"""LoRA smoke-test plumbing: sampling, masking, collation.

Offline and model-free. The parts that need weights (merge verification,
gradient flow, save/reload) are checked by scripts/13_lora_sanity.py against
the real model, since faking them would test the fake.
"""

import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.lora import (
    Encoded,
    LoraConfig_,
    Row,
    check_no_object_overlap,
    collate,
    encode_row,
    sample_pairs,
    split_stats,
)

ROLES = ("control", "counterfactual")
VARIANTS = ("exact", "loose", "distractor", "mixed")


def make_rows(n_pairs: int, *, objects_per_pair: int = 1) -> list[Row]:
    """n_pairs whole pairs, 8 rows each, as the real file is laid out."""
    rows = []
    for p in range(n_pairs):
        for role in ROLES:
            for variant in VARIANTS:
                rows.append(Row(
                    sample_id=f"pair{p:03d}:{role}:{variant}",
                    pair_id=f"pair{p:03d}",
                    object_id=f"obj{p // objects_per_pair:03d}",
                    role=role, variant=variant,
                    prompt=f"prompt {p}", target=f"1x2 (0,0,{p})\n",
                    n_tokens=100,
                ))
    return rows


class TestPairSampling:
    def test_takes_whole_pairs_only(self):
        got = sample_pairs(make_rows(20), n_pairs=5, seed=0)
        assert len(got) == 40
        by_pair = {}
        for r in got:
            by_pair.setdefault(r.pair_id, []).append(r)
        assert len(by_pair) == 5
        for rows in by_pair.values():
            assert len(rows) == 8, "a pair must never be split"
            assert {r.role for r in rows} == set(ROLES)
            assert {r.variant for r in rows} == set(VARIANTS)

    def test_is_deterministic_for_a_seed(self):
        rows = make_rows(30)
        a = [r.sample_id for r in sample_pairs(rows, 8, seed=0)]
        b = [r.sample_id for r in sample_pairs(rows, 8, seed=0)]
        assert a == b

    def test_seed_changes_the_selection(self):
        rows = make_rows(30)
        a = {r.pair_id for r in sample_pairs(rows, 8, seed=0)}
        b = {r.pair_id for r in sample_pairs(rows, 8, seed=1)}
        assert a != b

    def test_selection_does_not_depend_on_file_order(self):
        """Shuffling sorted ids, not file order, so a re-ordered file is fine."""
        rows = make_rows(30)
        shuffled = list(reversed(rows))
        a = {r.pair_id for r in sample_pairs(rows, 8, seed=0)}
        b = {r.pair_id for r in sample_pairs(shuffled, 8, seed=0)}
        assert a == b

    def test_roles_and_variants_stay_balanced(self):
        got = sample_pairs(make_rows(40), n_pairs=25, seed=3)
        s = split_stats(got)
        assert s["samples"] == 200 and s["pairs"] == 25
        assert s["roles"] == {"control": 100, "counterfactual": 100}
        assert s["variants"] == {v: 50 for v in VARIANTS}

    def test_asking_for_too_many_pairs_raises(self):
        with pytest.raises(ValueError, match="file holds"):
            sample_pairs(make_rows(3), n_pairs=10, seed=0)

    def test_a_ragged_pair_is_rejected(self):
        """A pair missing a variant means the file is not what we think."""
        rows = make_rows(5)
        rows = [r for r in rows if r.sample_id != "pair000:control:mixed"]
        with pytest.raises(ValueError, match="not all 8 rows"):
            sample_pairs(rows, n_pairs=5, seed=0)


class TestObjectOverlap:
    def test_disjoint_objects_pass(self):
        train = make_rows(4)
        val = [Row(**{**r.__dict__, "object_id": "held-out"}) for r in make_rows(2)]
        assert check_no_object_overlap(train, val)["shared"] == 0

    def test_a_shared_object_raises(self):
        """The split manifest guarantees this; a sampling bug would not."""
        train = make_rows(4)
        val = make_rows(2)          # same object ids
        with pytest.raises(ValueError, match="both train and val"):
            check_no_object_overlap(train, val)

    def test_counts_are_reported(self):
        train = make_rows(4)
        val = [Row(**{**r.__dict__, "object_id": f"v{i}"})
               for i, r in enumerate(make_rows(2))]
        out = check_no_object_overlap(train, val)
        assert out["train_objects"] == 4 and out["val_objects"] == 16


class FakeTokenizer:
    """Deterministic stand-in: one id per character, plus a chat wrapper."""

    eos_token_id = 999

    def apply_chat_template(self, msgs, add_generation_prompt=False,
                            return_dict=False, **kw):
        ids = [1] + [ord(c) % 500 for c in msgs[0]["content"]] + [2]
        return {"input_ids": ids} if return_dict else ids

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 500 for c in text]


class TestLossMask:
    def setup_method(self):
        self.tok = FakeTokenizer()
        self.row = Row("s", "p", "o", "control", "exact",
                       "a prompt", "1x2 (0,0,0)\n", 20)

    def test_prompt_is_fully_masked(self):
        e = encode_row(self.tok, self.row, max_length=2048)
        assert e.labels[:e.n_prompt_tokens] == [-100] * e.n_prompt_tokens

    def test_target_positions_are_supervised_verbatim(self):
        e = encode_row(self.tok, self.row, max_length=2048)
        assert e.labels[e.n_prompt_tokens:] == e.input_ids[e.n_prompt_tokens:]

    def test_eos_is_supervised(self):
        """A model that never learns to stop can only run out of budget."""
        e = encode_row(self.tok, self.row, max_length=2048)
        assert e.input_ids[-1] == self.tok.eos_token_id
        assert e.labels[-1] == self.tok.eos_token_id

    def test_no_masked_hole_inside_the_target(self):
        e = encode_row(self.tok, self.row, max_length=2048)
        assert -100 not in e.labels[e.n_prompt_tokens:]

    def test_supervised_count_equals_target_plus_eos(self):
        e = encode_row(self.tok, self.row, max_length=2048)
        n_supervised = sum(1 for x in e.labels if x != -100)
        assert n_supervised == len(self.tok.encode(self.row.target)) + 1

    def test_truncation_is_reported_not_hidden(self):
        e = encode_row(self.tok, self.row, max_length=5)
        assert e.truncated is True
        assert len(e.input_ids) == 5

    def test_a_row_within_budget_is_not_flagged(self):
        e = encode_row(self.tok, self.row, max_length=2048)
        assert e.truncated is False


class TestCollate:
    def test_padding_is_masked_from_loss_and_attention(self):
        a = Encoded([1, 2, 3, 4], [-100, -100, 3, 4], 2, False)
        b = Encoded([1, 2], [-100, 2], 1, False)
        out = collate([a, b], pad_id=0)

        assert out["input_ids"].shape == (2, 4)
        assert out["labels"][1].tolist() == [-100, 2, -100, -100]
        assert out["attention_mask"][1].tolist() == [1, 1, 0, 0]

    def test_no_padding_needed_is_a_no_op(self):
        a = Encoded([1, 2], [-100, 2], 1, False)
        out = collate([a, a], pad_id=0)
        assert out["attention_mask"].sum().item() == 4

    def test_pad_id_never_reaches_the_labels(self):
        """Padding with EOS is normal; supervising it would teach early stops."""
        a = Encoded([1, 2, 3], [-100, 2, 3], 1, False)
        b = Encoded([1], [-100], 1, False)
        out = collate([a, b], pad_id=999)
        padded = out["labels"][1].tolist()
        assert padded == [-100, -100, -100]
        assert 999 not in padded

    def test_dtypes_are_long(self):
        a = Encoded([1, 2], [-100, 2], 1, False)
        out = collate([a], pad_id=0)
        for v in out.values():
            assert v.dtype == torch.long


class TestDeclaredConfig:
    """The configuration is a commitment made before the run, not after."""

    def test_effective_batch_is_the_product(self):
        cfg = LoraConfig_()
        assert cfg.effective_batch == cfg.batch_size * cfg.grad_accum

    def test_config_is_frozen(self):
        cfg = LoraConfig_()
        with pytest.raises(Exception):
            cfg.rank = 64

    def test_no_quantisation_on_this_platform(self):
        cfg = LoraConfig_()
        assert cfg.dtype == "bfloat16"
        assert "4-bit" in cfg.quantization and "not used" in cfg.quantization

    def test_every_recorded_field_is_in_the_dict(self):
        d = LoraConfig_().as_dict()
        for key in ("rank", "alpha", "dropout", "learning_rate", "batch_size",
                    "grad_accum", "effective_batch", "max_length", "epochs",
                    "seed", "dtype", "quantization", "target_modules"):
            assert key in d, key


class TestStartingPointIsNotBareLlama:
    """The most expensive silent mistake available here."""

    def test_module_names_the_published_adapter(self):
        from src.training import lora

        assert lora.PUBLISHED_ADAPTER == "AvaLovelace/BrickGPT"
        assert len(lora.PUBLISHED_REVISION) == 40
        assert lora.BASE_MODEL == "meta-llama/Llama-3.2-1B-Instruct"

    def test_build_model_verifies_the_merge_changed_weights(self):
        """Guards the guard: the no-op merge check must stay in build_model."""
        src = (Path(__file__).resolve().parents[1]
               / "src" / "training" / "lora.py").read_text()
        assert "merge_and_unload" in src
        assert "_weight_fingerprint" in src
        assert "silently start from bare Llama" in src

    def test_fingerprint_notices_a_changed_projection(self):
        from src.training.lora import _weight_fingerprint

        class M:
            def __init__(self, scale):
                self.scale = scale

            def named_parameters(self):
                yield "model.layers.0.self_attn.q_proj.weight", torch.ones(4) * self.scale
                yield "model.layers.0.self_attn.v_proj.weight", torch.ones(4)
                yield "model.layers.0.mlp.down_proj.weight", torch.ones(9999)

        assert _weight_fingerprint(M(1)) != _weight_fingerprint(M(2))
        assert _weight_fingerprint(M(1)) == _weight_fingerprint(M(1))

    def test_fingerprint_ignores_untouched_modules(self):
        """Only q_proj/v_proj matter -- what the published adapter targets."""
        from src.training.lora import _weight_fingerprint

        class M:
            def __init__(self, mlp):
                self.mlp = mlp

            def named_parameters(self):
                yield "model.layers.0.self_attn.q_proj.weight", torch.ones(4)
                yield "model.layers.0.mlp.down_proj.weight", torch.ones(4) * self.mlp

        assert _weight_fingerprint(M(1)) == _weight_fingerprint(M(7))


class TestColdStartLoadOrder:
    """A saved adapter is only meaningful on the weights it was fitted to.

    Ours were fitted on merged BrickGPT. Applying them to bare Llama yields a
    model that loads, generates, and is wrong -- measured at 0.56 loss against
    0.10 for the correct path in scripts/13_lora_coldstart.py. These tests pin
    the order and the refusals; that script does the real weights.
    """

    def _fakes(self, monkeypatch, calls):
        import types

        import src.training.lora as L

        class FakeBase:
            def named_parameters(self):
                yield "model.layers.0.self_attn.q_proj.weight", torch.ones(4)

        class FakeMerged:
            def named_parameters(self):
                yield "model.layers.0.self_attn.q_proj.weight", torch.ones(4) * 3

            def to(self, d):
                return self

            def eval(self):
                return self

        class FakeAuto:
            @staticmethod
            def from_pretrained(name, revision=None, dtype=None):
                calls.append(("auto", name, revision))
                return FakeBase()

        class FakePeftModel:
            @staticmethod
            def from_pretrained(model, path, revision=None):
                calls.append(("peft", str(path), revision))
                out = FakeMerged()
                out.merge_and_unload = lambda: FakeMerged()
                return out

        tf = types.ModuleType("transformers")
        tf.AutoModelForCausalLM = FakeAuto
        peft = types.ModuleType("peft")
        peft.PeftModel = FakePeftModel
        monkeypatch.setitem(sys.modules, "transformers", tf)
        monkeypatch.setitem(sys.modules, "peft", peft)
        return L

    def _manifest(self, d: Path, **over):
        from src.training.lora import (BASE_MODEL, BASE_REVISION, LOAD_ORDER,
                                       MANIFEST_NAME, PUBLISHED_ADAPTER,
                                       PUBLISHED_REVISION)
        import json as _json

        m = {
            "load_order": list(LOAD_ORDER),
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
            "published_adapter": PUBLISHED_ADAPTER,
            "published_adapter_revision": PUBLISHED_REVISION,
            "tokenizer": PUBLISHED_ADAPTER,
            "tokenizer_revision": PUBLISHED_REVISION,
            "adapter_sha256": None,
        }
        m.update(over)
        (d / MANIFEST_NAME).write_text(_json.dumps(m))
        return m

    def test_order_is_base_published_merge_local(self, monkeypatch, tmp_path):
        from src.training.lora import LOAD_ORDER

        L = self._fakes(monkeypatch, calls := [])
        self._manifest(tmp_path)
        order = []
        L.load_finetuned(tmp_path, device="cpu", verify_digest=False,
                         _calls=order)
        assert [c[0] for c in order] == list(LOAD_ORDER)

    def test_the_local_adapter_is_applied_last(self, monkeypatch, tmp_path):
        """Not to the base: the published adapter must be merged in first."""
        L = self._fakes(monkeypatch, calls := [])
        self._manifest(tmp_path)
        L.load_finetuned(tmp_path, device="cpu", verify_digest=False)
        peft_calls = [c for c in calls if c[0] == "peft"]
        assert len(peft_calls) == 2
        assert peft_calls[0][1] == L.PUBLISHED_ADAPTER
        assert peft_calls[1][1] == str(tmp_path)

    def test_base_is_loaded_at_the_pinned_revision(self, monkeypatch, tmp_path):
        L = self._fakes(monkeypatch, calls := [])
        self._manifest(tmp_path)
        L.load_finetuned(tmp_path, device="cpu", verify_digest=False)
        auto = next(c for c in calls if c[0] == "auto")
        assert auto[2] == L.BASE_REVISION

    def test_published_adapter_is_loaded_at_its_own_revision(
        self, monkeypatch, tmp_path
    ):
        L = self._fakes(monkeypatch, calls := [])
        self._manifest(tmp_path)
        L.load_finetuned(tmp_path, device="cpu", verify_digest=False)
        pub = next(c for c in calls if c[0] == "peft")
        assert pub[2] == L.PUBLISHED_REVISION

    def test_a_directory_without_a_manifest_is_refused(self, monkeypatch, tmp_path):
        L = self._fakes(monkeypatch, [])
        with pytest.raises(FileNotFoundError, match="no brickagain_manifest"):
            L.load_finetuned(tmp_path, device="cpu")

    def test_a_wrong_load_order_in_the_manifest_is_refused(
        self, monkeypatch, tmp_path
    ):
        L = self._fakes(monkeypatch, [])
        self._manifest(tmp_path, load_order=["base", "local_adapter"])
        with pytest.raises(ValueError, match="load_order"):
            L.load_finetuned(tmp_path, device="cpu", verify_digest=False)

    def test_a_different_base_revision_is_refused(self, monkeypatch, tmp_path):
        """The delta would sit on weights it was not fitted to."""
        L = self._fakes(monkeypatch, [])
        self._manifest(tmp_path, base_revision="0" * 40)
        with pytest.raises(ValueError, match="base_revision"):
            L.load_finetuned(tmp_path, device="cpu", verify_digest=False)

    def test_a_different_published_revision_is_refused(self, monkeypatch, tmp_path):
        L = self._fakes(monkeypatch, [])
        self._manifest(tmp_path, published_adapter_revision="0" * 40)
        with pytest.raises(ValueError, match="published_adapter_revision"):
            L.load_finetuned(tmp_path, device="cpu", verify_digest=False)

    def test_a_changed_adapter_file_is_refused(self, monkeypatch, tmp_path):
        L = self._fakes(monkeypatch, [])
        (tmp_path / "adapter_model.safetensors").write_bytes(b"new weights")
        self._manifest(tmp_path, adapter_sha256="a" * 64)
        with pytest.raises(ValueError, match="does not\\s+match the manifest"):
            L.load_finetuned(tmp_path, device="cpu")

    def test_a_matching_adapter_file_passes(self, monkeypatch, tmp_path):
        from src.training.lora import sha256_file

        L = self._fakes(monkeypatch, [])
        f = tmp_path / "adapter_model.safetensors"
        f.write_bytes(b"weights")
        self._manifest(tmp_path, adapter_sha256=sha256_file(f))
        L.load_finetuned(tmp_path, device="cpu")


class TestBrickGPTRefusesTheWrongPath:
    """The unsafe call has to fail loudly, not merely be discouraged."""

    def test_a_local_adapter_with_a_manifest_is_refused(self, tmp_path):
        from src.generation.brickgpt import _refuse_locally_trained_adapter
        from src.training.lora import MANIFEST_NAME

        (tmp_path / MANIFEST_NAME).write_text("{}")
        with pytest.raises(ValueError, match="load_finetuned"):
            _refuse_locally_trained_adapter(str(tmp_path))

    def test_the_message_names_the_correct_loader(self, tmp_path):
        from src.generation.brickgpt import _refuse_locally_trained_adapter
        from src.training.lora import MANIFEST_NAME

        (tmp_path / MANIFEST_NAME).write_text("{}")
        with pytest.raises(ValueError) as e:
            _refuse_locally_trained_adapter(str(tmp_path))
        msg = str(e.value)
        assert "merged" in msg and "silently" in msg

    def test_the_published_hub_adapter_is_still_allowed(self):
        from src.generation.brickgpt import _refuse_locally_trained_adapter

        _refuse_locally_trained_adapter("AvaLovelace/BrickGPT")

    def test_a_plain_directory_without_a_manifest_is_allowed(self, tmp_path):
        """Only our own checkpoints carry the marker."""
        from src.generation.brickgpt import _refuse_locally_trained_adapter

        _refuse_locally_trained_adapter(str(tmp_path))


class TestManifest:
    def test_written_manifest_round_trips(self, tmp_path):
        import json as _json

        from src.training.lora import (LOAD_ORDER, MANIFEST_NAME, LoraConfig_,
                                       write_manifest)

        (tmp_path / "adapter_model.safetensors").write_bytes(b"w")
        m = write_manifest(tmp_path, {}, LoraConfig_())
        on_disk = _json.loads((tmp_path / MANIFEST_NAME).read_text())
        assert on_disk == m
        assert on_disk["load_order"] == list(LOAD_ORDER)
        assert len(on_disk["adapter_sha256"]) == 64

    def test_manifest_warns_about_the_wrong_load(self, tmp_path):
        from src.training.lora import LoraConfig_, write_manifest

        m = write_manifest(tmp_path, {}, LoraConfig_())
        assert "MERGED" in m["warning"]
        assert "load_finetuned" in m["warning"]

    def test_base_revision_is_pinned_to_a_full_sha(self):
        from src.training.lora import BASE_REVISION

        assert len(BASE_REVISION) == 40
        assert BASE_REVISION == "9213176726f574b556790deb65791e0c5aa438b6"

    def test_the_three_sources_are_pinned_independently(self):
        from src.training.lora import BASE_REVISION, PUBLISHED_REVISION

        assert BASE_REVISION != PUBLISHED_REVISION


class TestSharedModelIds:
    """Arms A/B/D and C/E must resolve the same base weights.

    Each side was previously self-consistent while defining its own constants,
    which is exactly how a controlled comparison drifts without anything
    looking wrong.
    """

    def test_inference_and_training_agree_on_every_id(self):
        import src.generation.brickgpt as B
        import src.model_ids as M
        import src.training.lora as L

        assert B.BASE_MODEL == L.BASE_MODEL == M.BASE_MODEL
        assert B.BASE_REVISION == L.BASE_REVISION == M.BASE_REVISION
        assert B.ADAPTER == L.PUBLISHED_ADAPTER == M.ADAPTER
        assert B.ADAPTER_REVISION == L.PUBLISHED_REVISION == M.ADAPTER_REVISION
        assert B.TOKENIZER_REVISION == L.TOKENIZER_REVISION

    def test_the_constants_are_defined_once(self):
        """Re-declaring a revision in either module reintroduces the drift."""
        for mod in ("src/generation/brickgpt.py", "src/training/lora.py"):
            src = (Path(__file__).resolve().parents[1] / mod).read_text()
            assert 'BASE_REVISION = "' not in src, mod
            assert 'BASE_MODEL = "' not in src, mod

    def test_model_ids_imports_nothing_heavy(self):
        """Inference must not need the training package to know its base.

        Checked over the parsed imports, not the raw text: the module's own
        docstring names ``src.training.lora`` when explaining why it exists,
        and a substring scan would fail on the explanation.
        """
        import ast

        tree = ast.parse((Path(__file__).resolve().parents[1]
                          / "src" / "model_ids.py").read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported <= {"__future__"}, f"unexpected imports: {imported}"

    def test_brickgpt_takes_a_base_revision_defaulting_to_the_shared_pin(self):
        import inspect

        import src.generation.brickgpt as B
        import src.model_ids as M

        p = inspect.signature(B.BrickGPT.__init__).parameters
        assert "base_revision" in p
        assert p["base_revision"].default == M.BASE_REVISION

    def test_brickgpt_passes_the_base_revision_to_from_pretrained(self, monkeypatch):
        """The parameter existing is not the same as it being used."""
        import src.generation.brickgpt as B
        import src.model_ids as M

        seen = {}

        class FakeModel:
            @staticmethod
            def from_pretrained(name, revision=None, dtype=None):
                seen["base"] = (name, revision)
                return FakeModel()

            def to(self, d):
                return self

            def eval(self):
                return self

        monkeypatch.setattr(B, "AutoTokenizer", type("T", (), {
            "from_pretrained": staticmethod(
                lambda n, revision=None: type("t", (), {"eos_token_id": 0})())}))
        monkeypatch.setattr(B, "AutoModelForCausalLM", FakeModel)
        monkeypatch.setattr(B.Slots, "build", classmethod(lambda cls, t: "slots"))

        B.BrickGPT(device="cpu", adapter=None)
        assert seen["base"] == (M.BASE_MODEL, M.BASE_REVISION)

    def test_an_explicit_base_revision_overrides_the_default(self, monkeypatch):
        import src.generation.brickgpt as B

        seen = {}

        class FakeModel:
            @staticmethod
            def from_pretrained(name, revision=None, dtype=None):
                seen["revision"] = revision
                return FakeModel()

            def to(self, d):
                return self

            def eval(self):
                return self

        monkeypatch.setattr(B, "AutoTokenizer", type("T", (), {
            "from_pretrained": staticmethod(
                lambda n, revision=None: type("t", (), {"eos_token_id": 0})())}))
        monkeypatch.setattr(B, "AutoModelForCausalLM", FakeModel)
        monkeypatch.setattr(B.Slots, "build", classmethod(lambda cls, t: "slots"))

        B.BrickGPT(device="cpu", adapter=None, base_revision="c0ffee")
        assert seen["revision"] == "c0ffee"


class TestBrickGPTConstructorRefusesTheWrongPath:
    """Not just the helper -- the constructor itself must stop."""

    def test_constructing_with_a_local_checkpoint_raises(self, monkeypatch, tmp_path):
        import src.generation.brickgpt as B
        from src.model_ids import LOCAL_ADAPTER_MANIFEST

        (tmp_path / LOCAL_ADAPTER_MANIFEST).write_text("{}")
        (tmp_path / "adapter_model.safetensors").write_bytes(b"w")

        loaded = {"model": False}

        class FakeModel:
            @staticmethod
            def from_pretrained(name, revision=None, dtype=None):
                loaded["model"] = True
                return FakeModel()

            def to(self, d):
                return self

            def eval(self):
                return self

        monkeypatch.setattr(B, "AutoTokenizer", type("T", (), {
            "from_pretrained": staticmethod(
                lambda n, revision=None: type("t", (), {"eos_token_id": 0})())}))
        monkeypatch.setattr(B, "AutoModelForCausalLM", FakeModel)
        monkeypatch.setattr(B.Slots, "build", classmethod(lambda cls, t: "slots"))

        with pytest.raises(ValueError, match="load_finetuned"):
            B.BrickGPT(device="cpu", adapter=str(tmp_path))

    def test_the_published_adapter_still_constructs(self, monkeypatch):
        """The guard must not block the normal A/B/D path."""
        import src.generation.brickgpt as B

        applied = {}

        class FakeModel:
            @staticmethod
            def from_pretrained(name, revision=None, dtype=None):
                return FakeModel()

            def to(self, d):
                return self

            def eval(self):
                return self

        class FakePeft:
            @staticmethod
            def from_pretrained(model, adapter, revision=None):
                applied["adapter"] = adapter
                return model

        peft = types.ModuleType("peft")
        peft.PeftModel = FakePeft
        monkeypatch.setitem(sys.modules, "peft", peft)
        monkeypatch.setattr(B, "AutoTokenizer", type("T", (), {
            "from_pretrained": staticmethod(
                lambda n, revision=None: type("t", (), {"eos_token_id": 0})())}))
        monkeypatch.setattr(B, "AutoModelForCausalLM", FakeModel)
        monkeypatch.setattr(B.Slots, "build", classmethod(lambda cls, t: "slots"))

        B.BrickGPT(device="cpu")
        assert applied["adapter"] == B.ADAPTER


class TestReportReplayRefusesDrift:
    """Re-rendering must describe the run's own inputs, not today's.

    Three things can move underneath a stored report: the instruction data, the
    adapter weights, and the manifest that says how those weights must be
    loaded. Each is checked separately because each fails differently -- new
    data, a retrain, or a changed load contract.
    """

    def _script(self, tmp_path, monkeypatch):
        import importlib.util

        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "s13x", root / "scripts" / "13_lora_smoke.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Point every path at a sandbox so nothing real is read or written.
        monkeypatch.setattr(mod, "OUT_DIR", tmp_path / "processed")
        monkeypatch.setattr(mod, "CKPT_DIR", tmp_path / "ckpt")
        (tmp_path / "processed").mkdir()
        (tmp_path / "ckpt").mkdir()
        return mod

    def _stored(self, mod, tmp_path):
        from src.model_ids import LOCAL_ADAPTER_MANIFEST
        from src.training.lora import sha256_file

        (mod.OUT_DIR / "instruct_inv_train.jsonl").write_text("train\n")
        (mod.OUT_DIR / "instruct_inv_val.jsonl").write_text("val\n")
        (mod.CKPT_DIR / "adapter_model.safetensors").write_bytes(b"weights")
        (mod.CKPT_DIR / LOCAL_ADAPTER_MANIFEST).write_text('{"load_order": []}')
        return {"data": {"provenance": {
            "instruction_sha256": {
                n: sha256_file(mod.OUT_DIR / n)
                for n in ("instruct_inv_train.jsonl", "instruct_inv_val.jsonl")},
            "adapter_sha256": sha256_file(
                mod.CKPT_DIR / "adapter_model.safetensors"),
            "manifest_sha256": sha256_file(
                mod.CKPT_DIR / LOCAL_ADAPTER_MANIFEST),
        }}}

    def test_an_unchanged_run_replays(self, tmp_path, monkeypatch):
        mod = self._script(tmp_path, monkeypatch)
        mod._check_replayable(self._stored(mod, tmp_path))

    def test_changed_instruction_data_is_refused(self, tmp_path, monkeypatch):
        mod = self._script(tmp_path, monkeypatch)
        stored = self._stored(mod, tmp_path)
        (mod.OUT_DIR / "instruct_inv_train.jsonl").write_text("rebuilt\n")
        with pytest.raises(SystemExit, match="instruct_inv_train.jsonl changed"):
            mod._check_replayable(stored)

    def test_changed_adapter_weights_are_refused(self, tmp_path, monkeypatch):
        mod = self._script(tmp_path, monkeypatch)
        stored = self._stored(mod, tmp_path)
        (mod.CKPT_DIR / "adapter_model.safetensors").write_bytes(b"retrained")
        with pytest.raises(SystemExit, match="adapter changed"):
            mod._check_replayable(stored)

    def test_a_changed_manifest_is_refused(self, tmp_path, monkeypatch):
        """The weights can be identical while the load contract has moved."""
        from src.model_ids import LOCAL_ADAPTER_MANIFEST

        mod = self._script(tmp_path, monkeypatch)
        stored = self._stored(mod, tmp_path)
        (mod.CKPT_DIR / LOCAL_ADAPTER_MANIFEST).write_text('{"load_order": ["x"]}')
        with pytest.raises(SystemExit, match="manifest changed"):
            mod._check_replayable(stored)

    def test_a_missing_file_is_refused(self, tmp_path, monkeypatch):
        mod = self._script(tmp_path, monkeypatch)
        stored = self._stored(mod, tmp_path)
        (mod.CKPT_DIR / "adapter_model.safetensors").unlink()
        with pytest.raises(SystemExit, match="adapter is gone"):
            mod._check_replayable(stored)

    def test_every_drift_is_named_not_just_the_first(self, tmp_path, monkeypatch):
        from src.model_ids import LOCAL_ADAPTER_MANIFEST

        mod = self._script(tmp_path, monkeypatch)
        stored = self._stored(mod, tmp_path)
        (mod.OUT_DIR / "instruct_inv_val.jsonl").write_text("new\n")
        (mod.CKPT_DIR / "adapter_model.safetensors").write_bytes(b"new")
        (mod.CKPT_DIR / LOCAL_ADAPTER_MANIFEST).write_text("{}")
        with pytest.raises(SystemExit) as e:
            mod._check_replayable(stored)
        msg = str(e.value)
        assert "instruct_inv_val.jsonl" in msg
        assert "adapter changed" in msg
        assert "manifest changed" in msg


def test_coldstart_report_refusal_message_uses_a_repository_relative_path():
    """A committed report must not expose the machine's absolute path."""
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "s13_portable", root / "scripts" / "13_lora_coldstart.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    message = f"{mod.CKPT_DIR} is a locally trained adapter"
    portable = mod._portable_refusal_message(message)
    assert portable.startswith("artifacts/checkpoints/lora_smoke ")
    assert str(root) not in portable


class TestDigestsAreDistinct:
    """Selection order and training order are different facts."""

    def test_training_order_differs_from_selection_order(self, tmp_path, monkeypatch):
        import importlib.util

        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "s13y", root / "scripts" / "13_lora_smoke.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        rows = make_rows(4)
        sel = mod.selection_digest(rows)
        shuffled = mod.training_order_digest(list(reversed(range(len(rows)))), rows)
        assert sel != shuffled

    def test_the_identity_permutation_matches_the_selection_digest(self):
        import importlib.util

        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "s13z", root / "scripts" / "13_lora_smoke.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        rows = make_rows(3)
        assert mod.training_order_digest(list(range(len(rows))), rows) == \
            mod.selection_digest(rows)
