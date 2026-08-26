"""The final training run: one arm, one epoch, one shot, and nothing inherited.

H2 was selected on held-out loss. This is the run that produces the model, and
it differs from the arms in three ways that all have to be enforced rather
than intended:

* it reads the **whole** training split -- 9,584 rows, not the frozen 2,000-row
  pool every measurement so far has used;
* it starts from a **freshly initialised** adapter on top of the merged
  BrickGPT, never from H2's weights, optimizer state or generator. "Fresh" is
  checkable: a run that restored anything says so in its own evidence, so the
  three restoration flags being false is a property this file pins rather than
  a promise;
* there is no resume, no stop, and no second attempt. A failure stops the
  sequence and leaves immutable failure evidence.

Nothing here loads a model, opens the dataset or touches a device.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training import gates

ROOT = Path(__file__).resolve().parents[1]

DETERMINISM = {"use_deterministic_algorithms": True, "warn_only": False,
               "cudnn_benchmark": False, "cudnn_deterministic": True,
               "cublas_workspace_config": ":4096:8",
               "tf32_matmul_allowed": False, "tf32_cudnn_allowed": False,
               "seed": 0}
ALLOC = "expandable_segments:True"

DEP_EVIDENCE = {
    "repositories": [
        {"repo_id": "a/b", "revision": "r" * 40,
         "files": [{"name": "tokenizer.json", "bytes": 10, "sha256": "1" * 64}]},
    ],
    "instruction_pool": {"path": "data/processed/x.jsonl", "sha256": "2" * 64},
}


def dep_checker():
    return lambda: {"ok": True, "problems": [], "evidence": DEP_EVIDENCE}


def dep_digest():
    from src.training.longrun import dependency_digest

    return dependency_digest(DEP_EVIDENCE)


def build_pack(tmp_path):
    from src.training import pack

    src = tmp_path / "src_tree"
    for rel, text in {
            "requirements.txt": "torch\n",
            "src/__init__.py": "",
            "src/training/__init__.py": "",
            "src/training/gates.py": "GATE = 8\n",
            "scripts/19_gpu_gate.py": "#!/usr/bin/env python3\n",
    }.items():
        path = src / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    dest = tmp_path / "pack"
    pack.build(dest, root=src)
    return dest


def digest_of(packdir) -> str:
    from src.training import pack

    return pack.read_manifest(packdir)[0]["pack_digest"]


def run_gate_once(gate, run_dir, packdir, *, rows, stop_after=None,
                  resume=False):
    from src.training import gate_suite
    from src.training.session import write_once_json

    evidence = gates.run_gate(
        gate, deps=gates.FakeGateDeps(rows=rows), run_dir=run_dir,
        pack_dir=packdir, expected_pack_digest=digest_of(packdir),
        expected_dependency_digest=dep_digest(),
        dependency_checker=dep_checker(), allocator_config=ALLOC,
        determinism=dict(DETERMINISM), stop_after=stop_after, resume=resume)
    write_once_json(gate_suite.evidence_path(run_dir, gate),
                    {"verdict": gates.verdict(gate, evidence),
                     "evidence": evidence})
    return evidence


def build_suite(root, packdir):
    from src.training import gate_suite

    root = Path(root)
    runs = {"gate_8": root / "gate_8"}
    run_gate_once("gate_8", runs["gate_8"], packdir, rows=8)
    for role in gate_suite.GATE_100_ROLES:
        runs[role] = root / role
        run_gate_once("gate_100", runs[role], packdir, rows=100)
    control = root / "gate_500_uninterrupted_control"
    run_gate_once("gate_500", control, packdir, rows=500)
    runs[gate_suite.GATE_500_CONTROL] = control
    resumed = root / "gate_500_resumed"
    with pytest.raises(gates.DeliberateStop):
        run_gate_once("gate_500", resumed, packdir, rows=500, stop_after=250)
    run_gate_once("gate_500", resumed, packdir, rows=500, resume=True)
    runs[gate_suite.GATE_500_RESUMED] = resumed
    return runs


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    root = tmp_path_factory.mktemp("final")
    packdir = build_pack(root)
    runs = build_suite(root / "runs", packdir)
    return {"runs": runs, "packdir": packdir,
            "carried": {"expected_pack_digest": digest_of(packdir),
                        "expected_dependency_digest": dep_digest(),
                        "allocator_config": ALLOC,
                        "determinism": dict(DETERMINISM)}}


def run_kwargs(suite, tmp_path, **over):
    out = {"run_dir": tmp_path / "final_run", "pack_dir": suite["packdir"],
           "gate_runs": dict(suite["runs"]), "dependency_checker": dep_checker(),
           **suite["carried"]}
    out.update(over)
    return out


class Spy:
    def __init__(self):
        self.built_with = []

    def __call__(self, cfg):
        self.built_with.append(cfg)
        return gates.FakeGateDeps(rows=8, cfg=cfg)


# ---------------------------------------------------------------------------

class TestTheFinalRunIsTheSelectedArmOnTheWholeSplit:

    def test_the_selected_arm_is_h2_and_is_declared(self):
        from src.training import final_run, hypotheses

        assert final_run.SELECTED_ARM == "H2"
        assert final_run.frozen_config() is hypotheses.config_for("H2")
        cfg = final_run.frozen_config()
        assert (cfg.rank, cfg.alpha, cfg.learning_rate) == (32, 16, 2e-3)

    def test_the_length_is_the_whole_training_split(self):
        from src.training import final_run

        assert final_run.FULL_TRAIN_ROWS == 9584
        assert final_run.FULL_TRAIN_PAIRS == 1198
        assert final_run.FINAL_EPOCHS == 1

    def test_the_step_count_is_derived_and_is_one_one_nine_eight(self):
        from src.training import final_run
        from src.training.lora import LoraConfig_

        assert final_run.expected_optimizer_steps() == 9584 // LoraConfig_().grad_accum
        assert final_run.expected_optimizer_steps() == 1198

    def test_a_split_that_changed_size_is_refused_at_load(self):
        """The declared shape is checked against the file, at load time.

        Asserted here against the loader rather than by opening the dataset:
        this suite travels in the pack and also runs in the public
        evidence-free tree, where there is no dataset to open -- and a test
        that can only skip teaches the operator that skipping is normal. The
        guarantee is stronger where it lives: the loader compares what it read
        against the declared shape and raises, on the node, against the actual
        bytes.
        """
        import inspect

        from src.training.longrun import DATA_SOURCES, ProductionChildDeps

        source = inspect.getsource(ProductionChildDeps.load)
        assert 'shape = DATA_SOURCES[self.source]' in source
        assert 'if len(pool) != shape["rows"]' in source
        assert "raise ValueError" in source
        assert DATA_SOURCES["full_train"]["pairs"] * 8 == \
            DATA_SOURCES["full_train"]["rows"]

    def test_an_unknown_source_is_refused_rather_than_defaulted(self):
        from src.training.longrun import ProductionChildDeps

        with pytest.raises(ValueError):
            ProductionChildDeps(source="everything")

    def test_the_checkpoint_stride_lands_on_an_optimizer_boundary(self):
        from src.training import final_run
        from src.training.lora import LoraConfig_

        assert final_run.FINAL_CHECKPOINT_EVERY % LoraConfig_().grad_accum == 0
        spec = final_run.final_spec()
        assert spec.rows == 9584
        assert spec.checkpoint_every == final_run.FINAL_CHECKPOINT_EVERY

    def test_the_loader_can_be_asked_for_the_whole_split(self):
        from src.training.longrun import DATA_SOURCES

        assert DATA_SOURCES["pool"] == {"pairs": 250, "rows": 2000}
        assert DATA_SOURCES["full_train"] == {"pairs": 1198, "rows": 9584}

    def test_the_pool_remains_the_default_source(self):
        import inspect

        from src.training.longrun import ProductionChildDeps

        sig = inspect.signature(ProductionChildDeps.__init__)
        assert sig.parameters["source"].default == "pool"


class TestNothingIsInherited:

    def test_there_is_no_resume_no_stop_and_no_predecessor(self):
        import inspect

        from src.training import final_run

        sig = inspect.signature(final_run.run_final)
        for forbidden in ("resume", "stop_after", "previous_run_dir",
                          "from_adapter", "init_from"):
            assert forbidden not in sig.parameters, forbidden

    def test_the_script_offers_no_way_to_continue_something(self):
        import importlib.util

        script = ROOT / "scripts" / "22_final_train.py"
        spec = importlib.util.spec_from_file_location("final_train", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        options = set()
        for action in module.build_parser()._actions:
            options.update(action.option_strings)
        for forbidden in ("--resume", "--stop-after", "--from-adapter",
                          "--init-from", "--continue", "--previous-arm-run-dir",
                          "--rank", "--alpha", "--lr", "--learning-rate",
                          "--dtype", "--rows", "--epochs", "--seed", "--arm",
                          "--force", "--unlock"):
            assert forbidden not in options, forbidden

    def test_a_run_that_restored_anything_is_refused(self):
        from src.training import final_run

        for field in ("model_state_restored", "rng_state_restored",
                      "optimizer_state_restored"):
            evidence = self.healthy()
            evidence[field] = True
            problems = final_run.operational_problems(evidence)
            assert any("restored" in p for p in problems), (field, problems)

    def test_a_fresh_run_says_so_in_all_three_flags(self):
        from src.training import final_run

        assert final_run.operational_problems(self.healthy()) == []

    def healthy(self, **over):
        from src.training import final_run

        e = {"gate": final_run.RUN_NAME,
             "rows_declared": final_run.FULL_TRAIN_ROWS,
             "rows_completed": final_run.FULL_TRAIN_ROWS,
             "losses_finite": True, "ledger_problems": [],
             "optimizer_steps": final_run.expected_optimizer_steps(),
             "attempts": 1, "truncated_rows": 0,
             "model_state_restored": False, "rng_state_restored": False,
             "optimizer_state_restored": False,
             "saved": {"sha256": "a" * 64},
             "cold_load": {"loaded": True, "matches_saved": True,
                           "sha256": "a" * 64},
             "trainable_digest": "b" * 64, "epochs": 1,
             "config": None}
        from src.training import hypotheses

        e["config"] = hypotheses.config_for("H2").as_dict()
        e.update(over)
        return e

    def test_more_than_one_attempt_is_refused(self):
        from src.training import final_run

        problems = final_run.operational_problems(self.healthy(attempts=2))
        assert any("attempt" in p for p in problems), problems

    def test_the_h1_configuration_is_refused(self):
        from src.training import final_run, hypotheses

        problems = final_run.operational_problems(
            self.healthy(config=hypotheses.config_for("H1").as_dict()))
        assert any("config" in p for p in problems), problems


class TestTruncationStopsTheRun:

    def test_any_truncated_row_is_refused(self):
        from src.training import final_run

        problems = final_run.truncation_problems({"truncated_rows": 1})
        assert problems
        assert any("truncat" in p for p in problems), problems

    def test_no_reading_at_all_is_refused_rather_than_assumed_zero(self):
        from src.training import final_run

        assert final_run.truncation_problems({})
        assert final_run.truncation_problems({"truncated_rows": None})

    def test_zero_truncated_rows_passes(self):
        from src.training import final_run

        assert final_run.truncation_problems({"truncated_rows": 0}) == []

    def test_the_evidence_check_also_refuses_truncation(self):
        from src.training import final_run

        e = TestNothingIsInherited().healthy(truncated_rows=3)
        assert any("truncat" in p
                   for p in final_run.operational_problems(e))

    def test_the_runner_checks_before_the_first_row(self, suite, tmp_path):
        """A truncated encoding is a data problem, not a training result."""
        from src.training import final_run

        def factory(cfg):
            deps = gates.FakeGateDeps(rows=64, cfg=cfg)
            inner = deps.load

            def load(*, rows):
                loaded = inner(rows=rows)
                loaded["truncated_rows"] = 5
                return loaded

            deps.load = load
            return deps

        with pytest.raises(final_run.FinalRunFailed) as exc:
            final_run.run_final(deps_factory=factory,
                                **run_kwargs(suite, tmp_path))
        assert "truncat" in str(exc.value)
        run_dir = tmp_path / "final_run"
        assert not final_run.evidence_path(run_dir).exists()
        assert final_run.failure_path(run_dir).is_file()
        assert gates.read_ledger(run_dir) == [], "a row was measured anyway"


class TestTheGateSuiteStillGuardsIt:

    def test_a_missing_role_stops_before_the_loader(self, suite, tmp_path):
        from src.training import final_run, hypotheses

        runs = {k: v for k, v in suite["runs"].items() if k != "gate_8"}
        spy = Spy()
        with pytest.raises(hypotheses.HypothesisLocked):
            final_run.run_final(deps_factory=spy,
                                **run_kwargs(suite, tmp_path, gate_runs=runs))
        assert spy.built_with == []
        assert not (tmp_path / "final_run").exists()

    def test_the_wrong_pack_digest_stops_before_the_loader(self, suite,
                                                           tmp_path):
        from src.training import final_run, hypotheses

        spy = Spy()
        with pytest.raises(hypotheses.HypothesisLocked):
            final_run.run_final(deps_factory=spy,
                                **run_kwargs(suite, tmp_path,
                                             expected_pack_digest="a" * 64))
        assert spy.built_with == []

    def test_it_unlocks_through_the_selected_arm(self):
        import inspect

        from src.training import final_run

        source = inspect.getsource(final_run.run_final)
        assert "require_unlocked" in source
        assert "SELECTED_ARM" in source


class TestTheRunItself:

    @pytest.fixture()
    def small(self, monkeypatch):
        from src.training import final_run

        monkeypatch.setattr(final_run, "FULL_TRAIN_ROWS", 64)
        monkeypatch.setattr(final_run, "FINAL_CHECKPOINT_EVERY", 32)
        return final_run

    def factory(self, cfg):
        return gates.FakeGateDeps(rows=64, cfg=cfg)

    def test_a_healthy_run_publishes_evidence(self, small, suite, tmp_path):
        evidence = small.run_final(deps_factory=self.factory,
                                   **run_kwargs(suite, tmp_path))
        run_dir = tmp_path / "final_run"
        assert small.evidence_path(run_dir).is_file()
        assert not small.failure_path(run_dir).exists()
        assert evidence["arm"] == "H2"
        assert evidence["operational_problems"] == []
        assert evidence["model_state_restored"] is False
        assert evidence["rng_state_restored"] is False
        assert evidence["optimizer_state_restored"] is False

    def test_the_evidence_carries_what_a_replay_needs(self, small, suite,
                                                      tmp_path):
        evidence = small.run_final(deps_factory=self.factory,
                                   **run_kwargs(suite, tmp_path))
        for field in ("arm", "config", "epochs", "rows_declared",
                      "rows_completed", "optimizer_steps", "truncated_rows",
                      "pack_digest", "dependency_digest", "allocator_config",
                      "determinism", "provenance", "order_digest",
                      "sample_ids", "per_row_loss", "trainable_digest",
                      "adapter", "losses_finite", "operational_problems"):
            assert field in evidence, field
        assert len(evidence["per_row_loss"]) == 64

    def test_it_declares_no_winner_and_no_next_round(self, small, suite,
                                                     tmp_path):
        evidence = small.run_final(deps_factory=self.factory,
                                   **run_kwargs(suite, tmp_path))
        for forbidden in ("verdict", "winner", "better", "threshold",
                          "next_round"):
            assert forbidden not in evidence, forbidden

    def test_a_directory_that_already_ran_is_not_reused(self, small, suite,
                                                        tmp_path):
        from src.training import final_run

        kw = run_kwargs(suite, tmp_path)
        small.run_final(deps_factory=self.factory, **kw)
        with pytest.raises((final_run.FinalRunRefused, gates.GateRefused)):
            small.run_final(deps_factory=self.factory, **kw)

    def test_a_directory_with_a_failure_marker_is_not_reused(self, small,
                                                             suite, tmp_path):
        from src.training import final_run

        run_dir = tmp_path / "final_run"
        run_dir.mkdir(parents=True)
        small.write_failure(run_dir, "something went wrong")
        with pytest.raises(final_run.FinalRunRefused) as exc:
            small.run_final(deps_factory=self.factory,
                            **run_kwargs(suite, tmp_path))
        assert "failure" in str(exc.value).lower()

    def test_a_failure_leaves_immutable_evidence_and_no_retry(self, small,
                                                              suite, tmp_path):
        from src.training import final_run

        calls = []

        def explode(cfg):
            calls.append(cfg)
            raise RuntimeError("the device fell over")

        with pytest.raises(RuntimeError):
            small.run_final(deps_factory=explode,
                            **run_kwargs(suite, tmp_path))
        assert len(calls) == 1, "something retried"
        run_dir = tmp_path / "final_run"
        assert not small.evidence_path(run_dir).exists()
        body = json.loads(small.failure_path(run_dir).read_text(
            encoding="utf-8"))
        assert "device fell over" in body["reason"]
        with pytest.raises(SystemExit):
            small.write_failure(run_dir, "again")


class TestItStaysInsideItsBoundaries:

    def test_the_test_split_is_unreachable(self):
        for path in (ROOT / "src" / "training" / "final_run.py",
                     ROOT / "scripts" / "22_final_train.py"):
            source = path.read_text(encoding="utf-8")
            assert "instruct_inv_test" not in source
            assert "_test.jsonl" not in source

    def test_nothing_is_discovered(self):
        import ast

        for path in (ROOT / "src" / "training" / "final_run.py",
                     ROOT / "scripts" / "22_final_train.py"):
            used = set()
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Attribute):
                    used.add(node.attr)
                elif isinstance(node, ast.Name):
                    used.add(node.id)
            for forbidden in ("glob", "rglob", "iterdir", "listdir",
                              "st_mtime", "walk"):
                assert forbidden not in used, (path.name, forbidden)

    def test_it_leaks_no_identifier(self):
        from src.training.longrun import leaked_identifiers

        for path in (ROOT / "src" / "training" / "final_run.py",
                     ROOT / "scripts" / "22_final_train.py"):
            assert leaked_identifiers(path.read_text(encoding="utf-8")) == []

    def test_the_runner_and_its_tests_travel_in_the_pack(self):
        from src.training import pack

        assert pack.classify("src/training/final_run.py")[0] == "include"
        assert pack.classify("scripts/22_final_train.py")[0] == "include"
        assert "tests/test_final_run.py" in pack.PACKED_TEST_SUITES
