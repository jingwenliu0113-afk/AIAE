"""The H1/H2 runner: what it refuses, and what it refuses to do first.

The arms are frozen settings. Turning one into a run is the single most
consequential thing this project can do on the node, and almost everything
that could go wrong about it goes wrong *before* the first row: the wrong
pack, the wrong dependency bytes, an allocator inherited from somewhere else,
a suite of gates that does not exist or does not agree, a configuration that
somebody nudged on the command line.

So the tests here are mostly about ordering. It is not enough that a bad run
fails; it has to fail before a model is built, before a run directory exists,
before a plan is written and before a row is measured -- because a spent boot
is not refundable and a half-made run directory is the thing the next person
resumes.

Nothing here loads a model, allocates a tensor or opens a socket. The loader
is a spy that records whether it was called at all, which is the assertion.
"""

from __future__ import annotations

import json
import subprocess
import sys
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
    """A verified pack to run against, built the way the node's is built."""
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
    """The six roles, built through the same runner the node uses.

    Self-contained on purpose: this file travels in the pack, and a packed
    suite that imports another test module is one that means something
    different depending on what else happened to be shipped.
    """
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
    """A verified six-role suite, built through the real runner."""
    root = tmp_path_factory.mktemp("arms")
    packdir = build_pack(root)
    runs = build_suite(root / "runs", packdir)
    return {"runs": runs, "packdir": packdir, "root": root,
            "carried": {"expected_pack_digest": digest_of(packdir),
                        "expected_dependency_digest": dep_digest(),
                        "allocator_config": ALLOC,
                        "determinism": dict(DETERMINISM)}}


class Spy:
    """A loader factory that records whether anything ever asked it to load."""

    def __init__(self):
        self.built_with = []

    def __call__(self, cfg):
        self.built_with.append(cfg)
        return gates.FakeGateDeps(rows=8)


def run_kwargs(suite, tmp_path, **over):
    out = {
        "run_dir": tmp_path / "arm_run",
        "pack_dir": suite["packdir"],
        "gate_runs": dict(suite["runs"]),
        "dependency_checker": dep_checker(),
        **suite["carried"],
    }
    out.update(over)
    return out


class TestTheArmsAreTheFrozenOnes:

    def test_only_h1_and_h2_exist(self):
        from src.training import arms

        assert arms.ARM_NAMES == ("H1", "H2")

    def test_an_unknown_arm_is_refused(self, suite, tmp_path):
        from src.training import arms

        spy = Spy()
        with pytest.raises(arms.ArmRefused):
            arms.run_arm("H3", deps_factory=spy,
                         **run_kwargs(suite, tmp_path))
        assert spy.built_with == [], "a loader was built for an unknown arm"

    def test_the_specification_is_the_frozen_row_count(self):
        from src.training import arms, hypotheses
        from src.training.lora import LoraConfig_

        for name in arms.ARM_NAMES:
            spec = arms.arm_spec(name)
            assert spec.rows == hypotheses.ROWS == 2000
            assert spec.checkpoint_every % LoraConfig_().grad_accum == 0

    def test_the_config_is_hypotheses_own_and_differs_in_three_fields(self):
        from src.training import arms, hypotheses

        for name in arms.ARM_NAMES:
            assert arms.frozen_config(name) is hypotheses.config_for(name)
        a = hypotheses.config_for("H1").as_dict()
        b = hypotheses.config_for("H2").as_dict()
        differing = {k for k in a if a[k] != b[k]}
        assert differing == {"rank", "alpha", "learning_rate"}

    def test_both_arms_run_one_epoch_over_the_same_batch_geometry(self):
        from src.training import hypotheses

        a, b = hypotheses.config_for("H1"), hypotheses.config_for("H2")
        for field in ("seed", "max_length", "target_modules", "grad_accum",
                      "effective_batch", "epochs", "dtype", "quantization"):
            assert a.as_dict()[field] == b.as_dict()[field], field
        assert a.epochs == 1 and a.effective_batch == 8


class TestNothingIsBuiltBeforeTheSuiteIsProved:

    def test_a_missing_role_stops_before_the_loader(self, suite, tmp_path):
        from src.training import arms, hypotheses

        runs = {k: v for k, v in suite["runs"].items() if k != "gate_100_r2"}
        spy = Spy()
        with pytest.raises(hypotheses.HypothesisLocked):
            arms.run_arm("H1", deps_factory=spy,
                         **run_kwargs(suite, tmp_path, gate_runs=runs))
        assert spy.built_with == []
        assert not (tmp_path / "arm_run").exists(), \
            "a run directory was created by a refused arm"

    def test_swapped_gate_500_roles_stop_before_the_loader(self, suite,
                                                           tmp_path):
        from src.training import arms, gate_suite, hypotheses

        runs = dict(suite["runs"])
        a, b = gate_suite.GATE_500_RESUMED, gate_suite.GATE_500_CONTROL
        runs[a], runs[b] = runs[b], runs[a]
        spy = Spy()
        with pytest.raises(hypotheses.HypothesisLocked):
            arms.run_arm("H1", deps_factory=spy,
                         **run_kwargs(suite, tmp_path, gate_runs=runs))
        assert spy.built_with == []

    def test_the_wrong_pack_digest_stops_before_the_loader(self, suite,
                                                           tmp_path):
        from src.training import arms, hypotheses

        spy = Spy()
        with pytest.raises(hypotheses.HypothesisLocked):
            arms.run_arm("H1", deps_factory=spy,
                         **run_kwargs(suite, tmp_path,
                                      expected_pack_digest="a" * 64))
        assert spy.built_with == []

    def test_the_wrong_dependency_digest_stops_before_the_loader(
            self, suite, tmp_path):
        from src.training import arms, hypotheses

        spy = Spy()
        with pytest.raises(hypotheses.HypothesisLocked):
            arms.run_arm("H1", deps_factory=spy,
                         **run_kwargs(suite, tmp_path,
                                      expected_dependency_digest="b" * 64))
        assert spy.built_with == []

    def test_the_wrong_allocator_config_stops_before_the_loader(
            self, suite, tmp_path):
        from src.training import arms, hypotheses

        spy = Spy()
        with pytest.raises(hypotheses.HypothesisLocked):
            arms.run_arm("H1", deps_factory=spy,
                         **run_kwargs(suite, tmp_path,
                                      allocator_config="expandable_segments:False"))
        assert spy.built_with == []

    def test_drifted_determinism_stops_before_the_loader(self, suite, tmp_path):
        from src.training import arms, hypotheses

        det = dict(DETERMINISM)
        det["seed"] = 7
        spy = Spy()
        with pytest.raises(hypotheses.HypothesisLocked):
            arms.run_arm("H1", deps_factory=spy,
                         **run_kwargs(suite, tmp_path, determinism=det))
        assert spy.built_with == []

    def test_bare_verdict_strings_cannot_be_passed_at_all(self, suite,
                                                          tmp_path):
        from src.training import arms

        spy = Spy()
        with pytest.raises(TypeError):
            arms.run_arm("H1", deps_factory=spy, gate_results={
                "gate_8": "passed", "gate_100": "passed",
                "gate_500": "passed"},
                **run_kwargs(suite, tmp_path))
        assert spy.built_with == []


class TestTheRunItself:

    def deps_factory(self, rows):
        def factory(cfg):
            deps = gates.FakeGateDeps(rows=rows)
            deps.provenance_override = {"lora_config": cfg.as_dict()}
            return deps
        return factory

    @pytest.fixture()
    def small(self, monkeypatch):
        """The arm specification, shortened so the test is a test.

        Only the row count moves, and only inside this fixture: the runner
        reads it from ``hypotheses.ROWS`` and nothing here changes what the
        arms *are*.
        """
        from src.training import arms

        monkeypatch.setattr(arms, "ARM_ROWS", 64)
        return arms

    def test_a_proved_suite_runs_the_arm(self, small, suite, tmp_path):
        evidence = small.run_arm(
            "H1", deps_factory=self.deps_factory(64),
            **run_kwargs(suite, tmp_path))
        assert evidence["arm"] == "H1"
        assert evidence["rows_completed"] == 64
        assert evidence["ledger_problems"] == []

    def test_the_evidence_carries_what_a_replay_needs(self, small, suite,
                                                      tmp_path):
        from src.training import hypotheses

        evidence = small.run_arm(
            "H1", deps_factory=self.deps_factory(64),
            **run_kwargs(suite, tmp_path))
        for field in ("arm", "config", "rows_declared", "epochs",
                      "pack_digest", "dependency_digest", "allocator_config",
                      "determinism", "provenance", "order_digest",
                      "sample_ids", "per_row_loss", "optimizer_steps",
                      "trainable_digest", "adapter", "losses_finite",
                      "peak_vram_gb", "model_state_restored",
                      "rng_state_restored", "optimizer_state_restored"):
            assert field in evidence, field
        assert evidence["config"] == hypotheses.config_for("H1").as_dict()
        assert len(evidence["per_row_loss"]) == 64
        assert len(evidence["sample_ids"]) == 64

    def test_it_declares_no_winner(self, small, suite, tmp_path):
        evidence = small.run_arm(
            "H1", deps_factory=self.deps_factory(64),
            **run_kwargs(suite, tmp_path))
        for forbidden in ("verdict", "passed", "winner", "better",
                          "threshold"):
            assert forbidden not in evidence, forbidden

    def test_the_two_arms_cannot_share_a_run_directory(self, small, suite,
                                                       tmp_path):
        from src.training import arms

        kw = run_kwargs(suite, tmp_path)
        small.run_arm("H1", deps_factory=self.deps_factory(64), **kw)
        with pytest.raises((arms.ArmRefused, gates.GateRefused)):
            small.run_arm("H2", deps_factory=self.deps_factory(64), **kw)

    def test_a_failure_leaves_immutable_evidence_and_stops(self, small, suite,
                                                           tmp_path):
        def explode(cfg):
            raise RuntimeError("the device fell over")

        with pytest.raises(RuntimeError):
            small.run_arm("H1", deps_factory=explode,
                          **run_kwargs(suite, tmp_path))
        failure = small.failure_path(tmp_path / "arm_run", "H1")
        assert failure.is_file()
        body = json.loads(failure.read_text(encoding="utf-8"))
        assert body["arm"] == "H1"
        assert "device fell over" in body["reason"]
        with pytest.raises(SystemExit):
            small.write_failure(tmp_path / "arm_run", "H1", "again")

    def test_a_failure_does_not_start_the_other_arm(self, small, suite,
                                                    tmp_path):
        started = []

        def explode(cfg):
            started.append(cfg)
            raise RuntimeError("no")

        with pytest.raises(RuntimeError):
            small.run_arm("H1", deps_factory=explode,
                          **run_kwargs(suite, tmp_path))
        assert len(started) == 1, "something retried or ran the second arm"

    def test_a_non_finite_loss_stops_the_arm_rather_than_being_reported(
            self, small, suite, tmp_path):
        """It used to come back as evidence with a flag set. It is a failure."""
        from src.training import arms

        def factory(cfg):
            deps = gates.FakeGateDeps(rows=64)
            inner = deps.load

            def load(*, rows):
                loaded = inner(rows=rows)
                step = loaded["step"]

                def bad(index, position):
                    out = step(index, position)
                    if position == 5:
                        out["loss"] = float("nan")
                    return out

                loaded["step"] = bad
                return loaded

            deps.load = load
            return deps

        with pytest.raises(arms.ArmFailed):
            small.run_arm("H1", deps_factory=factory,
                          **run_kwargs(suite, tmp_path))
        run_dir = tmp_path / "arm_run"
        assert not small.evidence_path(run_dir, "H1").exists()
        body = json.loads(small.failure_path(run_dir, "H1").read_text(
            encoding="utf-8"))
        assert body["measured"]["losses_finite"] is False


class TestTheNodeScript:

    SCRIPT = ROOT / "scripts" / "20_hypothesis_run.py"

    def source(self):
        return self.SCRIPT.read_text(encoding="utf-8")

    def run(self, *args):
        return subprocess.run(
            [sys.executable, "-B", str(self.SCRIPT), *args],
            cwd=ROOT, capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin", "HF_HUB_OFFLINE": "1",
                 "PYTHONDONTWRITEBYTECODE": "1"})

    def test_the_script_exists_and_is_packed(self):
        from src.training import pack

        assert self.SCRIPT.is_file()
        assert pack.classify("scripts/20_hypothesis_run.py")[0] == "include"

    def parser(self):
        """The parser itself, not the file it is written in.

        Asserting against the source text would fail on the docstring that
        *names* the flags this script deliberately does not have, and would
        pass on a flag added by any spelling the check did not anticipate.
        The option strings are the contract.
        """
        import importlib.util

        spec = importlib.util.spec_from_file_location("h_run", self.SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.build_parser()

    def options(self):
        out = set()
        for action in self.parser()._actions:
            out.update(action.option_strings)
        return out

    def test_there_is_no_override_or_bypass_flag(self):
        options = self.options()
        for forbidden in ("--force", "--unlock", "--skip-preflight",
                          "--no-preflight", "--rank", "--alpha",
                          "--learning-rate", "--lr", "--dtype",
                          "--quantization", "--rows", "--epochs", "--seed",
                          "--batch", "--grad-accum", "--max-length",
                          "--override", "--config", "--stop-after"):
            assert forbidden not in options, forbidden

    def test_the_six_roles_are_six_explicit_arguments(self):
        from src.training import gate_suite

        options = self.options()
        dests = {a.dest for a in self.parser()._actions}
        for role in gate_suite.ROLES:
            assert f"--{role.replace('_', '-')}" in options, role
            assert role in dests, role
        assert len([r for r in gate_suite.ROLES if r in dests]) == 6

    def test_the_only_directory_arguments_are_the_six_and_the_run(self):
        """Nothing takes a parent directory to look inside."""
        dests = {a.dest for a in self.parser()._actions}
        from src.training import gate_suite

        directory_like = {d for d in dests
                          if d.endswith("_dir") or d.endswith("_root")}
        assert directory_like == {"run_dir", "pack_dir", "data_root",
                                  "previous_arm_run_dir"}, directory_like
        assert set(gate_suite.ROLES) <= dests

    def test_it_discovers_nothing(self):
        source = self.source()
        for forbidden in ("glob(", "rglob(", "iterdir(", "listdir(",
                          "st_mtime", "sorted(Path"):
            assert forbidden not in source, forbidden

    def test_the_carried_digests_have_no_defaults(self):
        source = self.source()
        assert "--expected-pack-digest" in source
        assert "--expected-dependency-digest" in source
        assert 'default=None' not in source.split(
            "--expected-pack-digest")[1][:400]

    def test_summary_builds_no_model_and_needs_no_device(self):
        """It runs with no CUDA device visible at all.

        ``torch`` is imported by ``src.training.lora`` at module scope and has
        been since long before this script, so "does not import torch" is not
        the guarantee on offer. The guarantee is that no weights are read and
        no device is initialised, and a run that completes with
        ``CUDA_VISIBLE_DEVICES`` empty has demonstrated both.
        """
        result = subprocess.run(
            [sys.executable, "-B", str(self.SCRIPT), "--summary"],
            cwd=ROOT, capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin", "HF_HUB_OFFLINE": "1",
                 "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
        assert result.returncode == 0, result.stderr
        body = json.loads(result.stdout)
        assert body["arms_runnable_here"] is False
        assert set(body["arms"]) == {"H1", "H2"}
        assert body["contract"]["declares_a_winner"] is False

    def test_the_loader_is_constructed_in_exactly_one_place(self):
        """And that place is the factory ``run_arm`` calls after the unlock."""
        source = self.source()
        assert source.count("ProductionGateDeps(") == 1
        head, tail = source.split("ProductionGateDeps(")
        assert "deps_factory=lambda cfg:" in head[-120:], head[-160:]

    def test_verify_mode_stops_before_the_arm_runner(self):
        source = self.source()
        assert source.count("arms.run_arm(") == 1
        before = source.split("arms.run_arm(")[0]
        assert 'if args.verify:' in before
        assert "--verify stops here" in before

    def test_a_run_without_the_carried_digests_refuses(self):
        result = self.run("--arm", "H1", "--run-dir", "x")
        assert result.returncode != 0
        assert "digest" in (result.stderr + result.stdout).lower()

    def test_it_leaks_no_identifier(self):
        from src.training.longrun import leaked_identifiers

        assert leaked_identifiers(self.source()) == []

    def test_the_arm_choices_are_exactly_the_frozen_two(self):
        result = self.run("--help")
        assert result.returncode == 0
        assert "H1" in result.stdout and "H2" in result.stdout
        assert "H3" not in result.stdout

    def test_it_cannot_run_both_arms_in_one_process(self):
        from src.training import arms

        arm_action = next(a for a in self.parser()._actions
                          if a.dest == "arm")
        assert tuple(arm_action.choices) == arms.ARM_NAMES
        assert arm_action.nargs is None, "the arm argument accepts a list"
        assert "for arm in" not in self.source()

    def test_the_child_arguments_round_trip_through_its_own_parser(self):
        """What the parent would spawn, the child must be able to read back."""
        from src.training import arms

        argv = arms.child_argv(
            arm="H1", run_dir="runs/h1", pack_dir=".",
            expected_pack_digest="a" * 64,
            expected_dependency_digest="b" * 64,
            gate_runs={role: f"runs/{role}" for role in
                       __import__("src.training.gate_suite",
                                  fromlist=["x"]).ROLES})
        import importlib.util

        spec = importlib.util.spec_from_file_location("h_run", self.SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        parsed = module.build_parser().parse_args(argv)
        assert parsed.arm == "H1"
        assert parsed.expected_pack_digest == "a" * 64
        assert parsed.gate_8 == "runs/gate_8"
        assert parsed.gate_500_uninterrupted_control == \
            "runs/gate_500_uninterrupted_control"


class TestTheSeedIsSharedNotChosen:
    """Determinism is applied before an arm is picked, so the seed must be one."""

    def test_both_arms_declare_the_same_seed(self):
        from src.training import arms, hypotheses

        assert arms.shared_seed() == hypotheses.config_for("H1").seed
        assert arms.shared_seed() == hypotheses.config_for("H2").seed

    def test_two_different_seeds_would_be_refused(self, monkeypatch):
        from src.training import arms, hypotheses
        from src.training.lora import LoraConfig_

        monkeypatch.setitem(hypotheses.FROZEN, "H2",
                            LoraConfig_(rank=32, alpha=16,
                                        learning_rate=2e-3, seed=7))
        with pytest.raises(arms.ArmRefused):
            arms.shared_seed()

    def test_the_script_takes_the_seed_from_there(self):
        source = (ROOT / "scripts" / "20_hypothesis_run.py").read_text(
            encoding="utf-8")
        assert "seed=arms.shared_seed()" in source
        assert "seed=0" not in source


# ---------------------------------------------------------------------------
# What has to be true of a finished arm before its evidence is written at all.
# ---------------------------------------------------------------------------

class TestTheOperationalValidator:
    """A run that finished is not the same as a run that measured something.

    ``run_gate`` returns evidence whatever happened: a cold load that could
    not rebuild the adapter, a row whose loss was NaN, a ledger with a hole in
    it, four hundred rows instead of two thousand. Each of those produces a
    file that looks like a measurement and is not one, and the next person to
    read it has no way to tell.

    So there is one place that decides, and it decides before anything is
    published. A failed arm leaves ``H*_failure.json`` and no
    ``H*_evidence.json`` at all -- not an evidence file with a flag set, which
    is a thing somebody eventually reads past.
    """

    def deps_factory(self, rows, mutate=None):
        def factory(cfg):
            deps = gates.FakeGateDeps(rows=rows)
            inner = deps.load

            def load(*, rows):
                loaded = inner(rows=rows)
                if mutate is not None:
                    mutate(loaded)
                return loaded

            deps.load = load
            return deps
        return factory

    @pytest.fixture()
    def small(self, monkeypatch):
        from src.training import arms

        monkeypatch.setattr(arms, "ARM_ROWS", 64)
        return arms

    def expect_failure(self, small, suite, tmp_path, factory, fragment):
        from src.training import arms

        with pytest.raises(arms.ArmFailed) as exc:
            small.run_arm("H1", deps_factory=factory,
                          **run_kwargs(suite, tmp_path))
        run_dir = tmp_path / "arm_run"
        assert not small.evidence_path(run_dir, "H1").exists(), \
            "a failed arm published ordinary evidence"
        failure = small.failure_path(run_dir, "H1")
        assert failure.is_file(), "a failed arm left no failure evidence"
        body = json.loads(failure.read_text(encoding="utf-8"))
        assert body["arm"] == "H1"
        assert any(fragment in p for p in body["problems"]), body["problems"]
        assert fragment in str(exc.value)
        return body

    def test_the_expected_step_count_is_derived(self, small):
        from src.training.lora import LoraConfig_

        assert small.expected_optimizer_steps() == 64 // LoraConfig_().grad_accum

    def test_the_real_arm_expects_two_hundred_and_fifty_steps(self):
        from src.training import arms

        assert arms.ARM_ROWS == 2000
        assert arms.expected_optimizer_steps() == 250

    def test_a_healthy_arm_publishes_evidence_and_no_failure(self, small,
                                                             suite, tmp_path):
        evidence = small.run_arm("H1", deps_factory=self.deps_factory(64),
                                 **run_kwargs(suite, tmp_path))
        run_dir = tmp_path / "arm_run"
        assert small.evidence_path(run_dir, "H1").is_file()
        assert not small.failure_path(run_dir, "H1").exists()
        assert json.loads(small.evidence_path(run_dir, "H1").read_text(
            encoding="utf-8"))["arm"] == "H1"
        assert evidence["operational_problems"] == []

    def test_a_non_finite_loss_fails_the_arm(self, small, suite, tmp_path):
        def mutate(loaded):
            step = loaded["step"]

            def bad(index, position):
                out = step(index, position)
                if position == 5:
                    out["loss"] = float("nan")
                return out

            loaded["step"] = bad

        self.expect_failure(small, suite, tmp_path,
                            self.deps_factory(64, mutate), "finite")

    def test_a_cold_load_that_did_not_rebuild_fails_the_arm(self, small, suite,
                                                            tmp_path):
        def mutate(loaded):
            loaded["cold_load"] = lambda saved, d: {
                "loaded": False, "sha256": None, "matches_saved": False,
                "reason": "the adapter would not load"}

        self.expect_failure(small, suite, tmp_path,
                            self.deps_factory(64, mutate), "cold load")

    def test_a_cold_load_whose_digest_disagrees_fails_the_arm(self, small,
                                                              suite, tmp_path):
        def mutate(loaded):
            loaded["cold_load"] = lambda saved, d: {
                "loaded": True, "sha256": "0" * 64, "matches_saved": False}

        self.expect_failure(small, suite, tmp_path,
                            self.deps_factory(64, mutate), "cold load")

    def test_an_adapter_saved_without_a_digest_fails_the_arm(self, small,
                                                             suite, tmp_path):
        def mutate(loaded):
            inner = loaded["save_adapter"]
            loaded["save_adapter"] = lambda dest: {
                **inner(dest), "sha256": None}

        self.expect_failure(small, suite, tmp_path,
                            self.deps_factory(64, mutate), "adapter")

    def test_an_unusable_trainable_digest_fails_the_arm(self, small, suite,
                                                        tmp_path):
        def mutate(loaded):
            loaded["trainable_digest"] = lambda: "not-a-digest"

        self.expect_failure(small, suite, tmp_path,
                            self.deps_factory(64, mutate), "trainable digest")

    def test_the_wrong_row_count_fails_the_arm(self, small, suite, tmp_path,
                                               monkeypatch):
        """The spec says 64 rows; the loader hands back an order of 32."""
        from src.training import arms, gates as g

        def factory(cfg):
            deps = g.FakeGateDeps(rows=64)
            inner = deps.load

            def load(*, rows):
                loaded = inner(rows=rows)
                loaded["order"] = loaded["order"][:32]
                return loaded

            deps.load = load
            return deps

        with pytest.raises((arms.ArmFailed, g.GateRefused)):
            small.run_arm("H1", deps_factory=factory,
                          **run_kwargs(suite, tmp_path))
        assert not small.evidence_path(tmp_path / "arm_run", "H1").exists()

    def test_a_wrong_optimizer_step_count_fails_the_arm(self, small, suite,
                                                        tmp_path):
        from src.training import arms

        evidence = {"arm": "H1", "rows_declared": 64, "rows_completed": 64,
                    "losses_finite": True, "ledger_problems": [],
                    "optimizer_steps": 7,
                    "saved": {"sha256": "a" * 64},
                    "cold_load": {"loaded": True, "matches_saved": True},
                    "trainable_digest": "b" * 64,
                    "per_row_loss": [1.0] * 64, "sample_ids": ["s"] * 64}
        problems = arms.operational_problems(evidence)
        assert any("optimizer_steps" in p for p in problems), problems

    def test_a_ledger_problem_fails_the_arm(self, small):
        from src.training import arms

        evidence = {"arm": "H1", "rows_declared": 64, "rows_completed": 64,
                    "losses_finite": True,
                    "ledger_problems": ["row 12 records no sample_id"],
                    "optimizer_steps": 8, "saved": {"sha256": "a" * 64},
                    "cold_load": {"loaded": True, "matches_saved": True},
                    "trainable_digest": "b" * 64,
                    "per_row_loss": [1.0] * 64, "sample_ids": ["s"] * 64}
        problems = arms.operational_problems(evidence)
        assert any("ledger" in p for p in problems), problems

    def test_every_check_is_reported_not_just_the_first(self, small):
        from src.training import arms

        problems = arms.operational_problems({"arm": "H1"})
        assert len(problems) >= 5, problems

    def test_a_directory_that_already_failed_is_not_reused(self, small, suite,
                                                           tmp_path):
        from src.training import arms

        self.expect_failure(
            small, suite, tmp_path,
            self.deps_factory(64, lambda loaded: loaded.__setitem__(
                "cold_load", lambda saved, d: {"loaded": False,
                                               "matches_saved": False})),
            "cold load")
        with pytest.raises(arms.ArmRefused) as exc:
            small.run_arm("H1", deps_factory=self.deps_factory(64),
                          **run_kwargs(suite, tmp_path))
        assert "failure" in str(exc.value).lower()

    def test_the_other_arm_s_failure_marker_also_blocks_the_directory(
            self, small, suite, tmp_path):
        from src.training import arms

        run_dir = tmp_path / "arm_run"
        run_dir.mkdir(parents=True)
        small.write_failure(run_dir, "H2", "something went wrong")
        with pytest.raises(arms.ArmRefused):
            small.run_arm("H1", deps_factory=self.deps_factory(64),
                          **run_kwargs(suite, tmp_path))


class TestTheOperationalValidatorPinsAbsoluteValues:
    """Evidence must not be allowed to set its own pass mark.

    ``rows_declared`` came out of the same file as ``rows_completed``. Reading
    the target from the thing being measured means a run that declares 8 rows
    and completes 8 rows agrees with itself perfectly and is not the arm.
    """

    def base(self, **over):
        from src.training import arms

        e = {"arm": "H1", "rows_declared": arms.ARM_ROWS,
             "rows_completed": arms.ARM_ROWS, "losses_finite": True,
             "ledger_problems": [],
             "optimizer_steps": arms.expected_optimizer_steps(),
             "saved": {"sha256": "a" * 64},
             "cold_load": {"loaded": True, "matches_saved": True,
                           "sha256": "a" * 64},
             "trainable_digest": "b" * 64}
        e.update(over)
        return e

    def test_a_self_consistent_short_run_is_refused(self):
        from src.training import arms

        e = self.base(rows_declared=8, rows_completed=8, optimizer_steps=1)
        problems = arms.operational_problems(e)
        assert any("rows" in p for p in problems), problems

    def test_rows_declared_must_be_the_frozen_length(self):
        from src.training import arms

        problems = arms.operational_problems(self.base(rows_declared=8))
        assert any("rows_declared" in p for p in problems), problems

    def test_the_healthy_shape_passes(self):
        from src.training import arms

        assert arms.operational_problems(self.base()) == []

    @pytest.mark.parametrize("value", [None, "", "not-a-digest", "A" * 64,
                                       "a" * 63, 12345, True])
    def test_an_adapter_digest_that_is_not_a_sha256_is_refused(self, value):
        from src.training import arms

        problems = arms.operational_problems(
            self.base(saved={"sha256": value},
                      cold_load={"loaded": True, "matches_saved": True,
                                 "sha256": value}))
        assert any("adapter" in p for p in problems), (value, problems)

    def test_the_cold_load_digest_must_equal_the_saved_one(self):
        from src.training import arms

        problems = arms.operational_problems(
            self.base(cold_load={"loaded": True, "matches_saved": True,
                                 "sha256": "c" * 64}))
        assert any("cold load" in p for p in problems), problems

    def test_a_cold_load_with_no_digest_at_all_is_refused(self):
        from src.training import arms

        problems = arms.operational_problems(
            self.base(cold_load={"loaded": True, "matches_saved": True}))
        assert any("cold load" in p for p in problems), problems


class TestTheArmOrderIsFrozen:
    """H1 then H2, once each.

    What "after H1" means -- and how thoroughly it is checked -- is
    :class:`TestThePredecessorIsVerifiedNotBelieved` below. This is only about
    the order being declared rather than chosen later.
    """

    def test_the_order_is_declared(self):
        from src.training import arms

        assert arms.ARM_ORDER == ("H1", "H2")
        assert arms.RUNS_PER_ARM == 1
        assert arms.preceding_arm("H1") is None
        assert arms.preceding_arm("H2") == "H1"

    def test_the_order_is_the_arm_list_and_not_a_second_one(self):
        from src.training import arms

        assert set(arms.ARM_ORDER) == set(arms.ARM_NAMES)
        assert len(arms.ARM_ORDER) == len(arms.ARM_NAMES) == 2

    def test_the_script_requires_the_predecessor_for_h2_only(self):
        source = (ROOT / "scripts" / "20_hypothesis_run.py").read_text(
            encoding="utf-8")
        assert "--previous-arm-run-dir" in source
        assert "predecessor_problems" in source


# ---------------------------------------------------------------------------
# H2 runs after H1. "After" has to mean something a file cannot simply claim.
# ---------------------------------------------------------------------------

class TestThePredecessorIsVerifiedNotBelieved:
    """``{"arm": "H1", "operational_problems": []}`` used to unlock H2.

    Two keys, typed by anyone, naming no pack, no dataset, no configuration
    and no measurement. The predecessor check re-reads H1's evidence, plan,
    ledger and adapter, re-runs the operational validator over them, and binds
    the whole thing to the same carried values H2 itself is being run against.
    """

    @pytest.fixture()
    def small(self, monkeypatch):
        from src.training import arms

        monkeypatch.setattr(arms, "ARM_ROWS", 64)
        return arms

    def h1(self, small, suite, tmp_path, *, factory=None):
        """A real, complete H1 run, driven through the real runner."""
        run_dir = tmp_path / "h1"
        factory = factory or (lambda cfg: gates.FakeGateDeps(rows=64, cfg=cfg))
        small.run_arm("H1", deps_factory=factory,
                      **run_kwargs(suite, tmp_path, run_dir=run_dir))
        return run_dir

    def check(self, small, suite, previous):
        return small.predecessor_problems(
            "H2", previous_run_dir=previous, **suite["carried"])

    # -- the shape of the rule -------------------------------------------

    def test_h1_takes_no_predecessor(self, small, suite):
        assert small.predecessor_problems(
            "H1", previous_run_dir=None, **suite["carried"]) == []

    def test_h1_given_one_is_refused(self, small, suite, tmp_path):
        problems = small.predecessor_problems(
            "H1", previous_run_dir=tmp_path, **suite["carried"])
        assert any("H1" in p for p in problems), problems

    def test_h2_without_one_is_refused(self, small, suite):
        assert self.check(small, suite, None)

    # -- the fail-open this closes ---------------------------------------

    def test_two_keys_no_longer_unlock_h2(self, small, suite, tmp_path):
        from src.training.session import write_once_json

        fake = tmp_path / "claimed"
        write_once_json(small.evidence_path(fake, "H1"),
                        {"arm": "H1", "operational_problems": []})
        problems = self.check(small, suite, fake)
        assert problems, "a two-key file unlocked H2"

    def test_a_self_consistent_eight_row_h1_is_refused(self, small, suite,
                                                       tmp_path):
        """Complete, internally coherent, and not the arm."""
        from src.training.session import write_once_json

        fake = tmp_path / "eight"
        write_once_json(small.evidence_path(fake, "H1"), {
            "arm": "H1", "rows_declared": 8, "rows_completed": 8,
            "losses_finite": True, "ledger_problems": [],
            "optimizer_steps": 1, "operational_problems": [],
            "saved": {"sha256": "a" * 64},
            "cold_load": {"loaded": True, "matches_saved": True,
                          "sha256": "a" * 64},
            "trainable_digest": "b" * 64})
        assert self.check(small, suite, fake)

    def test_a_stored_empty_problem_list_is_not_believed(self, small, suite,
                                                         tmp_path):
        previous = self.h1(small, suite, tmp_path)
        path = small.evidence_path(previous, "H1")
        body = json.loads(path.read_text(encoding="utf-8"))
        body["losses_finite"] = False
        body["operational_problems"] = []
        path.unlink(); path.write_text(json.dumps(body, indent=2))
        problems = self.check(small, suite, previous)
        assert any("finite" in p for p in problems), problems

    # -- the happy path ---------------------------------------------------

    def test_a_complete_h1_from_the_same_pack_unlocks_h2(self, small, suite,
                                                         tmp_path):
        previous = self.h1(small, suite, tmp_path)
        assert self.check(small, suite, previous) == []

    def test_and_then_h2_actually_runs(self, small, suite, tmp_path):
        previous = self.h1(small, suite, tmp_path)
        evidence = small.run_arm(
            "H2", deps_factory=lambda cfg: gates.FakeGateDeps(rows=64, cfg=cfg),
            previous_run_dir=previous,
            **run_kwargs(suite, tmp_path, run_dir=tmp_path / "h2"))
        assert evidence["arm"] == "H2"
        assert evidence["config"]["rank"] == 32
        assert evidence["operational_problems"] == []

    # -- binding to this H2's carried values -------------------------------

    def test_an_h1_from_another_pack_is_refused(self, small, suite, tmp_path):
        previous = self.h1(small, suite, tmp_path)
        carried = dict(suite["carried"])
        carried["expected_pack_digest"] = "f" * 64
        problems = small.predecessor_problems(
            "H2", previous_run_dir=previous, **carried)
        assert any("pack" in p for p in problems), problems

    def test_an_h1_bound_to_a_different_dependency_set_is_refused(
            self, small, suite, tmp_path):
        previous = self.h1(small, suite, tmp_path)
        path = previous / gates.PLAN_NAME
        body = json.loads(path.read_text(encoding="utf-8"))
        body["dependency_digest"] = "c" * 64
        path.unlink(); path.write_text(json.dumps(body, indent=2))
        assert any("dependency" in p
                   for p in self.check(small, suite, previous))

    @pytest.mark.parametrize("field,value", [
        ("allocator_config", "expandable_segments:False"),
        ("determinism", {"seed": 9}),
    ])
    def test_an_h1_under_a_different_runtime_is_refused(
            self, small, suite, tmp_path, field, value):
        previous = self.h1(small, suite, tmp_path)
        path = small.evidence_path(previous, "H1")
        body = json.loads(path.read_text(encoding="utf-8"))
        body[field] = value
        path.unlink(); path.write_text(json.dumps(body, indent=2))
        assert self.check(small, suite, previous)

    def test_an_h1_that_recorded_the_wrong_config_is_refused(
            self, small, suite, tmp_path):
        previous = self.h1(small, suite, tmp_path)
        path = small.evidence_path(previous, "H1")
        body = json.loads(path.read_text(encoding="utf-8"))
        body["config"]["rank"] = 32
        path.unlink(); path.write_text(json.dumps(body, indent=2))
        assert any("config" in p for p in self.check(small, suite, previous))

    def test_an_h1_that_ran_the_wrong_arm_is_refused(self, small, suite,
                                                     tmp_path):
        previous = self.h1(small, suite, tmp_path)
        path = small.evidence_path(previous, "H1")
        body = json.loads(path.read_text(encoding="utf-8"))
        body["epochs"] = 3
        path.unlink(); path.write_text(json.dumps(body, indent=2))
        assert any("epoch" in p for p in self.check(small, suite, previous))

    # -- the record has to agree with itself -------------------------------

    def test_a_tampered_ledger_loss_is_refused(self, small, suite, tmp_path):
        previous = self.h1(small, suite, tmp_path)
        path = previous / gates.LEDGER_NAME
        lines = path.read_text(encoding="utf-8").splitlines()
        entry = json.loads(lines[3]); entry["loss"] += 1.0
        lines[3] = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert any("loss" in p for p in self.check(small, suite, previous))

    def test_a_tampered_sample_id_is_refused(self, small, suite, tmp_path):
        previous = self.h1(small, suite, tmp_path)
        path = previous / gates.LEDGER_NAME
        lines = path.read_text(encoding="utf-8").splitlines()
        entry = json.loads(lines[7]); entry["sample_id"] = "somebody-else"
        lines[7] = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert any("sample" in p for p in self.check(small, suite, previous))

    def test_a_missing_ledger_is_refused(self, small, suite, tmp_path):
        previous = self.h1(small, suite, tmp_path)
        (previous / gates.LEDGER_NAME).unlink()
        assert self.check(small, suite, previous)

    def test_a_missing_plan_is_refused(self, small, suite, tmp_path):
        previous = self.h1(small, suite, tmp_path)
        (previous / gates.PLAN_NAME).unlink()
        assert any("plan" in p for p in self.check(small, suite, previous))

    def test_an_edited_order_is_refused(self, small, suite, tmp_path):
        previous = self.h1(small, suite, tmp_path)
        path = previous / gates.PLAN_NAME
        body = json.loads(path.read_text(encoding="utf-8"))
        body["order"] = list(reversed(body["order"]))
        path.unlink(); path.write_text(json.dumps(body, indent=2))
        assert any("order" in p for p in self.check(small, suite, previous))

    # -- the adapter --------------------------------------------------------

    def test_a_tampered_adapter_is_refused(self, small, suite, tmp_path):
        previous = self.h1(small, suite, tmp_path)
        blob = previous / "adapter" / "adapter_model.json"
        blob.write_bytes(blob.read_bytes() + b" ")
        assert any("adapter" in p for p in self.check(small, suite, previous))

    def test_a_missing_adapter_is_refused(self, small, suite, tmp_path):
        previous = self.h1(small, suite, tmp_path)
        (previous / "adapter" / "adapter_model.json").unlink()
        assert any("adapter" in p for p in self.check(small, suite, previous))

    def test_an_adapter_manifest_for_the_other_arm_is_refused(
            self, small, suite, tmp_path):
        """Rank 32 weights beside a manifest that says 16, one level up."""
        previous = self.h1(small, suite, tmp_path)
        path = previous / "adapter" / "brickagain_manifest.json"
        body = json.loads(path.read_text(encoding="utf-8"))
        body["lora"]["r"] = 32
        path.write_text(json.dumps(body, indent=2))
        assert any("manifest" in p for p in self.check(small, suite, previous))

    def test_a_missing_adapter_manifest_is_refused(self, small, suite,
                                                   tmp_path):
        previous = self.h1(small, suite, tmp_path)
        (previous / "adapter" / "brickagain_manifest.json").unlink()
        assert any("manifest" in p for p in self.check(small, suite, previous))

    # -- failure markers ----------------------------------------------------

    @pytest.mark.parametrize("arm", ["H1", "H2"])
    def test_any_failure_marker_in_the_predecessor_is_refused(
            self, small, suite, tmp_path, arm):
        previous = self.h1(small, suite, tmp_path)
        small.write_failure(previous, arm, "something went wrong")
        assert any("failure" in p.lower()
                   for p in self.check(small, suite, previous))

    # -- one verifier, two callers -----------------------------------------

    def test_the_runner_and_the_script_call_the_same_verifier(self):
        import inspect

        from src.training import arms

        source = inspect.getsource(arms.run_arm)
        assert "predecessor_problems(" in source
        script = (ROOT / "scripts" / "20_hypothesis_run.py").read_text(
            encoding="utf-8")
        assert "predecessor_problems(" in script
        assert "order_problems(" not in script

    def test_run_arm_refuses_a_bad_predecessor_before_the_loader(
            self, small, suite, tmp_path):
        from src.training import arms
        from src.training.session import write_once_json

        fake = tmp_path / "claimed"
        write_once_json(small.evidence_path(fake, "H1"),
                        {"arm": "H1", "operational_problems": []})
        spy = Spy()
        with pytest.raises(arms.ArmRefused):
            small.run_arm("H2", deps_factory=spy, previous_run_dir=fake,
                          **run_kwargs(suite, tmp_path, run_dir=tmp_path / "h2"))
        assert spy.built_with == []
        assert not (tmp_path / "h2").exists()

    def test_the_child_argv_round_trips_the_predecessor(self):
        import importlib.util

        from src.training import arms, gate_suite

        argv = arms.child_argv(
            arm="H2", run_dir="runs/h2", pack_dir=".",
            expected_pack_digest="a" * 64,
            expected_dependency_digest="b" * 64,
            gate_runs={role: f"runs/{role}" for role in gate_suite.ROLES},
            previous_run_dir="runs/h1")
        script = ROOT / "scripts" / "20_hypothesis_run.py"
        spec = importlib.util.spec_from_file_location("h_run", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        parsed = module.build_parser().parse_args(argv)
        assert parsed.arm == "H2"
        assert parsed.previous_arm_run_dir == "runs/h1"

    def test_h1_child_argv_carries_no_predecessor(self):
        from src.training import arms, gate_suite

        argv = arms.child_argv(
            arm="H1", run_dir="runs/h1", pack_dir=".",
            expected_pack_digest="a" * 64,
            expected_dependency_digest="b" * 64,
            gate_runs={role: f"runs/{role}" for role in gate_suite.ROLES})
        assert "--previous-arm-run-dir" not in argv
