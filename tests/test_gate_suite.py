"""The formal gate suite: six named runs, or no unlock.

Three gates passing is not the same claim as *this* node having passed them.
The gates each answer one question about one run; unlocking H1 and H2 asks
something none of them asks on its own -- that six specific runs, all against
the same pack, the same dependency bytes, the same allocator and the same
determinism settings, exist and agree with each other.

So the verifier takes six **roles**, by name, and never a directory listing:

* ``gate_8``
* ``gate_100_r1`` / ``gate_100_r2`` / ``gate_100_r3``
* ``gate_500_resumed``
* ``gate_500_uninterrupted_control``

A glob would make the answer depend on what happened to be on disk; sorting
would make it depend on names somebody chose afterwards; ``mtime`` would make
it depend on which file was copied last. All three turn "which runs prove
this" into "which runs are lying around", and the whole point of the suite is
that the caller has to say, out loud, which run plays which part.

Everything below is driven through the same runner the node uses, with the
model and device supplied by the deterministic fake. Nothing here loads a
model, opens a socket or reads the dataset.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.training import gates

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


def dep_checker(evidence=None):
    body = DEP_EVIDENCE if evidence is None else evidence
    return lambda: {"ok": True, "problems": [], "evidence": body}


def dep_digest(evidence=None):
    from src.training.longrun import dependency_digest

    return dependency_digest(DEP_EVIDENCE if evidence is None else evidence)


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
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    dest = tmp_path / "pack"
    pack.build(dest, root=src)
    return dest


def digest_of(packdir) -> str:
    from src.training import pack

    return pack.read_manifest(packdir)[0]["pack_digest"]


# ---------------------------------------------------------------------------
# Six real runs, built once.
# ---------------------------------------------------------------------------

def run_one(gate, run_dir, packdir, *, rows, stop_after=None, resume=False):
    from src.training import gate_suite

    kw = dict(pack_dir=packdir, expected_pack_digest=digest_of(packdir),
              expected_dependency_digest=dep_digest(),
              dependency_checker=dep_checker(),
              allocator_config=ALLOC, determinism=dict(DETERMINISM))
    evidence = gates.run_gate(gate, deps=gates.FakeGateDeps(rows=rows),
                              run_dir=run_dir, stop_after=stop_after,
                              resume=resume, **kw)
    from src.training.session import write_once_json

    write_once_json(gate_suite.evidence_path(run_dir, gate),
                    {"verdict": gates.verdict(gate, evidence),
                     "evidence": evidence})
    return evidence


def build_suite(root, packdir):
    """The six runs, in the shape the node produces them."""
    from src.training import gate_suite

    root = Path(root)
    runs = {}
    runs["gate_8"] = root / "gate_8"
    run_one("gate_8", runs["gate_8"], packdir, rows=8)

    for role in gate_suite.GATE_100_ROLES:
        runs[role] = root / role
        run_one("gate_100", runs[role], packdir, rows=100)

    control = root / "gate_500_control"
    run_one("gate_500", control, packdir, rows=500)
    runs[gate_suite.GATE_500_CONTROL] = control

    resumed = root / "gate_500_resumed"
    with pytest.raises(gates.DeliberateStop):
        run_one("gate_500", resumed, packdir, rows=500, stop_after=250)
    run_one("gate_500", resumed, packdir, rows=500, resume=True)
    runs[gate_suite.GATE_500_RESUMED] = resumed
    return runs


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("suite")
    packdir = build_pack(root)
    runs = build_suite(root / "runs", packdir)
    return {"runs": runs, "packdir": packdir, "root": root}


@pytest.fixture()
def suite(built):
    return dict(built["runs"])


@pytest.fixture()
def carried(built):
    return {"expected_pack_digest": digest_of(built["packdir"]),
            "expected_dependency_digest": dep_digest(),
            "allocator_config": ALLOC,
            "determinism": dict(DETERMINISM)}


@pytest.fixture()
def mutable(built, tmp_path):
    """A private copy of all six, for tests that have to break one."""
    from src.training import gate_suite

    out = {}
    for role, src in built["runs"].items():
        dest = tmp_path / "copy" / role
        shutil.copytree(src, dest)
        out[role] = dest
    return out


def problems(runs, carried):
    from src.training import gate_suite

    return gate_suite.suite_problems(runs, **carried)


def rewrite_evidence(run_dir, gate, mutate):
    from src.training import gate_suite

    path = gate_suite.evidence_path(run_dir, gate)
    body = json.loads(path.read_text(encoding="utf-8"))
    mutate(body)
    path.unlink()
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")


def rewrite_plan(run_dir, mutate):
    path = Path(run_dir) / gates.PLAN_NAME
    body = json.loads(path.read_text(encoding="utf-8"))
    mutate(body)
    path.unlink()
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------

class TestTheSuiteIsSixNamedRoles:

    def test_the_roles_are_declared_and_exactly_six(self):
        from src.training import gate_suite

        assert gate_suite.ROLES == (
            "gate_8", "gate_100_r1", "gate_100_r2", "gate_100_r3",
            "gate_500_resumed", "gate_500_uninterrupted_control")

    def test_each_role_names_the_gate_it_plays(self):
        from src.training import gate_suite

        assert gate_suite.ROLE_GATE == {
            "gate_8": "gate_8",
            "gate_100_r1": "gate_100",
            "gate_100_r2": "gate_100",
            "gate_100_r3": "gate_100",
            "gate_500_resumed": "gate_500",
            "gate_500_uninterrupted_control": "gate_500"}

    def test_the_honest_suite_verifies(self, suite, carried):
        assert problems(suite, carried) == []

    def test_a_missing_role_is_refused(self, suite, carried):
        for role in list(suite):
            short = {k: v for k, v in suite.items() if k != role}
            found = problems(short, carried)
            assert found, role
            assert any(role in p for p in found), (role, found)

    def test_an_unknown_role_is_refused(self, suite, carried):
        suite["gate_1000"] = suite["gate_8"]
        found = problems(suite, carried)
        assert any("gate_1000" in p for p in found), found

    def test_the_same_directory_cannot_play_two_roles(self, suite, carried):
        suite["gate_100_r2"] = suite["gate_100_r1"]
        found = problems(suite, carried)
        assert found
        assert any("gate_100_r1" in p and "gate_100_r2" in p for p in found), found

    def test_the_two_gate_500_roles_cannot_be_swapped(self, suite, carried):
        from src.training import gate_suite

        a, b = gate_suite.GATE_500_RESUMED, gate_suite.GATE_500_CONTROL
        suite[a], suite[b] = suite[b], suite[a]
        found = problems(suite, carried)
        assert found, "a swapped pair unlocked the hypotheses"

    def test_a_gate_100_run_cannot_stand_in_for_gate_8(self, suite, carried):
        suite["gate_8"] = suite["gate_100_r1"]
        found = problems(suite, carried)
        assert any("gate_8" in p for p in found), found

    def test_a_directory_that_is_not_a_run_is_refused(self, suite, carried,
                                                      tmp_path):
        empty = tmp_path / "nothing"
        empty.mkdir()
        suite["gate_8"] = empty
        assert problems(suite, carried)

    def test_nothing_is_found_by_globbing_or_sorting(self):
        """The reader must not contain a discovery mechanism at all."""
        import inspect

        from src.training import gate_suite

        source = inspect.getsource(gate_suite)
        for forbidden in ("glob(", "rglob(", "iterdir(", "listdir(",
                          "st_mtime", "latest_checkpoint("):
            assert forbidden not in source, forbidden


class TestVerdictsAreRecomputedNotRead:

    def test_a_tampered_stored_verdict_does_not_unlock(self, mutable, carried):
        """A file that says 'passed' is a file, not a verdict."""
        rewrite_evidence(mutable["gate_8"], "gate_8",
                         lambda b: b.__setitem__("verdict", "failed"))
        found = problems(mutable, carried)
        assert any("verdict" in p for p in found), found

    def test_a_failing_evidence_labelled_passed_does_not_unlock(
            self, mutable, carried):
        def spoil(body):
            body["evidence"]["losses_finite"] = False
            body["verdict"] = "passed"

        rewrite_evidence(mutable["gate_100_r1"], "gate_100", spoil)
        found = problems(mutable, carried)
        assert any("gate_100_r1" in p for p in found), found

    def test_three_bare_passed_strings_are_not_an_argument(self):
        """The old API took them. Nothing may take them now."""
        from src.training import hypotheses

        with pytest.raises(TypeError):
            hypotheses.require_unlocked(
                "H1", gate_results={"gate_8": "passed", "gate_100": "passed",
                                    "gate_500": "passed"})
        assert not hasattr(hypotheses, "gate_problems"), (
            "the verdict-string entry point is still reachable")

    def test_evidence_naming_the_wrong_gate_is_refused(self, mutable, carried):
        rewrite_evidence(mutable["gate_100_r2"], "gate_100",
                         lambda b: b["evidence"].__setitem__("gate", "gate_8"))
        found = problems(mutable, carried)
        assert any("gate_100_r2" in p for p in found), found

    def test_a_plan_naming_the_wrong_gate_is_refused(self, mutable, carried):
        rewrite_plan(mutable["gate_8"],
                     lambda b: b.__setitem__("gate", "gate_500"))
        found = problems(mutable, carried)
        assert any("gate_8" in p for p in found), found

    def test_a_tampered_ledger_is_caught_even_when_the_evidence_is_clean(
            self, mutable, carried):
        """``ledger_problems`` inside the evidence is also just a file."""
        path = Path(mutable["gate_100_r3"]) / gates.LEDGER_NAME
        lines = path.read_text(encoding="utf-8").splitlines()
        entry = json.loads(lines[10])
        entry["index"] = entry["index"] + 1
        lines[10] = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert problems(mutable, carried)


class TestEveryRunIsBoundToTheSameCarriedValues:

    def test_a_wrong_carried_pack_digest_refuses(self, suite, carried):
        carried["expected_pack_digest"] = "f" * 64
        assert problems(suite, carried)

    def test_a_malformed_carried_value_refuses(self, suite, carried):
        carried["expected_pack_digest"] = "not-a-digest"
        assert problems(suite, carried)

    def test_a_run_from_an_older_pack_refuses(self, mutable, carried):
        rewrite_plan(mutable["gate_100_r1"],
                     lambda b: b.__setitem__("pack_digest", "a" * 64))
        found = problems(mutable, carried)
        assert any("gate_100_r1" in p and "pack" in p for p in found), found

    def test_a_mixed_dependency_digest_refuses(self, mutable, carried):
        rewrite_plan(mutable["gate_500_resumed"],
                     lambda b: b.__setitem__("dependency_digest", "b" * 64))
        found = problems(mutable, carried)
        assert any("dependency" in p for p in found), found

    def test_a_different_allocator_config_refuses(self, mutable, carried):
        rewrite_plan(mutable["gate_8"],
                     lambda b: b.__setitem__("allocator_config",
                                             "expandable_segments:True,x:1"))
        found = problems(mutable, carried)
        assert any("allocator" in p for p in found), found

    def test_a_different_determinism_record_refuses(self, mutable, carried):
        rewrite_plan(mutable["gate_100_r2"],
                     lambda b: b["determinism"].__setitem__("seed", 7))
        found = problems(mutable, carried)
        assert any("determinism" in p for p in found), found

    def test_a_missing_binding_field_refuses(self, mutable, carried):
        rewrite_plan(mutable["gate_8"], lambda b: b.pop("allocator_config"))
        assert problems(mutable, carried)

    def test_drifted_provenance_refuses(self, mutable, carried):
        rewrite_plan(mutable["gate_500_resumed"],
                     lambda b: b["provenance"].__setitem__("device", "cpu"))
        found = problems(mutable, carried)
        assert any("provenance" in p for p in found), found

    def test_the_evidence_must_agree_with_its_own_plan(self, mutable, carried):
        rewrite_evidence(
            mutable["gate_100_r1"], "gate_100",
            lambda b: b["evidence"].__setitem__("pack_digest", "c" * 64))
        assert problems(mutable, carried)


class TestTheThreeHundredRunsMustAgreeExactly:

    def test_all_three_must_pass_individually(self, mutable, carried):
        rewrite_evidence(
            mutable["gate_100_r2"], "gate_100",
            lambda b: b["evidence"].__setitem__("rows_completed", 99))
        found = problems(mutable, carried)
        assert any("gate_100_r2" in p for p in found), found

    def test_a_differing_input_order_refuses(self, mutable, carried):
        path = Path(mutable["gate_100_r3"]) / gates.LEDGER_NAME
        lines = path.read_text(encoding="utf-8").splitlines()
        a, b = json.loads(lines[0]), json.loads(lines[1])
        a["index"], b["index"] = b["index"], a["index"]
        lines[0] = json.dumps(a, sort_keys=True, separators=(",", ":"))
        lines[1] = json.dumps(b, sort_keys=True, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        found = problems(mutable, carried)
        assert any("order" in p for p in found), found

    def test_a_differing_sample_id_refuses(self, mutable, carried):
        path = Path(mutable["gate_100_r2"]) / gates.LEDGER_NAME
        lines = path.read_text(encoding="utf-8").splitlines()
        entry = json.loads(lines[3])
        entry["sample_id"] = "somebody-else"
        lines[3] = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        found = problems(mutable, carried)
        assert any("sample" in p for p in found), found

    def test_a_single_differing_loss_refuses(self, mutable, carried):
        path = Path(mutable["gate_100_r3"]) / gates.LEDGER_NAME
        lines = path.read_text(encoding="utf-8").splitlines()
        entry = json.loads(lines[57])
        entry["loss"] = entry["loss"] + 1e-12
        lines[57] = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        found = problems(mutable, carried)
        assert any("loss" in p for p in found), found

    def test_a_differing_final_digest_refuses(self, mutable, carried):
        rewrite_evidence(
            mutable["gate_100_r1"], "gate_100",
            lambda b: b["evidence"].__setitem__("trainable_digest", "d" * 64))
        found = problems(mutable, carried)
        assert any("trainable" in p for p in found), found

    def test_the_comparison_reads_the_ledger_not_the_evidence(
            self, mutable, carried):
        """A run that reported 100 identical rows and recorded other ones."""
        path = Path(mutable["gate_100_r1"]) / gates.LEDGER_NAME
        lines = path.read_text(encoding="utf-8").splitlines()
        entry = json.loads(lines[20])
        entry["loss"] = 99.0
        lines[20] = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert problems(mutable, carried)


class TestTheTwoFiveHundredRunsMustAgreeExactly:

    def test_the_resumed_run_must_pass(self, mutable, carried):
        rewrite_evidence(
            mutable["gate_500_resumed"], "gate_500",
            lambda b: b["evidence"].__setitem__("rng_state_restored", False))
        found = problems(mutable, carried)
        assert any("gate_500_resumed" in p for p in found), found

    def test_the_control_must_fail_and_a_passing_one_is_refused(
            self, mutable, carried):
        """A control that passes gate 500 was interrupted, so it is not a control."""
        def make_it_pass(body):
            e = body["evidence"]
            e.update({"stopped_at": 250, "resumed_from": 192, "attempts": 2,
                      "model_state_restored": True, "rng_state_restored": True,
                      "optimizer_state_restored": True})
            body["verdict"] = "passed"

        rewrite_evidence(mutable["gate_500_uninterrupted_control"], "gate_500",
                         make_it_pass)
        found = problems(mutable, carried)
        assert any("control" in p for p in found), found

    def test_the_control_may_only_fail_for_never_being_interrupted(
            self, mutable, carried):
        rewrite_evidence(
            mutable["gate_500_uninterrupted_control"], "gate_500",
            lambda b: b["evidence"].__setitem__("losses_finite", False))
        found = problems(mutable, carried)
        assert any("control" in p for p in found), found

    def test_a_control_with_a_missing_row_is_refused(self, mutable, carried):
        rewrite_evidence(
            mutable["gate_500_uninterrupted_control"], "gate_500",
            lambda b: b["evidence"].__setitem__("missing_positions", [17]))
        assert problems(mutable, carried)

    def test_a_differing_five_hundred_row_loss_refuses(self, mutable, carried):
        path = Path(mutable["gate_500_uninterrupted_control"]) / gates.LEDGER_NAME
        lines = path.read_text(encoding="utf-8").splitlines()
        entry = json.loads(lines[499])
        entry["loss"] = entry["loss"] * 2
        lines[499] = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        found = problems(mutable, carried)
        assert any("loss" in p for p in found), found

    def test_a_differing_final_digest_refuses(self, mutable, carried):
        rewrite_evidence(
            mutable["gate_500_resumed"], "gate_500",
            lambda b: b["evidence"].__setitem__("trainable_digest", "e" * 64))
        found = problems(mutable, carried)
        assert any("trainable" in p for p in found), found

    def test_differing_optimizer_steps_refuse(self, mutable, carried):
        rewrite_evidence(
            mutable["gate_500_uninterrupted_control"], "gate_500",
            lambda b: b["evidence"].__setitem__("optimizer_steps", 61))
        found = problems(mutable, carried)
        assert any("optimizer_steps" in p for p in found), found

    def test_the_re_executed_span_must_reproduce_attempt_one(
            self, mutable, carried):
        """Rows measured twice must have been measured from the same state."""
        path = Path(mutable["gate_500_resumed"]) / gates.LEDGER_NAME
        lines = path.read_text(encoding="utf-8").splitlines()
        changed = 0
        for i, line in enumerate(lines):
            entry = json.loads(line)
            if entry["attempt"] == 1 and entry["position"] == 200:
                entry["loss"] = entry["loss"] + 0.5
                lines[i] = json.dumps(entry, sort_keys=True,
                                      separators=(",", ":"))
                changed += 1
        assert changed == 1, "the fixture no longer has a re-executed row 200"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        found = problems(mutable, carried)
        assert any("re-executed" in p or "attempt" in p for p in found), found

    def test_the_span_that_is_checked_is_the_one_that_was_re_executed(
            self, suite, carried):
        from src.training import gate_suite

        span = gate_suite.re_executed_span(suite[gate_suite.GATE_500_RESUMED])
        assert span == list(range(193, 251)), span


class TestTheUnlockGoesThroughTheVerifier:

    def test_the_honest_suite_unlocks_both_arms(self, suite, carried):
        from src.training import hypotheses
        from src.training.lora import LoraConfig_

        for name in ("H1", "H2"):
            cfg = hypotheses.require_unlocked(name, runs=suite, **carried)
            assert isinstance(cfg, LoraConfig_)

    def test_one_broken_role_locks_both_arms(self, mutable, carried):
        from src.training import hypotheses

        rewrite_evidence(
            mutable["gate_100_r2"], "gate_100",
            lambda b: b["evidence"].__setitem__("trainable_digest", "f" * 64))
        for name in ("H1", "H2"):
            with pytest.raises(hypotheses.HypothesisLocked):
                hypotheses.require_unlocked(name, runs=mutable, **carried)

    def test_the_carried_values_cannot_be_omitted(self):
        """No defaults. A trust check with a default is one that gets skipped."""
        import inspect

        from src.training import gate_suite, hypotheses

        for fn in (gate_suite.suite_problems, hypotheses.require_unlocked):
            sig = inspect.signature(fn)
            for name in ("expected_pack_digest", "expected_dependency_digest",
                         "allocator_config", "determinism"):
                param = sig.parameters[name]
                assert param.default is inspect.Parameter.empty, (fn, name)
                assert param.kind is inspect.Parameter.KEYWORD_ONLY

    def test_the_runs_argument_cannot_be_omitted(self):
        import inspect

        from src.training import hypotheses

        sig = inspect.signature(hypotheses.require_unlocked)
        assert sig.parameters["runs"].default is inspect.Parameter.empty

    def test_the_node_script_still_has_no_way_to_run_an_arm(self):
        """The unlock exists; the execution path deliberately does not.

        The script may *print* the frozen declaration -- that is what makes it
        reviewable before it is runnable. What it must not have is any way to
        turn one into a run: no unlock call, no arm lookup, no arm name.
        """
        source = (Path(__file__).resolve().parents[1]
                  / "scripts" / "19_gpu_gate.py").read_text(encoding="utf-8")
        for forbidden in ("require_unlocked", "config_for", "hypotheses.H1",
                          "hypotheses.H2", "hypotheses.FROZEN"):
            assert forbidden not in source, forbidden
        for arm in ("H1", "H2"):
            assert f'"{arm}"' not in source, arm
            assert f"'{arm}'" not in source, arm
        assert "hypotheses_runnable_here" in source

    def test_the_node_script_says_the_arms_are_not_runnable_there(self):
        source = (Path(__file__).resolve().parents[1]
                  / "scripts" / "19_gpu_gate.py").read_text(encoding="utf-8")
        assert '"hypotheses_runnable_here": False' in source

    def test_the_verifier_imports_no_torch(self):
        import inspect

        from src.training import gate_suite

        source = inspect.getsource(gate_suite)
        assert "import torch" not in source
        assert "from torch" not in source


# ---------------------------------------------------------------------------
# Three places the verifier answered "nothing wrong" when it had not looked.
# ---------------------------------------------------------------------------

def rewrite_provenance(run_dir, gate, mutate):
    """Change the provenance in the plan *and* the evidence, together.

    Separately would trip the "the evidence's provenance is not the one its
    own plan froze" check, and every test below would pass for the wrong
    reason -- the thing being tested would never be reached.
    """
    rewrite_plan(run_dir, lambda b: mutate(b["provenance"]))
    rewrite_evidence(run_dir, gate, lambda b: mutate(b["evidence"]["provenance"]))


def all_six(runs, mutate):
    from src.training import gate_suite

    for role, run_dir in runs.items():
        rewrite_provenance(run_dir, gate_suite.ROLE_GATE[role], mutate)


def accum() -> int:
    from src.training.lora import LoraConfig_

    return LoraConfig_().grad_accum


class TestTheDeclaredRowCountIsCheckedNotSkipped:
    """``measurement_intervals.max_rows`` is exempt from one check, not from all.

    It is the only provenance field allowed to differ across the six, because
    it is the gate's own row count. Exempting it from the identity comparison
    and then only pinning it *if it happens to be there* means a run whose
    provenance says nothing is a run that agrees with everything.
    """

    def test_provenance_without_it_at_all_is_refused(self, mutable, carried):
        all_six(mutable, lambda p: p.pop("measurement_intervals", None))
        found = problems(mutable, carried)
        assert found, "six runs that declare no row count unlocked the arms"
        assert any("max_rows" in p for p in found), found

    def test_it_is_required_of_every_role(self, mutable, carried):
        from src.training import gate_suite

        for role in gate_suite.ROLES:
            fresh = dict(mutable)
            rewrite_provenance(fresh[role], gate_suite.ROLE_GATE[role],
                               lambda p: p.pop("measurement_intervals", None))
            found = problems(fresh, carried)
            assert any(role in p and "max_rows" in p for p in found), (role, found)
            # put it back for the next role
            rewrite_provenance(
                fresh[role], gate_suite.ROLE_GATE[role],
                lambda p: p.__setitem__(
                    "measurement_intervals",
                    {"window": 20, "max_rows": gates.GATES[
                        gate_suite.ROLE_GATE[role]].rows}))

    @pytest.mark.parametrize("value", [None, 8, "intervals", True, [8], ()])
    def test_measurement_intervals_that_is_not_a_mapping_is_refused(
            self, mutable, carried, value):
        all_six(mutable,
                lambda p: p.__setitem__("measurement_intervals", value))
        found = problems(mutable, carried)
        assert any("measurement_intervals" in p or "max_rows" in p
                   for p in found), found

    @pytest.mark.parametrize("value", [None, True, False, "100", 100.0, [100]])
    def test_a_max_rows_that_is_not_a_strict_int_is_refused(
            self, mutable, carried, value):
        rewrite_provenance(
            mutable["gate_100_r1"], "gate_100",
            lambda p: p["measurement_intervals"].__setitem__("max_rows", value))
        found = problems(mutable, carried)
        assert any("max_rows" in p for p in found), (value, found)

    def test_a_max_rows_that_is_not_this_gate_s_row_count_is_refused(
            self, mutable, carried):
        rewrite_provenance(
            mutable["gate_500_resumed"], "gate_500",
            lambda p: p["measurement_intervals"].__setitem__("max_rows", 100))
        found = problems(mutable, carried)
        assert any("max_rows" in p for p in found), found

    def test_the_evidence_s_copy_is_checked_too(self, mutable, carried):
        """Not only the plan's. Both are files, and both are read."""
        rewrite_evidence(
            mutable["gate_8"], "gate_8",
            lambda b: b["evidence"]["provenance"]["measurement_intervals"]
            .__setitem__("max_rows", 500))
        assert problems(mutable, carried)

    def test_the_honest_suite_still_verifies(self, suite, carried):
        assert problems(suite, carried) == []


class TestTheOptimizerStepCountIsAValueNotJustAnAgreement:
    """Two runs agreeing on 61 agree about something that did not happen.

    500 rows at ``grad_accum`` 8 is 62 optimizer steps. Checking only that the
    two gate 500 runs report the same number accepts any number they both
    report -- including one they both got wrong the same way, which is exactly
    what a shared bug produces.
    """

    def test_the_expected_count_is_derived_not_typed(self):
        from src.training import gate_suite

        assert gate_suite.expected_optimizer_steps("gate_500") == 500 // accum()
        assert gate_suite.expected_optimizer_steps("gate_500") == 62

    @pytest.mark.parametrize("value", [61, 63, 0])
    def test_both_runs_reporting_the_same_wrong_count_is_refused(
            self, mutable, carried, value):
        for role in ("gate_500_resumed", "gate_500_uninterrupted_control"):
            rewrite_evidence(
                mutable[role], "gate_500",
                lambda b: b["evidence"].__setitem__("optimizer_steps", value))
        found = problems(mutable, carried)
        assert any("optimizer_steps" in p for p in found), (value, found)

    @pytest.mark.parametrize("value", [None, "62", 62.0, True])
    def test_a_count_that_is_not_a_strict_int_is_refused(
            self, mutable, carried, value):
        for role in ("gate_500_resumed", "gate_500_uninterrupted_control"):
            rewrite_evidence(
                mutable[role], "gate_500",
                lambda b: b["evidence"].__setitem__("optimizer_steps", value))
        found = problems(mutable, carried)
        assert any("optimizer_steps" in p for p in found), (value, found)

    def test_each_run_is_checked_on_its_own(self, mutable, carried):
        for role in ("gate_500_resumed", "gate_500_uninterrupted_control"):
            fresh = dict(mutable)
            rewrite_evidence(
                fresh[role], "gate_500",
                lambda b: b["evidence"].__setitem__("optimizer_steps", 61))
            found = problems(fresh, carried)
            assert any(role in p for p in found), (role, found)
            rewrite_evidence(
                fresh[role], "gate_500",
                lambda b: b["evidence"].__setitem__("optimizer_steps", 62))


class TestTheResumedRunHasExactlyTwoAttempts:
    """The re-run interval only means something if both attempts covered it."""

    def read_ledger_lines(self, run_dir):
        return (Path(run_dir) / gates.LEDGER_NAME).read_text(
            encoding="utf-8").splitlines()

    def write_ledger_lines(self, run_dir, lines):
        (Path(run_dir) / gates.LEDGER_NAME).write_text(
            "\n".join(lines) + "\n", encoding="utf-8")

    def test_a_third_attempt_is_refused(self, mutable, carried):
        run_dir = mutable["gate_500_resumed"]
        lines = self.read_ledger_lines(run_dir)
        extra = json.loads(lines[-1])
        extra["attempt"] = 3
        lines.append(json.dumps(extra, sort_keys=True, separators=(",", ":")))
        self.write_ledger_lines(run_dir, lines)
        found = problems(mutable, carried)
        assert any("attempt" in p for p in found), found

    def test_the_evidence_must_say_two_attempts(self, mutable, carried):
        rewrite_evidence(mutable["gate_500_resumed"], "gate_500",
                         lambda b: b["evidence"].__setitem__("attempts", 3))
        found = problems(mutable, carried)
        assert any("attempts" in p for p in found), found

    @pytest.mark.parametrize("value", [None, "2", 2.0, True, 1])
    def test_an_attempts_field_that_is_not_two_is_refused(
            self, mutable, carried, value):
        rewrite_evidence(mutable["gate_500_resumed"], "gate_500",
                         lambda b: b["evidence"].__setitem__("attempts", value))
        found = problems(mutable, carried)
        assert found, value

    def test_a_span_row_missing_from_the_first_attempt_is_refused(
            self, mutable, carried):
        run_dir = mutable["gate_500_resumed"]
        lines = [line for line in self.read_ledger_lines(run_dir)
                 if not (json.loads(line)["attempt"] == 1
                         and json.loads(line)["position"] == 210)]
        self.write_ledger_lines(run_dir, lines)
        found = problems(mutable, carried)
        assert found, "a half-covered re-run interval unlocked the arms"

    def test_a_span_row_missing_from_the_second_attempt_is_refused(
            self, mutable, carried):
        run_dir = mutable["gate_500_resumed"]
        lines = [line for line in self.read_ledger_lines(run_dir)
                 if not (json.loads(line)["attempt"] == 2
                         and json.loads(line)["position"] == 210)]
        self.write_ledger_lines(run_dir, lines)
        found = problems(mutable, carried)
        assert found, "a half-covered re-run interval unlocked the arms"

    def test_only_a_partial_overlap_is_refused(self, mutable, carried):
        """Attempt 2 starting later than it claims to have resumed from."""
        run_dir = mutable["gate_500_resumed"]
        lines = [line for line in self.read_ledger_lines(run_dir)
                 if not (json.loads(line)["attempt"] == 2
                         and 193 <= json.loads(line)["position"] <= 220)]
        self.write_ledger_lines(run_dir, lines)
        found = problems(mutable, carried)
        assert found

    def test_the_losses_are_compared_over_the_whole_declared_span(
            self, mutable, carried):
        run_dir = mutable["gate_500_resumed"]
        lines = self.read_ledger_lines(run_dir)
        changed = 0
        for i, line in enumerate(lines):
            entry = json.loads(line)
            if entry["attempt"] == 2 and entry["position"] == 249:
                entry["loss"] = entry["loss"] + 1e-9
                lines[i] = json.dumps(entry, sort_keys=True,
                                      separators=(",", ":"))
                changed += 1
        assert changed == 1
        self.write_ledger_lines(run_dir, lines)
        found = problems(mutable, carried)
        assert any("re-executed" in p for p in found), found

    def test_a_span_that_disagrees_with_the_evidence_is_refused(
            self, mutable, carried):
        """The ledger's account of the re-run and the evidence's must match.

        Reached without breaking the ledger's own contiguity rules, so it is
        this check that answers rather than the one before it.
        """
        rewrite_evidence(mutable["gate_500_resumed"], "gate_500",
                         lambda b: b["evidence"].__setitem__("stopped_at", 300))
        found = problems(mutable, carried)
        assert any("measured more than once" in p for p in found), found

    @pytest.mark.parametrize("field", ["resumed_from", "stopped_at"])
    @pytest.mark.parametrize("value", [None, "192", 192.0, True])
    def test_a_non_integer_interval_bound_is_refused(
            self, mutable, carried, field, value):
        rewrite_evidence(mutable["gate_500_resumed"], "gate_500",
                         lambda b: b["evidence"].__setitem__(field, value))
        assert problems(mutable, carried), (field, value)
