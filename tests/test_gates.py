"""The three gates, and the ledger that makes a resumed run checkable.

The node has never trained anything. Before either frozen hypothesis may run,
three gates have to pass on it, and each answers a question the previous one
cannot:

* **8 rows** -- does the machinery exist at all? A real load, a real forward, a
  real backward, a real optimizer step, a real save, and a real *cold* load of
  what was saved. Eight rows is one optimizer boundary at ``grad_accum=8``, so
  it is the smallest run in which the optimizer is exercised even once.
* **100 rows** -- what does it cost, and does it stay sane? Speed, peak VRAM,
  loss and stability. Peak VRAM is the one thresholded reading, because the
  card's capacity is a physical bound rather than an opinion.
* **500 rows** -- does it survive being interrupted? Checkpoint part-way, stop
  on purpose, resume, and prove afterwards that no row was skipped and no row
  was trained on twice.

Most of this file is about the third. Interruption is where a training loop
tells its most convincing lie: the run finishes, the loss looks fine, and the
optimizer saw row 250 twice and row 251 never. So the ledger is append-only
and self-describing -- every entry records which attempt wrote it and which
checkpoint that attempt resumed from -- and the invariants are checked against
*that*, not against a count of lines.

Nothing here loads a model, allocates a tensor, opens a socket or reads the
dataset. Every run below is driven through the same code path the real one
uses, with the model, optimizer and device supplied by a deterministic fake.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training import gates

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# A verified pack to run against, built the way the node's is built.
# ---------------------------------------------------------------------------

@pytest.fixture()
def packdir(tmp_path):
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


@pytest.fixture()
def run_dir(tmp_path):
    return tmp_path / "runs" / "gate"


def digest_of(packdir) -> str:
    """The value the build machine printed, as the operator would carry it.

    Read from the manifest here only because in a test the two are the same
    directory. On the node they are not: the digest travels by a route the
    pack did not, which is the entire reason the check means anything.
    """
    from src.training import pack

    return pack.read_manifest(packdir)[0]["pack_digest"]


#: What a checkpoint records about the weights it saved beside it. Only the
#: shape matters where ``write_checkpoint`` is called directly: these tests are
#: about the record, and the file it names is written by ``_take_checkpoint``.
MODEL_STATE = {"name": "model_state.safetensors", "sha256": "b" * 64,
               "bytes": 4096}
RNG_STATE = {"name": "rng_state.pt", "sha256": "e" * 64, "bytes": 152}


def entry(attempt, position, index, *, resumed_from=0, loss=1.0, sample=None):
    return {"attempt": attempt, "position": position, "index": index,
            "resumed_from": resumed_from, "loss": loss,
            "sample_id": sample or f"s{index}", "tokens": 10,
            "supervised_tokens": 4, "seconds": 0.5}


class TestGateDefinitions:

    def test_the_three_gates_are_the_declared_sizes(self):
        assert gates.GATES["gate_8"].rows == 8
        assert gates.GATES["gate_100"].rows == 100
        assert gates.GATES["gate_500"].rows == 500

    def test_eight_rows_is_one_optimizer_boundary(self):
        """Fewer than ``grad_accum`` rows never steps the optimizer."""
        from src.training.lora import LoraConfig_

        assert gates.GATES["gate_8"].rows == LoraConfig_().grad_accum

    def test_each_gate_says_what_it_proves(self):
        for name, gate in gates.GATES.items():
            assert gate.proves, f"{name} proves nothing in particular"

    def test_the_eight_row_gate_names_the_six_stages(self):
        proves = " ".join(gates.GATES["gate_8"].proves).lower()
        for stage in ("load", "forward", "backward", "optimizer", "save",
                      "cold"):
            assert stage in proves, stage

    def test_the_five_hundred_row_gate_checkpoints_more_than_once(self):
        gate = gates.GATES["gate_500"]
        assert gate.rows // gate.checkpoint_every >= 2

    def test_checkpoints_land_on_optimizer_boundaries(self):
        """A checkpoint mid-accumulation saves gradients nothing will apply."""
        from src.training.lora import LoraConfig_

        accum = LoraConfig_().grad_accum
        for gate in gates.GATES.values():
            assert gate.checkpoint_every % accum == 0, gate.name


class TestLedgerInvariants:
    """No missing row, no duplicated row -- checked, not counted."""

    ORDER = list(range(100, 110))

    def test_a_clean_single_attempt_ledger_has_no_problems(self):
        entries = [entry(1, p, self.ORDER[p - 1]) for p in range(1, 6)]
        assert gates.ledger_problems(entries, order=self.ORDER) == []

    def test_a_gap_is_refused(self):
        entries = [entry(1, p, self.ORDER[p - 1]) for p in (1, 2, 4, 5)]
        problems = gates.ledger_problems(entries, order=self.ORDER)
        assert problems
        assert any("3" in p for p in problems)

    def test_a_repeat_inside_one_attempt_is_refused(self):
        entries = [entry(1, p, self.ORDER[p - 1]) for p in (1, 2, 3)]
        entries.append(entry(1, 3, self.ORDER[2]))
        problems = gates.ledger_problems(entries, order=self.ORDER)
        assert problems
        assert any("twice" in p or "repeat" in p for p in problems)

    def test_re_executing_a_row_the_checkpoint_already_covered_is_refused(self):
        """The realistic duplicate: resume too late and train a row twice."""
        first = [entry(1, p, self.ORDER[p - 1]) for p in range(1, 7)]
        second = [entry(2, p, self.ORDER[p - 1], resumed_from=4)
                  for p in range(3, 8)]
        problems = gates.ledger_problems(first + second, order=self.ORDER,
                                         checkpoint_positions=(4,))
        assert problems
        assert any("resumed_from" in p or "already" in p for p in problems)

    def test_re_executing_a_row_after_the_checkpoint_is_allowed(self):
        """Rows measured after the last checkpoint were never saved.

        Their optimizer effect is not in the restored state, so re-running
        them is the correct behaviour, not a duplicate -- and the ledger keeps
        the discarded work visible instead of quietly dropping it.
        """
        first = [entry(1, p, self.ORDER[p - 1]) for p in range(1, 7)]
        second = [entry(2, p, self.ORDER[p - 1], resumed_from=4)
                  for p in range(5, 9)]
        assert gates.ledger_problems(first + second, order=self.ORDER,
                                     checkpoint_positions=(4,)) == []

    def test_the_effective_ledger_has_each_position_exactly_once(self):
        first = [entry(1, p, self.ORDER[p - 1]) for p in range(1, 7)]
        second = [entry(2, p, self.ORDER[p - 1], resumed_from=4)
                  for p in range(5, 9)]
        eff = gates.effective_ledger(first + second)
        assert sorted(eff) == list(range(1, 9))
        assert eff[5]["attempt"] == 2, "the later attempt must win"
        assert eff[3]["attempt"] == 1

    def test_resuming_past_what_was_ever_done_is_refused(self):
        """Attempt 1 reached row 5; attempt 2 starts at 9. Rows 6-8 vanish."""
        first = [entry(1, p, self.ORDER[p - 1]) for p in range(1, 6)]
        second = [entry(2, p, self.ORDER[p - 1], resumed_from=8)
                  for p in range(9, 11)]
        problems = gates.ledger_problems(first + second, order=self.ORDER,
                                         checkpoint_positions=(8,))
        assert problems

    def test_a_changed_input_order_is_refused(self):
        entries = [entry(1, p, self.ORDER[p - 1]) for p in range(1, 5)]
        entries[2]["index"] = 999
        problems = gates.ledger_problems(entries, order=self.ORDER)
        assert problems
        assert any("order" in p for p in problems)

    def test_the_same_index_may_not_become_a_different_sample(self):
        first = [entry(1, p, self.ORDER[p - 1]) for p in range(1, 7)]
        second = [entry(2, p, self.ORDER[p - 1], resumed_from=4)
                  for p in range(5, 9)]
        second[0]["sample_id"] = "something-else"
        problems = gates.ledger_problems(first + second, order=self.ORDER,
                                         checkpoint_positions=(4,))
        assert problems
        assert any("sample" in p for p in problems)

    def test_a_resume_point_that_is_not_a_checkpoint_is_refused(self):
        first = [entry(1, p, self.ORDER[p - 1]) for p in range(1, 7)]
        second = [entry(2, p, self.ORDER[p - 1], resumed_from=5)
                  for p in range(6, 9)]
        problems = gates.ledger_problems(first + second, order=self.ORDER,
                                         checkpoint_positions=(4,))
        assert problems
        assert any("checkpoint" in p for p in problems)

    def test_a_non_contiguous_attempt_number_is_refused(self):
        first = [entry(1, p, self.ORDER[p - 1]) for p in range(1, 5)]
        third = [entry(3, p, self.ORDER[p - 1], resumed_from=4)
                 for p in range(5, 7)]
        assert gates.ledger_problems(first + third, order=self.ORDER,
                                     checkpoint_positions=(4,))

    def test_declared_rows_makes_an_unfinished_run_a_problem(self):
        entries = [entry(1, p, self.ORDER[p - 1]) for p in range(1, 6)]
        assert gates.ledger_problems(entries, order=self.ORDER) == []
        assert gates.ledger_problems(entries, order=self.ORDER,
                                     declared_rows=10)

    def test_an_empty_ledger_with_declared_rows_is_a_problem_not_a_pass(self):
        assert gates.ledger_problems([], order=self.ORDER, declared_rows=10)

    def test_a_malformed_entry_is_refused_rather_than_skipped(self):
        entries = [entry(1, 1, self.ORDER[0]), {"position": "two"}]
        assert gates.ledger_problems(entries, order=self.ORDER)

    def test_a_non_finite_loss_is_refused(self):
        entries = [entry(1, p, self.ORDER[p - 1]) for p in range(1, 4)]
        entries[1]["loss"] = float("nan")
        problems = gates.ledger_problems(entries, order=self.ORDER)
        assert any("finite" in p for p in problems), problems


class TestCheckpointsAreWriteOnce:

    def test_a_checkpoint_records_what_resume_needs(self, run_dir, packdir):
        from src.training import pack

        plan = gates.write_plan(
            run_dir, gate="gate_8", order=[1, 2, 3, 4, 5, 6, 7, 8],
            pack_digest=pack.read_manifest(packdir)[0]["pack_digest"],
            config={"rank": 16}, provenance={"device": "cuda"},
            expected_dependency_digest=dep_digest(),
            allocator_config=ALLOC, determinism=dict(DETERMINISM))
        body = gates.write_checkpoint(
            run_dir, position=8, attempt=1, plan=plan,
            optimizer_sha256="a" * 64, model_state=MODEL_STATE,
            rng_state=RNG_STATE,
            trainable_digest="c" * 64, ledger_entries=[],
            provenance={"device": "cuda"})
        for field in ("position", "attempt", "order_digest", "pack_digest",
                      "dependency_digest", "optimizer_sha256", "model_state",
                      "rng_state", "trainable_digest", "ledger_digest",
                      "provenance", "written_at"):
            assert field in body, field

    def test_a_second_write_to_the_same_position_refuses(self, run_dir, packdir):
        from src.training import pack

        plan = gates.write_plan(
            run_dir, gate="gate_8", order=list(range(8)),
            pack_digest=pack.read_manifest(packdir)[0]["pack_digest"],
            config={}, provenance={},
            expected_dependency_digest=dep_digest(),
            allocator_config=ALLOC, determinism=dict(DETERMINISM))
        gates.write_checkpoint(run_dir, position=8, attempt=1, plan=plan,
                               optimizer_sha256="a" * 64,
                               model_state=MODEL_STATE,
                               rng_state=RNG_STATE,
                               trainable_digest="c" * 64, ledger_entries=[],
                               provenance={})
        with pytest.raises(SystemExit):
            gates.write_checkpoint(run_dir, position=8, attempt=1, plan=plan,
                                   optimizer_sha256="b" * 64,
                                   model_state=MODEL_STATE,
                                   rng_state=RNG_STATE,
                                   trainable_digest="d" * 64,
                                   ledger_entries=[], provenance={})

    def test_the_plan_cannot_be_rewritten(self, run_dir, packdir):
        gates.write_plan(run_dir, gate="gate_8", order=[1], pack_digest="x",
                         config={}, provenance={},
            expected_dependency_digest=dep_digest(),
            allocator_config=ALLOC, determinism=dict(DETERMINISM))
        with pytest.raises(SystemExit):
            gates.write_plan(run_dir, gate="gate_8", order=[2],
                             pack_digest="x", config={}, provenance={},
            expected_dependency_digest=dep_digest(),
            allocator_config=ALLOC, determinism=dict(DETERMINISM))

    def test_latest_checkpoint_is_the_highest_position_not_the_newest_file(
            self, run_dir, packdir):
        plan = gates.write_plan(run_dir, gate="gate_500",
                                order=list(range(500)), pack_digest="x",
                                config={}, provenance={},
                                expected_dependency_digest=dep_digest(),
            allocator_config=ALLOC, determinism=dict(DETERMINISM))
        for pos in (64, 192, 128):
            gates.write_checkpoint(run_dir, position=pos, attempt=1, plan=plan,
                                   optimizer_sha256="a" * 64,
                                   model_state=MODEL_STATE,
                                   rng_state=RNG_STATE,
                                   trainable_digest="c" * 64,
                                   ledger_entries=[], provenance={})
        assert gates.latest_checkpoint(run_dir)["position"] == 192


class TestResumeVerifiesBeforeItTrusts:

    @pytest.fixture()
    def stopped(self, run_dir, packdir):
        """A 500-row gate stopped on purpose part-way through."""
        deps = gates.FakeGateDeps(rows=500)
        with pytest.raises(gates.DeliberateStop):
            gates.run_gate("gate_500", deps=deps, run_dir=run_dir,
                           pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir), stop_after=200,
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        return run_dir

    def test_the_stop_left_a_checkpoint_and_a_ledger(self, stopped):
        assert gates.latest_checkpoint(stopped)
        assert gates.read_ledger(stopped)

    def test_resume_reports_where_it_will_start(self, stopped, packdir):
        point = gates.resume_point(stopped, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        assert point["resume_from"] == gates.latest_checkpoint(stopped)["position"]
        assert point["next_position"] == point["resume_from"] + 1
        assert point["attempt"] == 2

    def test_a_drifted_packed_file_refuses_the_resume(self, stopped, packdir):
        (packdir / "src" / "training" / "gates.py").write_text("GATE = 9\n")
        problems = gates.resume_problems(stopped, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        assert problems
        assert any("gates.py" in p for p in problems)
        with pytest.raises(gates.GateRefused):
            gates.resume_point(stopped, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))

    def test_the_manifest_is_checked_file_by_file_not_just_by_digest(
            self, stopped, packdir):
        """Any drift. Not "the interesting files", not a spot check."""
        (packdir / "requirements.txt").write_text("torch\nnumpy\n")
        assert gates.resume_problems(stopped, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))

    def test_a_missing_optimizer_state_refuses(self, stopped, packdir):
        ckpt = gates.latest_checkpoint(stopped)
        (Path(ckpt["dir"]) / gates.OPTIMIZER_NAME).unlink()
        problems = gates.resume_problems(stopped, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        assert problems
        assert any("optimizer" in p for p in problems)

    def test_a_tampered_optimizer_state_refuses(self, stopped, packdir):
        ckpt = gates.latest_checkpoint(stopped)
        (Path(ckpt["dir"]) / gates.OPTIMIZER_NAME).write_bytes(b"not the state")
        problems = gates.resume_problems(stopped, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        assert problems
        assert any("optimizer" in p for p in problems)

    def test_a_changed_input_order_refuses(self, stopped, packdir):
        plan_path = Path(stopped) / gates.PLAN_NAME
        body = json.loads(plan_path.read_text())
        body["order"] = list(reversed(body["order"]))
        plan_path.unlink()
        plan_path.write_text(json.dumps(body, indent=2))
        problems = gates.resume_problems(stopped, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        assert problems
        assert any("order" in p for p in problems)

    def test_a_pack_that_is_not_the_one_the_plan_was_made_against_refuses(
            self, stopped, tmp_path):
        from src.training import pack

        other_src = tmp_path / "other_src"
        (other_src / "src").mkdir(parents=True)
        (other_src / "requirements.txt").write_text("torch\n")
        (other_src / "src" / "__init__.py").write_text("")
        other = tmp_path / "other_pack"
        pack.build(other, root=other_src)
        # ``other``'s own digest, so the carried-digest check is satisfied and
        # what fails is the question this test is about: this is a different
        # pack from the one the plan was made against.
        problems = gates.resume_problems(
            stopped, pack_dir=other, expected_pack_digest=digest_of(other),
            expected_dependency_digest=dep_digest(),
            dependency_checker=dep_checker(),
            allocator_config=ALLOC,
            determinism=dict(DETERMINISM))
        assert any("pack_digest" in p for p in problems), problems

    def test_no_plan_means_no_resume(self, run_dir, packdir):
        run_dir.mkdir(parents=True)
        assert gates.resume_problems(run_dir, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))

    def test_no_checkpoint_means_no_resume(self, run_dir, packdir):
        from src.training import pack

        gates.write_plan(
            run_dir, gate="gate_500", order=list(range(500)),
            pack_digest=pack.read_manifest(packdir)[0]["pack_digest"],
            config={}, provenance={},
            expected_dependency_digest=dep_digest(),
            allocator_config=ALLOC, determinism=dict(DETERMINISM))
        problems = gates.resume_problems(run_dir, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        assert problems
        assert any("checkpoint" in p for p in problems)


class TestTheFiveHundredRowGateSurvivesAStop:

    def test_stop_then_resume_covers_every_row_exactly_once(
            self, run_dir, packdir):
        deps = gates.FakeGateDeps(rows=500)
        with pytest.raises(gates.DeliberateStop):
            gates.run_gate("gate_500", deps=deps, run_dir=run_dir,
                           pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir), stop_after=200,
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))

        evidence = gates.run_gate("gate_500", deps=gates.FakeGateDeps(rows=500),
                                  run_dir=run_dir, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM),
                                  resume=True)

        eff = gates.effective_ledger(gates.read_ledger(run_dir))
        assert sorted(eff) == list(range(1, 501)), "a row was skipped or doubled"
        assert evidence["rows_completed"] == 500
        assert gates.verdict("gate_500", evidence) == "passed"

    def test_the_resumed_run_reloaded_the_optimizer_state(self, run_dir, packdir):
        deps = gates.FakeGateDeps(rows=500)
        with pytest.raises(gates.DeliberateStop):
            gates.run_gate("gate_500", deps=deps, run_dir=run_dir,
                           pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir), stop_after=200,
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        resumed = gates.FakeGateDeps(rows=500)
        gates.run_gate("gate_500", deps=resumed, run_dir=run_dir,
                       pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir), resume=True,
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        assert resumed.loaded_optimizer_from is not None, \
            "the resumed run started from a fresh optimizer"

    def test_the_rows_after_the_last_checkpoint_are_re_executed_not_skipped(
            self, run_dir, packdir):
        deps = gates.FakeGateDeps(rows=500)
        with pytest.raises(gates.DeliberateStop):
            gates.run_gate("gate_500", deps=deps, run_dir=run_dir,
                           pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir), stop_after=200,
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        ckpt = gates.latest_checkpoint(run_dir)["position"]
        assert ckpt < 200, "the fixture no longer exercises discarded rows"

        gates.run_gate("gate_500", deps=gates.FakeGateDeps(rows=500),
                       run_dir=run_dir, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir), resume=True,
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        raw = gates.read_ledger(run_dir)
        superseded = [e for e in raw
                      if e["attempt"] == 1 and e["position"] > ckpt]
        assert superseded, "nothing was discarded, so nothing was re-executed"
        assert all(e["position"] > ckpt for e in superseded)

    def test_the_raw_ledger_keeps_the_discarded_work_visible(
            self, run_dir, packdir):
        deps = gates.FakeGateDeps(rows=500)
        with pytest.raises(gates.DeliberateStop):
            gates.run_gate("gate_500", deps=deps, run_dir=run_dir,
                           pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir), stop_after=200,
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        before = len(gates.read_ledger(run_dir))
        gates.run_gate("gate_500", deps=gates.FakeGateDeps(rows=500),
                       run_dir=run_dir, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir), resume=True,
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        after = gates.read_ledger(run_dir)
        assert len(after) > before
        assert after[:before] == gates.read_ledger(run_dir)[:before], \
            "the ledger was rewritten rather than appended to"

    def test_resuming_a_run_that_already_finished_is_refused(
            self, run_dir, packdir):
        deps = gates.FakeGateDeps(rows=500)
        with pytest.raises(gates.DeliberateStop):
            gates.run_gate("gate_500", deps=deps, run_dir=run_dir,
                           pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir), stop_after=200,
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        gates.run_gate("gate_500", deps=gates.FakeGateDeps(rows=500),
                       run_dir=run_dir, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir), resume=True,
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        with pytest.raises(gates.GateRefused):
            gates.run_gate("gate_500", deps=gates.FakeGateDeps(rows=500),
                           run_dir=run_dir, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir), resume=True,
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))

    def test_starting_fresh_over_an_existing_run_is_refused(
            self, run_dir, packdir):
        deps = gates.FakeGateDeps(rows=500)
        with pytest.raises(gates.DeliberateStop):
            gates.run_gate("gate_500", deps=deps, run_dir=run_dir,
                           pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir), stop_after=200,
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        with pytest.raises(gates.GateRefused):
            gates.run_gate("gate_500", deps=gates.FakeGateDeps(rows=500),
                           run_dir=run_dir, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))


class TestTheEightRowGate:

    def test_a_clean_eight_row_run_passes(self, run_dir, packdir):
        evidence = gates.run_gate("gate_8", deps=gates.FakeGateDeps(rows=8),
                                  run_dir=run_dir, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        assert gates.gate_problems("gate_8", evidence) == []
        assert gates.verdict("gate_8", evidence) == "passed"

    def test_it_records_all_six_stages(self, run_dir, packdir):
        evidence = gates.run_gate("gate_8", deps=gates.FakeGateDeps(rows=8),
                                  run_dir=run_dir, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        assert evidence["model_load_seconds"] > 0
        assert evidence["forward_rows"] == 8
        assert evidence["backward_rows"] == 8
        assert evidence["optimizer_steps"] == 1
        assert evidence["saved"]["sha256"]
        assert evidence["cold_load"]["matches_saved"] is True

    @pytest.mark.parametrize("field", [
        "model_load_seconds", "forward_rows", "backward_rows",
        "optimizer_steps", "saved", "cold_load",
    ])
    def test_missing_evidence_fails_rather_than_passes(self, field):
        evidence = _good_eight()
        del evidence[field]
        assert gates.gate_problems("gate_8", evidence)
        assert gates.verdict("gate_8", evidence) == "failed"

    def test_a_run_that_never_stepped_the_optimizer_fails(self):
        evidence = _good_eight()
        evidence["optimizer_steps"] = 0
        problems = gates.gate_problems("gate_8", evidence)
        assert any("optimizer" in p for p in problems), problems

    def test_a_cold_load_that_did_not_match_fails(self):
        evidence = _good_eight()
        evidence["cold_load"] = {"loaded": True, "matches_saved": False,
                                 "sha256": "a" * 64}
        problems = gates.gate_problems("gate_8", evidence)
        assert any("cold" in p for p in problems), problems

    def test_a_save_with_no_digest_fails(self):
        evidence = _good_eight()
        evidence["saved"] = {"path": "adapter"}
        assert gates.gate_problems("gate_8", evidence)

    def test_an_empty_evidence_object_is_failed_not_indeterminate(self):
        assert gates.verdict("gate_8", {}) == "failed"


class TestTheHundredRowGate:

    def test_a_clean_hundred_row_run_passes(self, run_dir, packdir):
        evidence = gates.run_gate("gate_100", deps=gates.FakeGateDeps(rows=100),
                                  run_dir=run_dir, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        assert gates.gate_problems("gate_100", evidence) == []

    def test_it_measures_all_four_things(self, run_dir, packdir):
        evidence = gates.run_gate("gate_100", deps=gates.FakeGateDeps(rows=100),
                                  run_dir=run_dir, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        assert evidence["seconds_per_row"] is not None
        assert evidence["peak_vram_gb"] is not None
        assert evidence["loss_first_window"] is not None
        assert evidence["loss_last_window"] is not None
        assert evidence["windows"]

    def test_an_unreadable_peak_vram_fails(self):
        evidence = _good_hundred()
        evidence["peak_vram_gb"] = None
        problems = gates.gate_problems("gate_100", evidence)
        assert any("vram" in p.lower() for p in problems), problems

    def test_peak_vram_over_the_card_fails(self):
        evidence = _good_hundred()
        evidence["peak_vram_gb"] = 15.8
        assert gates.gate_problems("gate_100", evidence)

    def test_a_non_finite_loss_fails(self):
        evidence = _good_hundred()
        evidence["loss_last_window"] = float("nan")
        problems = gates.gate_problems("gate_100", evidence)
        assert any("finite" in p for p in problems), problems

    def test_a_diverged_loss_fails(self):
        evidence = _good_hundred()
        evidence["loss_last_window"] = \
            evidence["loss_first_window"] * (gates.DIVERGENCE_FACTOR + 1)
        problems = gates.gate_problems("gate_100", evidence)
        assert any("diverg" in p for p in problems), problems

    def test_a_merely_slower_loss_does_not_fail(self):
        """A gate that also judged learning rate would be judging H1 vs H2."""
        evidence = _good_hundred()
        evidence["loss_last_window"] = evidence["loss_first_window"] * 1.2
        assert gates.gate_problems("gate_100", evidence) == []

    def test_speed_is_recorded_without_a_threshold_and_says_why(self):
        assert gates.SPEED_THRESHOLD_SECONDS_PER_ROW is None
        assert gates.SPEED_THRESHOLD_REASON
        assert "calibrat" in gates.SPEED_THRESHOLD_REASON.lower()

    def test_an_unreadable_speed_still_fails(self):
        """No threshold is not the same as no reading."""
        evidence = _good_hundred()
        evidence["seconds_per_row"] = None
        assert gates.gate_problems("gate_100", evidence)


class TestVerdictsFailClosed:

    @pytest.mark.parametrize("name", ["gate_8", "gate_100", "gate_500"])
    def test_no_evidence_is_failed(self, name):
        assert gates.verdict(name, {}) == "failed"

    @pytest.mark.parametrize("name", ["gate_8", "gate_100", "gate_500"])
    def test_none_evidence_is_failed(self, name):
        assert gates.verdict(name, None) == "failed"

    def test_an_unknown_gate_is_refused_not_defaulted(self):
        with pytest.raises(KeyError):
            gates.gate_problems("gate_42", {})

    def test_the_wrong_row_count_fails(self):
        evidence = _good_eight()
        evidence["rows_completed"] = 7
        assert gates.gate_problems("gate_8", evidence)

    def test_ledger_problems_carried_in_evidence_fail_the_gate(self):
        evidence = _good_five_hundred()
        evidence["ledger_problems"] = ["row 3 appears twice"]
        assert gates.gate_problems("gate_500", evidence)

    def test_a_gate_run_against_an_unverified_pack_refuses(
            self, run_dir, packdir):
        (packdir / "src" / "training" / "gates.py").write_text("GATE = 9\n")
        with pytest.raises(gates.GateRefused):
            gates.run_gate("gate_8", deps=gates.FakeGateDeps(rows=8),
                           run_dir=run_dir, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))


class TestEvidenceIsSafeToRead:

    def test_no_evidence_carries_an_identifier(self, run_dir, packdir):
        from src.training.longrun import leaked_identifiers

        evidence = gates.run_gate("gate_8", deps=gates.FakeGateDeps(rows=8),
                                  run_dir=run_dir, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        assert leaked_identifiers(json.dumps(evidence, default=str)) == []

    def test_no_checkpoint_carries_an_identifier(self, run_dir, packdir):
        from src.training.longrun import leaked_identifiers

        deps = gates.FakeGateDeps(rows=500)
        with pytest.raises(gates.DeliberateStop):
            gates.run_gate("gate_500", deps=deps, run_dir=run_dir,
                           pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir), stop_after=200,
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        ckpt = gates.latest_checkpoint(run_dir)
        body = json.loads(
            (Path(ckpt["dir"]) / gates.CHECKPOINT_STATE).read_text())
        assert leaked_identifiers(json.dumps(body, default=str)) == []


class TestProvenanceTravelsWithTheCheckpoint:

    def test_the_checkpoint_carries_the_provenance_the_replay_needs(
            self, run_dir, packdir):
        deps = gates.FakeGateDeps(rows=500)
        with pytest.raises(gates.DeliberateStop):
            gates.run_gate("gate_500", deps=deps, run_dir=run_dir,
                           pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir), stop_after=200,
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        ckpt = gates.latest_checkpoint(run_dir)
        for field in ("device", "dtype", "lora_config", "optimizer"):
            assert field in ckpt["provenance"], field

    def test_provenance_that_changed_across_the_stop_refuses(
            self, run_dir, packdir):
        deps = gates.FakeGateDeps(rows=500)
        with pytest.raises(gates.DeliberateStop):
            gates.run_gate("gate_500", deps=deps, run_dir=run_dir,
                           pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir), stop_after=200,
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        changed = gates.FakeGateDeps(rows=500)
        changed.provenance_override = {"dtype": "float16"}
        with pytest.raises(gates.GateRefused) as exc:
            gates.run_gate("gate_500", deps=changed, run_dir=run_dir,
                           pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir), resume=True,
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        assert "dtype" in str(exc.value)

    def test_the_ledger_digest_pins_what_had_been_measured(
            self, run_dir, packdir):
        deps = gates.FakeGateDeps(rows=500)
        with pytest.raises(gates.DeliberateStop):
            gates.run_gate("gate_500", deps=deps, run_dir=run_dir,
                           pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir), stop_after=200,
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        ckpt = gates.latest_checkpoint(run_dir)
        assert len(ckpt["ledger_digest"]) == 64
        ledger = Path(run_dir) / gates.LEDGER_NAME
        ledger.write_text(ledger.read_text() + json.dumps(
            entry(1, 999, 999)) + "\n")
        problems = gates.resume_problems(run_dir, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        assert problems


# ---------------------------------------------------------------------------
# Evidence fixtures, so a verdict test does not have to run a gate.
# ---------------------------------------------------------------------------

def _good_eight() -> dict:
    return {
        "gate": "gate_8", "rows_completed": 8, "model_load_seconds": 12.5,
        "forward_rows": 8, "backward_rows": 8, "optimizer_steps": 1,
        "saved": {"path": "adapter", "sha256": "a" * 64, "bytes": 1024},
        "cold_load": {"loaded": True, "matches_saved": True,
                      "sha256": "a" * 64},
        "ledger_problems": [], "losses_finite": True,
    }


def _good_hundred() -> dict:
    return {
        "gate": "gate_100", "rows_completed": 100, "seconds_per_row": 0.42,
        "peak_vram_gb": 9.1, "loss_first_window": 1.8, "loss_last_window": 1.5,
        "windows": [{"window": 0, "seconds_per_row": 0.42, "loss": 1.8}],
        "ledger_problems": [], "losses_finite": True,
    }


def _good_five_hundred() -> dict:
    return {
        "gate": "gate_500", "rows_completed": 500, "checkpoints": [64, 128],
        "stopped_at": 200, "resumed_from": 192, "attempts": 2,
        "model_state_restored": True, "rng_state_restored": True,
        "optimizer_state_restored": True, "ledger_problems": [],
        "losses_finite": True, "duplicate_positions": [], "missing_positions": [],
    }


class TestTheEvidenceFixturesAreHonest:
    """If these stop passing, every verdict test above proves nothing."""

    def test_the_good_eight_passes(self):
        assert gates.gate_problems("gate_8", _good_eight()) == []

    def test_the_good_hundred_passes(self):
        assert gates.gate_problems("gate_100", _good_hundred()) == []

    def test_the_good_five_hundred_passes(self):
        assert gates.gate_problems("gate_500", _good_five_hundred()) == []


class _StubModel:
    def save_pretrained(self, dest):
        Path(dest).mkdir(parents=True, exist_ok=True)


class _StubOptimizer:
    def __init__(self):
        self.restored = None

    def state_dict(self):
        return {"step": 1}

    def load_state_dict(self, state):
        self.restored = state


class TestTheProductionLoaderIsAWrapperNotASecondLoader:
    """The real one. Its model half is report 16's, already tested there.

    What is new is the wrapper: the save, the cold load, the optimizer state
    and the memory reading. Those are what these tests drive. Nothing here
    imports torch or builds a model -- the base loader and the device are
    injected, which is the only way the wrapper can be tested at all on a Mac.
    """

    class StubTorch:
        class cuda:
            @staticmethod
            def max_memory_allocated():
                return 9 * 1024 ** 3

            @staticmethod
            def max_memory_reserved():
                return 10 * 1024 ** 3

            @staticmethod
            def reset_peak_memory_stats():
                return None

    class StubBase:
        def __init__(self, *, holder=True):
            self.holder = holder

        def load(self, *, rows):
            out = {
                "order": list(range(rows)),
                "step": lambda index, position: {
                    "loss": 1.0, "tokens": 1, "supervised_tokens": 1,
                    "sample_id": f"s{index}"},
                "provenance": {"device": "cuda", "dtype": "bfloat16",
                               "lora_config": {}, "optimizer": {}},
                "sample_ids": [f"s{i}" for i in range(rows)],
                "model_load_seconds": 3.0,
                "teardown": lambda: 0.0,
                "clear": lambda: 0.0,
                "probe": dict,
            }
            if self.holder:
                out["holder"] = {"model": _StubModel(),
                                 "optimizer": _StubOptimizer()}
            return out

    def deps(self, **kw):
        return gates.ProductionGateDeps(
            device="cuda", base=kw.pop("base", self.StubBase()),
            torch_mod=kw.pop("torch_mod", self.StubTorch), **kw)

    @pytest.mark.parametrize("device", ["cpu", "mps", "cuda:1", ""])
    def test_any_device_but_cuda_is_refused_at_construction(self, device):
        with pytest.raises(gates.GateRefused):
            gates.ProductionGateDeps(device=device)

    def test_it_supplies_every_key_the_runner_requires(self):
        loaded = self.deps().load(rows=8)
        for key in gates._LOADED_REQUIRED:
            assert key in loaded, key
        for key in ("save_adapter", "save_optimizer", "load_optimizer",
                    "cold_load", "peak_memory"):
            assert callable(loaded[key]), key

    def test_it_keeps_the_base_loader_s_provenance_rather_than_writing_its_own(self):
        loaded = self.deps().load(rows=8)
        assert loaded["provenance"]["device"] == "cuda"
        assert loaded["model_load_seconds"] == 3.0

    def test_a_base_loader_that_hides_the_model_is_refused(self):
        with pytest.raises(gates.GateRefused) as exc:
            self.deps(base=self.StubBase(holder=False)).load(rows=8)
        assert "model" in str(exc.value)

    def test_peak_memory_reads_the_reserved_high_water_mark(self):
        loaded = self.deps().load(rows=8)
        assert loaded["peak_memory"]()["peak_vram_gb"] == pytest.approx(10.0)

    def test_an_unreadable_peak_memory_is_none_not_zero(self):
        class Broken:
            class cuda:
                @staticmethod
                def max_memory_reserved():
                    raise RuntimeError("no driver")

        loaded = self.deps(torch_mod=Broken).load(rows=8)
        assert loaded["peak_memory"]()["peak_vram_gb"] is None

    def test_the_optimizer_state_round_trips_through_the_injected_torch(
            self, tmp_path):
        saved = {}

        class Recording:
            class cuda:
                @staticmethod
                def max_memory_reserved():
                    return 0

            @staticmethod
            def save(obj, path):
                saved["path"] = str(path)
                Path(path).write_text("state")

            @staticmethod
            def load(path, map_location=None, weights_only=True):
                saved["loaded"] = str(path)
                return {"state": {}}

        base = self.StubBase()
        loaded = self.deps(base=base, torch_mod=Recording).load(rows=8)
        target = tmp_path / "optimizer.pt"
        loaded["save_optimizer"](target)
        assert saved["path"] == str(target)
        loaded["load_optimizer"](target)
        assert saved["loaded"] == str(target)

    def test_the_default_device_is_cuda(self):
        import inspect

        sig = inspect.signature(gates.ProductionGateDeps.__init__)
        assert sig.parameters["device"].default == "cuda"


class TestTheRunnerChecksItsFoundationsFirst:
    """Two things the runner assumed and did not check."""

    def test_teardown_is_required_by_the_contract_not_just_called(
            self, run_dir, packdir):
        """It is called after the last row, which is the worst time to find out.

        A loader without ``teardown`` used to run the whole gate, save the
        adapter, and then raise KeyError -- with every row already measured
        and the model still resident.
        """
        assert "teardown" in gates._LOADED_REQUIRED

        class NoTeardown(gates.FakeGateDeps):
            def load(self, *, rows):
                loaded = super().load(rows=rows)
                del loaded["teardown"]
                return loaded

        with pytest.raises(gates.GateRefused) as exc:
            gates.run_gate("gate_8", deps=NoTeardown(rows=8),
                           run_dir=run_dir, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        assert "teardown" in str(exc.value)
        assert not (Path(run_dir) / gates.LEDGER_NAME).exists(), \
            "it measured rows before noticing"

    def test_a_filesystem_without_hard_links_is_refused(self, tmp_path):
        """Every write-once guarantee here is ``os.link`` and nothing else.

        On a filesystem that cannot link, ``write_once_json`` degrades to
        "SystemExit at the first checkpoint" -- after the rows are measured.
        Worth knowing before, not during: on this node the realistic cause is
        putting the pack on the Windows filesystem, where the run would also
        be too slow to measure.
        """
        def cannot_link(src, dst):
            raise OSError("operation not supported")

        assert gates.link_support_problems(tmp_path) == []
        problems = gates.link_support_problems(tmp_path, linker=cannot_link)
        assert problems
        assert "hard link" in problems[0]

    def test_the_link_check_leaves_nothing_behind(self, tmp_path):
        before = sorted(p.name for p in tmp_path.iterdir())
        gates.link_support_problems(tmp_path)
        assert sorted(p.name for p in tmp_path.iterdir()) == before

    def test_the_runner_refuses_before_measuring_when_links_are_unavailable(
            self, run_dir, packdir, monkeypatch):
        monkeypatch.setattr(gates, "link_support_problems",
                            lambda d, **kw: ["this filesystem cannot hard link"])
        with pytest.raises(gates.GateRefused):
            gates.run_gate("gate_8", deps=gates.FakeGateDeps(rows=8),
                           run_dir=run_dir, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        assert not (Path(run_dir) / gates.LEDGER_NAME).exists()

    def test_an_unwritable_run_directory_is_refused(self, tmp_path):
        problems = gates.link_support_problems(tmp_path / "does" / "not" / "x",
                                               make_dirs=False)
        assert problems


class TestTheRunnerRefusesWithoutACarriedDigest:
    """The trust check is un-bypassable and happens before anything exists.

    Not "before the model loads" -- before the run directory is created. A
    refusal that has already made a directory has already started a run, and
    the next thing anybody does with a half-made run directory is resume it.
    """

    def digest(self, packdir):
        from src.training import pack

        return pack.read_manifest(packdir)[0]["pack_digest"]

    @pytest.mark.parametrize("fn", ["run_gate", "resume_problems",
                                    "resume_point"])
    def test_the_parameter_cannot_be_omitted(self, fn):
        import inspect

        param = inspect.signature(
            getattr(gates, fn)).parameters["expected_pack_digest"]
        assert param.default is inspect.Parameter.empty
        assert param.kind is inspect.Parameter.KEYWORD_ONLY

    @pytest.mark.parametrize("bad", [None, "", "a" * 63, "A" * 64, "b" * 64])
    def test_a_bad_digest_refuses_before_the_run_dir_exists(
            self, run_dir, packdir, bad):
        with pytest.raises(gates.GateRefused):
            gates.run_gate("gate_8", deps=gates.FakeGateDeps(rows=8),
                           run_dir=run_dir, pack_dir=packdir,
                           expected_pack_digest=bad,
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        assert not Path(run_dir).exists(), \
            "it created the run directory before refusing"

    def test_a_bad_digest_writes_no_evidence(self, run_dir, packdir):
        with pytest.raises(gates.GateRefused):
            gates.run_gate("gate_8", deps=gates.FakeGateDeps(rows=8),
                           run_dir=run_dir, pack_dir=packdir,
                           expected_pack_digest="b" * 64,
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        assert not Path(run_dir).exists()

    def test_a_bad_digest_never_reaches_the_loader(self, run_dir, packdir):
        """No model is built, because ``load`` is never called."""
        class Counting(gates.FakeGateDeps):
            loads = 0

            def load(self, *, rows):
                type(self).loads += 1
                return super().load(rows=rows)

        with pytest.raises(gates.GateRefused):
            gates.run_gate("gate_8", deps=Counting(rows=8), run_dir=run_dir,
                           pack_dir=packdir, expected_pack_digest="b" * 64,
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        assert Counting.loads == 0

    def test_the_matching_digest_runs(self, run_dir, packdir):
        evidence = gates.run_gate(
            "gate_8", deps=gates.FakeGateDeps(rows=8), run_dir=run_dir,
            pack_dir=packdir, expected_pack_digest=self.digest(packdir),
            expected_dependency_digest=dep_digest(),
            dependency_checker=dep_checker(),
            allocator_config=ALLOC,
            determinism=dict(DETERMINISM))
        assert gates.verdict("gate_8", evidence) == "passed"

    def test_a_resume_needs_the_digest_too(self, run_dir, packdir):
        digest = self.digest(packdir)
        with pytest.raises(gates.DeliberateStop):
            gates.run_gate("gate_500", deps=gates.FakeGateDeps(rows=500),
                           run_dir=run_dir, pack_dir=packdir,
                           expected_pack_digest=digest, stop_after=200,
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        assert gates.resume_problems(run_dir, pack_dir=packdir,
                                     expected_pack_digest="b" * 64,
                                     expected_dependency_digest=dep_digest(),
                                     dependency_checker=dep_checker(),
                                     allocator_config=ALLOC,
                                     determinism=dict(DETERMINISM))
        with pytest.raises(gates.GateRefused):
            gates.resume_point(run_dir, pack_dir=packdir,
                               expected_pack_digest=None,
                               expected_dependency_digest=dep_digest(),
                               dependency_checker=dep_checker(),
                               allocator_config=ALLOC,
                               determinism=dict(DETERMINISM))
        assert gates.resume_problems(
            run_dir, pack_dir=packdir, expected_pack_digest=digest,
            expected_dependency_digest=dep_digest(),
            dependency_checker=dep_checker(),
            allocator_config=ALLOC,
            determinism=dict(DETERMINISM)) == []

    def test_a_completely_re_signed_pack_cannot_be_resumed(
            self, run_dir, packdir):
        """Stop, rewrite everything, re-sign, resume. Refused on the digest."""
        import json

        from src.training import pack as packmod
        from src.training.session import manifest_digest, sha256_file

        carried = self.digest(packdir)
        with pytest.raises(gates.DeliberateStop):
            gates.run_gate("gate_500", deps=gates.FakeGateDeps(rows=500),
                           run_dir=run_dir, pack_dir=packdir,
                           expected_pack_digest=carried, stop_after=200,
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))

        body = json.loads((packdir / packmod.MANIFEST_NAME).read_text())
        for rel, entry in body["files"].items():
            path = packdir / rel
            path.write_text(f"# replaced\n{path.name}\n", encoding="utf-8")
            entry["sha256"] = sha256_file(path)
            entry["bytes"] = path.stat().st_size
        body["files_digest"] = manifest_digest(body)
        body["pack_digest"] = packmod.pack_digest(body)
        (packdir / packmod.MANIFEST_NAME).unlink()
        (packdir / packmod.MANIFEST_NAME).write_text(json.dumps(body, indent=2))

        assert packmod.verify(packdir) == [], "should be internally consistent"
        problems = gates.resume_problems(run_dir, pack_dir=packdir,
                                         expected_pack_digest=carried,
                                         expected_dependency_digest=dep_digest(),
                                         dependency_checker=dep_checker(),
                                         allocator_config=ALLOC,
                                         determinism=dict(DETERMINISM))
        assert problems
        assert any("pack_digest" in p or "expected" in p for p in problems), \
            problems


#: Fixed synthetic dependency evidence, owned by this file. The runner has to
#: be bound to *these* bytes for the drift tests below to mean anything.
DEP_EVIDENCE = {
    "schema_version": 1, "kind": "longrun_dependency_preflight",
    "network_used": False, "tensors_loaded": False, "device_initialised": False,
    "repositories": [
        {"repo_id": "Vendor/Tok", "revision": "a" * 40,
         "files": [{"name": "tokenizer.json", "bytes": 10,
                    "sha256": "1" * 64}]},
    ],
    "instruction_pool": {"path": "data/processed/instruct_inv_train.jsonl",
                         "sha256": "5" * 64},
}


def dep_checker(evidence=None):
    body = DEP_EVIDENCE if evidence is None else evidence
    return lambda: {"ok": True, "problems": [], "evidence": body}


def dep_digest(evidence=None):
    from src.training.longrun import dependency_digest

    return dependency_digest(DEP_EVIDENCE if evidence is None else evidence)


class TestTheRunnerRefusesWithoutADependencyDigest:
    """Same discipline as the pack digest, one layer down.

    A pack whose every byte is correct still trains against whatever tokenizer
    and base weights happen to be in the node's cache. Binding those is what
    this adds -- and it has to be un-bypassable and re-checked on every resume,
    because the cache is on a machine somebody uses and a resume can be days
    later.
    """

    @pytest.mark.parametrize("fn", ["run_gate", "resume_problems",
                                    "resume_point"])
    def test_the_parameter_cannot_be_omitted(self, fn):
        import inspect

        param = inspect.signature(
            getattr(gates, fn)).parameters["expected_dependency_digest"]
        assert param.default is inspect.Parameter.empty
        assert param.kind is inspect.Parameter.KEYWORD_ONLY

    @pytest.mark.parametrize("bad", [None, "", "a" * 63, "A" * 64, "b" * 64])
    def test_a_bad_digest_refuses_before_the_run_dir_exists(
            self, run_dir, packdir, bad):
        with pytest.raises(gates.GateRefused):
            gates.run_gate("gate_8", deps=gates.FakeGateDeps(rows=8),
                           run_dir=run_dir, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=bad,
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        assert not Path(run_dir).exists(), \
            "it created the run directory before refusing"

    def test_a_bad_digest_never_reaches_the_loader(self, run_dir, packdir):
        class Counting(gates.FakeGateDeps):
            loads = 0

            def load(self, *, rows):
                type(self).loads += 1
                return super().load(rows=rows)

        with pytest.raises(gates.GateRefused):
            gates.run_gate("gate_8", deps=Counting(rows=8), run_dir=run_dir,
                           pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest="b" * 64,
                           dependency_checker=dep_checker(),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        assert Counting.loads == 0

    def test_a_resolved_but_different_dependency_refuses(self, run_dir, packdir):
        """``ok=True``, every file present, one of them not the Mac's."""
        import copy

        drifted = copy.deepcopy(DEP_EVIDENCE)
        drifted["repositories"][0]["files"][0]["sha256"] = "9" * 64
        with pytest.raises(gates.GateRefused) as exc:
            gates.run_gate("gate_8", deps=gates.FakeGateDeps(rows=8),
                           run_dir=run_dir, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(drifted),
            allocator_config=ALLOC, determinism=dict(DETERMINISM))
        # "dependencies digest to ..." -- matched on the stem so the assertion
        # does not pin the exact wording of a sentence meant for a human.
        assert "dependenc" in str(exc.value).lower()
        assert "different file under the same name" in str(exc.value)
        assert not Path(run_dir).exists()

    def test_the_matching_digest_runs(self, run_dir, packdir):
        evidence = gates.run_gate(
            "gate_8", deps=gates.FakeGateDeps(rows=8), run_dir=run_dir,
            pack_dir=packdir, expected_pack_digest=digest_of(packdir),
            expected_dependency_digest=dep_digest(),
            dependency_checker=dep_checker(),
            allocator_config=ALLOC,
            determinism=dict(DETERMINISM))
        assert gates.verdict("gate_8", evidence) == "passed"

    def test_a_resume_revalidates_the_dependencies(self, run_dir, packdir):
        """Requirement: every resume, not just the first start.

        The cache sat on a working machine between the stop and the resume.
        Trusting the check the first attempt passed would trust a reading
        taken before the gap that made it worth re-taking.
        """
        import copy

        with pytest.raises(gates.DeliberateStop):
            gates.run_gate("gate_500", deps=gates.FakeGateDeps(rows=500),
                           run_dir=run_dir, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(), stop_after=200,
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))

        drifted = copy.deepcopy(DEP_EVIDENCE)
        drifted["repositories"][0]["files"][0]["sha256"] = "9" * 64

        problems = gates.resume_problems(
            run_dir, pack_dir=packdir,
            expected_pack_digest=digest_of(packdir),
            expected_dependency_digest=dep_digest(),
            dependency_checker=dep_checker(drifted),
            allocator_config=ALLOC, determinism=dict(DETERMINISM))
        assert problems
        assert any("dependenc" in p for p in problems), problems

        with pytest.raises(gates.GateRefused):
            gates.run_gate("gate_500", deps=gates.FakeGateDeps(rows=500),
                           run_dir=run_dir, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(drifted),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM), resume=True)

    def test_an_undrifted_resume_still_works(self, run_dir, packdir):
        with pytest.raises(gates.DeliberateStop):
            gates.run_gate("gate_500", deps=gates.FakeGateDeps(rows=500),
                           run_dir=run_dir, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(), stop_after=200,
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM))
        evidence = gates.run_gate(
            "gate_500", deps=gates.FakeGateDeps(rows=500), run_dir=run_dir,
            pack_dir=packdir, expected_pack_digest=digest_of(packdir),
            expected_dependency_digest=dep_digest(),
            dependency_checker=dep_checker(), resume=True,
            allocator_config=ALLOC,
            determinism=dict(DETERMINISM))
        assert evidence["rows_completed"] == 500
        assert gates.verdict("gate_500", evidence) == "passed"


#: A second, different dependency reading. Self-consistent, perfectly valid,
#: and not the one the run was planned against -- which is the whole point.
DEP_EVIDENCE_B = {
    "schema_version": 1, "kind": "longrun_dependency_preflight",
    "network_used": False, "tensors_loaded": False, "device_initialised": False,
    "repositories": [
        {"repo_id": "Vendor/Tok", "revision": "a" * 40,
         "files": [{"name": "tokenizer.json", "bytes": 11,
                    "sha256": "7" * 64}]},
    ],
    "instruction_pool": {"path": "data/processed/instruct_inv_train.jsonl",
                         "sha256": "5" * 64},
}


class TestTheDependencyDigestIsFrozenIntoTheRun:
    """Re-checking is not the same as freezing.

    Every resume already recomputes the dependency digest and compares it to
    the value the operator carried. Both of those can be *B*: swap the cache,
    carry the new digest, and the pair agrees with itself perfectly -- while
    attempt 1 trained against *A*. The run would splice two different
    tokenizers together and every check would pass.

    So the value the run was planned against is written into ``plan.json``,
    copied into every checkpoint, and everything else is compared against
    *that* rather than against whatever this invocation happens to see.
    """

    def start_and_stop(self, run_dir, packdir, *, evidence=None):
        """Attempt 1, against evidence A, stopped part-way."""
        with pytest.raises(gates.DeliberateStop):
            gates.run_gate("gate_500", deps=gates.FakeGateDeps(rows=500),
                           run_dir=run_dir, pack_dir=packdir,
                           expected_pack_digest=digest_of(packdir),
                           expected_dependency_digest=dep_digest(evidence),
                           dependency_checker=dep_checker(evidence),
                           allocator_config=ALLOC,
                           determinism=dict(DETERMINISM),
                           stop_after=200)

    def resume_kwargs(self, packdir, *, evidence=None):
        return {"pack_dir": packdir,
                "expected_pack_digest": digest_of(packdir),
                "expected_dependency_digest": dep_digest(evidence),
                "dependency_checker": dep_checker(evidence),
                "allocator_config": ALLOC,
                "determinism": dict(DETERMINISM)}

    # -- 1. the plan freezes it ------------------------------------------

    def test_write_plan_requires_the_digest(self):
        import inspect

        param = inspect.signature(
            gates.write_plan).parameters["expected_dependency_digest"]
        assert param.default is inspect.Parameter.empty
        assert param.kind is inspect.Parameter.KEYWORD_ONLY

    def test_the_plan_records_it(self, run_dir, packdir):
        self.start_and_stop(run_dir, packdir)
        plan = gates.read_plan(run_dir)
        assert plan["dependency_digest"] == dep_digest()

    def test_the_plan_on_disk_records_it(self, run_dir, packdir):
        self.start_and_stop(run_dir, packdir)
        body = json.loads((Path(run_dir) / gates.PLAN_NAME).read_text())
        assert body["dependency_digest"] == dep_digest()

    # -- 2. every checkpoint carries it ----------------------------------

    def test_every_checkpoint_records_it(self, run_dir, packdir):
        self.start_and_stop(run_dir, packdir)
        found = gates.read_checkpoints(run_dir)
        assert found, "the fixture took no checkpoint"
        for ckpt in found:
            assert ckpt["dependency_digest"] == dep_digest()

    def test_the_checkpoint_takes_it_from_the_plan_not_from_the_moment(
            self, run_dir, packdir):
        """It is the run's value, not a reading taken when it was written."""
        plan = gates.write_plan(
            run_dir, gate="gate_8", order=list(range(8)),
            pack_digest="x", config={}, provenance={},
            expected_dependency_digest="c" * 64,
            allocator_config=ALLOC, determinism=dict(DETERMINISM))
        body = gates.write_checkpoint(
            run_dir, position=8, attempt=1, plan=plan,
            optimizer_sha256="a" * 64, model_state=MODEL_STATE,
            rng_state=RNG_STATE,
            trainable_digest="c" * 64, ledger_entries=[], provenance={})
        assert body["dependency_digest"] == "c" * 64

    # -- 3. the resume compares against the frozen value ------------------

    def test_a_self_consistent_but_different_dependency_set_is_refused(
            self, run_dir, packdir):
        """The case this whole change exists for.

        Attempt 1 ran against A. The resume presents B: a different cache, a
        different digest, and a carried value that matches it exactly. Every
        check that only compares "carried" against "recomputed" passes.
        """
        self.start_and_stop(run_dir, packdir)

        from src.training import pack

        assert dep_digest(DEP_EVIDENCE_B) != dep_digest()
        assert pack.expected_digest_problems(
            dep_digest(DEP_EVIDENCE_B)) == [], "B must be a valid digest"

        # B agrees with itself completely...
        assert gates.dependency_digest_problems(
            dep_digest(DEP_EVIDENCE_B),
            checker=dep_checker(DEP_EVIDENCE_B)) == []

        # ...and the resume still refuses, because it is not the plan's.
        problems = gates.resume_problems(
            run_dir, **self.resume_kwargs(packdir, evidence=DEP_EVIDENCE_B))
        assert problems
        assert any("plan" in p and "dependency" in p.lower()
                   for p in problems), problems

    def test_that_refusal_happens_before_the_loader_and_before_any_new_row(
            self, run_dir, packdir):
        self.start_and_stop(run_dir, packdir)
        before = (Path(run_dir) / gates.LEDGER_NAME).read_bytes()

        class Counting(gates.FakeGateDeps):
            loads = 0

            def load(self, *, rows):
                type(self).loads += 1
                return super().load(rows=rows)

        with pytest.raises(gates.GateRefused):
            gates.run_gate("gate_500", deps=Counting(rows=500),
                           run_dir=run_dir, resume=True,
                           **self.resume_kwargs(packdir,
                                                evidence=DEP_EVIDENCE_B))
        assert Counting.loads == 0, "it built a model before refusing"
        assert (Path(run_dir) / gates.LEDGER_NAME).read_bytes() == before, \
            "it appended to the ledger before refusing"

    def test_a_carried_value_that_disagrees_with_the_plan_is_refused(
            self, run_dir, packdir):
        """Right cache, wrong carried value: still refused.

        The node recomputes A and the plan says A, but the operator typed a
        different valid digest. Nothing here silently prefers one over the
        other.
        """
        self.start_and_stop(run_dir, packdir)
        problems = gates.resume_problems(
            run_dir, pack_dir=packdir,
            expected_pack_digest=digest_of(packdir),
            expected_dependency_digest=dep_digest(DEP_EVIDENCE_B),
            dependency_checker=dep_checker(),
            allocator_config=ALLOC,
            determinism=dict(DETERMINISM))
        assert problems

    def test_a_matching_resume_still_works(self, run_dir, packdir):
        self.start_and_stop(run_dir, packdir)
        assert gates.resume_problems(
            run_dir, **self.resume_kwargs(packdir)) == []

    # -- missing, malformed, drifted --------------------------------------

    def test_a_plan_with_no_frozen_digest_is_refused(self, run_dir, packdir):
        """A run planned before this existed cannot be vouched for."""
        self.start_and_stop(run_dir, packdir)
        path = Path(run_dir) / gates.PLAN_NAME
        body = json.loads(path.read_text())
        del body["dependency_digest"]
        path.unlink()
        path.write_text(json.dumps(body, indent=2))
        problems = gates.resume_problems(
            run_dir, **self.resume_kwargs(packdir))
        assert problems
        assert any("dependency" in p.lower() for p in problems), problems

    @pytest.mark.parametrize("bad", [None, "", "a" * 63, "A" * 64, "g" * 64,
                                     12345, True])
    def test_a_malformed_frozen_digest_is_refused(self, run_dir, packdir, bad):
        self.start_and_stop(run_dir, packdir)
        path = Path(run_dir) / gates.PLAN_NAME
        body = json.loads(path.read_text())
        body["dependency_digest"] = bad
        path.unlink()
        path.write_text(json.dumps(body, indent=2))
        assert gates.resume_problems(run_dir,
                                     **self.resume_kwargs(packdir))

    def test_a_checkpoint_whose_digest_drifted_is_refused(
            self, run_dir, packdir):
        self.start_and_stop(run_dir, packdir)
        ckpt = gates.latest_checkpoint(run_dir)
        state = Path(ckpt["dir"]) / gates.CHECKPOINT_STATE
        body = json.loads(state.read_text())
        body["dependency_digest"] = "d" * 64
        state.unlink()
        state.write_text(json.dumps(body, indent=2))
        problems = gates.resume_problems(
            run_dir, **self.resume_kwargs(packdir))
        assert problems
        assert any("checkpoint" in p for p in problems), problems

    def test_a_checkpoint_with_no_digest_is_refused(self, run_dir, packdir):
        self.start_and_stop(run_dir, packdir)
        ckpt = gates.latest_checkpoint(run_dir)
        state = Path(ckpt["dir"]) / gates.CHECKPOINT_STATE
        body = json.loads(state.read_text())
        del body["dependency_digest"]
        state.unlink()
        state.write_text(json.dumps(body, indent=2))
        assert gates.resume_problems(run_dir,
                                     **self.resume_kwargs(packdir))

    # -- 4. the evidence that travels back --------------------------------

    def test_the_final_evidence_records_it(self, run_dir, packdir):
        evidence = gates.run_gate(
            "gate_8", deps=gates.FakeGateDeps(rows=8), run_dir=run_dir,
            pack_dir=packdir, expected_pack_digest=digest_of(packdir),
            expected_dependency_digest=dep_digest(),
            dependency_checker=dep_checker(),
            allocator_config=ALLOC,
            determinism=dict(DETERMINISM))
        assert evidence["dependency_digest"] == dep_digest()

    def test_the_resumed_evidence_records_the_plans_value(
            self, run_dir, packdir):
        """What went back to the Mac has to name the bytes that were used."""
        self.start_and_stop(run_dir, packdir)
        evidence = gates.run_gate(
            "gate_500", deps=gates.FakeGateDeps(rows=500), run_dir=run_dir,
            resume=True, **self.resume_kwargs(packdir))
        assert evidence["dependency_digest"] == dep_digest()
        assert evidence["rows_completed"] == 500

    def test_the_evidence_names_the_pack_and_the_dependencies_together(
            self, run_dir, packdir):
        """One without the other cannot say what produced a number."""
        evidence = gates.run_gate(
            "gate_8", deps=gates.FakeGateDeps(rows=8), run_dir=run_dir,
            pack_dir=packdir, expected_pack_digest=digest_of(packdir),
            expected_dependency_digest=dep_digest(),
            dependency_checker=dep_checker(),
            allocator_config=ALLOC,
            determinism=dict(DETERMINISM))
        assert evidence["pack_digest"] == digest_of(packdir)
        assert evidence["dependency_digest"] == dep_digest()


class TestTheDeviceTeardownMatchesTheDevice:
    """The failure the node found on its first real run.

    ``_empty_cache`` guarded itself with ``hasattr(torch, "mps")`` and
    ``hasattr(torch.mps, "empty_cache")``. Both are true on a **CUDA** build --
    the module and the function exist on every build -- so the guard let the
    call through and the driver refused it. Nothing on the Mac could have
    caught this: there, MPS really is available and the guard is accidentally
    correct.

    So the test is what the guard should have asked all along: is the backend
    *available*, not is the attribute *present*.
    """

    class MpsPresentButUnavailable:
        """A CUDA build: torch.mps exists, MPS does not."""
        called = False

        class mps:
            @staticmethod
            def empty_cache():
                TestTheDeviceTeardownMatchesTheDevice.MpsPresentButUnavailable.called = True
                raise RuntimeError("Cannot execute emptyCache() without MPS backend.")

        class backends:
            class mps:
                @staticmethod
                def is_available():
                    return False

    class MpsAvailable:
        called = False

        class mps:
            @staticmethod
            def empty_cache():
                TestTheDeviceTeardownMatchesTheDevice.MpsAvailable.called = True

        class backends:
            class mps:
                @staticmethod
                def is_available():
                    return True

    def test_it_does_not_call_mps_empty_cache_on_a_cuda_build(self):
        from src.training.longrun import _empty_cache

        stub = self.MpsPresentButUnavailable
        stub.called = False
        elapsed = _empty_cache(stub)          # must not raise
        assert stub.called is False, \
            "it called torch.mps.empty_cache() on a build with no MPS backend"
        assert elapsed >= 0

    def test_it_still_calls_mps_empty_cache_where_mps_is_available(self):
        """The Mac path must not be broken by fixing the node path."""
        from src.training.longrun import _empty_cache

        stub = self.MpsAvailable
        stub.called = False
        _empty_cache(stub)
        assert stub.called is True

    def test_a_torch_that_cannot_answer_is_treated_as_unavailable(self):
        from src.training.longrun import _empty_cache

        class Broken:
            class mps:
                @staticmethod
                def empty_cache():
                    raise AssertionError("must not be reached")

            class backends:
                class mps:
                    @staticmethod
                    def is_available():
                        raise RuntimeError("no idea")

        _empty_cache(Broken)   # must not raise

    def test_the_production_teardown_clears_the_cuda_cache(self):
        """And does it *after* dropping the model, not before.

        A clear measured while a merged 1B model, an adapter and Adam's two
        moment tensors are still referenced measures what cannot be freed.
        """
        order = []

        class RecordingTorch:
            class cuda:
                @staticmethod
                def is_available():
                    return True

                @staticmethod
                def empty_cache():
                    order.append("empty_cache")

                @staticmethod
                def max_memory_reserved():
                    return 0

                @staticmethod
                def max_memory_allocated():
                    return 0

        holder_seen = {}

        class Base:
            def load(self, *, rows):
                holder = {"model": _StubModel(), "optimizer": _StubOptimizer()}
                holder_seen["h"] = holder
                return {
                    "order": list(range(rows)),
                    "step": lambda i, p: {"loss": 1.0, "tokens": 1,
                                          "supervised_tokens": 1,
                                          "sample_id": f"s{i}"},
                    "provenance": {"device": "cuda", "dtype": "bfloat16",
                                   "lora_config": {}, "optimizer": {}},
                    "sample_ids": [f"s{i}" for i in range(rows)],
                    "model_load_seconds": 1.0,
                    "teardown": lambda: (_ for _ in ()).throw(
                        AssertionError("the MPS teardown must not be used")),
                    "clear": lambda: 0.0, "probe": dict,
                    "holder": holder,
                }

        deps = gates.ProductionGateDeps(device="cuda", base=Base(),
                                        torch_mod=RecordingTorch)
        loaded = deps.load(rows=8)
        loaded["teardown"]()

        assert order == ["empty_cache"], \
            "the CUDA cache was not cleared by teardown"
        holder = holder_seen["h"]
        assert "model" not in holder, "the model was not released"
        assert "optimizer" not in holder, "the optimizer was not released"

    def test_the_production_teardown_fails_closed_when_the_clear_fails(self):
        class Refusing:
            class cuda:
                @staticmethod
                def is_available():
                    return True

                @staticmethod
                def empty_cache():
                    raise RuntimeError("driver busy")

                @staticmethod
                def max_memory_reserved():
                    return 0

        class Base:
            def load(self, *, rows):
                return {
                    "order": list(range(rows)),
                    "step": lambda i, p: {"loss": 1.0, "tokens": 1,
                                          "supervised_tokens": 1,
                                          "sample_id": f"s{i}"},
                    "provenance": {}, "sample_ids": [f"s{i}" for i in range(rows)],
                    "model_load_seconds": 1.0, "teardown": lambda: 0.0,
                    "clear": lambda: 0.0, "probe": dict,
                    "holder": {"model": _StubModel(),
                               "optimizer": _StubOptimizer()},
                }

        deps = gates.ProductionGateDeps(device="cuda", base=Base(),
                                        torch_mod=Refusing)
        loaded = deps.load(rows=8)
        holder = loaded["holder"]
        # A clear that did not happen leaves the card in a state the next
        # measurement cannot be compared against, and swallowing it would let
        # a run report a peak that was never actually released. It raises --
        # but only *after* the references are gone, so a failed clear does not
        # also strand the model on the device.
        with pytest.raises(Exception) as exc:
            loaded["teardown"]()
        assert "driver busy" in str(exc.value) or "empty_cache" in str(exc.value)
        assert "model" not in holder, "it raised without releasing the model"
        assert "optimizer" not in holder


class TestPeakAllocatedIsRecordedButNotJudged:
    """Reserved says what the process took from the driver. Allocated says
    what was actually live.

    Gate 100 refused at 15.477 GB reserved on a 15.92 GB card. That reading
    alone cannot distinguish "the run really needs 15.5 GB" from "the caching
    allocator grew to 15.5 GB and would have released it under pressure" --
    and those call for opposite responses. ``peak_memory()`` has always
    computed both; only one was ever written down.

    Recorded, **not** judged. The threshold stays on reserved, because
    reserved is what has to fit in the card, and ``MAX_PEAK_VRAM_GB`` was
    declared before this machine produced any number at all. Adding a second
    reading is a diagnostic; moving the bound after seeing 15.477 would be
    choosing a threshold from the result.
    """

    def digest(self, packdir):
        from src.training import pack

        return pack.read_manifest(packdir)[0]["pack_digest"]

    def test_the_evidence_records_peak_allocated(self, run_dir, packdir):
        evidence = gates.run_gate(
            "gate_8", deps=gates.FakeGateDeps(rows=8), run_dir=run_dir,
            pack_dir=packdir, expected_pack_digest=self.digest(packdir),
            expected_dependency_digest=dep_digest(),
            dependency_checker=dep_checker(),
            allocator_config=ALLOC,
            determinism=dict(DETERMINISM))
        assert "peak_allocated_gb" in evidence

    def test_it_is_the_value_the_loader_reported(self, run_dir, packdir):
        """Taken from ``peak_memory()``, not recomputed from somewhere else."""
        class Deps(gates.FakeGateDeps):
            def load(self, *, rows):
                loaded = super().load(rows=rows)
                loaded["peak_memory"] = lambda: {"peak_vram_gb": 9.1,
                                                 "peak_allocated_gb": 7.25}
                return loaded

        evidence = gates.run_gate(
            "gate_8", deps=Deps(rows=8), run_dir=run_dir, pack_dir=packdir,
            expected_pack_digest=self.digest(packdir),
            expected_dependency_digest=dep_digest(),
            dependency_checker=dep_checker(),
            allocator_config=ALLOC,
            determinism=dict(DETERMINISM))
        assert evidence["peak_allocated_gb"] == 7.25
        assert evidence["peak_vram_gb"] == 9.1

    def test_an_unreadable_allocated_reading_is_none_not_absent(
            self, run_dir, packdir):
        class Deps(gates.FakeGateDeps):
            def load(self, *, rows):
                loaded = super().load(rows=rows)
                loaded["peak_memory"] = lambda: {"peak_vram_gb": 9.1}
                return loaded

        evidence = gates.run_gate(
            "gate_8", deps=Deps(rows=8), run_dir=run_dir, pack_dir=packdir,
            expected_pack_digest=self.digest(packdir),
            expected_dependency_digest=dep_digest(),
            dependency_checker=dep_checker(),
            allocator_config=ALLOC,
            determinism=dict(DETERMINISM))
        assert "peak_allocated_gb" in evidence
        assert evidence["peak_allocated_gb"] is None

    def test_the_fake_supplies_both_readings(self):
        """The stand-in has to honour the contract the real one does."""
        loaded = gates.FakeGateDeps(rows=8).load(rows=8)
        memory = loaded["peak_memory"]()
        assert "peak_vram_gb" in memory
        assert "peak_allocated_gb" in memory

    # -- the verdict must not move -------------------------------------

    def test_the_threshold_is_untouched(self):
        assert gates.MAX_PEAK_VRAM_GB == 15.0

    def test_the_verdict_still_judges_reserved_not_allocated(self):
        """Reserved over the bound fails, however small allocated was."""
        e = _good_hundred()
        e["peak_vram_gb"] = 15.477
        e["peak_allocated_gb"] = 6.0
        problems = gates.gate_problems("gate_100", e)
        assert any("peak VRAM" in p for p in problems), problems

    def test_a_high_allocated_reading_alone_does_not_fail_the_gate(self):
        """It is a diagnostic. Judging it would be a second, undeclared bound."""
        e = _good_hundred()
        e["peak_allocated_gb"] = 99.0
        assert gates.gate_problems("gate_100", e) == []

    def test_a_missing_allocated_reading_does_not_fail_the_gate(self):
        e = _good_hundred()
        e.pop("peak_allocated_gb", None)
        assert gates.gate_problems("gate_100", e) == []

    def test_the_previously_passing_evidence_still_passes(self):
        """Adding a field must not change any verdict that already stood."""
        for name, fixture in (("gate_8", _good_eight),
                              ("gate_100", _good_hundred),
                              ("gate_500", _good_five_hundred)):
            assert gates.gate_problems(name, fixture()) == [], name


class TestAllocatorDiagnosticsAreRecordedOnly:
    """Why reserved outran allocated, in numbers rather than inference.

    Gate 100 reserved 15.477 GB while only 7.150 GB was live. Two readings
    cannot say *why*: a caching allocator that is merely holding freed blocks
    and one that is fragmenting badly and retrying look identical from
    outside. ``num_alloc_retries`` and ``num_ooms`` separate them -- a
    allocator that never retried was never under pressure -- and
    ``inactive_split_bytes`` says how much of the cache is split fragments
    rather than whole reusable blocks.

    Every one of these is **recorded and nothing else**. The bound stays on
    reserved at 15.0 GB, every verdict is unchanged, and no allocator
    environment variable is set: a diagnostic that alters what it measures is
    not a diagnostic.
    """

    FIELDS = ("allocator_backend", "inactive_split_bytes_current",
              "inactive_split_bytes_peak", "num_alloc_retries", "num_ooms")

    def digest(self, packdir):
        from src.training import pack

        return pack.read_manifest(packdir)[0]["pack_digest"]

    class DiagTorch:
        """A CUDA build that answers the allocator questions."""
        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def empty_cache():
                return None

            @staticmethod
            def max_memory_reserved():
                return 15 * 1024 ** 3

            @staticmethod
            def max_memory_allocated():
                return 7 * 1024 ** 3

            @staticmethod
            def get_allocator_backend():
                return "native"

            @staticmethod
            def memory_stats():
                return {"inactive_split_bytes.all.current": 3 * 1024 ** 3,
                        "inactive_split_bytes.all.peak": 8 * 1024 ** 3,
                        "num_alloc_retries": 4,
                        "num_ooms": 1}

    def base(self):
        class Base:
            def load(self, *, rows):
                return {
                    "order": list(range(rows)),
                    "step": lambda i, p: {"loss": 1.0, "tokens": 1,
                                          "supervised_tokens": 1,
                                          "sample_id": f"s{i}"},
                    "provenance": {}, "sample_ids": [f"s{i}" for i in range(rows)],
                    "model_load_seconds": 1.0, "teardown": lambda: 0.0,
                    "clear": lambda: 0.0, "probe": dict,
                    "holder": {"model": _StubModel(),
                               "optimizer": _StubOptimizer()},
                }
        return Base()

    def test_peak_memory_reports_every_diagnostic(self):
        deps = gates.ProductionGateDeps(device="cuda", base=self.base(),
                                        torch_mod=self.DiagTorch)
        memory = deps.load(rows=8)["peak_memory"]()
        for field in self.FIELDS:
            assert field in memory, field
        assert memory["allocator_backend"] == "native"
        assert memory["num_alloc_retries"] == 4
        assert memory["num_ooms"] == 1
        assert memory["inactive_split_bytes_current"] == pytest.approx(3.0)
        assert memory["inactive_split_bytes_peak"] == pytest.approx(8.0)

    def test_the_byte_counts_are_reported_in_gigabytes(self):
        """Same unit as the two readings beside them, or they cannot be compared."""
        deps = gates.ProductionGateDeps(device="cuda", base=self.base(),
                                        torch_mod=self.DiagTorch)
        memory = deps.load(rows=8)["peak_memory"]()
        assert memory["inactive_split_bytes_peak"] < 100, \
            "this looks like raw bytes, not GB"

    def test_an_allocator_that_cannot_answer_gives_none_not_absent(self):
        class Mute:
            class cuda:
                @staticmethod
                def is_available():
                    return True

                @staticmethod
                def max_memory_reserved():
                    return 0

                @staticmethod
                def get_allocator_backend():
                    raise RuntimeError("not supported on this build")

                @staticmethod
                def memory_stats():
                    raise RuntimeError("no stats")

        deps = gates.ProductionGateDeps(device="cuda", base=self.base(),
                                        torch_mod=Mute)
        memory = deps.load(rows=8)["peak_memory"]()
        for field in self.FIELDS:
            assert field in memory, field
            assert memory[field] is None, field

    def test_missing_keys_in_memory_stats_are_none_not_zero(self):
        """Zero retries and "the field was not there" are different facts."""
        class Sparse:
            class cuda:
                @staticmethod
                def is_available():
                    return True

                @staticmethod
                def max_memory_reserved():
                    return 0

                @staticmethod
                def get_allocator_backend():
                    return "native"

                @staticmethod
                def memory_stats():
                    return {}

        deps = gates.ProductionGateDeps(device="cuda", base=self.base(),
                                        torch_mod=Sparse)
        memory = deps.load(rows=8)["peak_memory"]()
        assert memory["num_alloc_retries"] is None
        assert memory["num_ooms"] is None

    def test_the_evidence_carries_them(self, run_dir, packdir):
        class Deps(gates.FakeGateDeps):
            def load(self, *, rows):
                loaded = super().load(rows=rows)
                loaded["peak_memory"] = lambda: {
                    "peak_vram_gb": 9.1, "peak_allocated_gb": 7.4,
                    "allocator_backend": "native",
                    "inactive_split_bytes_current": 1.5,
                    "inactive_split_bytes_peak": 2.5,
                    "num_alloc_retries": 0, "num_ooms": 0}
                return loaded

        evidence = gates.run_gate(
            "gate_8", deps=Deps(rows=8), run_dir=run_dir, pack_dir=packdir,
            expected_pack_digest=self.digest(packdir),
            expected_dependency_digest=dep_digest(),
            dependency_checker=dep_checker(),
            allocator_config=ALLOC,
            determinism=dict(DETERMINISM))
        for field in self.FIELDS:
            assert field in evidence, field
        assert evidence["num_alloc_retries"] == 0
        assert evidence["allocator_backend"] == "native"

    def test_the_fake_supplies_them_too(self):
        memory = gates.FakeGateDeps(rows=8).load(rows=8)["peak_memory"]()
        for field in self.FIELDS:
            assert field in memory, field

    # -- nothing may be judged, and nothing may be configured -------------

    def test_the_threshold_is_still_fifteen(self):
        assert gates.MAX_PEAK_VRAM_GB == 15.0

    @pytest.mark.parametrize("field,value", [
        ("num_alloc_retries", 9999), ("num_ooms", 42),
        ("inactive_split_bytes_peak", 99.0),
        ("inactive_split_bytes_current", 99.0),
        ("allocator_backend", "cudaMallocAsync"),
    ])
    def test_no_diagnostic_can_change_a_verdict(self, field, value):
        e = _good_hundred()
        e[field] = value
        assert gates.gate_problems("gate_100", e) == [], \
            f"{field} changed the verdict; it is a diagnostic, not a bound"

    def test_the_verdict_still_turns_only_on_reserved(self):
        e = _good_hundred()
        e["peak_vram_gb"] = 15.477
        e.update({"num_alloc_retries": 0, "num_ooms": 0,
                  "inactive_split_bytes_peak": 8.3})
        problems = gates.gate_problems("gate_100", e)
        assert len(problems) == 1
        assert "peak VRAM" in problems[0]

    def test_the_pack_sets_no_allocator_environment_variable(self):
        """Measuring must not reconfigure the thing being measured."""
        included = [e["path"] for e in pack_module().manifest(ROOT)["include"]]
        for rel in included:
            # Tests are excluded because this one names the variable in order
            # to assert about it; scanning itself would make the check
            # self-failing in the same way the data-pin check was once
            # self-satisfying. What matters is that no *module* sets it.
            if not rel.endswith(".py") or rel.startswith("tests/"):
                continue
            text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
            # Naming it in order to *check* it is the whole point of
            # gpu_node; what must not happen is a module setting it.
            if rel == "src/training/gpu_node.py":
                continue
            assert "PYTORCH_CUDA_ALLOC_CONF" not in text, (
                f"{rel} names the allocator config variable; a diagnostic "
                "round must not change allocator behaviour")


def pack_module():
    from src.training import pack

    return pack


DETERMINISM = {"use_deterministic_algorithms": True, "warn_only": False,
               "cudnn_benchmark": False, "cudnn_deterministic": True,
               "cublas_workspace_config": ":4096:8",
               "tf32_matmul_allowed": False, "tf32_cudnn_allowed": False,
               "seed": 0}
ALLOC = "expandable_segments:True"


def runtime_kwargs(**over):
    out = {"allocator_config": ALLOC, "determinism": dict(DETERMINISM)}
    out.update(over)
    return out


class TestRuntimeProvenanceIsFrozenAndCrossVerified:
    """The allocator config and the determinism settings are provenance.

    A run that reserved 7.6 GB and one that reserved 15.5 GB differ only in an
    environment variable read before either started, and both report
    ``backend=native``. A run that cannot say which one it was cannot be
    compared with anything -- and a resume that silently changed it would
    splice two different machines into one result.
    """

    def digest(self, packdir):
        from src.training import pack

        return pack.read_manifest(packdir)[0]["pack_digest"]

    def start(self, run_dir, packdir, *, stop_after=None, **over):
        return gates.run_gate(
            "gate_500" if stop_after else "gate_8",
            deps=gates.FakeGateDeps(rows=500 if stop_after else 8),
            run_dir=run_dir, pack_dir=packdir,
            expected_pack_digest=self.digest(packdir),
            expected_dependency_digest=dep_digest(),
            dependency_checker=dep_checker(),
            stop_after=stop_after, **runtime_kwargs(**over))

    # -- required, no defaults -------------------------------------------

    @pytest.mark.parametrize("fn", ["write_plan", "run_gate",
                                    "resume_problems", "resume_point"])
    @pytest.mark.parametrize("param", ["allocator_config", "determinism"])
    def test_the_parameters_cannot_be_omitted(self, fn, param):
        import inspect

        p = inspect.signature(getattr(gates, fn)).parameters[param]
        assert p.default is inspect.Parameter.empty
        assert p.kind is inspect.Parameter.KEYWORD_ONLY

    # -- frozen everywhere ------------------------------------------------

    def test_the_plan_freezes_both(self, run_dir, packdir):
        self.start(run_dir, packdir)
        plan = gates.read_plan(run_dir)
        assert plan["allocator_config"] == ALLOC
        assert plan["determinism"] == DETERMINISM

    def test_every_checkpoint_carries_both(self, run_dir, packdir):
        with pytest.raises(gates.DeliberateStop):
            self.start(run_dir, packdir, stop_after=200)
        found = gates.read_checkpoints(run_dir)
        assert found
        for ck in found:
            assert ck["allocator_config"] == ALLOC
            assert ck["determinism"] == DETERMINISM

    def test_the_evidence_carries_both(self, run_dir, packdir):
        e = self.start(run_dir, packdir)
        assert e["allocator_config"] == ALLOC
        assert e["determinism"] == DETERMINISM

    # -- fail closed ------------------------------------------------------

    @pytest.mark.parametrize("bad", [None, "", "max_split_size_mb:128",
                                     "expandable_segments:False"])
    def test_a_bad_allocator_config_refuses_before_the_run_dir_exists(
            self, run_dir, packdir, bad):
        with pytest.raises(gates.GateRefused):
            self.start(run_dir, packdir, allocator_config=bad)
        assert not Path(run_dir).exists()

    @pytest.mark.parametrize("broken", [
        {"warn_only": True},
        {"use_deterministic_algorithms": False},
        {"cudnn_benchmark": True},
        {"cudnn_deterministic": False},
        {"cublas_workspace_config": ":2:2"},
    ])
    def test_a_non_strict_determinism_record_refuses(self, run_dir, packdir,
                                                     broken):
        settings = dict(DETERMINISM); settings.update(broken)
        with pytest.raises(gates.GateRefused):
            self.start(run_dir, packdir, determinism=settings)
        assert not Path(run_dir).exists()

    def test_an_incomplete_determinism_record_refuses(self, run_dir, packdir):
        settings = dict(DETERMINISM); settings.pop("seed")
        with pytest.raises(gates.GateRefused):
            self.start(run_dir, packdir, determinism=settings)

    # -- resume must match ------------------------------------------------

    def test_a_resume_with_a_different_allocator_config_refuses(
            self, run_dir, packdir):
        with pytest.raises(gates.DeliberateStop):
            self.start(run_dir, packdir, stop_after=200)
        problems = gates.resume_problems(
            run_dir, pack_dir=packdir,
            expected_pack_digest=self.digest(packdir),
            expected_dependency_digest=dep_digest(),
            dependency_checker=dep_checker(),
            **runtime_kwargs(
                allocator_config="expandable_segments:True,max_split_size_mb:64"))
        assert problems
        assert any("allocator" in p for p in problems), problems

    def test_a_resume_with_a_different_seed_refuses(self, run_dir, packdir):
        with pytest.raises(gates.DeliberateStop):
            self.start(run_dir, packdir, stop_after=200)
        settings = dict(DETERMINISM); settings["seed"] = 1
        problems = gates.resume_problems(
            run_dir, pack_dir=packdir,
            expected_pack_digest=self.digest(packdir),
            expected_dependency_digest=dep_digest(),
            dependency_checker=dep_checker(),
            **runtime_kwargs(determinism=settings))
        assert problems
        assert any("determinism" in p or "seed" in p for p in problems), problems

    def test_a_resume_with_changed_tf32_refuses(self, run_dir, packdir):
        with pytest.raises(gates.DeliberateStop):
            self.start(run_dir, packdir, stop_after=200)
        settings = dict(DETERMINISM); settings["tf32_matmul_allowed"] = True
        problems = gates.resume_problems(
            run_dir, pack_dir=packdir,
            expected_pack_digest=self.digest(packdir),
            expected_dependency_digest=dep_digest(),
            dependency_checker=dep_checker(),
            **runtime_kwargs(determinism=settings))
        assert problems

    def test_a_plan_missing_the_fields_refuses(self, run_dir, packdir):
        with pytest.raises(gates.DeliberateStop):
            self.start(run_dir, packdir, stop_after=200)
        path = Path(run_dir) / gates.PLAN_NAME
        body = json.loads(path.read_text()); del body["allocator_config"]
        path.unlink(); path.write_text(json.dumps(body, indent=2))
        assert gates.resume_problems(
            run_dir, pack_dir=packdir,
            expected_pack_digest=self.digest(packdir),
            expected_dependency_digest=dep_digest(),
            dependency_checker=dep_checker(), **runtime_kwargs())

    def test_a_drifted_checkpoint_refuses(self, run_dir, packdir):
        with pytest.raises(gates.DeliberateStop):
            self.start(run_dir, packdir, stop_after=200)
        ck = gates.latest_checkpoint(run_dir)
        state = Path(ck["dir"]) / gates.CHECKPOINT_STATE
        body = json.loads(state.read_text())
        body["allocator_config"] = "expandable_segments:False"
        state.unlink(); state.write_text(json.dumps(body, indent=2))
        problems = gates.resume_problems(
            run_dir, pack_dir=packdir,
            expected_pack_digest=self.digest(packdir),
            expected_dependency_digest=dep_digest(),
            dependency_checker=dep_checker(), **runtime_kwargs())
        assert any("checkpoint" in p for p in problems), problems

    def test_a_matching_resume_still_works(self, run_dir, packdir):
        with pytest.raises(gates.DeliberateStop):
            self.start(run_dir, packdir, stop_after=200)
        assert gates.resume_problems(
            run_dir, pack_dir=packdir,
            expected_pack_digest=self.digest(packdir),
            expected_dependency_digest=dep_digest(),
            dependency_checker=dep_checker(), **runtime_kwargs()) == []

    # -- the threshold is still untouched ---------------------------------

    def test_the_vram_threshold_is_unchanged(self):
        assert gates.MAX_PEAK_VRAM_GB == 15.0


class TestTheTrainableDigestIsRecorded:
    """The repeatability criterion needs the weights, not just the losses.

    Identical per-row losses can still end on different weights if anything
    after the last measured row differs. The criterion is stated over three
    things -- order, per-row loss, and the final trainable tensor content --
    so the third has to be measured and written down.
    """

    def digest(self, packdir):
        from src.training import pack

        return pack.read_manifest(packdir)[0]["pack_digest"]

    def test_it_is_part_of_the_loader_contract(self):
        assert "trainable_digest" in gates._LOADED_REQUIRED

    def test_the_fake_supplies_one(self):
        loaded = gates.FakeGateDeps(rows=8).load(rows=8)
        value = loaded["trainable_digest"]()
        assert isinstance(value, str) and len(value) == 64

    def test_the_evidence_records_it(self, run_dir, packdir):
        e = gates.run_gate(
            "gate_8", deps=gates.FakeGateDeps(rows=8), run_dir=run_dir,
            pack_dir=packdir, expected_pack_digest=self.digest(packdir),
            expected_dependency_digest=dep_digest(),
            dependency_checker=dep_checker(), **runtime_kwargs())
        assert len(e["trainable_digest"]) == 64

    def test_it_is_taken_before_teardown_drops_the_model(self, run_dir, packdir):
        """After teardown there is no model left to digest."""
        order = []

        class Deps(gates.FakeGateDeps):
            def load(self, *, rows):
                loaded = super().load(rows=rows)
                base_td = loaded["teardown"]
                loaded["trainable_digest"] = lambda: (
                    order.append("digest") or "d" * 64)
                loaded["teardown"] = lambda: (order.append("teardown")
                                              or base_td())
                return loaded

        gates.run_gate("gate_8", deps=Deps(rows=8), run_dir=run_dir,
                       pack_dir=packdir,
                       expected_pack_digest=self.digest(packdir),
                       expected_dependency_digest=dep_digest(),
                       dependency_checker=dep_checker(), **runtime_kwargs())
        assert order == ["digest", "teardown"], order

    def test_a_loader_without_one_is_refused(self, run_dir, packdir):
        class Deps(gates.FakeGateDeps):
            def load(self, *, rows):
                loaded = super().load(rows=rows)
                del loaded["trainable_digest"]
                return loaded

        with pytest.raises(gates.GateRefused) as exc:
            gates.run_gate("gate_8", deps=Deps(rows=8), run_dir=run_dir,
                           pack_dir=packdir,
                           expected_pack_digest=self.digest(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(), **runtime_kwargs())
        assert "trainable_digest" in str(exc.value)


# ---------------------------------------------------------------------------
# The checkpoint has to carry the weights, not only Adam's moments.
# ---------------------------------------------------------------------------

class TestTheCheckpointCarriesTheModelNotJustTheOptimizer:
    """``optimizer.state_dict()`` does not contain a single model parameter.

    A checkpoint holding only that restores Adam's moments onto weights that
    were rebuilt from scratch. The run then continues from the wrong point
    while every invariant this file already checks stays green -- contiguous
    positions, no duplicates, the planned order, unchanged provenance --
    because the ledger records *which* rows were measured and never what they
    were measured against. The failure is invisible in exactly the record
    built to make failures visible.

    So the checkpoint carries the trainable tensors too, and the resume has to
    prove it got them back: reload, re-digest, compare against the value the
    checkpoint recorded, and only then touch the optimizer.

    And not only the tensors. ``lora_dropout`` is 0.05 and the model is in
    train mode, so every forward pass draws from the generator. A resume that
    restores the weights and Adam's moments into a fresh process still starts
    its dropout stream from the seed rather than from wherever the rows
    already measured left it -- so it measures different losses from its first
    row onwards and ends on different weights, with everything else agreeing.
    The position of a run is three things, and a checkpoint carrying two of
    them is exactly as unresumable as one carrying one.
    """

    def digest(self, packdir):
        from src.training import pack

        return pack.read_manifest(packdir)[0]["pack_digest"]

    def stop_at(self, run_dir, packdir, rows=500, at=250):
        deps = gates.FakeGateDeps(rows=rows)
        with pytest.raises(gates.DeliberateStop):
            gates.run_gate("gate_500", deps=deps, run_dir=run_dir,
                           pack_dir=packdir,
                           expected_pack_digest=self.digest(packdir),
                           expected_dependency_digest=dep_digest(),
                           dependency_checker=dep_checker(),
                           stop_after=at, **runtime_kwargs())
        return deps

    def resume(self, run_dir, packdir, deps=None):
        return gates.run_gate(
            "gate_500", deps=deps or gates.FakeGateDeps(rows=500),
            run_dir=run_dir, pack_dir=packdir,
            expected_pack_digest=self.digest(packdir),
            expected_dependency_digest=dep_digest(),
            dependency_checker=dep_checker(), resume=True, **runtime_kwargs())

    # -- the fake must be able to tell the two states apart ----------------

    def test_the_fake_s_weights_move_on_every_row_not_only_on_steps(self):
        """A fake whose digest tracks the optimizer counter proves nothing.

        If ``trainable_digest`` is a function of ``optimizer_steps``, then
        restoring the optimizer alone restores the digest too, and every test
        below would pass against the broken checkpoint it is meant to catch.
        """
        deps = gates.FakeGateDeps(rows=8)
        loaded = deps.load(rows=8)
        before = loaded["trainable_digest"]()
        loaded["step"](loaded["order"][0], 1)
        assert deps.optimizer_steps == 0, "row 1 stepped the optimizer"
        assert loaded["trainable_digest"]() != before, \
            "the weights did not move on a row that took no optimizer step"

    def test_the_two_states_are_saved_and_restored_independently(self, tmp_path):
        deps = gates.FakeGateDeps(rows=8)
        loaded = deps.load(rows=8)
        for position in range(1, 9):
            loaded["step"](loaded["order"][position - 1], position)
        loaded["save_model_state"](tmp_path / "model")
        loaded["save_optimizer"](tmp_path / "opt")
        saved_digest, saved_steps = loaded["trainable_digest"](), deps.optimizer_steps

        other = gates.FakeGateDeps(rows=8)
        fresh = other.load(rows=8)
        assert fresh["trainable_digest"]() != saved_digest
        fresh["load_optimizer"](tmp_path / "opt")
        assert other.optimizer_steps == saved_steps
        assert fresh["trainable_digest"]() != saved_digest, \
            "loading the optimizer moved the weights, so the fake conflates them"
        fresh["load_model_state"](tmp_path / "model")
        assert fresh["trainable_digest"]() == saved_digest

    # -- the finding itself -----------------------------------------------

    def test_only_all_three_land_where_an_uninterrupted_run_lands(
            self, tmp_path):
        """The bug, demonstrated at the level where it happens.

        Four drives of the same fake over the same 500 rows: uninterrupted;
        stopped at 192 and continued with the optimizer alone; with the
        weights and the optimizer; and with all three. Only the last ends
        where the first did, and each of the two failures is a checkpoint that
        looks complete from every other angle.
        """
        def drive(loaded, positions):
            for position in positions:
                loaded["step"](loaded["order"][position - 1], position)
            return loaded

        want = drive(gates.FakeGateDeps(rows=500).load(rows=500),
                     range(1, 501))["trainable_digest"]()

        stopped = drive(gates.FakeGateDeps(rows=500).load(rows=500),
                        range(1, 193))
        stopped["save_model_state"](tmp_path / "model")
        stopped["save_optimizer"](tmp_path / "opt")
        stopped["save_rng_state"](tmp_path / "rng")
        drive(stopped, range(193, 251))          # discarded by the resume

        optimizer_only = gates.FakeGateDeps(rows=500).load(rows=500)
        optimizer_only["load_optimizer"](tmp_path / "opt")
        drive(optimizer_only, range(193, 501))
        assert optimizer_only["trainable_digest"]() != want, (
            "the fixture cannot distinguish an optimizer-only resume from an "
            "uninterrupted run, so nothing else here proves anything")

        no_stream = gates.FakeGateDeps(rows=500).load(rows=500)
        no_stream["load_model_state"](tmp_path / "model")
        no_stream["load_optimizer"](tmp_path / "opt")
        drive(no_stream, range(193, 501))
        assert no_stream["trainable_digest"]() != want, (
            "the fixture draws nothing per row, so it cannot show what a "
            "dropout stream restarted from the seed does")

        everything = gates.FakeGateDeps(rows=500).load(rows=500)
        everything["load_model_state"](tmp_path / "model")
        everything["load_optimizer"](tmp_path / "opt")
        everything["load_rng_state"](tmp_path / "rng")
        drive(everything, range(193, 501))
        assert everything["trainable_digest"]() == want

    # -- what the checkpoint has to hold ----------------------------------

    def test_it_is_part_of_the_loader_contract(self):
        for key in ("save_model_state", "load_model_state",
                    "save_rng_state", "load_rng_state"):
            assert key in gates._LOADED_REQUIRED, key

    def test_the_fake_draws_once_per_row(self):
        """Standing in for the dropout mask every forward pass consumes."""
        deps = gates.FakeGateDeps(rows=8)
        loaded = deps.load(rows=8)
        before = dict(deps.rng)
        loaded["step"](loaded["order"][0], 1)
        assert deps.rng != before, "the row took no draw"

    def test_the_same_row_from_the_same_weights_differs_by_the_stream_alone(
            self, tmp_path):
        """If it did not, restoring the stream would not need proving."""
        first = gates.FakeGateDeps(rows=8)
        a = first.load(rows=8)
        loss_a = a["step"](a["order"][0], 1)["loss"]

        second = gates.FakeGateDeps(rows=8)
        b = second.load(rows=8)
        for position in range(1, 5):
            b["step"](b["order"][position - 1], position)
        # Same weights, same row, four draws further down the stream.
        second.weights = {"w": 0.0, "rows_trained": 0}
        assert b["step"](b["order"][0], 1)["loss"] != loss_a

    def test_the_checkpoint_records_the_model_state_and_its_digest(
            self, run_dir, packdir):
        from src.training.session import sha256_file

        self.stop_at(run_dir, packdir)
        ckpt = gates.latest_checkpoint(run_dir)
        assert ckpt["position"] == 192

        state = ckpt.get("model_state")
        assert isinstance(state, dict), "the checkpoint records no model state"
        assert state["name"] == gates.MODEL_STATE_NAME
        assert isinstance(state["sha256"], str) and len(state["sha256"]) == 64
        assert isinstance(state["bytes"], int) and state["bytes"] > 0
        assert isinstance(ckpt.get("trainable_digest"), str)
        assert len(ckpt["trainable_digest"]) == 64

        blob = Path(ckpt["dir"]) / gates.MODEL_STATE_NAME
        assert blob.is_file(), "the checkpoint names a file that is not there"
        assert sha256_file(blob) == state["sha256"]
        assert blob.stat().st_size == state["bytes"]

        stream = ckpt.get("rng_state")
        assert isinstance(stream, dict), "the checkpoint records no rng state"
        assert stream["name"] == gates.RNG_STATE_NAME
        assert isinstance(stream["sha256"], str) and len(stream["sha256"]) == 64
        rng_blob = Path(ckpt["dir"]) / gates.RNG_STATE_NAME
        assert rng_blob.is_file()
        assert sha256_file(rng_blob) == stream["sha256"]

    def test_every_checkpoint_carries_one_not_just_the_last(
            self, run_dir, packdir):
        self.stop_at(run_dir, packdir)
        found = gates.read_checkpoints(run_dir)
        assert len(found) >= 3
        for ckpt in found:
            assert (Path(ckpt["dir"]) / gates.MODEL_STATE_NAME).is_file(), \
                f"checkpoint {ckpt['position']} saved no model state"
            assert ckpt.get("trainable_digest")

    def test_the_recorded_digest_is_the_weights_at_that_row(
            self, run_dir, packdir, tmp_path):
        """Not the weights at some other row, and not a constant."""
        self.stop_at(run_dir, packdir)
        found = gates.read_checkpoints(run_dir)
        digests = [c["trainable_digest"] for c in found]
        assert len(set(digests)) == len(digests), \
            "two checkpoints recorded the same weights"

        first = found[0]
        replay = gates.FakeGateDeps(rows=500).load(rows=500)
        for position in range(1, first["position"] + 1):
            replay["step"](replay["order"][position - 1], position)
        assert replay["trainable_digest"]() == first["trainable_digest"]

    def test_the_model_state_file_cannot_be_rewritten(self, run_dir, packdir):
        self.stop_at(run_dir, packdir)
        ckpt = gates.latest_checkpoint(run_dir)
        blob = Path(ckpt["dir"]) / gates.MODEL_STATE_NAME
        before = blob.read_bytes()

        plan = gates.read_plan(run_dir)
        loaded = gates.FakeGateDeps(rows=500).load(rows=500)
        with pytest.raises(SystemExit):
            gates._take_checkpoint(run_dir, position=ckpt["position"],
                                   attempt=9, plan=plan, loaded=loaded,
                                   provenance=plan["provenance"])
        assert blob.read_bytes() == before, \
            "a second checkpoint at the same row replaced the weights"

    def test_no_staging_file_is_left_inside_the_checkpoint(
            self, run_dir, packdir):
        self.stop_at(run_dir, packdir)
        for ckpt in gates.read_checkpoints(run_dir):
            names = sorted(p.name for p in Path(ckpt["dir"]).iterdir())
            assert names == sorted([gates.CHECKPOINT_STATE,
                                    gates.MODEL_STATE_NAME,
                                    gates.RNG_STATE_NAME,
                                    gates.OPTIMIZER_NAME]), names

    # -- fail closed on anything wrong with it ----------------------------

    def problems(self, run_dir, packdir):
        return gates.resume_problems(
            run_dir, pack_dir=packdir,
            expected_pack_digest=self.digest(packdir),
            expected_dependency_digest=dep_digest(),
            dependency_checker=dep_checker(), **runtime_kwargs())

    def rewrite_state(self, ckpt, **changes):
        path = Path(ckpt["dir"]) / gates.CHECKPOINT_STATE
        body = json.loads(path.read_text(encoding="utf-8"))
        body.update(changes)
        path.unlink()
        path.write_text(json.dumps(body, indent=2), encoding="utf-8")

    def test_a_checkpoint_that_records_no_model_state_refuses(
            self, run_dir, packdir):
        self.stop_at(run_dir, packdir)
        ckpt = gates.latest_checkpoint(run_dir)
        self.rewrite_state(ckpt, model_state=None)
        problems = self.problems(run_dir, packdir)
        assert any("model state" in p for p in problems), problems

    def test_a_checkpoint_that_records_no_trainable_digest_refuses(
            self, run_dir, packdir):
        self.stop_at(run_dir, packdir)
        ckpt = gates.latest_checkpoint(run_dir)
        self.rewrite_state(ckpt, trainable_digest=None)
        problems = self.problems(run_dir, packdir)
        assert any("trainable" in p for p in problems), problems

    def test_a_missing_model_state_file_refuses(self, run_dir, packdir):
        self.stop_at(run_dir, packdir)
        ckpt = gates.latest_checkpoint(run_dir)
        (Path(ckpt["dir"]) / gates.MODEL_STATE_NAME).unlink()
        problems = self.problems(run_dir, packdir)
        assert any("model state" in p for p in problems), problems

    def test_a_missing_rng_state_file_refuses(self, run_dir, packdir):
        self.stop_at(run_dir, packdir)
        ckpt = gates.latest_checkpoint(run_dir)
        (Path(ckpt["dir"]) / gates.RNG_STATE_NAME).unlink()
        problems = self.problems(run_dir, packdir)
        assert any("rng state" in p for p in problems), problems

    def test_a_tampered_rng_state_file_refuses(self, run_dir, packdir):
        self.stop_at(run_dir, packdir)
        ckpt = gates.latest_checkpoint(run_dir)
        blob = Path(ckpt["dir"]) / gates.RNG_STATE_NAME
        blob.write_bytes(blob.read_bytes() + b" ")
        problems = self.problems(run_dir, packdir)
        assert any("rng state" in p for p in problems), problems

    def test_a_checkpoint_that_records_no_rng_state_refuses(
            self, run_dir, packdir):
        self.stop_at(run_dir, packdir)
        ckpt = gates.latest_checkpoint(run_dir)
        self.rewrite_state(ckpt, rng_state=None)
        problems = self.problems(run_dir, packdir)
        assert any("rng state" in p for p in problems), problems

    def test_a_tampered_model_state_file_refuses(self, run_dir, packdir):
        self.stop_at(run_dir, packdir)
        ckpt = gates.latest_checkpoint(run_dir)
        blob = Path(ckpt["dir"]) / gates.MODEL_STATE_NAME
        blob.write_bytes(blob.read_bytes() + b" ")
        problems = self.problems(run_dir, packdir)
        assert any("model state" in p for p in problems), problems

    def test_a_model_state_named_outside_the_checkpoint_refuses(
            self, run_dir, packdir):
        """The name comes off disk, so it is input, not a constant."""
        self.stop_at(run_dir, packdir)
        ckpt = gates.latest_checkpoint(run_dir)
        state = dict(ckpt["model_state"])
        state["name"] = "../../../etc/passwd"
        self.rewrite_state(ckpt, model_state=state)
        problems = self.problems(run_dir, packdir)
        assert any("model state" in p for p in problems), problems

    def test_a_resume_is_refused_when_the_restored_weights_do_not_match(
            self, run_dir, packdir):
        """The load claimed to work; the digest says it did not."""
        self.stop_at(run_dir, packdir)

        class Forgetful(gates.FakeGateDeps):
            def load(self, *, rows):
                loaded = super().load(rows=rows)
                loaded["load_model_state"] = lambda path: None
                return loaded

        with pytest.raises(gates.GateRefused) as exc:
            self.resume(run_dir, packdir, deps=Forgetful(rows=500))
        assert "trainable" in str(exc.value).lower()

    def test_the_refusal_happens_before_another_row_is_measured(
            self, run_dir, packdir):
        self.stop_at(run_dir, packdir)
        before = len(gates.read_ledger(run_dir))

        class Forgetful(gates.FakeGateDeps):
            def load(self, *, rows):
                loaded = super().load(rows=rows)
                loaded["load_model_state"] = lambda path: None
                return loaded

        with pytest.raises(gates.GateRefused):
            self.resume(run_dir, packdir, deps=Forgetful(rows=500))
        assert len(gates.read_ledger(run_dir)) == before, \
            "a refused resume still appended rows"

    # -- the order the resume does things in ------------------------------

    def test_the_resume_loads_the_weights_then_the_optimizer_then_steps(
            self, run_dir, packdir):
        """Order is the whole point.

        Loading the optimizer first and the weights second would leave Adam's
        state pointing at parameters that were replaced underneath it, and
        measuring a row before either would train the fresh model for one row
        and then overwrite the result.
        """
        self.stop_at(run_dir, packdir)
        calls: list[str] = []

        class Recording(gates.FakeGateDeps):
            def load(self, *, rows):
                loaded = super().load(rows=rows)
                for key in ("load_model_state", "load_optimizer",
                            "load_rng_state"):
                    inner = loaded[key]
                    loaded[key] = (lambda fn, name: lambda path: (
                        calls.append(name), fn(path))[1])(inner, key)
                inner_step = loaded["step"]
                loaded["step"] = lambda index, position: (
                    calls.append(f"step:{position}"),
                    inner_step(index, position))[1]
                return loaded

        self.resume(run_dir, packdir, deps=Recording(rows=500))
        assert calls[:4] == ["load_model_state", "load_optimizer",
                             "load_rng_state", "step:193"], calls[:6]

    # -- and the whole thing, end to end ----------------------------------

    def test_an_interrupted_run_ends_exactly_where_an_uninterrupted_one_does(
            self, tmp_path, packdir):
        """The criterion, stated before the numbers exist.

        Four equalities, all of them exact: the effective order, every
        effective per-row loss, the optimizer steps actually taken, and the
        final trainable tensor content. Nothing here is a tolerance.
        """
        straight_dir = tmp_path / "runs" / "straight"
        straight_deps = gates.FakeGateDeps(rows=500)
        straight = gates.run_gate(
            "gate_500", deps=straight_deps, run_dir=straight_dir,
            pack_dir=packdir, expected_pack_digest=self.digest(packdir),
            expected_dependency_digest=dep_digest(),
            dependency_checker=dep_checker(), **runtime_kwargs())

        broken_dir = tmp_path / "runs" / "interrupted"
        self.stop_at(broken_dir, packdir, at=250)
        resumed_deps = gates.FakeGateDeps(rows=500)
        resumed = self.resume(broken_dir, packdir, deps=resumed_deps)

        a = gates.effective_ledger(gates.read_ledger(straight_dir))
        b = gates.effective_ledger(gates.read_ledger(broken_dir))
        assert sorted(a) == sorted(b) == list(range(1, 501))
        assert [a[p]["index"] for p in sorted(a)] == \
               [b[p]["index"] for p in sorted(b)], "the input order diverged"
        assert [a[p]["sample_id"] for p in sorted(a)] == \
               [b[p]["sample_id"] for p in sorted(b)]
        assert [a[p]["loss"] for p in sorted(a)] == \
               [b[p]["loss"] for p in sorted(b)], "a per-row loss differs"
        assert straight_deps.optimizer_steps == resumed_deps.optimizer_steps
        assert straight["trainable_digest"] == resumed["trainable_digest"], \
            "the interrupted run ended on different weights"

    def test_the_evidence_records_that_the_weights_were_restored(
            self, run_dir, packdir):
        self.stop_at(run_dir, packdir)
        evidence = self.resume(run_dir, packdir)
        assert evidence["model_state_restored"] is True
        assert evidence["optimizer_state_restored"] is True
        assert evidence["rng_state_restored"] is True
        assert gates.verdict("gate_500", evidence) == "passed"

    def test_a_run_that_was_never_resumed_says_so(self, run_dir, packdir):
        evidence = gates.run_gate(
            "gate_8", deps=gates.FakeGateDeps(rows=8), run_dir=run_dir,
            pack_dir=packdir, expected_pack_digest=self.digest(packdir),
            expected_dependency_digest=dep_digest(),
            dependency_checker=dep_checker(), **runtime_kwargs())
        assert evidence["model_state_restored"] is False
        assert evidence["rng_state_restored"] is False

    def test_the_five_hundred_verdict_needs_both_restorations(self):
        evidence = dict(_good_five_hundred())
        assert gates.gate_problems("gate_500", evidence) == []
        for key in ("model_state_restored", "optimizer_state_restored",
                    "rng_state_restored"):
            spoiled = dict(evidence)
            spoiled[key] = False
            problems = gates.gate_problems("gate_500", spoiled)
            assert problems, key
            spoiled.pop(key)
            assert gates.gate_problems("gate_500", spoiled), key

    def test_the_production_loader_supplies_both(self):
        loaded = TestTheProductionLoaderIsAWrapperNotASecondLoader().deps().load(
            rows=8)
        for key in ("save_model_state", "load_model_state",
                    "save_rng_state", "load_rng_state"):
            assert callable(loaded[key]), key


class TestTheAdapterManifestDescribesTheAdapterThatWasSaved:
    """``write_manifest`` was called with the default configuration.

    The manifest beside a saved adapter is what ``load_finetuned`` checks
    against before it builds anything: it records the rank, the alpha and the
    target modules the adapter was fitted with. Building the run with H2's
    configuration and writing the manifest from ``LoraConfig_()`` produces a
    directory whose weights are rank 32 and whose manifest says rank 16 --
    and the cold load then validates the adapter against the wrong shape, or
    passes and leaves a permanently mislabelled checkpoint behind.
    """

    class Torch:
        class cuda:
            @staticmethod
            def max_memory_reserved():
                return 0

    def build(self, tmp_path, cfg):
        base = TestTheProductionLoaderIsAWrapperNotASecondLoader.StubBase()
        deps = gates.ProductionGateDeps(device="cuda", base=base,
                                        torch_mod=self.Torch, cfg=cfg)
        loaded = deps.load(rows=8)
        dest = tmp_path / "adapter"
        loaded["save_adapter"](dest)
        return json.loads((dest / "brickagain_manifest.json").read_text())

    def test_the_manifest_records_the_configuration_the_run_was_built_with(
            self, tmp_path):
        from src.training import hypotheses

        manifest = self.build(tmp_path, hypotheses.config_for("H2"))
        assert manifest["lora"]["r"] == 32, manifest["lora"]
        assert manifest["lora"]["alpha"] == 16, manifest["lora"]

    def test_the_other_arm_is_recorded_as_itself(self, tmp_path):
        from src.training import hypotheses

        manifest = self.build(tmp_path, hypotheses.config_for("H1"))
        assert manifest["lora"]["r"] == 16
        assert manifest["lora"]["alpha"] == 32

    def test_no_configuration_still_means_the_project_default(self, tmp_path):
        from src.training.lora import LoraConfig_

        manifest = self.build(tmp_path, None)
        assert manifest["lora"]["r"] == LoraConfig_().rank
        assert manifest["lora"]["alpha"] == LoraConfig_().alpha

    def test_the_two_arms_do_not_produce_the_same_manifest(self, tmp_path):
        from src.training import hypotheses

        a = self.build(tmp_path / "a", hypotheses.config_for("H1"))
        b = self.build(tmp_path / "b", hypotheses.config_for("H2"))
        assert a["lora"] != b["lora"]
