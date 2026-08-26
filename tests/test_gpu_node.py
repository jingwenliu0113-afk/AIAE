"""The execution node's preflight, and the two hypotheses it is not allowed to run.

The Taichung machine is an *execution node*. It runs a pack the Mac built and
signed, and it is not a second place where the project is developed. Both
halves of that sentence are enforced here rather than asked for politely:

* a preflight that **fails closed**. Every check answers from a reading, and a
  reading that could not be taken is a failure -- never a pass, never a
  warning, never a quiet fallback to the CPU. The interesting tests below are
  the ones where something is *unreadable*, because that is the shape a real
  broken node has;
* a refusal to be a development source. A pack has no git history and no
  modifiable working tree, and the preflight says so.

The hypotheses are here for the opposite reason: to be inert. H1 and H2 are
frozen configurations, and until all three gates have passed on this node they
cannot be turned into a run. The tests assert the refusal, and assert that the
two differ in exactly the three numbers the design says they differ in.

Nothing here imports torch, loads a model, opens a socket, or reads the
dataset. Every device reading is injected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.training import gpu_node, hypotheses
from src.training.lora import LoraConfig_

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# A node that passes everything, so each test can break exactly one thing.
# ---------------------------------------------------------------------------

def good_probe(**overrides) -> dict:
    base = {
        "os_system": "Linux",
        "wsl2": True,
        "wsl_evidence": "kernel release names a Microsoft build",
        "torch_version": "2.13.0+cu128",
        "torch_cuda_build": "12.8",
        "cuda_available": True,
        "device_count": 1,
        "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
        "vram_total_gb": 15.9,
        "system_ram_gb": 31.2,
        "offline_env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "HF_HUB_DISABLE_TELEMETRY": "1"},
        # Inherited from the environment before launch, never set by us.
        "alloc_env": {"PYTORCH_ALLOC_CONF": "expandable_segments:True",
                      "PYTORCH_CUDA_ALLOC_CONF": None},
        "cublas_workspace_config": ":4096:8",
        "allocator_backend": "native",
    }
    base.update(overrides)
    return base


@pytest.fixture()
def node(tmp_path):
    """A verified pack directory, with the data beside it."""
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
    for rel in pack.REQUIRED_DATA:
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"sample_id":"x"}\n', encoding="utf-8")
    dest = tmp_path / "pack"
    pack.build(dest, root=src)
    for rel in pack.REQUIRED_DATA:
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"sample_id":"x"}\n', encoding="utf-8")
    return dest


def trusted_digest(node) -> str:
    """The value the Mac printed, as the operator would have carried it."""
    from src.training import pack

    return pack.read_manifest(node)[0]["pack_digest"]


#: A fixed, synthetic dependency evidence block. Nothing here is read from
#: this machine: the point is that the node's answer is bound to *these* bytes,
#: so the test has to own them.
DEP_EVIDENCE = {
    "schema_version": 1, "kind": "longrun_dependency_preflight",
    "network_used": False, "tensors_loaded": False, "device_initialised": False,
    "repositories": [
        {"repo_id": "Vendor/Tok", "revision": "a" * 40,
         "files": [{"name": "tokenizer.json", "bytes": 10, "sha256": "1" * 64}]},
        {"repo_id": "Vendor/Base", "revision": "b" * 40,
         "files": [{"name": "config.json", "bytes": 30, "sha256": "3" * 64}]},
    ],
    "instruction_pool": {"path": "data/processed/instruct_inv_train.jsonl",
                         "sha256": "5" * 64},
}


def dep_digest(evidence=None):
    from src.training.longrun import dependency_digest

    return dependency_digest(DEP_EVIDENCE if evidence is None else evidence)


def ok_dependencies(evidence=None):
    """Every pinned dependency resolves locally. Injected: nothing is read."""
    body = DEP_EVIDENCE if evidence is None else evidence
    return lambda: {"ok": True, "problems": [], "evidence": body}


def run(node, **kw):
    kw.setdefault("probe", good_probe())
    kw.setdefault("pack_dir", node)
    kw.setdefault("data_root", node)
    kw.setdefault("expected_pack_digest", trusted_digest(node))
    kw.setdefault("dependency_checker", ok_dependencies())
    kw.setdefault("expected_dependency_digest", dep_digest())
    return gpu_node.preflight(**kw)


class TestTheHappyPath:

    def test_a_good_node_passes(self, node):
        result = run(node)
        assert result["passed"], result["failed"]
        assert result["failed"] == []

    def test_every_check_says_what_it_read(self, node):
        for name, check in run(node)["checks"].items():
            assert check["detail"], f"{name} passed without saying why"
            assert "passed" in check

    def test_the_result_carries_no_personal_identifier(self, node):
        import json

        from src.training.longrun import leaked_identifiers

        assert leaked_identifiers(json.dumps(run(node), default=str)) == []

    def test_a_failing_result_carries_no_personal_identifier_either(
            self, node, tmp_path):
        """The failure path is the one that quotes paths back at you.

        A passing preflight has nothing to say about the disk. A failing one
        names what it could not find -- and that message ends up in the
        evidence, which travels.
        """
        import json

        from src.training.longrun import leaked_identifiers

        result = gpu_node.preflight(probe=good_probe(),
                                    pack_dir=tmp_path / "no_pack_here",
                                    expected_pack_digest="a" * 64,
                                    data_root=tmp_path / "no_data_here",
                                    dependency_checker=ok_dependencies(),
                                    expected_dependency_digest=dep_digest())
        assert not result["passed"]
        leaks = leaked_identifiers(json.dumps(result, default=str))
        assert leaks == [], f"the refusal leaked {leaks}"

    def test_macos_is_named_rather_than_called_unreadable(self, node):
        """"Could not be read" is wrong when it plainly was read."""
        result = gpu_node.preflight(
            probe=good_probe(os_system="Darwin", wsl2=None,
                             wsl_evidence=None),
            pack_dir=node, expected_pack_digest=trusted_digest(node),
            data_root=node, dependency_checker=ok_dependencies(),
            expected_dependency_digest=dep_digest())
        detail = result["checks"]["platform"]["detail"]
        assert "Darwin" in detail
        assert "could not be read" not in detail


class TestFailsClosed:
    """A reading that could not be taken has not been satisfied."""

    @pytest.mark.parametrize("field", [
        "os_system", "wsl2", "torch_cuda_build", "cuda_available",
        "device_count", "gpu_name", "vram_total_gb", "system_ram_gb",
    ])
    def test_an_unreadable_probe_value_fails_rather_than_passes(self, node, field):
        result = run(node, probe=good_probe(**{field: None}))
        assert not result["passed"], f"{field}=None was treated as a pass"
        assert result["failed"], "nothing was named as the reason"

    def test_a_missing_probe_key_is_not_a_pass_either(self, node):
        probe = good_probe()
        del probe["cuda_available"]
        assert not run(node, probe=probe)["passed"]

    def test_an_empty_probe_fails_everything_it_can(self, node):
        result = run(node, probe={})
        assert not result["passed"]
        assert len(result["failed"]) >= 6


class TestItIsTheRightMachine:

    def test_macos_is_refused(self, node):
        result = run(node, probe=good_probe(os_system="Darwin", wsl2=False))
        assert not result["passed"]
        assert "platform" in result["failed"]
        assert "develop" in result["checks"]["platform"]["detail"].lower() \
            or "execution node" in result["checks"]["platform"]["detail"]

    def test_bare_linux_without_wsl_is_refused(self, node):
        result = run(node, probe=good_probe(wsl2=False, wsl_evidence=None))
        assert not result["passed"]
        assert "platform" in result["failed"]

    def test_the_wrong_gpu_is_refused(self, node):
        result = run(node, probe=good_probe(gpu_name="NVIDIA GeForce GTX 1060"))
        assert not result["passed"]
        assert "gpu_model" in result["failed"]

    def test_too_little_vram_is_refused(self, node):
        result = run(node, probe=good_probe(vram_total_gb=11.5))
        assert not result["passed"]
        assert "vram" in result["failed"]

    def test_too_little_system_ram_is_refused(self, node):
        result = run(node, probe=good_probe(system_ram_gb=15.0))
        assert not result["passed"]
        assert "system_ram" in result["failed"]

    def test_the_declared_hardware_is_written_down(self):
        spec = gpu_node.NODE_SPEC
        assert "5070 Ti" in spec["gpu"]
        assert spec["vram_gb"] == 16
        assert spec["system_ram_gb"] == 32
        assert "Ryzen 5 7600" in spec["cpu"]


class TestNoSilentFallback:

    def test_cuda_unavailable_is_refused_not_downgraded_to_cpu(self, node):
        result = run(node, probe=good_probe(cuda_available=False))
        assert not result["passed"]
        assert "cuda" in result["failed"]

    def test_a_cpu_only_torch_build_is_refused(self, node):
        result = run(node, probe=good_probe(torch_cuda_build=None))
        assert not result["passed"]
        assert "torch_cuda_build" in result["failed"]

    def test_zero_devices_is_refused(self, node):
        assert not run(node, probe=good_probe(device_count=0))["passed"]

    def test_asking_for_any_device_but_cuda_is_refused(self, node):
        result = run(node, requested_device="cpu")
        assert not result["passed"]
        assert "requested_device" in result["failed"]
        result = run(node, requested_device="mps")
        assert not result["passed"]

    def test_a_changed_dtype_is_refused(self, node):
        result = run(node, requested_dtype="float16")
        assert not result["passed"]
        assert "dtype" in result["failed"]

    def test_quantization_is_refused(self, node):
        result = run(node, requested_quantization="4bit")
        assert not result["passed"]
        assert "quantization" in result["failed"]

    def test_the_declared_dtype_is_the_frozen_one(self, node):
        assert gpu_node.REQUIRED_DTYPE == LoraConfig_().dtype


class TestStaysOffline:

    def test_an_unpinned_offline_environment_is_refused(self, node):
        result = run(node, probe=good_probe(offline_env={}))
        assert not result["passed"]
        assert "offline" in result["failed"]

    def test_a_partially_pinned_environment_is_refused(self, node):
        result = run(node, probe=good_probe(
            offline_env={"HF_HUB_OFFLINE": "1"}))
        assert not result["passed"]
        assert "offline" in result["failed"]

    def test_offline_is_pinned_to_the_existing_definition(self):
        from src.training.longrun import PRODUCTION_OFFLINE_ENV

        assert gpu_node.REQUIRED_OFFLINE_ENV == PRODUCTION_OFFLINE_ENV


class TestThePackIsTheOnlyThingItRuns:

    def test_a_drifted_pack_is_refused(self, node):
        (node / "src" / "training" / "gates.py").write_text("GATE = 9\n")
        result = run(node)
        assert not result["passed"]
        assert "pack" in result["failed"]

    def test_an_extra_file_in_the_pack_is_refused(self, node):
        (node / "src" / "training" / "hand_edit.py").write_text("x = 1\n")
        assert not run(node)["passed"]

    def test_a_missing_pack_is_refused(self, node, tmp_path):
        assert not run(node, pack_dir=tmp_path / "nothing")["passed"]

    def test_data_that_does_not_match_the_pin_is_refused(self, node):
        from src.training import pack as packmod

        (node / packmod.REQUIRED_DATA[0]).write_text('{"sample_id":"z"}\n')
        result = run(node)
        assert not result["passed"]
        assert "pack" in result["failed"] or "data" in result["failed"]

    def test_a_pack_with_a_git_directory_is_refused(self, node):
        """A node that can commit is a node that can diverge."""
        (node / ".git").mkdir()
        (node / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        result = run(node)
        assert not result["passed"]
        assert "execution_node_only" in result["failed"]

    def test_the_refusal_explains_why_a_repository_is_the_problem(self, node):
        (node / ".git").mkdir()
        result = run(node)
        detail = result["checks"]["execution_node_only"]["detail"]
        assert "development" in detail or "diverge" in detail


class TestTheShapeMatchesTheExistingGate:
    """Report 15's preflight already decided what a gate result looks like."""

    def test_it_reports_passed_checks_and_failed(self, node):
        result = run(node)
        assert set(result) >= {"passed", "checks", "failed"}
        assert isinstance(result["failed"], list)

    def test_failed_lists_exactly_the_checks_that_did_not_pass(self, node):
        result = run(node, probe=good_probe(cuda_available=False,
                                            vram_total_gb=None))
        expected = sorted(k for k, v in result["checks"].items()
                          if not v["passed"])
        assert result["failed"] == expected


class TestProbeReadsRatherThanGuesses:

    def test_it_reports_none_when_torch_is_absent(self):
        probe = gpu_node.probe(torch_mod=None, proc_version="", meminfo="",
                               env={}, os_system="Linux")
        assert probe["cuda_available"] is None
        assert probe["gpu_name"] is None
        assert probe["vram_total_gb"] is None

    def test_it_detects_wsl2_from_the_kernel_string(self):
        p = gpu_node.probe(torch_mod=None, env={}, os_system="Linux",
                           proc_version="Linux version 5.15.0-microsoft-standard-WSL2",
                           meminfo="")
        assert p["wsl2"] is True
        assert p["wsl_evidence"]

    def test_a_plain_linux_kernel_is_not_wsl2(self):
        p = gpu_node.probe(torch_mod=None, env={}, os_system="Linux",
                           proc_version="Linux version 6.8.0-generic",
                           meminfo="")
        assert p["wsl2"] is False

    def test_an_unreadable_kernel_string_is_unknown_not_false(self):
        p = gpu_node.probe(torch_mod=None, env={}, os_system="Linux",
                           proc_version=None, meminfo="")
        assert p["wsl2"] is None

    def test_it_reads_total_ram_from_meminfo(self):
        p = gpu_node.probe(torch_mod=None, env={}, os_system="Linux",
                           proc_version="", meminfo="MemTotal:       32611248 kB\n")
        assert p["system_ram_gb"] == pytest.approx(31.1, abs=0.2)

    def test_a_torch_that_raises_is_reported_as_unreadable(self):
        class Exploding:
            class cuda:
                @staticmethod
                def is_available():
                    raise RuntimeError("driver missing")
            class version:
                cuda = None

        p = gpu_node.probe(torch_mod=Exploding, env={}, os_system="Linux",
                           proc_version="", meminfo="")
        assert p["cuda_available"] is None

    def test_the_probe_names_no_path_and_no_account(self):
        import json

        from src.training.longrun import leaked_identifiers

        p = gpu_node.probe(torch_mod=None, env={}, os_system="Linux",
                           proc_version="", meminfo="")
        assert leaked_identifiers(json.dumps(p, default=str)) == []


class TestHypothesesAreFrozenAndInert:

    def test_h1_is_the_design(self):
        cfg = hypotheses.config_for("H1")
        assert cfg.rank == 16
        assert cfg.alpha == 32
        assert cfg.learning_rate == 1e-4
        assert cfg.effective_batch == 8
        assert cfg.epochs == 1

    def test_h2_is_the_design(self):
        cfg = hypotheses.config_for("H2")
        assert cfg.rank == 32
        assert cfg.alpha == 16
        assert cfg.learning_rate == 2e-3
        assert cfg.effective_batch == 8
        assert cfg.epochs == 1

    def test_both_run_two_thousand_rows_for_one_epoch(self):
        assert hypotheses.ROWS == 2000
        for name in ("H1", "H2"):
            assert hypotheses.config_for(name).epochs == 1

    def test_they_differ_in_exactly_three_numbers(self):
        """Everything else controlled. That is what makes it a comparison."""
        assert hypotheses.differences() == {"rank", "alpha", "learning_rate"}

    def test_nothing_else_drifted(self):
        a, b = hypotheses.config_for("H1"), hypotheses.config_for("H2")
        for field in a.as_dict():
            if field in ("rank", "alpha", "learning_rate"):
                continue
            assert a.as_dict()[field] == b.as_dict()[field], field

    def test_they_use_the_existing_config_type_not_a_new_one(self):
        assert isinstance(hypotheses.config_for("H1"), LoraConfig_)

    def test_the_dtype_is_not_quietly_different(self):
        for name in ("H1", "H2"):
            cfg = hypotheses.config_for(name)
            assert cfg.dtype == "bfloat16"
            assert "4-bit" not in cfg.quantization or "not used" in cfg.quantization

    def unlock_kwargs(self, **over):
        out = {"runs": {}, "expected_pack_digest": "a" * 64,
               "expected_dependency_digest": "b" * 64,
               "allocator_config": "expandable_segments:True",
               "determinism": {"use_deterministic_algorithms": True,
                               "warn_only": False, "cudnn_benchmark": False,
                               "cudnn_deterministic": True,
                               "cublas_workspace_config": ":4096:8",
                               "tf32_matmul_allowed": False,
                               "tf32_cudnn_allowed": False, "seed": 0}}
        out.update(over)
        return out

    def test_running_one_is_refused_while_no_suite_has_been_supplied(self):
        with pytest.raises(hypotheses.HypothesisLocked) as exc:
            hypotheses.require_unlocked("H1", **self.unlock_kwargs())
        assert "gate_8" in str(exc.value)

    def test_a_partial_suite_does_not_unlock(self, tmp_path):
        """Five of six is not five-sixths of a proof."""
        runs = {role: tmp_path / role
                for role in hypotheses.REQUIRED_ROLES[:-1]}
        with pytest.raises(hypotheses.HypothesisLocked) as exc:
            hypotheses.require_unlocked("H1", **self.unlock_kwargs(runs=runs))
        assert "gate_500_uninterrupted_control" in str(exc.value)

    def test_three_bare_verdict_strings_are_not_accepted_at_all(self):
        """The old entry point. A claim anybody could type."""
        with pytest.raises(TypeError):
            hypotheses.require_unlocked("H1", gate_results={
                "gate_8": "passed", "gate_100": "passed",
                "gate_500": "passed"})
        assert not hasattr(hypotheses, "gate_problems")

    def test_the_lock_names_six_roles_not_three_gates(self):
        assert hypotheses.REQUIRED_ROLES == (
            "gate_8", "gate_100_r1", "gate_100_r2", "gate_100_r3",
            "gate_500_resumed", "gate_500_uninterrupted_control")
        assert "required_roles" in hypotheses.summary()

    def test_an_unknown_name_is_refused(self):
        with pytest.raises(KeyError):
            hypotheses.config_for("H3")

    def test_the_module_does_not_execute_anything_on_import(self):
        """Importing must not train. Asserted by what it does not import."""
        import inspect

        source = inspect.getsource(hypotheses)
        assert "import torch" not in source
        assert "from torch" not in source


class TestTheDigestMustBeCarriedNotRead:
    """The node cannot establish trust by reading the thing it is verifying.

    Everything in a manifest is computed from the manifest. Whoever can write
    it can make it agree with itself. The only value that means anything is
    the one the Mac printed, carried by a different route, and compared here.
    """

    def test_the_parameter_cannot_be_omitted(self, node):
        """Not a default. A trust check with a default is not a trust check."""
        import inspect

        sig = inspect.signature(gpu_node.preflight)
        param = sig.parameters["expected_pack_digest"]
        assert param.default is inspect.Parameter.empty
        assert param.kind is inspect.Parameter.KEYWORD_ONLY

    def test_a_missing_digest_fails_the_preflight(self, node):
        result = run(node, expected_pack_digest=None)
        assert not result["passed"]
        assert "expected_digest" in result["failed"]

    @pytest.mark.parametrize("bad", ["", "a" * 63, "A" * 64, "g" * 64, 12345])
    def test_a_malformed_digest_fails_the_preflight(self, node, bad):
        result = run(node, expected_pack_digest=bad)
        assert not result["passed"]
        assert "expected_digest" in result["failed"]

    def test_a_mismatched_digest_fails_the_preflight(self, node):
        result = run(node, expected_pack_digest="b" * 64)
        assert not result["passed"]
        assert "expected_digest" in result["failed"]

    def test_the_matching_digest_passes(self, node):
        assert run(node)["passed"]

    def test_a_re_signed_pack_still_fails_against_the_carried_digest(self, node):
        """Rewrite every file and re-sign the manifest completely."""
        import json

        from src.training import pack
        from src.training.session import manifest_digest, sha256_file

        carried = trusted_digest(node)
        body = json.loads((node / pack.MANIFEST_NAME).read_text())
        for rel, entry in body["files"].items():
            path = node / rel
            path.write_text(f"# replaced\n{path.name}\n", encoding="utf-8")
            entry["sha256"] = sha256_file(path)
            entry["bytes"] = path.stat().st_size
        body["files_digest"] = manifest_digest(body)
        body["pack_digest"] = pack.pack_digest(body)
        (node / pack.MANIFEST_NAME).unlink()
        (node / pack.MANIFEST_NAME).write_text(json.dumps(body, indent=2))

        result = run(node, expected_pack_digest=carried)
        assert not result["passed"]
        assert "expected_digest" in result["failed"]

    def test_the_refusal_names_no_personal_identifier(self, node):
        import json

        from src.training.longrun import leaked_identifiers

        result = run(node, expected_pack_digest="b" * 64)
        assert leaked_identifiers(json.dumps(result, default=str)) == []


class TestTheDependenciesAreResolvedBeforeAnythingIsSpent:
    """Gate 8 must not be where a missing tokenizer is discovered.

    Report 16 already learned this the expensive way: its first measured run
    passed the gate, spent the boot, spawned the child, and only then found
    the tokenizer could not be resolved. Same check, same implementation,
    moved in front of the GPU preflight rather than written again.
    """

    def test_it_reuses_report_16s_implementation(self):
        from src.training.longrun import dependency_preflight

        assert gpu_node.DEFAULT_DEPENDENCY_CHECKER is dependency_preflight

    def test_a_good_node_with_dependencies_present_passes(self, node):
        assert run(node)["passed"]

    def test_a_missing_dependency_fails_the_preflight(self, node):
        def missing():
            return {"ok": False,
                    "problems": ["AvaLovelace/BrickGPT@19737def7bfe is "
                                 "missing tokenizer.json in the local cache"],
                    "evidence": {}}

        result = run(node, dependency_checker=missing)
        assert not result["passed"]
        assert "dependencies" in result["failed"]
        assert "tokenizer.json" in result["checks"]["dependencies"]["detail"]

    def test_a_checker_that_raises_fails_closed(self, node):
        def explodes():
            raise ImportError("no module named huggingface_hub")

        result = run(node, dependency_checker=explodes)
        assert not result["passed"]
        assert "dependencies" in result["failed"]

    def test_a_checker_returning_nonsense_fails_closed(self, node):
        for bad in (None, {}, {"ok": "yes"}, [], "fine"):
            result = run(node, dependency_checker=lambda b=bad: b)
            assert not result["passed"], bad
            assert "dependencies" in result["failed"], bad

    def test_the_evidence_is_carried_but_never_the_cache_path(self, node):
        import json

        from src.training.longrun import leaked_identifiers

        evidence = {"repositories": [
            {"repo_id": "AvaLovelace/BrickGPT",
             "revision": "19737def7bfe",
             "files": [{"name": "tokenizer.json", "bytes": 9,
                        "sha256": "a" * 64}]}]}

        # The carried digest has to be the one *this* evidence produces, now
        # that the node binds contents rather than only presence.
        result = run(node, dependency_checker=ok_dependencies(evidence),
                     expected_dependency_digest=dep_digest(evidence))
        assert result["passed"], result["failed"]
        assert leaked_identifiers(json.dumps(result, default=str)) == []

    def test_it_declares_that_it_touched_no_network_and_no_device(self, node):
        result = run(node)
        detail = result["checks"]["dependencies"]["detail"]
        assert detail


class TestTheDependencyContentIsBoundToTheMacValue:
    """Resolving is not the same question as resolving to the right bytes.

    Until now the node asked only "is tokenizer.json present". A tokenizer.json
    of the correct name and the wrong contents answers yes -- and then trains
    against a vocabulary nobody compared. The Mac prints one digest over the
    whole portable evidence; the node recomputes it from its own cache and the
    two have to agree.
    """

    def test_the_parameter_cannot_be_omitted(self):
        import inspect

        param = inspect.signature(
            gpu_node.preflight).parameters["expected_dependency_digest"]
        assert param.default is inspect.Parameter.empty
        assert param.kind is inspect.Parameter.KEYWORD_ONLY

    def test_the_matching_digest_passes(self, node):
        result = run(node)
        assert result["passed"], result["failed"]
        assert result["checks"]["dependency_digest"]["passed"]

    def test_a_missing_digest_fails(self, node):
        result = run(node, expected_dependency_digest=None)
        assert not result["passed"]
        assert "dependency_digest" in result["failed"]

    @pytest.mark.parametrize("bad", ["", "a" * 63, "a" * 65, "A" * 64,
                                     "g" * 64, 12345, True, ["a" * 64],
                                     "sha256:" + "a" * 64])
    def test_a_malformed_digest_fails(self, node, bad):
        result = run(node, expected_dependency_digest=bad)
        assert not result["passed"]
        assert "dependency_digest" in result["failed"]

    def test_a_mismatched_digest_fails(self, node):
        result = run(node, expected_dependency_digest="b" * 64)
        assert not result["passed"]
        assert "dependency_digest" in result["failed"]

    def test_a_resolved_but_different_file_is_still_refused(self, node):
        """The case this whole check exists for.

        The checker is happy: ``ok=True``, no problems, every pinned file
        present. One of them is simply not the file the Mac had.
        """
        import copy

        drifted = copy.deepcopy(DEP_EVIDENCE)
        drifted["repositories"][0]["files"][0]["sha256"] = "9" * 64

        result = run(node, dependency_checker=ok_dependencies(drifted),
                     expected_dependency_digest=dep_digest())
        assert result["checks"]["dependencies"]["passed"], \
            "the checker itself reported success, which is the premise"
        assert not result["checks"]["dependency_digest"]["passed"]
        assert not result["passed"]
        assert "dependency_digest" in result["failed"]

    def test_a_drifted_byte_count_is_refused_too(self, node):
        import copy

        drifted = copy.deepcopy(DEP_EVIDENCE)
        drifted["repositories"][0]["files"][0]["bytes"] = 999
        result = run(node, dependency_checker=ok_dependencies(drifted),
                     expected_dependency_digest=dep_digest())
        assert not result["passed"]
        assert "dependency_digest" in result["failed"]

    def test_a_drifted_instruction_pool_is_refused(self, node):
        import copy

        drifted = copy.deepcopy(DEP_EVIDENCE)
        drifted["instruction_pool"]["sha256"] = "9" * 64
        result = run(node, dependency_checker=ok_dependencies(drifted),
                     expected_dependency_digest=dep_digest())
        assert not result["passed"]

    def test_the_result_shows_what_the_node_recomputed(self, node):
        """Requirement 3: the operator has to be able to compare by eye."""
        result = run(node, expected_dependency_digest="b" * 64)
        assert result["dependency_digest"] == dep_digest()
        assert result["checks"]["dependency_digest"]["detail"]
        assert dep_digest()[:16] in result["checks"]["dependency_digest"]["detail"]

    def test_a_failing_checker_does_not_pretend_to_have_a_digest(self, node):
        def broken():
            raise ImportError("no module named huggingface_hub")

        result = run(node, dependency_checker=broken)
        assert not result["passed"]
        assert "dependencies" in result["failed"]
        assert "dependency_digest" in result["failed"]

    def test_the_refusal_names_no_personal_identifier(self, node):
        import json

        from src.training.longrun import leaked_identifiers

        result = run(node, expected_dependency_digest="b" * 64)
        assert leaked_identifiers(json.dumps(result, default=str)) == []

    def test_the_two_checks_answer_different_questions(self, node):
        """Kept separate on purpose: "present" and "the same" are not one fact."""
        result = run(node)
        assert "dependencies" in result["checks"]
        assert "dependency_digest" in result["checks"]


class TestAllocatorConfigIsRequiredNotSuppliedByUs:
    """``expandable_segments`` is the difference between 15.477 and 7.635 GB.

    It has to be in the environment *before* the process starts, because a
    caching allocator reads its configuration once and a program that sets it
    later has already lost. That makes it provenance: a run's numbers mean
    nothing without knowing which allocator produced them, and a run that set
    it for itself could not be told from one that inherited it.

    So: required, checked before the model loads, and never written by us.
    """

    def test_the_primary_and_alias_names_are_both_known(self):
        assert gpu_node.ALLOC_ENV_PRIMARY == "PYTORCH_ALLOC_CONF"
        assert gpu_node.ALLOC_ENV_ALIAS == "PYTORCH_CUDA_ALLOC_CONF"

    @pytest.mark.parametrize("raw,expected", [
        ("expandable_segments:True", "expandable_segments:True"),
        ("expandable_segments:true", "expandable_segments:True"),
        ("  expandable_segments : True  ", "expandable_segments:True"),
        ("max_split_size_mb:128,expandable_segments:True",
         "expandable_segments:True,max_split_size_mb:128"),
        ("expandable_segments:1", "expandable_segments:True"),
    ])
    def test_normalisation_is_canonical(self, raw, expected):
        assert gpu_node.normalize_alloc_conf(raw) == expected

    def test_a_good_environment_has_no_problems(self):
        env = {gpu_node.ALLOC_ENV_PRIMARY: "expandable_segments:True"}
        config, problems = gpu_node.allocator_config_from_env(env)
        assert problems == []
        assert config == "expandable_segments:True"

    def test_the_alias_alone_is_accepted(self):
        env = {gpu_node.ALLOC_ENV_ALIAS: "expandable_segments:True"}
        config, problems = gpu_node.allocator_config_from_env(env)
        assert problems == []
        assert config == "expandable_segments:True"

    def test_both_set_and_agreeing_is_accepted(self):
        env = {gpu_node.ALLOC_ENV_PRIMARY: "expandable_segments:True",
               gpu_node.ALLOC_ENV_ALIAS: "expandable_segments:true"}
        config, problems = gpu_node.allocator_config_from_env(env)
        assert problems == []

    def test_an_alias_conflict_fails_closed(self):
        env = {gpu_node.ALLOC_ENV_PRIMARY: "expandable_segments:True",
               gpu_node.ALLOC_ENV_ALIAS: "expandable_segments:False"}
        config, problems = gpu_node.allocator_config_from_env(env)
        assert problems
        assert any("conflict" in p or "disagree" in p for p in problems)

    def test_an_unset_environment_fails_closed(self):
        config, problems = gpu_node.allocator_config_from_env({})
        assert problems
        assert config is None

    @pytest.mark.parametrize("value", ["expandable_segments:False",
                                       "expandable_segments:false",
                                       "expandable_segments:0",
                                       "max_split_size_mb:128"])
    def test_a_config_without_expandable_segments_true_fails_closed(self, value):
        _, problems = gpu_node.allocator_config_from_env(
            {gpu_node.ALLOC_ENV_PRIMARY: value})
        assert problems, value

    def test_the_preflight_check_exists_and_fails_closed(self, node):
        result = run(node, probe=good_probe(alloc_env={}))
        assert not result["passed"]
        assert "allocator_config" in result["failed"]

    def test_the_preflight_reports_the_normalised_config(self, node):
        result = run(node)
        assert result["allocator_config"] == "expandable_segments:True"
        assert result["checks"]["allocator_config"]["passed"]

    def test_the_backend_is_recorded_but_cannot_substitute(self, node):
        """"native" is a fact about the allocator, not the configuration.

        Reporting backend=native and calling that provenance would hide
        whether expandable segments were on -- and those two runs differ by
        8 GB of reserved memory.
        """
        result = run(node, probe=good_probe(
            alloc_env={}, allocator_backend="native"))
        assert not result["passed"], \
            "a recorded backend was allowed to stand in for the config"
        assert "allocator_config" in result["failed"]

    def test_no_packed_module_sets_the_allocator_environment(self):
        """We check it. We never supply it."""
        from src.training import pack

        for entry in pack.manifest(ROOT)["include"]:
            rel = entry["path"]
            if not rel.endswith(".py") or rel.startswith("tests/"):
                continue
            text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
            for name in (gpu_node.ALLOC_ENV_PRIMARY, gpu_node.ALLOC_ENV_ALIAS):
                for setter in (f'environ["{name}"] =', f"environ['{name}'] =",
                               f'setdefault("{name}"', f"setdefault('{name}'"):
                    assert setter not in text, f"{rel} sets {name}"


class TestDeterminismIsStrictOrItIsNothing:
    """warn_only turns "this operation is not deterministic" into a log line.

    The whole point of the mode is to find out whether repeated runs agree; a
    mode that silently falls back to a non-deterministic kernel and says so in
    passing answers the question wrong and looks like it answered it right.
    """

    def test_the_required_cublas_workspace_values(self):
        assert ":4096:8" in gpu_node.CUBLAS_WORKSPACE_VALUES

    def test_a_good_environment_passes(self):
        problems = gpu_node.determinism_env_problems(
            {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
        assert problems == []

    @pytest.mark.parametrize("env", [
        {}, {"CUBLAS_WORKSPACE_CONFIG": ""},
        {"CUBLAS_WORKSPACE_CONFIG": ":2:2"},
        {"CUBLAS_WORKSPACE_CONFIG": "4096:8"},
    ])
    def test_a_bad_or_missing_workspace_config_fails_closed(self, env):
        assert gpu_node.determinism_env_problems(env)

    def test_settings_are_reported_from_torch_not_assumed(self):
        class T:
            @staticmethod
            def use_deterministic_algorithms(flag, warn_only=False):
                T.called = (flag, warn_only)

            @staticmethod
            def are_deterministic_algorithms_enabled():
                return True

            @staticmethod
            def is_deterministic_algorithms_warn_only_enabled():
                return False

            @staticmethod
            def manual_seed(s):
                T.seed = s

            class backends:
                class cudnn:
                    benchmark = True
                    deterministic = False
                    allow_tf32 = True

                class cuda:
                    class matmul:
                        allow_tf32 = True

        got = gpu_node.apply_determinism(T, seed=7,
                                         env={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
        assert T.called == (True, False), "warn_only must never be requested"
        assert got["use_deterministic_algorithms"] is True
        assert got["warn_only"] is False
        assert got["cudnn_benchmark"] is False
        assert got["cudnn_deterministic"] is True
        assert got["seed"] == 7
        assert got["cublas_workspace_config"] == ":4096:8"
        assert "tf32_matmul_allowed" in got
        assert "tf32_cudnn_allowed" in got
        assert T.backends.cudnn.benchmark is False
        assert T.backends.cudnn.deterministic is True

    def test_a_torch_that_reports_warn_only_fails_closed(self):
        class T:
            @staticmethod
            def use_deterministic_algorithms(flag, warn_only=False):
                return None

            @staticmethod
            def are_deterministic_algorithms_enabled():
                return True

            @staticmethod
            def is_deterministic_algorithms_warn_only_enabled():
                return True          # the thing that must never be tolerated

            @staticmethod
            def manual_seed(s):
                return None

            class backends:
                class cudnn:
                    benchmark = True
                    deterministic = False
                    allow_tf32 = False

                class cuda:
                    class matmul:
                        allow_tf32 = False

        with pytest.raises(RuntimeError) as exc:
            gpu_node.apply_determinism(T, seed=0,
                                       env={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
        assert "warn_only" in str(exc.value)

    def test_a_torch_that_will_not_enable_it_fails_closed(self):
        class T:
            @staticmethod
            def use_deterministic_algorithms(flag, warn_only=False):
                return None

            @staticmethod
            def are_deterministic_algorithms_enabled():
                return False          # asked for, not in effect

            @staticmethod
            def is_deterministic_algorithms_warn_only_enabled():
                return False

            @staticmethod
            def manual_seed(s):
                return None

            class backends:
                class cudnn:
                    benchmark = True
                    deterministic = False
                    allow_tf32 = False

                class cuda:
                    class matmul:
                        allow_tf32 = False

        with pytest.raises(RuntimeError):
            gpu_node.apply_determinism(T, seed=0,
                                       env={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})

    def test_a_bad_workspace_env_refuses_before_touching_torch(self):
        class T:
            @staticmethod
            def use_deterministic_algorithms(*a, **k):
                raise AssertionError("must not be reached")

        with pytest.raises(RuntimeError):
            gpu_node.apply_determinism(T, seed=0, env={})
