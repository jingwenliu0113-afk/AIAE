"""Configuration, augmentation, device tuning and the checkpoint contract.

No weights are downloaded and no backbone is built: ``build_model`` is replaced
by a small stand-in wherever a model object is needed. That keeps this suite
self-contained -- it means the same thing on the CUDA node as it does here, and
it introduces no new skip category into the public snapshot's suite.

What is actually pinned here is the part that makes a fitted checkpoint
trustworthy: it cannot be written without the digests, the log and the seed, it
cannot be loaded if its weights or its class order have moved, and the
augmentation it was fitted under is reproducible from the configuration alone.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.vision import model as model_module
from src.vision.classes import CLASS_ORDER, N_CLASSES
from src.vision.model import (BATCH_DEFAULT, DeviceTuning, ModelError,
                              _late_stage_prefixes, freeze_parameters,
                              TrainConfig, Augmentation, apply_tuning, augment,
                              autocast_context, code_digest, device_report,
                              dependency_versions, preprocessing_record,
                              read_manifest, resolve_device, save, softmax,
                              suggested_batch_size, suggested_loader_workers,
                              tuning_for)
from src.vision.model_ids import VISION_MANIFEST


class Stand(torch.nn.Module):
    """A model-shaped object with the one attribute the code reads."""

    def __init__(self, labels=N_CLASSES):
        super().__init__()
        self.classifier = torch.nn.Linear(4, labels)
        self.config = type("C", (), {"num_labels": labels})()

    def forward(self, pixel_values=None):
        flat = pixel_values.reshape(pixel_values.shape[0], -1)[:, :4]
        return type("O", (), {"logits": self.classifier(flat)})()


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

class TestConfigurationIsChecked:
    def test_the_defaults_are_valid(self):
        TrainConfig()

    @pytest.mark.parametrize("kw", [
        {"epochs": 0}, {"batch_size": 0}, {"learning_rate": 0},
        {"learning_rate": 1.0}, {"weight_decay": -0.1}, {"seed": -1},
        {"label_smoothing": 0.5}, {"label_smoothing": -0.01},
        {"trainable_stages": -1},
    ])
    def test_a_bad_value_is_refused(self, kw):
        with pytest.raises(ModelError):
            TrainConfig(**kw)

    def test_it_serialises_everything_a_rerun_needs(self):
        body = TrainConfig().as_dict()
        for key in ("epochs", "batch_size", "learning_rate", "weight_decay",
                    "seed", "freeze_backbone", "trainable_stages",
                    "label_smoothing", "augmentation"):
            assert key in body


class TestAugmentationIsReproducible:
    def _image(self):
        rng = np.random.default_rng(7)
        return (rng.random((48, 64, 3)) * 255).astype(np.uint8)

    def test_the_same_seed_epoch_and_index_give_the_same_pixels(self):
        image = self._image()
        spec = Augmentation()
        first = augment(image, spec, seed=3, epoch=2, index=11)
        again = augment(image, spec, seed=3, epoch=2, index=11)
        assert np.array_equal(first, again)

    def test_a_different_epoch_gives_different_pixels(self):
        image = self._image()
        spec = Augmentation()
        assert not np.array_equal(
            augment(image, spec, seed=3, epoch=2, index=11),
            augment(image, spec, seed=3, epoch=3, index=11))

    def test_a_different_index_gives_different_pixels(self):
        image = self._image()
        spec = Augmentation()
        assert not np.array_equal(
            augment(image, spec, seed=3, epoch=2, index=11),
            augment(image, spec, seed=3, epoch=2, index=12))

    def test_disabled_augmentation_returns_the_image_untouched(self):
        image = self._image()
        out = augment(image, Augmentation(enabled=False), seed=1, epoch=1,
                      index=1)
        assert out is image

    def test_the_pixel_range_is_preserved(self):
        image = self._image()
        out = augment(image, Augmentation(), seed=5, epoch=1, index=2)
        assert out.min() >= 0.0 and out.max() <= 255.0

    def test_only_quarter_turns_are_used(self):
        """A free rotation would teach the head that a 1x4 seen end-on is
        a 1x1; the class is invariant to quarter turns and to nothing more."""
        image = np.zeros((40, 80, 3), dtype=np.uint8)
        image[10:30, 20:60] = 200
        shapes = {augment(image, Augmentation(brightness=0, contrast=0),
                          seed=s, epoch=1, index=1).shape
                  for s in range(12)}
        assert shapes <= {(40, 80, 3), (80, 40, 3)}


# --------------------------------------------------------------------------
# device tuning
# --------------------------------------------------------------------------

class TestTuningIsDerivedFromTheDevice:
    def test_the_cpu_gets_no_autocast_and_no_channels_last(self):
        tuning = tuning_for("cpu")
        assert tuning.autocast_dtype is None
        assert tuning.channels_last is False
        assert tuning.tf32 is False
        assert tuning.cudnn_benchmark is False

    def test_mps_gets_no_autocast_either(self):
        assert tuning_for("mps").autocast_dtype is None

    def test_deterministic_turns_everything_off(self):
        tuning = tuning_for("cpu", deterministic=True)
        assert tuning.deterministic is True
        assert tuning.autocast_dtype is None
        assert tuning.cudnn_benchmark is False

    def test_an_unknown_device_is_refused(self):
        with pytest.raises(ModelError, match="not one of"):
            tuning_for("tpu")

    def test_the_loader_worker_count_leaves_the_machine_a_core(self):
        import os

        workers = suggested_loader_workers()
        assert 2 <= workers <= 8
        assert workers <= max(2, (os.cpu_count() or 4))

    def test_the_tuning_records_that_it_changes_the_arithmetic(self):
        note = tuning_for("cpu").as_dict()["note"]
        assert "not bit-identical" in note

    def test_applying_reports_what_was_set_not_what_was_asked(self):
        applied = apply_tuning(DeviceTuning(loader_workers=3), "cpu")
        assert applied["device"] == "cpu"
        assert applied["loader_workers"] == 3

    def test_autocast_off_is_a_usable_context_manager(self):
        with autocast_context(DeviceTuning(), "cpu"):
            pass

    def test_an_unknown_autocast_dtype_is_refused(self):
        """Refused before any device is touched, so this holds everywhere."""
        with pytest.raises(ModelError, match="not a torch dtype"):
            autocast_context(DeviceTuning(autocast_dtype="bfloat9"), "cuda")

    def test_the_batch_size_is_the_default_off_cuda(self):
        assert suggested_batch_size("cpu") == BATCH_DEFAULT
        assert suggested_batch_size("mps") == BATCH_DEFAULT

    def test_the_device_report_names_the_device(self):
        report = device_report("cpu")
        assert report["device"] == "cpu"
        assert report["torch"] == torch.__version__
        assert "name" in report

    def test_resolving_an_unknown_device_is_refused(self):
        with pytest.raises(ModelError, match="not one of"):
            resolve_device("tpu")

    def test_asking_for_cuda_where_there_is_none_is_refused(self,
                                                             monkeypatch):
        """Forced rather than conditional, so it runs on every machine."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(ModelError, match="cuda was asked for"):
            resolve_device("cuda")

    def test_asking_for_mps_where_there_is_none_is_refused(self, monkeypatch):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        with pytest.raises(ModelError, match="mps was asked for"):
            resolve_device("mps")

    def test_with_nothing_available_the_cpu_is_chosen(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert resolve_device() == "cpu"


# --------------------------------------------------------------------------
# softmax
# --------------------------------------------------------------------------

class TestSoftmax:
    def test_rows_sum_to_one(self):
        values = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
        assert np.allclose(values.sum(axis=1), 1.0)

    def test_it_is_stable_for_large_logits(self):
        values = softmax(np.array([[1000.0, 999.0]]))
        assert np.isfinite(values).all()
        assert abs(float(values.sum()) - 1.0) < 1e-6


# --------------------------------------------------------------------------
# the checkpoint contract
# --------------------------------------------------------------------------

def write(tmp_path, **overrides):
    kw = dict(config=TrainConfig(), epoch_log=[{"epoch": 1,
                                                "validation_loss": 1.0}],
              selected_epoch=1, selection_criterion="lowest validation loss",
              data_manifest_sha256="a" * 64, device="cpu")
    kw.update(overrides)
    return save(Stand(), tmp_path, **kw)


class TestSavingRequiresTheEvidence:
    def test_a_complete_manifest_is_written(self, tmp_path):
        manifest = write(tmp_path)
        for key in model_module.REQUIRED_MANIFEST_KEYS:
            assert key in manifest
        assert (tmp_path / VISION_MANIFEST).is_file()
        assert (tmp_path / manifest["weights"]["file"]).is_file()

    def test_a_short_data_digest_is_refused(self, tmp_path):
        with pytest.raises(ModelError, match="digest of the data manifest"):
            write(tmp_path, data_manifest_sha256="abc")

    def test_an_empty_epoch_log_is_refused(self, tmp_path):
        with pytest.raises(ModelError, match="per-epoch log"):
            write(tmp_path, epoch_log=[])

    def test_the_weights_digest_matches_the_file(self, tmp_path):
        manifest = write(tmp_path)
        import hashlib

        payload = (tmp_path / manifest["weights"]["file"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == \
            manifest["weights"]["sha256"]

    def test_the_class_order_is_recorded(self, tmp_path):
        assert write(tmp_path)["class_order"] == list(CLASS_ORDER)

    def test_the_preprocessing_is_recorded(self, tmp_path):
        recorded = write(tmp_path)["preprocessing"]
        assert recorded == preprocessing_record()
        assert "bilinear" in recorded["resample"]

    def test_the_tuning_and_device_are_recorded(self, tmp_path):
        manifest = write(tmp_path, tuning=tuning_for("cpu"),
                         applied_tuning={"device": "cpu"})
        assert manifest["tuning"]["applied"]["device"] == "cpu"
        assert manifest["device_report"]["device"] == "cpu"

    def test_the_dependency_versions_are_recorded(self, tmp_path):
        recorded = write(tmp_path)["dependencies"]
        assert recorded == dependency_versions()
        assert recorded["torch"] != "absent"

    def test_the_boundary_note_separates_it_from_the_generation_track(
            self, tmp_path):
        note = write(tmp_path)["boundary"]
        assert "final_H2" in note and "Phase 2" in note


class TestReadingAManifestBack:
    def test_a_written_manifest_reads_back(self, tmp_path):
        written = write(tmp_path)
        assert read_manifest(tmp_path)["weights"]["sha256"] == \
            written["weights"]["sha256"]

    def test_a_missing_manifest_is_refused(self, tmp_path):
        with pytest.raises(ModelError, match="no brickagain_vision_manifest"):
            read_manifest(tmp_path)

    def test_a_manifest_that_is_not_one_is_refused(self, tmp_path):
        (tmp_path / VISION_MANIFEST).write_text(json.dumps({"kind": "other"}),
                                                encoding="utf-8")
        with pytest.raises(ModelError, match="declare itself"):
            read_manifest(tmp_path)

    def test_invalid_json_is_refused(self, tmp_path):
        (tmp_path / VISION_MANIFEST).write_text("{", encoding="utf-8")
        with pytest.raises(ModelError, match="not valid JSON"):
            read_manifest(tmp_path)

    def test_a_missing_required_key_is_refused(self, tmp_path):
        write(tmp_path)
        body = json.loads((tmp_path / VISION_MANIFEST).read_text("utf-8"))
        del body["seed"]
        (tmp_path / VISION_MANIFEST).write_text(json.dumps(body),
                                                encoding="utf-8")
        with pytest.raises(ModelError, match="is missing"):
            read_manifest(tmp_path)

    def test_a_different_class_order_is_refused(self, tmp_path):
        write(tmp_path)
        body = json.loads((tmp_path / VISION_MANIFEST).read_text("utf-8"))
        body["class_order"] = list(reversed(CLASS_ORDER))
        (tmp_path / VISION_MANIFEST).write_text(json.dumps(body),
                                                encoding="utf-8")
        with pytest.raises(ModelError, match="mislabelled"):
            read_manifest(tmp_path)


class TestTheCodeDigest:
    def test_it_covers_the_files_that_decide_what_a_checkpoint_means(self):
        assert len(code_digest()) == 64

    def test_it_is_stable_across_calls(self):
        assert code_digest() == code_digest()

    def test_a_missing_file_is_refused(self):
        with pytest.raises(ModelError, match="code digest needs"):
            code_digest(("src/vision/not_a_file.py",))

    def test_two_different_file_sets_give_different_digests(self):
        one = code_digest(("src/vision/classes.py",))
        two = code_digest(("src/vision/schema.py",))
        assert one != two


# --------------------------------------------------------------------------
# which parameters actually train
# --------------------------------------------------------------------------

class Stages(torch.nn.Module):
    """A backbone shaped like the real one: ``resnet.encoder.stages.N``.

    The depth is the point. The first version of the stage finder required a
    three-segment path and this shape has four, so it matched nothing and
    every run trained the classifier head alone whatever the configuration
    asked for -- a linear probe wearing a fine-tune's label, with nothing
    raising.
    """

    def __init__(self, stages=4, labels=N_CLASSES):
        super().__init__()
        self.resnet = torch.nn.Module()
        self.resnet.encoder = torch.nn.Module()
        self.resnet.encoder.stages = torch.nn.ModuleList(
            [torch.nn.Linear(8, 8) for _ in range(stages)])
        self.classifier = torch.nn.Linear(8, labels)
        self.config = type("C", (), {"num_labels": labels})()


class Flat(torch.nn.Module):
    """A backbone with no stages at all, to exercise the refusal."""

    def __init__(self, labels=N_CLASSES):
        super().__init__()
        self.trunk = torch.nn.Linear(8, 8)
        self.classifier = torch.nn.Linear(8, labels)
        self.config = type("C", (), {"num_labels": labels})()


class TestTheStageFinderIsNotDepthDependent:
    def test_it_finds_stages_four_segments_deep(self):
        found = _late_stage_prefixes(Stages(), 1)
        assert found == ("resnet.encoder.stages.3",)

    def test_two_stages_are_the_last_two_in_order(self):
        assert _late_stage_prefixes(Stages(), 2) == (
            "resnet.encoder.stages.2", "resnet.encoder.stages.3")

    def test_asking_for_none_finds_none(self):
        assert _late_stage_prefixes(Stages(), 0) == ()

    def test_a_backbone_with_no_stages_finds_none(self):
        assert _late_stage_prefixes(Flat(), 1) == ()


class TestFreezing:
    def test_head_only_leaves_just_the_head_trainable(self):
        model = Stages()
        report = freeze_parameters(model, TrainConfig(trainable_stages=0))
        trainable = {name for name, p in model.named_parameters()
                     if p.requires_grad}
        assert trainable == {"classifier.weight", "classifier.bias"}
        assert report["trainable_stages_requested"] == 0

    def test_one_stage_unfreezes_that_stage_as_well(self):
        model = Stages()
        report = freeze_parameters(model, TrainConfig(trainable_stages=1))
        trainable = {name for name, p in model.named_parameters()
                     if p.requires_grad}
        assert "resnet.encoder.stages.3.weight" in trainable
        assert "resnet.encoder.stages.0.weight" not in trainable
        assert "resnet.encoder.stages.3" in report["trainable_prefixes"]
        assert report["trainable_parameters"] > 0
        assert report["frozen_parameters"] > 0

    def test_two_stages_unfreeze_more_than_one(self):
        one = freeze_parameters(Stages(), TrainConfig(trainable_stages=1))
        two = freeze_parameters(Stages(), TrainConfig(trainable_stages=2))
        assert two["trainable_parameters"] > one["trainable_parameters"]

    def test_asking_for_a_stage_and_finding_none_is_refused(self):
        """Rather than quietly training the head alone."""
        with pytest.raises(ModelError, match="linear probe and a fine-tune"):
            freeze_parameters(Flat(), TrainConfig(trainable_stages=1))

    def test_a_full_backbone_fit_trains_everything(self):
        model = Stages()
        freeze_parameters(model, TrainConfig(freeze_backbone=False))
        assert all(p.requires_grad for p in model.parameters())

    def test_head_only_is_still_allowed_when_asked_for_explicitly(self):
        freeze_parameters(Flat(), TrainConfig(trainable_stages=0))


# ---------------------------------------------------------------------------
# Round 49: the selection record, and what it refuses to claim
#
# The round reported that this checkpoint was chosen from among several
# configurations by a pre-frozen criterion.  Only one configuration's
# artefacts were ever returned to this machine, so nothing here can check
# that; the record withdraws it in writing and refuses to be edited back.
# What it *does* check is everything the returned artefacts can support, and
# it re-derives rather than repeats.
# ---------------------------------------------------------------------------

import hashlib

from src.vision import selection as selection_module
from src.vision.selection import (EPOCH_CRITERION, RECORD_FILE, RECORD_KIND,
                                  SelectionError, best_epoch, split_log)


def a_checkpoint(tmp_path, *, log=None, selected=2, weights=b"weights"):
    """A minimal returned-artefact pair, in the shape the node produced."""
    directory = tmp_path / "classifier"
    directory.mkdir()
    (directory / "vision_head.pt").write_bytes(weights)
    epoch_log = log if log is not None else [
        {"epoch": 1, "validation_loss": 0.9, "train_loss": 1.1},
        {"epoch": 2, "validation_loss": 0.4, "train_loss": 0.8},
        {"epoch": 3, "validation_loss": 0.5, "train_loss": 0.7},
        {"applied_tuning": {"device": "cuda"},
         "frozen_parameters": {"trainable_stages_requested": 2,
                               "trainable_parameters": 10,
                               "trainable_prefixes": ["classifier"]}},
    ]
    manifest = {
        "kind": "brickagain.vision_classifier",
        "class_order": list(CLASS_ORDER),
        "config": {"epochs": 3, "trainable_stages": 2, "seed": 0},
        "seed": 0,
        "selected_epoch": selected,
        "selection_criterion": "lowest validation loss",
        "epoch_log": epoch_log,
        "code_sha256": code_digest(),
        "data_manifest_sha256": "d" * 64,
        "split_manifest_sha256": "s" * 64,
        "device": "cuda",
        "device_report": {"name": "a card"},
        "dependencies": {"torch": "2.11.0+cu130"},
        "tuning": {"autocast_dtype": "bfloat16"},
        "backbone": {"repo": "microsoft/resnet-18"},
        "weights": {"file": "vision_head.pt", "bytes": len(weights),
                    "sha256": hashlib.sha256(weights).hexdigest()},
    }
    (directory / model_module.VISION_MANIFEST).write_text(
        json.dumps(manifest), encoding="utf-8")
    (directory / "run_summary.json").write_text(json.dumps({
        "weights_sha256": manifest["weights"]["sha256"],
        "data_manifest_sha256": "d" * 64,
        "split_manifest_sha256": "s" * 64,
        "selected_epoch": selected,
    }), encoding="utf-8")
    return directory


class TestTheSelectionRecordChecksWhatItCan:
    def test_a_sound_pair_produces_a_record_with_no_problems(self, tmp_path):
        body = selection_module.build(a_checkpoint(tmp_path))
        assert body["kind"] == RECORD_KIND
        assert body["problems"] == []
        assert all(body["checks"].values())

    def test_the_epoch_is_re_derived_not_copied(self, tmp_path):
        """A manifest that names the wrong epoch fails rather than passes."""
        body = selection_module.build(a_checkpoint(tmp_path, selected=3))
        assert body["epoch_selection"]["selected_epoch_stated"] == 3
        assert body["epoch_selection"]["selected_epoch_rederived"] == 2
        assert "selected_epoch_is_the_argmin_of_the_epoch_log" in \
            body["problems"]

    def test_a_moved_weight_file_is_caught(self, tmp_path):
        directory = a_checkpoint(tmp_path)
        (directory / "vision_head.pt").write_bytes(b"something else")
        body = selection_module.build(directory)
        assert "weights_digest_matches_manifest" in body["problems"]

    def test_a_summary_that_disagrees_with_the_manifest_is_caught(self,
                                                                  tmp_path):
        directory = a_checkpoint(tmp_path)
        summary = json.loads(
            (directory / "run_summary.json").read_text(encoding="utf-8"))
        summary["split_manifest_sha256"] = "z" * 64
        (directory / "run_summary.json").write_text(json.dumps(summary),
                                                    encoding="utf-8")
        body = selection_module.build(directory)
        assert "summary_and_manifest_agree_on_the_split" in body["problems"]

    def test_a_code_digest_from_another_tree_is_caught(self, tmp_path):
        body = selection_module.build(a_checkpoint(tmp_path),
                                      code_digest="0" * 64)
        assert "code_digest_matches_this_tree" in body["problems"]

    def test_the_trainable_prefixes_travel_in_the_record(self, tmp_path):
        body = selection_module.build(a_checkpoint(tmp_path))
        assert body["trainable_parameters"][
            "trainable_stages_requested"] == 2
        assert body["trainable_parameters"]["trainable_prefixes"] == [
            "classifier"]

    def test_the_criterion_is_stated_and_matches_the_derivation(self,
                                                                tmp_path):
        body = selection_module.build(a_checkpoint(tmp_path))
        assert body["epoch_selection"]["criterion"] == EPOCH_CRITERION
        losses = body["epoch_selection"]["validation_losses"]
        best = min(losses, key=lambda key: losses[key])
        assert int(best) == body["epoch_selection"]["selected_epoch_rederived"]


class TestTheCrossConfigurationClaimIsWithdrawn:
    def test_the_record_says_so_in_a_field_not_only_in_prose(self, tmp_path):
        body = selection_module.build(a_checkpoint(tmp_path))
        assert body["cross_configuration_selection"] is None
        assert "withdrawn" in body["cross_configuration_reason"]
        assert "were not\nreturned" in body["cross_configuration_reason"] or \
            "were not returned" in body["cross_configuration_reason"]

    def test_a_record_that_fills_the_field_in_is_refused(self, tmp_path):
        directory = a_checkpoint(tmp_path)
        body = selection_module.build(directory)
        body["cross_configuration_selection"] = {
            "winner": "stage2_e16", "beat": ["head_only", "stage1"]}
        problems = selection_module.verify(body, directory)
        assert any("cross_configuration_selection is filled in" in problem
                   for problem in problems)

    def test_verify_passes_on_the_record_build_produces(self, tmp_path):
        directory = a_checkpoint(tmp_path)
        assert selection_module.verify(
            selection_module.build(directory), directory) == []

    def test_verify_catches_an_edited_record(self, tmp_path):
        directory = a_checkpoint(tmp_path)
        body = selection_module.build(directory)
        body["configuration"]["trainable_stages"] = 3
        assert any("configuration" in problem for problem in
                   selection_module.verify(body, directory))

    def test_a_file_that_is_not_a_record_is_refused(self, tmp_path):
        assert selection_module.verify({"kind": "something"}, tmp_path) == [
            "the file is not a BrickAgain vision selection record"]


class TestWritingTheRecord:
    def test_it_writes_beside_the_checkpoint_and_returns_its_digest(
            self, tmp_path):
        directory = a_checkpoint(tmp_path)
        path, digest = selection_module.write(directory)
        assert path.name == RECORD_FILE
        assert digest == hashlib.sha256(path.read_bytes()).hexdigest()

    def test_it_refuses_to_write_a_record_whose_checks_fail(self, tmp_path):
        directory = a_checkpoint(tmp_path, selected=3)
        with pytest.raises(SelectionError, match="do not check out"):
            selection_module.write(directory)
        assert not (directory / RECORD_FILE).exists()

    def test_a_missing_manifest_is_named(self, tmp_path):
        with pytest.raises(SelectionError, match="is missing"):
            selection_module.build(tmp_path / "nothing")


class TestTheEpochLogHelpers:
    def test_provenance_rows_are_separated_from_epochs(self):
        epochs, provenance = split_log([
            {"epoch": 1, "validation_loss": 0.5},
            {"applied_tuning": {}},
        ])
        assert len(epochs) == 1 and len(provenance) == 1

    def test_an_empty_log_is_refused(self):
        with pytest.raises(SelectionError, match="empty"):
            best_epoch([])

    def test_an_epoch_with_no_validation_loss_is_refused(self):
        with pytest.raises(SelectionError, match="no validation_loss"):
            best_epoch([{"epoch": 1, "train_loss": 0.2}])

    @pytest.mark.parametrize("loss", [float("nan"), float("inf"),
                                      float("-inf")])
    def test_a_non_finite_loss_is_refused(self, loss):
        with pytest.raises(SelectionError, match="not finite"):
            best_epoch([{"epoch": 1, "validation_loss": loss}])

    def test_an_exact_tie_keeps_the_earlier_epoch(self):
        assert best_epoch([{"epoch": 1, "validation_loss": 0.4},
                           {"epoch": 2, "validation_loss": 0.4}]) == 1


# ---------------------------------------------------------------------------
# Round 50: verification is total, and every malformed input is a refusal
#
# The round-49 verifier compared five fields and the re-derived epoch. Eight
# top-level fields and five of epoch_selection's own were never looked at, an
# extra field could be added to the file and a required one removed, and a
# malformed log could return a number or raise a raw TypeError. Each test
# below was run against that version first; the ones marked with the field
# name were green there and are red here.
# ---------------------------------------------------------------------------

UNCOMPARED_IN_ROUND_49 = ("checkpoint", "epoch_log", "run_provenance",
                          "environment", "checks", "problems",
                          "cross_configuration_reason", "boundary")


def tamper(value):
    """Change a value into a different one of a comparable shape."""
    if isinstance(value, list):
        return value[:-1] if value else ["injected"]
    if isinstance(value, dict):
        return {**value, "injected": True}
    return "TAMPERED"


class TestTheSelectionRecordIsComparedInFull:
    @pytest.mark.parametrize("field", UNCOMPARED_IN_ROUND_49)
    def test_every_previously_uncompared_field_is_now_compared(self, field,
                                                               tmp_path):
        directory = a_checkpoint(tmp_path)
        record = selection_module.build(directory)
        record[field] = tamper(record[field])
        problems = selection_module.verify(record, directory)
        assert any(field in problem for problem in problems), problems

    @pytest.mark.parametrize("sub", ["criterion", "manifest_criterion",
                                     "selected_epoch_stated",
                                     "validation_losses", "note",
                                     "selected_epoch_rederived"])
    def test_every_epoch_selection_sub_field_is_compared(self, sub, tmp_path):
        directory = a_checkpoint(tmp_path)
        record = selection_module.build(directory)
        record["epoch_selection"][sub] = "TAMPERED"
        problems = selection_module.verify(record, directory)
        assert any("epoch_selection" in problem for problem in problems)

    @pytest.mark.parametrize("field", ["configuration", "seed", "epochs_run",
                                       "digests", "trainable_parameters",
                                       "kind"])
    def test_the_fields_round_49_did_compare_are_still_compared(self, field,
                                                                tmp_path):
        directory = a_checkpoint(tmp_path)
        record = selection_module.build(directory)
        record[field] = tamper(record[field])
        assert selection_module.verify(record, directory) != []

    def test_a_field_added_to_the_file_is_refused_as_an_extra(self, tmp_path):
        directory = a_checkpoint(tmp_path)
        record = selection_module.build(directory)
        record["cross_configuration_summary"] = "stage2_e16 won"
        problems = selection_module.verify(record, directory)
        assert any("cross_configuration_summary" in problem
                   for problem in problems)
        assert any("added to the file rather than derived" in problem
                   for problem in problems)

    @pytest.mark.parametrize("field", selection_module.RECORD_FIELDS)
    def test_a_field_removed_from_the_file_is_refused_as_missing(self, field,
                                                                 tmp_path):
        directory = a_checkpoint(tmp_path)
        record = selection_module.build(directory)
        record.pop(field)
        problems = selection_module.verify(record, directory)
        if field == "kind":
            assert problems == ["the file is not a BrickAgain vision "
                                "selection record"]
            return
        assert any(field in problem for problem in problems), problems

    def test_the_builder_cannot_grow_a_field_the_comparison_does_not_know(
            self, tmp_path, monkeypatch):
        """RECORD_FIELDS is the contract, and the builder is held to it."""
        directory = a_checkpoint(tmp_path)
        monkeypatch.setattr(
            selection_module, "RECORD_FIELDS",
            tuple(f for f in selection_module.RECORD_FIELDS
                  if f != "boundary"))
        with pytest.raises(SelectionError, match="RECORD_FIELDS does not"):
            selection_module.build(directory)

    def test_differences_reports_both_directions(self):
        problems = selection_module.differences(
            {"a": 1, "gone": 2}, {"a": 2, "new": 3})
        assert any("missing 'new'" in p for p in problems)
        assert any("carries 'gone'" in p for p in problems)
        assert any("'a' does not match" in p for p in problems)

    def test_a_sound_record_still_verifies_clean(self, tmp_path):
        directory = a_checkpoint(tmp_path)
        assert selection_module.verify(
            selection_module.build(directory), directory) == []

    # The real returned checkpoint is verified by
    # ``scripts/32_vision_train.py --verify-selection`` rather than here: it
    # lives under ``runs/``, which neither the public snapshot nor the vision
    # pack carries, so a test that read it could only skip in both of the
    # trees this file travels to -- and a suite that mostly skips teaches an
    # operator that skipping is normal.


class TestMalformedInputIsARefusalNotATraceback:
    @pytest.mark.parametrize("loss,fragment", [
        (float("inf"), "not finite"),
        (float("-inf"), "not finite"),
        (float("nan"), "not finite"),
        ("0.4", "must be a number"),
        (None, "must be a number"),
        (True, "must be a number"),
        ([0.4], "must be a number"),
    ])
    def test_a_malformed_loss_is_named(self, loss, fragment):
        with pytest.raises(SelectionError, match=fragment):
            best_epoch([{"epoch": 1, "validation_loss": loss}])

    @pytest.mark.parametrize("epoch,fragment", [
        ("1", "must be a whole number"),
        (True, "must be a whole number"),
        (1.0, "must be a whole number"),
        (None, "must be a whole number"),
        (0, "not a positive epoch"),
        (-3, "not a positive epoch"),
    ])
    def test_a_malformed_epoch_number_is_named(self, epoch, fragment):
        with pytest.raises(SelectionError, match=fragment):
            best_epoch([{"epoch": epoch, "validation_loss": 0.4}])

    def test_a_repeated_epoch_is_refused(self):
        with pytest.raises(SelectionError, match="appears twice"):
            best_epoch([{"epoch": 1, "validation_loss": 0.4},
                        {"epoch": 1, "validation_loss": 0.3}])

    @pytest.mark.parametrize("log,fragment", [
        (None, "carries no epoch log"),
        ("not a log", "must be a list"),
        ({"epoch": 1}, "must be a list"),
        ([1, 2, 3], "row 0 is a int"),
        ([{"epoch": 1, "validation_loss": 0.4}, "x"], "row 1 is a str"),
        ([], "empty"),
    ])
    def test_a_malformed_log_is_named(self, log, fragment):
        with pytest.raises(SelectionError, match=fragment):
            best_epoch(log)

    def test_nothing_escapes_as_a_bare_exception(self, tmp_path):
        """Whatever is wrong, a caller sees one error type."""
        broken = [
            None, "text", 5, [1], [{"epoch": 1}],
            [{"epoch": 1, "validation_loss": float("nan")}],
            [{"epoch": None, "validation_loss": 1.0}],
            [{"epoch": 1, "validation_loss": {}}],
        ]
        for log in broken:
            with pytest.raises(SelectionError):
                best_epoch(log)

    @pytest.mark.parametrize("mutate,fragment", [
        (lambda m: m.pop("weights"), "has no 'weights'"),
        (lambda m: m.update(weights={"sha256": "x"}), "has no 'file'"),
        (lambda m: m.update(weights={"file": "", "sha256": "x"}),
         "not a file name"),
        (lambda m: m.update(weights={"file": "../escape.pt", "sha256": "x",
                                     "bytes": 1}), "not a file in"),
        (lambda m: m.update(weights=["vision_head.pt"]), "must be an object"),
        (lambda m: m.pop("selected_epoch"), "has no 'selected_epoch'"),
        (lambda m: m.update(selected_epoch="two"), "must be a whole number"),
        (lambda m: m.pop("config"), "has no 'config'"),
        (lambda m: m.update(config=[1, 2]), "must be an object"),
        (lambda m: m.pop("epoch_log"), "carries no epoch log"),
    ])
    def test_a_malformed_manifest_is_a_named_refusal(self, mutate, fragment,
                                                     tmp_path):
        directory = a_checkpoint(tmp_path)
        path = directory / model_module.VISION_MANIFEST
        body = json.loads(path.read_text(encoding="utf-8"))
        mutate(body)
        path.write_text(json.dumps(body), encoding="utf-8")
        with pytest.raises(SelectionError, match=fragment):
            selection_module.build(directory)

    def test_a_json_constant_in_an_artefact_is_refused_at_the_boundary(
            self, tmp_path):
        """``NaN`` is legal JSON to Python's decoder and is stopped here."""
        directory = a_checkpoint(tmp_path)
        path = directory / model_module.VISION_MANIFEST
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace('"validation_loss": 0.4',
                                     '"validation_loss": NaN'),
                        encoding="utf-8")
        with pytest.raises(SelectionError, match="JSON constant"):
            selection_module.build(directory)

    def test_an_unreadable_record_file_is_a_named_refusal(self, tmp_path):
        directory = a_checkpoint(tmp_path)
        (directory / selection_module.RECORD_FILE).write_text(
            "{not json", encoding="utf-8")
        with pytest.raises(SelectionError, match="not readable JSON"):
            selection_module.read(directory)

    def test_a_record_that_is_not_json_serialisable_is_refused(self):
        with pytest.raises(SelectionError, match="cannot be written as JSON"):
            selection_module.canonical({"kind": RECORD_KIND, "x": {1, 2}})

    def test_a_nan_inside_a_record_is_refused_by_canonical(self):
        with pytest.raises(SelectionError, match="cannot be written as JSON"):
            selection_module.canonical({"x": float("nan")})
