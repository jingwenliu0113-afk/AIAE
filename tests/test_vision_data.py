"""Dataset selection, provenance rules and label parsing.

The provenance rules are read out of filenames, and a rule that quietly returns
"no group" for a name shape nobody tested would put those images somewhere. So
every name form the two archives actually use is pinned here, and so is the
refusal for a name that fits none of them.
"""

from __future__ import annotations

import json
import zipfile

import pytest

from src.vision import datasets
from src.vision.datasets import (CLASSIFICATION, DETECTION, KIND_PHOTO,
                                 KIND_RENDER, POPULATION_REAL,
                                 POPULATION_SYNTHETIC, DatasetError,
                                 build_manifest, classification_group,
                                 classification_records, detection_group,
                                 detection_records, label_for,
                                 manifest_digest, parse_voc, read_manifest,
                                 records_from_manifest, render_colour,
                                 render_pose, source_token, write_manifest)
from src.vision.source import local_file_fetcher, read_central_directory


class TestSourceTokens:
    @pytest.mark.parametrize("stem,token", [
        ("c1_6_LnM8_original_3005_1609807264126", "LnM8"),
        ("c1_3_B11_qNen_original_1618304497893", "qNen"),
        ("c1_5_W3_IHYa_original_1618302600489", "IHYa"),
        ("c1_3_pWOd_original-B1_1621422299779", "pWOd"),
        ("c0_2_P2_FKzN_original_1618999026118", "FKzN"),
        ("0_EnU9_original_1608917990980", "EnU9"),
    ])
    def test_the_token_is_the_segment_before_original(self, stem, token):
        assert source_token(stem) == token

    @pytest.mark.parametrize("stem", [
        "IMG_20201211_171151", "12345_flash_01", "nothing_here",
        "original_only", "",
    ])
    def test_a_name_with_no_token_returns_none(self, stem):
        assert source_token(stem) is None


class TestClassificationGrouping:
    def test_crops_of_one_photograph_share_a_group(self):
        first = classification_group(
            KIND_PHOTO, "3005", "c0_6_LnM8_original_3005_1609807264126.jpg")
        second = classification_group(
            KIND_PHOTO, "3005", "c3_6_LnM8_original_3005_1609807264999.jpg")
        assert first == second == "3005/photo/LnM8"

    def test_different_photographs_are_different_groups(self):
        assert classification_group(
            KIND_PHOTO, "3005", "c0_6_LnM8_original_3005_1.jpg") != \
            classification_group(
                KIND_PHOTO, "3005", "c0_6_ZZZZ_original_3005_1.jpg")

    def test_a_photograph_with_no_token_is_refused(self):
        with pytest.raises(DatasetError, match="provenance is unknown"):
            classification_group(KIND_PHOTO, "3005", "mystery.jpg")

    def test_renders_group_by_design_and_colour(self):
        assert classification_group(
            KIND_RENDER, "2456", "44237_Earth Blue_1_1620367898.jpeg") == \
            "2456/render/Earth Blue"

    def test_every_pose_of_one_colour_is_one_group(self):
        groups = {
            classification_group(KIND_RENDER, "3001",
                                f"3001_Aqua_{pose}_16085149{pose}5.jpeg")
            for pose in range(4)}
        assert len(groups) == 1

    def test_a_render_name_that_is_not_the_shape_is_refused(self):
        with pytest.raises(DatasetError, match="is not"):
            classification_group(KIND_RENDER, "3001", "short.jpeg")

    def test_colour_names_normalise_across_the_two_archives(self):
        assert render_colour("3008_Bright-Blue_0_1608479929.jpg") == \
            render_colour("3008_Bright Blue_0_1608479929.jpeg") == \
            "Bright Blue"

    def test_the_pose_index_is_read(self):
        assert render_pose("3001_Aqua_2_1608398303.jpeg") == 2

    def test_a_render_with_no_pose_index_is_refused(self):
        with pytest.raises(DatasetError, match="no pose index"):
            render_pose("3001_Aqua_x_1608398303.jpeg")


class TestDetectionGrouping:
    def test_a_camera_file_groups_by_the_day(self):
        assert detection_group("IMG_20201211_171151.jpg") == \
            detection_group("IMG_20201211_235959.jpg") == "session/20201211"

    def test_two_days_are_two_groups(self):
        assert detection_group("IMG_20201211_171151.jpg") != \
            detection_group("IMG_20201212_171151.jpg")

    def test_flash_and_no_flash_of_one_arrangement_are_one_group(self):
        assert detection_group("1234_flash_01.jpg") == \
            detection_group("1234_no_flash_07.jpg") == "arrangement/1234"

    def test_a_token_name_groups_by_token(self):
        assert detection_group("0_EnU9_original_1608917990980.jpg") == \
            "photo/EnU9"

    def test_an_unrecognised_name_is_refused(self):
        with pytest.raises(DatasetError, match="provenance is unknown"):
            detection_group("random-name.jpg")


class TestSelectingTheEightClasses:
    def _archive(self, tmp_path, extra=()):
        members = {}
        for design in ("3005", "3004", "3010", "3009", "3008", "3003",
                       "3001", "2456"):
            members[f"photos/{design}/c0_1_AB{design[-1]}A_original_"
                    f"{design}_1600000000000.jpg"] = b"p" * 40
            members[f"renders/{design}/{design}_Bright Red_0_1600000000"
                    ".jpeg"] = b"r" * 30
        for name in extra:
            members[name] = b"x" * 10
        path = tmp_path / "cls.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, payload in members.items():
                archive.writestr(name, payload)
        return read_central_directory(local_file_fetcher(path),
                                     path.stat().st_size)

    def test_all_eight_classes_are_selected(self, tmp_path):
        zipped = self._archive(tmp_path)
        records, summary = classification_records(zipped.entries)
        assert len(records) == 16
        assert summary["designs_missing"] == []
        assert set(summary["designs_selected"]) == set(
            datasets.design_numbers())

    def test_other_classes_are_counted_and_dropped(self, tmp_path):
        zipped = self._archive(tmp_path, extra=[
            "photos/3062/c0_1_QQQQ_original_3062_1.jpg",
            "renders/3062/3062_Black_0_1.jpeg"])
        records, summary = classification_records(zipped.entries)
        assert len(records) == 16
        assert summary["skipped_other_class_members"] == 2

    def test_a_missing_class_is_refused(self, tmp_path):
        members = {"photos/3005/c0_1_AAAA_original_3005_1.jpg": b"p"}
        path = tmp_path / "partial.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name, payload in members.items():
                archive.writestr(name, payload)
        zipped = read_central_directory(local_file_fetcher(path),
                                       path.stat().st_size)
        with pytest.raises(DatasetError, match="does not contain"):
            classification_records(zipped.entries)

    def test_populations_are_labelled(self, tmp_path):
        zipped = self._archive(tmp_path)
        records, _summary = classification_records(zipped.entries)
        photos = [r for r in records if r.kind == KIND_PHOTO]
        renders = [r for r in records if r.kind == KIND_RENDER]
        assert all(r.population == POPULATION_REAL for r in photos)
        assert all(r.population == POPULATION_SYNTHETIC for r in renders)

    def test_the_class_count_note_reports_what_was_seen(self, tmp_path):
        zipped = self._archive(tmp_path, extra=[
            "photos/9999/c0_1_QQQQ_original_9999_1.jpg"])
        _records, summary = classification_records(zipped.entries)
        assert summary["archive_photo_classes"] == 9
        assert summary["archive_render_classes"] == 8
        assert summary["archive_classes_seen"] == 9
        assert summary["archive_classes_photos_only"] == ["9999"]


class TestDetectionSelectionIsHonestAboutSampling:
    def _archive(self, tmp_path):
        members = {}
        for index in range(12):
            stem = f"IMG_2020120{index % 3}_1200{index:02d}"
            members[f"photos/1/{stem}.jpg"] = b"p" * 20
            members[f"photos/1/{stem}.xml"] = _voc(stem + ".jpg", 1)
        for index in range(4):
            stem = f"IMG_20201210_1300{index:02d}"
            members[f"photos/7/{stem}.jpg"] = b"p" * 20
            members[f"photos/7/{stem}.xml"] = _voc(stem + ".jpg", 7)
        for design in datasets.design_numbers():
            stem = f"{design}_Bright-Red_0_1608000000"
            members[f"renders/1/{stem}.jpg"] = b"r" * 15
            members[f"renders/1/{stem}.xml"] = _voc(stem + ".jpg", 1)
        path = tmp_path / "det.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, payload in members.items():
                archive.writestr(name, payload)
        return read_central_directory(local_file_fetcher(path),
                                     path.stat().st_size)

    def test_the_sampling_records_what_it_dropped(self, tmp_path):
        zipped = self._archive(tmp_path)
        records, labels, summary = detection_records(
            zipped.entries,
            sampling=((1, 1, 4, "sparse"), (5, 10 ** 6, 1, "all")))
        buckets = {(b["bricks_from"], b["stride"]): b
                   for b in summary["photo_buckets"]}
        assert buckets[(1, 4)]["available"] == 12
        assert buckets[(1, 4)]["taken"] == 3
        assert buckets[(1, 4)]["dropped"] == 9
        assert buckets[(5, 1)]["taken"] == 4
        assert summary["photos_dropped"] == 9
        assert len(labels) == len(records)

    def test_a_photograph_carries_no_part_label(self, tmp_path):
        zipped = self._archive(tmp_path)
        records, _labels, _summary = detection_records(zipped.entries)
        photos = [r for r in records
                  if r.kind == KIND_PHOTO and r.dataset == DETECTION.key]
        assert photos and all(r.part == "" for r in photos)

    def test_a_render_carries_its_class_from_its_filename(self, tmp_path):
        zipped = self._archive(tmp_path)
        records, _labels, _summary = detection_records(zipped.entries)
        renders = [r for r in records if r.kind == KIND_RENDER]
        assert len(renders) == 8
        assert {r.part for r in renders} == {
            datasets.part_of_design(design)
            for design in datasets.design_numbers()}

    def test_an_image_with_no_label_file_is_refused(self, tmp_path):
        path = tmp_path / "nolabel.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("photos/7/IMG_20201210_120000.jpg", b"p")
        zipped = read_central_directory(local_file_fetcher(path),
                                       path.stat().st_size)
        with pytest.raises(DatasetError, match="no label file"):
            detection_records(zipped.entries)

    def test_a_photograph_outside_a_count_directory_is_refused(self, tmp_path):
        path = tmp_path / "odd.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("photos/misc/IMG_20201210_120000.jpg", b"p")
            archive.writestr("photos/misc/IMG_20201210_120000.xml", b"<x/>")
        zipped = read_central_directory(local_file_fetcher(path),
                                       path.stat().st_size)
        with pytest.raises(DatasetError, match="bricks-per-image"):
            detection_records(zipped.entries)


def _voc(filename, boxes, width=100, height=80):
    objects = "".join(
        f"<object><name>brick</name><bndbox><xmin>{5 + i}</xmin>"
        f"<ymin>{5 + i}</ymin><xmax>{20 + i}</xmax><ymax>{20 + i}</ymax>"
        "</bndbox></object>" for i in range(boxes))
    return (f"<annotation><filename>{filename}</filename><size>"
            f"<width>{width}</width><height>{height}</height>"
            f"<depth>3</depth></size>{objects}</annotation>").encode("utf-8")


class TestVocParsing:
    def test_it_reads_boxes_and_size(self):
        annotation = parse_voc(_voc("a.jpg", 3).decode("utf-8"))
        assert annotation.width == 100 and annotation.height == 80
        assert len(annotation.boxes) == 3
        assert annotation.filename == "a.jpg"

    def test_the_far_edge_becomes_exclusive(self):
        annotation = parse_voc(_voc("a.jpg", 1).decode("utf-8"))
        box = annotation.boxes[0]
        assert (box.x0, box.y0, box.x1, box.y1) == (5, 5, 21, 21)

    def test_a_box_is_clamped_into_the_image(self):
        text = ("<annotation><size><width>10</width><height>10</height>"
                "</size><object><name>brick</name><bndbox><xmin>-4</xmin>"
                "<ymin>2</ymin><xmax>99</xmax><ymax>99</ymax></bndbox>"
                "</object></annotation>")
        box = parse_voc(text).boxes[0]
        assert (box.x0, box.y0, box.x1, box.y1) == (0, 2, 10, 10)

    def test_float_coordinates_are_accepted(self):
        text = ("<annotation><size><width>50</width><height>50</height>"
                "</size><object><name>brick</name><bndbox><xmin>1.4</xmin>"
                "<ymin>2.6</ymin><xmax>10.2</xmax><ymax>11.8</ymax></bndbox>"
                "</object></annotation>")
        box = parse_voc(text).boxes[0]
        assert (box.x0, box.y0) == (1, 3)

    def test_an_empty_annotation_has_no_boxes(self):
        assert parse_voc(_voc("a.jpg", 0).decode("utf-8")).boxes == ()

    def test_broken_xml_is_refused(self):
        with pytest.raises(DatasetError, match="not valid XML"):
            parse_voc("<annotation>")

    def test_the_wrong_root_is_refused(self):
        with pytest.raises(DatasetError, match="not <annotation>"):
            parse_voc("<other/>")

    def test_a_missing_size_is_refused(self):
        with pytest.raises(DatasetError, match="no <size>"):
            parse_voc("<annotation><object/></annotation>")

    def test_a_non_numeric_coordinate_is_refused(self):
        text = ("<annotation><size><width>10</width><height>10</height>"
                "</size><object><name>brick</name><bndbox><xmin>a</xmin>"
                "<ymin>0</ymin><xmax>5</xmax><ymax>5</ymax></bndbox>"
                "</object></annotation>")
        with pytest.raises(DatasetError, match="not a number"):
            parse_voc(text)

    def test_an_object_with_no_box_is_refused(self):
        text = ("<annotation><size><width>10</width><height>10</height>"
                "</size><object><name>brick</name></object></annotation>")
        with pytest.raises(DatasetError, match="no <bndbox>"):
            parse_voc(text)

    def test_label_for_swaps_the_extension(self):
        assert label_for("photos/7/a.jpg") == "photos/7/a.xml"
        with pytest.raises(DatasetError, match="no extension"):
            label_for("noextension")


class TestTheManifest:
    def _built(self, tmp_path):
        members = {}
        for design in datasets.design_numbers():
            members[f"photos/{design}/c0_1_AB{design[-1]}A_original_"
                    f"{design}_1.jpg"] = b"p" * 40
            members[f"renders/{design}/{design}_Black_0_1.jpeg"] = b"r" * 30
        path = tmp_path / "a.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, payload in members.items():
                archive.writestr(name, payload)
        zipped = read_central_directory(local_file_fetcher(path),
                                       path.stat().st_size)
        records, summary = classification_records(zipped.entries)
        return build_manifest(CLASSIFICATION, archive=zipped, records=records,
                              summary=summary)

    def test_it_carries_the_source_licence_and_the_archive_identity(self,
                                                                   tmp_path):
        manifest = self._built(tmp_path)
        assert manifest["source"]["licence"] == "CC BY 4.0"
        assert manifest["source"]["doi"] == "10.34808/rcza-jy08"
        assert len(manifest["archive"]["central_directory_sha256"]) == 64

    def test_the_class_filter_is_recorded_as_derived(self, tmp_path):
        manifest = self._built(tmp_path)
        assert manifest["eight_class_filter"]["derived_from"] == \
            "src.rendering.ldr.PART_TO_LDRAW"
        assert len(manifest["eight_class_filter"]["design_numbers"]) == 8

    def test_it_round_trips_through_disk_with_a_stable_digest(self, tmp_path):
        manifest = self._built(tmp_path)
        path, digest = write_manifest(manifest, tmp_path / "m.json")
        assert digest == manifest_digest(manifest)
        again = read_manifest(path, expected_digest=digest)
        assert records_from_manifest(again)[0].member == \
            records_from_manifest(manifest)[0].member

    def test_a_wrong_expected_digest_is_refused(self, tmp_path):
        manifest = self._built(tmp_path)
        path, _digest = write_manifest(manifest, tmp_path / "m.json")
        with pytest.raises(DatasetError, match="not the expected"):
            read_manifest(path, expected_digest="0" * 64)

    def test_a_file_that_is_not_a_data_manifest_is_refused(self, tmp_path):
        target = tmp_path / "x.json"
        target.write_text(json.dumps({"kind": "other"}), encoding="utf-8")
        with pytest.raises(DatasetError, match="not a vision data manifest"):
            read_manifest(target)

    def test_the_boundary_note_is_present(self, tmp_path):
        manifest = self._built(tmp_path)
        assert "never enter git" in manifest["boundary"]
