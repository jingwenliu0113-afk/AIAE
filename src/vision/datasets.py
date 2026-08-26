"""The two public datasets: what they are, what is taken, and how it groups.

Everything the rest of the project needs to know about the public LEGO image
data is decided here and nowhere else: which archive, which version, which
licence, which eight classes, which members are taken, and -- the part that
actually protects the result -- which capture group each image belongs to.

**Provenance is read out of the filenames, and it is checked.**  The archives
carry no metadata file, but their names are systematic, and the systematic part
is what says which images are near-duplicates of each other:

*Single-brick classification archive*
    ``photos/<design>/c3_4_48NF_original_3001_1609710554145.jpg``.  The four
    characters before ``original`` identify the **source photograph**; the
    leading ``c3`` is one crop of it.  Thirty crops of one photograph share a
    token, so the token is the group and the crops never straddle a split.
    ``renders/<design>/2456_Earth Blue_1_1620367898.jpeg`` names the colour and
    a pose index, so a rendered instance groups by ``(design, colour)`` and all
    its poses stay together.

*Multi-brick detection archive*
    ``photos/<n_bricks>/IMG_20201211_171151.jpg`` is a camera file: everything
    shot on one day is one session.  ``12345_flash_01.jpg`` is one physical
    arrangement photographed with and without flash.  A token name groups the
    same way as in the other archive.

If a name does not yield a group, :func:`classification_records` and
:func:`detection_records` **raise**.  There is deliberately no
"ungrouped" bucket and no per-image fallback: an image whose provenance is
unknown cannot be placed on either side of a split without risking the leak
the whole exercise is meant to avoid.

**What is not downloaded is written down.**  The detection photographs are 6.0
GB and carry no per-brick class, so a documented deterministic subset is taken
rather than all of them.  :data:`DETECTION_PHOTO_SAMPLING` is that rule, and
the manifest records how many images each bucket had and how many were taken,
so nothing is quietly truncated into looking like full coverage.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.vision.classes import (DESIGN_TO_PART, check_contract,
                                design_numbers, part_of_design)
from src.vision.split import SplitRecord

KIND_PHOTO = "photo"
KIND_RENDER = "render"
KINDS = (KIND_PHOTO, KIND_RENDER)

#: ``photo`` is a real photograph; ``render`` is synthetic.  They are counted
#: and reported separately, everywhere, without exception.
POPULATION_REAL = "real"
POPULATION_SYNTHETIC = "synthetic"


class DatasetError(ValueError):
    """The archive, a member name, or a provenance rule did not hold."""


@dataclass(frozen=True)
class DatasetSource:
    """One public archive: how to reach it and what it is licensed as."""

    key: str
    title: str
    doi: str
    version: str
    landing: str
    download: str
    licence: str
    attribution: str
    describing_paper: str

    def as_dict(self) -> dict:
        return {"key": self.key, "title": self.title, "doi": self.doi,
                "version": self.version, "landing": self.landing,
                "licence": self.licence, "attribution": self.attribution,
                "describing_paper": self.describing_paper}


#: The single-brick classification archive.  Version 1.1 of the record, which
#: is the current one; the 2021 release is a separate DOI and is not used.  The
#: description says 447 classes; the earlier paper says 431.  Neither number is
#: quoted as this project's own: what is recorded is the count actually seen in
#: the archive that was read, which the manifest carries.
CLASSIFICATION = DatasetSource(
    key="classification",
    title="LEGO bricks for training classification network",
    doi="10.34808/rcza-jy08",
    version="1.1",
    landing=("https://mostwiedzy.pl/en/open-research-data/"
             "lego-bricks-for-training-classification-network,"
             "202309140842198941751-0"),
    download=("https://mostwiedzy.pl/en/open-research-data/"
              "lego-bricks-for-training-classification-network,"
              "202309140842198941751-0/download"),
    licence="CC BY 4.0",
    attribution=("Boinski, Zarazinski, Sledz; Gdansk University of "
                 "Technology, Bridge of Knowledge"),
    describing_paper="https://www.nature.com/articles/s41597-023-02682-2",
)

#: The multi-brick detection archive: photographs and renders with PASCAL VOC
#: bounding boxes.
DETECTION = DatasetSource(
    key="detection",
    title="Tagged images with LEGO bricks",
    doi="10.34808/anq4-rn44",
    version="1.1",
    landing=("https://mostwiedzy.pl/en/open-research-data/"
             "tagged-images-with-lego-bricks,202309140833448152311-0"),
    download=("https://mostwiedzy.pl/en/open-research-data/"
              "tagged-images-with-lego-bricks,202309140833448152311-0"
              "/download"),
    licence="CC BY 4.0",
    attribution=("Zawora, Zarazinski, Sledz, Lobacz, Boinski; Gdansk "
                 "University of Technology, Bridge of Knowledge"),
    describing_paper="https://www.nature.com/articles/s41597-023-02682-2",
)

SOURCES = {source.key: source for source in (CLASSIFICATION, DETECTION)}

#: Deterministic subset of the detection photographs, keyed by the archive's
#: own bricks-per-image directory.  ``(stride, reason)``: every ``stride``-th
#: name in sorted order is taken.  Single-brick photographs are the bulk of the
#: 6.0 GB and are the least informative for multi-brick counting, so they are
#: sampled most sparsely.
DETECTION_PHOTO_SAMPLING: tuple[tuple[int, int, int, str], ...] = (
    (1, 1, 60, "single-brick photographs: 4.9 GB of the archive, and one box "
               "per image; sampled sparsely because multi-brick counting is "
               "what this population is for"),
    (2, 4, 6, "two to four bricks: the common multi-brick case, sampled to "
              "keep the download proportionate"),
    (5, 10 ** 6, 1, "five or more bricks: taken in full; these are the images "
                    "that actually exercise counting and occlusion"),
)

#: Renders in the detection archive whose leading filename token is one of the
#: eight design numbers.  Small enough to take in full.
DETECTION_RENDER_STRIDE = 1

_TOKEN = re.compile(r"\A[A-Za-z0-9]{3,6}\Z")
_IMG_NAME = re.compile(r"\AIMG_(\d{8})_\d{6}(?:_\d+)?\Z")
_FLASH_NAME = re.compile(r"\A(\d+)_(?:no_)?flash_\d+\Z")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def source_token(stem: str) -> str | None:
    """The capture token in a name of the ``..._<TOKEN>_original...`` form.

    The token is the segment immediately before the one beginning
    ``original``, which is the only position that holds across every variant
    the archives use -- some names carry an extra batch segment, some carry a
    design number after ``original``, some spell it ``original-B1``.  Anchoring
    on the neighbour rather than on a fixed field index is what makes the rule
    survive those.
    """
    parts = stem.split("_")
    for i, part in enumerate(parts):
        if part.startswith("original") and i > 0:
            candidate = parts[i - 1]
            return candidate if _TOKEN.match(candidate) else None
    return None


def classification_group(kind: str, design: str, filename: str) -> str:
    """The capture group for one member of the classification archive."""
    stem = filename.rsplit(".", 1)[0]
    if kind == KIND_PHOTO:
        capture = source_token(stem)
        if capture is None:
            raise DatasetError(
                f"no source-photograph token could be read from "
                f"{filename!r}. Its provenance is unknown, so it cannot be "
                "placed in a split; this is refused rather than guessed")
        return f"{design}/photo/{capture}"
    parts = stem.split("_")
    if len(parts) < 4:
        raise DatasetError(
            f"render name {filename!r} is not "
            "<design>_<colour>_<pose>_<stamp>")
    colour = "_".join(parts[1:-2]).strip()
    if not colour:
        raise DatasetError(f"render name {filename!r} carries no colour")
    return f"{design}/render/{colour}"


def render_colour(filename: str) -> str:
    """The colour name a render's own filename declares.

    The detection archive hyphenates its colour names and the classification
    archive spaces them; both are normalised to spaces so one colour is one
    label across the two.
    """
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) < 4:
        raise DatasetError(
            f"render name {filename!r} is not "
            "<design>_<colour>_<pose>_<stamp>")
    return "_".join(parts[1:-2]).replace("-", " ").strip()


def render_pose(filename: str) -> int:
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) < 4 or not parts[-2].isdigit():
        raise DatasetError(f"render name {filename!r} has no pose index")
    return int(parts[-2])


def detection_group(filename: str) -> str:
    """The capture group for one detection photograph.

    Three name forms, three groups, and a refusal for anything else.  A camera
    file groups by the day it was taken, because a day of shooting is one
    session and its frames are of the same arrangements under the same light.
    """
    stem = filename.rsplit(".", 1)[0]
    match = _IMG_NAME.match(stem)
    if match:
        return f"session/{match.group(1)}"
    match = _FLASH_NAME.match(stem)
    if match:
        # The same arrangement with the flash on and off is one arrangement.
        return f"arrangement/{match.group(1)}"
    capture = source_token(stem)
    if capture is not None:
        return f"photo/{capture}"
    raise DatasetError(
        f"no capture group could be read from detection photograph "
        f"{filename!r}; its provenance is unknown and it is refused rather "
        "than assigned at random")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageRecord:
    """One selected archive member, with everything the split needs."""

    member: str
    dataset: str
    kind: str
    population: str
    design: str
    part: str
    group: str
    uncompressed_bytes: int
    crc32: int
    colour: str | None = None
    pose: int | None = None
    n_bricks: int | None = None

    @property
    def item_id(self) -> str:
        return self.member

    @property
    def local_name(self) -> str:
        """Where this member is stored under the private raw directory."""
        return f"{self.dataset}/{self.member}"

    def split_record(self, *, stratum: str | None = None) -> SplitRecord:
        return SplitRecord(item_id=self.item_id, group=self.group,
                           stratum=stratum or self.part or "all",
                           label=self.part or None)

    def as_dict(self) -> dict:
        return {"member": self.member, "dataset": self.dataset,
                "kind": self.kind, "population": self.population,
                "design": self.design, "part": self.part, "group": self.group,
                "bytes": self.uncompressed_bytes, "crc32": self.crc32,
                "colour": self.colour, "pose": self.pose,
                "n_bricks": self.n_bricks}


def classification_records(entries) -> tuple[list[ImageRecord], dict]:
    """Select the eight classes out of the classification archive.

    ``entries`` are :class:`~src.vision.source.ZipEntry` objects from the
    archive's central directory.  Everything outside ``photos/<design>/`` and
    ``renders/<design>/`` for the eight design numbers is skipped, and the
    counts of what was skipped come back in the summary so "we took the eight"
    is a statement with a number behind it.
    """
    check_contract()
    wanted = set(design_numbers())
    out: list[ImageRecord] = []
    seen_designs: dict[str, set[str]] = {"photos": set(), "renders": set()}
    skipped_other_class = 0
    skipped_shape = 0
    for entry in entries:
        if entry.is_directory:
            continue
        parts = entry.name.split("/")
        if len(parts) != 3 or parts[0] not in ("photos", "renders"):
            skipped_shape += 1
            continue
        top, design, filename = parts
        seen_designs[top].add(design)
        if design not in wanted:
            skipped_other_class += 1
            continue
        kind = KIND_PHOTO if top == "photos" else KIND_RENDER
        colour = render_colour(filename) if kind == KIND_RENDER else None
        pose = render_pose(filename) if kind == KIND_RENDER else None
        out.append(ImageRecord(
            member=entry.name, dataset=CLASSIFICATION.key, kind=kind,
            population=(POPULATION_REAL if kind == KIND_PHOTO
                        else POPULATION_SYNTHETIC),
            design=design, part=part_of_design(design),
            group=classification_group(kind, design, filename),
            uncompressed_bytes=entry.uncompressed_bytes, crc32=entry.crc32,
            colour=colour, pose=pose))
    out.sort(key=lambda record: record.member)
    photo_classes = seen_designs["photos"] - {""}
    render_classes = seen_designs["renders"] - {""}
    summary = {
        "archive_photo_classes": len(photo_classes),
        "archive_render_classes": len(render_classes),
        "archive_classes_seen": len(photo_classes | render_classes),
        "archive_classes_photos_only": sorted(photo_classes - render_classes),
        "archive_classes_renders_only": sorted(render_classes - photo_classes),
        "archive_classes_note": (
            "counted from the archive that was actually read, not quoted from "
            "a description. The record says 447 classes and an earlier paper "
            "says 431; what is here is 447 photo directories and 447 render "
            "directories that are not the same 447 -- one class has renders "
            "and no photographs and another has photographs and no renders, "
            "so the union is 448. This project quotes none of the published "
            "numbers as its own"),
        "selected": len(out),
        "skipped_other_class_members": skipped_other_class,
        "skipped_unexpected_shape": skipped_shape,
        "designs_selected": sorted(
            {record.design for record in out}, key=int),
        "designs_missing": sorted(wanted - {record.design for record in out},
                                 key=int),
    }
    if summary["designs_missing"]:
        raise DatasetError(
            "the archive does not contain "
            f"{summary['designs_missing']}; every one of the eight classes has "
            "to be present or the class list is not the one this project uses")
    return out, summary


def detection_records(entries, *,
                      sampling=DETECTION_PHOTO_SAMPLING,
                      render_stride: int = DETECTION_RENDER_STRIDE
                      ) -> tuple[list[ImageRecord], list[str], dict]:
    """Select detection members: sampled photographs, plus eight-class renders.

    Returns the image records, the label files that go with them, and a
    summary that records the sampling honestly: per bucket, how many the
    archive holds and how many were taken.

    A photograph's boxes carry no per-brick class in this archive, so a
    photograph record has ``part=""``.  That is not a gap to be filled in
    later by guessing: it is the reason the detection evaluation reports
    per-class counting as unavailable for this population.
    """
    check_contract()
    wanted = set(design_numbers())
    photos: dict[int, list] = {}
    renders: list = []
    labels: dict[str, str] = {}
    for entry in entries:
        if entry.is_directory:
            continue
        parts = entry.name.split("/")
        if len(parts) != 3:
            continue
        top, bucket, filename = parts
        lower = filename.lower()
        if lower.endswith(".xml"):
            labels[entry.name] = entry.name
            continue
        if not lower.endswith((".jpg", ".jpeg", ".png")):
            continue
        if top == "photos":
            if not bucket.isdigit():
                raise DatasetError(
                    f"detection photograph {entry.name!r} is not under a "
                    "bricks-per-image directory")
            photos.setdefault(int(bucket), []).append(entry)
        elif top == "renders":
            renders.append(entry)

    out: list[ImageRecord] = []
    buckets: list[dict] = []
    for low, high, stride, reason in sampling:
        available = 0
        taken = 0
        for count in sorted(photos):
            if not low <= count <= high:
                continue
            members = sorted(photos[count], key=lambda e: e.name)
            available += len(members)
            for index, entry in enumerate(members):
                if index % stride:
                    continue
                taken += 1
                out.append(ImageRecord(
                    member=entry.name, dataset=DETECTION.key, kind=KIND_PHOTO,
                    population=POPULATION_REAL, design="", part="",
                    group=detection_group(entry.name.split("/")[-1]),
                    uncompressed_bytes=entry.uncompressed_bytes,
                    crc32=entry.crc32, n_bricks=count))
        buckets.append({"bricks_from": low,
                        "bricks_to": (None if high >= 10 ** 6 else high),
                        "stride": stride, "available": available,
                        "taken": taken, "dropped": available - taken,
                        "reason": reason})

    render_taken = 0
    render_other_class = 0
    for index, entry in enumerate(sorted(renders, key=lambda e: e.name)):
        filename = entry.name.split("/")[-1]
        design = filename.split("_")[0]
        if design not in wanted:
            render_other_class += 1
            continue
        if index % render_stride:
            continue
        render_taken += 1
        out.append(ImageRecord(
            member=entry.name, dataset=DETECTION.key, kind=KIND_RENDER,
            population=POPULATION_SYNTHETIC, design=design,
            part=part_of_design(design),
            group=f"{design}/render/{render_colour(filename)}",
            uncompressed_bytes=entry.uncompressed_bytes, crc32=entry.crc32,
            colour=render_colour(filename), pose=render_pose(filename),
            n_bricks=1))

    out.sort(key=lambda record: record.member)
    needed = {label_for(record.member) for record in out}
    wanted_labels = sorted(name for name in labels if name in needed)
    missing = sorted(needed - set(labels))
    summary = {
        "photo_buckets": buckets,
        "photos_taken": sum(b["taken"] for b in buckets),
        "photos_available": sum(b["available"] for b in buckets),
        "photos_dropped": sum(b["dropped"] for b in buckets),
        "sampling_note": (
            "the photographs are 6.0 GB and their boxes carry no per-brick "
            "class, so a deterministic every-nth subset is taken. What was "
            "dropped is counted here rather than left to look like full "
            "coverage"),
        "renders_taken": render_taken,
        "renders_skipped_other_class": render_other_class,
        "label_files": len(wanted_labels),
        "label_files_missing": missing,
    }
    if missing:
        raise DatasetError(
            f"{len(missing)} selected image(s) have no label file, first: "
            f"{missing[:3]}. An image with no boxes cannot be scored and is "
            "not silently treated as an empty scene")
    return out, wanted_labels, summary


def label_for(member: str) -> str:
    """The PASCAL VOC label member that belongs to an image member."""
    head, _dot, _ext = member.rpartition(".")
    if not head:
        raise DatasetError(f"{member!r} has no extension to replace")
    return f"{head}.xml"


# ---------------------------------------------------------------------------
# PASCAL VOC labels
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VocBox:
    label: str
    x0: int
    y0: int
    x1: int
    y1: int
    truncated: bool = False
    difficult: bool = False


@dataclass(frozen=True)
class VocAnnotation:
    filename: str
    width: int
    height: int
    boxes: tuple[VocBox, ...]


def parse_voc(text: str) -> VocAnnotation:
    """Parse one PASCAL VOC annotation, refusing a box that cannot be one.

    ``xml.etree`` with no entity resolution and no DTD, which is what makes it
    safe to run over a file that came from a download: an external-entity
    reference in a label file would otherwise be a way to read this machine.
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise DatasetError(f"the annotation is not valid XML: {exc}") from exc
    if root.tag != "annotation":
        raise DatasetError(
            f"the annotation root is <{root.tag}>, not <annotation>")
    size = root.find("size")
    if size is None:
        raise DatasetError("the annotation has no <size>")

    def whole(node, name: str) -> int:
        found = node.find(name)
        if found is None or found.text is None:
            raise DatasetError(f"the annotation has no <{name}>")
        text_value = found.text.strip()
        try:
            # Some tools write float coordinates; round rather than refuse,
            # and refuse anything that is not a number at all.
            return int(round(float(text_value)))
        except ValueError:
            raise DatasetError(
                f"<{name}> is {text_value!r}, which is not a number") from None

    width, height = whole(size, "width"), whole(size, "height")
    if width < 1 or height < 1:
        raise DatasetError(
            f"the annotation declares a {width}x{height} image")
    boxes: list[VocBox] = []
    for obj in root.findall("object"):
        name_node = obj.find("name")
        label = (name_node.text or "").strip() if name_node is not None else ""
        box = obj.find("bndbox")
        if box is None:
            raise DatasetError("an <object> has no <bndbox>")
        x0, y0 = whole(box, "xmin"), whole(box, "ymin")
        x1, y1 = whole(box, "xmax"), whole(box, "ymax")
        # VOC coordinates are inclusive and one-based in the original spec and
        # zero-based inclusive as labelImg writes them. Either way the far edge
        # is inclusive, so it is incremented once here to match the exclusive
        # convention this project uses everywhere else.
        x0, y0 = max(0, min(x0, width - 1)), max(0, min(y0, height - 1))
        x1 = max(x0 + 1, min(x1 + 1, width))
        y1 = max(y0 + 1, min(y1 + 1, height))
        boxes.append(VocBox(
            label=label or "brick", x0=x0, y0=y0, x1=x1, y1=y1,
            truncated=(obj.findtext("truncated") or "0").strip() == "1",
            difficult=(obj.findtext("difficult") or "0").strip() == "1"))
    filename = (root.findtext("filename") or "").strip()
    return VocAnnotation(filename=filename, width=width, height=height,
                         boxes=tuple(boxes))


# ---------------------------------------------------------------------------
# The data manifest
# ---------------------------------------------------------------------------

MANIFEST_KIND = "brickagain.vision_data"


def build_manifest(source: DatasetSource, *, archive, records, summary: dict,
                   extracted: dict[str, dict] | None = None,
                   labels: dict[str, str] | None = None) -> dict:
    """The record of exactly what was taken from one archive.

    ``archive`` is the :class:`~src.vision.source.RemoteZip`.  Its central
    directory digest is the archive's identity here: the mirror serves the file
    through expiring signed URLs, so the URL names nothing durable, and the
    published S3 ETag is a composite of part digests rather than a digest of
    the file. The central directory changes if any member does.
    """
    per_class: dict[str, dict[str, int]] = {}
    for record in records:
        bucket = per_class.setdefault(
            record.part or "unlabelled",
            {KIND_PHOTO: 0, KIND_RENDER: 0, "bytes": 0})
        bucket[record.kind] += 1
        bucket["bytes"] += record.uncompressed_bytes
    groups = {record.group for record in records}
    return {
        "kind": MANIFEST_KIND,
        "source": source.as_dict(),
        "archive": {
            "total_bytes": archive.total_bytes,
            "entry_count": archive.entry_count,
            "central_directory_sha256": archive.central_directory_sha256,
            "identity_note": (
                "the archive is identified by the SHA-256 of its central "
                "directory, which changes if any member is added, removed, "
                "renamed or recompressed. The mirror's own published checksum "
                "is a composite of 512 MB part digests, not a digest of the "
                "file"),
        },
        "eight_class_filter": {
            "design_numbers": list(design_numbers()),
            "design_to_part": dict(sorted(DESIGN_TO_PART.items())),
            "derived_from": "src.rendering.ldr.PART_TO_LDRAW",
        },
        "selection": summary,
        "per_class": {name: per_class[name] for name in sorted(per_class)},
        "capture_groups": len(groups),
        "records": [record.as_dict() for record in records],
        "labels": dict(sorted((labels or {}).items())),
        "extracted": dict(sorted((extracted or {}).items())),
        "boundary": (
            "raw images live under data/raw and never enter git, the public "
            "snapshot or a GPU pack. This manifest is the only thing that "
            "leaves that directory"),
    }


def manifest_digest(manifest: dict) -> str:
    """SHA-256 over the canonical serialisation, excluding nothing."""
    body = json.dumps(manifest, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def write_manifest(manifest: dict, path) -> tuple[Path, str]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(manifest, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")
    target.write_bytes(body)
    return target, hashlib.sha256(body).hexdigest()


def read_manifest(path, *, expected_digest: str | None = None) -> dict:
    target = Path(path)
    if not target.is_file():
        raise DatasetError(f"there is no vision data manifest at {target}")
    raw = target.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if expected_digest is not None and actual != expected_digest:
        raise DatasetError(
            f"the data manifest digest is {actual}, not the expected "
            f"{expected_digest}")
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"{target} is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("kind") != MANIFEST_KIND:
        raise DatasetError(f"{target} is not a vision data manifest")
    return manifest


def records_from_manifest(manifest: dict) -> list[ImageRecord]:
    out = []
    for row in manifest.get("records", []):
        out.append(ImageRecord(
            member=row["member"], dataset=row["dataset"], kind=row["kind"],
            population=row["population"], design=row["design"],
            part=row["part"], group=row["group"],
            uncompressed_bytes=int(row["bytes"]), crc32=int(row["crc32"]),
            colour=row.get("colour"), pose=row.get("pose"),
            n_bricks=row.get("n_bricks")))
    return out
