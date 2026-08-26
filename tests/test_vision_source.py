"""The selective ZIP reader, driven over a real archive on disk.

No network anywhere in this file. Every test builds a real ZIP in a temporary
directory and reads it through :func:`src.vision.source.local_file_fetcher`,
which is the same code path the range reader uses -- so the parsing, the
verification and the span coalescing are all exercised offline.

The refusals are the point. A ZIP member's name, declared size and CRC are all
attacker-controlled data, and this reader is pointed at a six-gigabyte download
from a public mirror.
"""

from __future__ import annotations

import struct
import zipfile
import zlib

import pytest

from src.vision import source
from src.vision.source import (MAX_MEMBER_BYTES, SourceError, ZipEntry,
                               fetch_member, fetch_members,
                               local_file_fetcher, plan_spans,
                               read_central_directory, safe_member_name,
                               select)


def build_zip(path, members, *, compression=zipfile.ZIP_DEFLATED):
    with zipfile.ZipFile(path, "w", compression) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


@pytest.fixture
def archive(tmp_path):
    members = {
        "photos/3001/a.jpg": b"A" * 1000,
        "photos/3001/b.jpg": b"B" * 2000,
        "photos/3003/c.jpg": b"C" * 1500,
        "renders/3001/d.jpeg": b"D" * 800,
        "renders/30011/e.jpeg": b"E" * 700,
        "other/f.txt": b"F" * 100,
    }
    path = build_zip(tmp_path / "test.zip", members)
    return path, members


class TestReadingTheDirectory:
    def test_it_finds_every_member(self, archive):
        path, members = archive
        fetch = local_file_fetcher(path)
        zipped = read_central_directory(fetch, path.stat().st_size)
        names = {entry.name for entry in zipped.entries}
        assert names == set(members)
        assert zipped.entry_count == len(members)

    def test_the_digest_is_of_the_directory_not_the_file(self, archive, tmp_path):
        path, members = archive
        first = read_central_directory(
            local_file_fetcher(path), path.stat().st_size)
        again = read_central_directory(
            local_file_fetcher(path), path.stat().st_size)
        assert first.central_directory_sha256 == \
            again.central_directory_sha256
        other = build_zip(tmp_path / "other.zip",
                          {**members, "photos/3001/z.jpg": b"Z"})
        changed = read_central_directory(
            local_file_fetcher(other), other.stat().st_size)
        assert changed.central_directory_sha256 != \
            first.central_directory_sha256

    def test_a_small_chunk_size_still_reads_the_whole_directory(self, archive):
        path, members = archive
        zipped = read_central_directory(
            local_file_fetcher(path), path.stat().st_size, chunk_bytes=17)
        assert {entry.name for entry in zipped.entries} == set(members)

    def test_a_file_that_is_not_a_zip_is_refused(self, tmp_path):
        target = tmp_path / "no.bin"
        target.write_bytes(b"not a zip at all" * 100)
        with pytest.raises(SourceError, match="end-of-central-directory"):
            read_central_directory(local_file_fetcher(target),
                                  target.stat().st_size)

    def test_a_short_range_is_refused_rather_than_parsed(self, archive):
        path, _members = archive

        def stingy(start, end):
            return local_file_fetcher(path)(start, end)[:-1]

        with pytest.raises(SourceError, match="short range|returned"):
            read_central_directory(stingy, path.stat().st_size)

    def test_a_fetcher_returning_the_wrong_type_is_refused(self, archive):
        path, _members = archive
        with pytest.raises(SourceError, match="not bytes"):
            read_central_directory(lambda a, b: "text",
                                   path.stat().st_size)

    def test_a_directory_larger_than_the_cap_is_refused(self, archive,
                                                       monkeypatch):
        path, _members = archive
        monkeypatch.setattr(source, "MAX_CENTRAL_DIRECTORY_BYTES", 4)
        with pytest.raises(SourceError, match="over the"):
            read_central_directory(local_file_fetcher(path),
                                  path.stat().st_size)


class TestReadingOneMember:
    def test_the_bytes_come_back_exactly(self, archive):
        path, members = archive
        fetch = local_file_fetcher(path)
        zipped = read_central_directory(fetch, path.stat().st_size)
        for name, payload in members.items():
            entry = zipped.by_name(name)
            assert fetch_member(fetch, entry,
                                total_bytes=path.stat().st_size) == payload

    def test_a_stored_member_reads_too(self, tmp_path):
        path = build_zip(tmp_path / "stored.zip", {"a/b.jpg": b"X" * 50},
                        compression=zipfile.ZIP_STORED)
        fetch = local_file_fetcher(path)
        zipped = read_central_directory(fetch, path.stat().st_size)
        assert fetch_member(fetch, zipped.by_name("a/b.jpg")) == b"X" * 50

    def test_a_wrong_crc_is_refused(self, archive):
        path, _members = archive
        fetch = local_file_fetcher(path)
        zipped = read_central_directory(fetch, path.stat().st_size)
        entry = zipped.by_name("photos/3001/a.jpg")
        tampered = ZipEntry(entry.name, entry.method, entry.crc32 ^ 0xFF,
                            entry.compressed_bytes, entry.uncompressed_bytes,
                            entry.local_offset)
        with pytest.raises(SourceError, match="CRC-32"):
            fetch_member(fetch, tampered)

    def test_a_wrong_declared_size_is_refused(self, archive):
        path, _members = archive
        fetch = local_file_fetcher(path)
        zipped = read_central_directory(fetch, path.stat().st_size)
        entry = zipped.by_name("photos/3001/a.jpg")
        lying = ZipEntry(entry.name, entry.method, entry.crc32,
                         entry.compressed_bytes,
                         entry.uncompressed_bytes + 500, entry.local_offset)
        with pytest.raises(SourceError, match="inflated to"):
            fetch_member(fetch, lying)

    def test_an_unsupported_compression_method_is_refused(self, archive):
        path, _members = archive
        fetch = local_file_fetcher(path)
        zipped = read_central_directory(fetch, path.stat().st_size)
        entry = zipped.by_name("photos/3001/a.jpg")
        odd = ZipEntry(entry.name, 14, entry.crc32, entry.compressed_bytes,
                       entry.uncompressed_bytes, entry.local_offset)
        with pytest.raises(SourceError, match="compression method"):
            fetch_member(fetch, odd)

    def test_a_member_at_the_wrong_offset_is_refused(self, archive):
        path, _members = archive
        fetch = local_file_fetcher(path)
        zipped = read_central_directory(fetch, path.stat().st_size)
        entry = zipped.by_name("photos/3001/b.jpg")
        moved = ZipEntry(entry.name, entry.method, entry.crc32,
                         entry.compressed_bytes, entry.uncompressed_bytes,
                         entry.local_offset + 3)
        with pytest.raises(SourceError, match="local file header"):
            fetch_member(fetch, moved)


class TestTheBombGuards:
    def test_a_member_declaring_more_than_the_cap_is_refused(self, archive):
        path, _members = archive
        fetch = local_file_fetcher(path)
        zipped = read_central_directory(fetch, path.stat().st_size)
        entry = zipped.by_name("photos/3001/a.jpg")
        huge = ZipEntry(entry.name, entry.method, entry.crc32,
                        entry.compressed_bytes, MAX_MEMBER_BYTES + 1,
                        entry.local_offset)
        with pytest.raises(SourceError, match="cap this reader will inflate"):
            fetch_member(fetch, huge)

    def test_a_member_declaring_a_huge_compressed_size_is_refused(self,
                                                                  archive):
        path, _members = archive
        fetch = local_file_fetcher(path)
        zipped = read_central_directory(fetch, path.stat().st_size)
        entry = zipped.by_name("photos/3001/a.jpg")
        huge = ZipEntry(entry.name, entry.method, entry.crc32,
                        MAX_MEMBER_BYTES + 1, entry.uncompressed_bytes,
                        entry.local_offset)
        with pytest.raises(SourceError, match="compressed bytes"):
            fetch_member(fetch, huge)

    def test_a_highly_compressible_member_cannot_expand_past_its_claim(
            self, tmp_path):
        """The classic bomb: a small payload that inflates enormously.

        The declared size is what bounds the decompressor, and the inflated
        length is then checked against that declaration, so a payload that
        expands further is stopped at the cap rather than after it.
        """
        payload = b"\0" * (4 * 1024 * 1024)
        path = build_zip(tmp_path / "bomb.zip", {"a/b.jpg": payload})
        fetch = local_file_fetcher(path)
        zipped = read_central_directory(fetch, path.stat().st_size)
        entry = zipped.by_name("a/b.jpg")
        understated = ZipEntry(entry.name, entry.method,
                               zlib.crc32(payload[:16]) & 0xFFFFFFFF,
                               entry.compressed_bytes, 16, entry.local_offset)
        with pytest.raises(SourceError, match="inflates past|inflated to"):
            fetch_member(fetch, understated)


class TestAMemberNameIsNotAPath:
    @pytest.mark.parametrize("name", [
        "../escape.jpg", "a/../../escape.jpg", "/absolute.jpg",
        "C:\\windows\\x.jpg", "back\\slash.jpg", "./here.jpg",
        "a/./b.jpg", "null\x00.jpg",
    ])
    def test_an_escaping_name_is_refused(self, name):
        with pytest.raises(SourceError):
            safe_member_name(name)

    @pytest.mark.parametrize("name", [
        "photos/3001/a.jpg", "renders/2456/x y z.jpeg", "a.jpg",
    ])
    def test_an_ordinary_name_passes(self, name):
        assert safe_member_name(name) == name

    def test_an_empty_name_is_refused(self):
        with pytest.raises(SourceError, match="non-empty"):
            safe_member_name("")

    def test_fetching_an_escaping_member_is_refused_before_any_read(self,
                                                                   archive):
        path, _members = archive
        calls = []

        def counting(start, end):
            calls.append((start, end))
            return local_file_fetcher(path)(start, end)

        bad = ZipEntry("../x.jpg", 8, 0, 1, 1, 0)
        with pytest.raises(SourceError, match="relative segment"):
            fetch_member(counting, bad)
        assert calls == [], "the name is checked before anything is fetched"


class TestSpansAreCoalescedWithoutChangingTheResult:
    def test_every_member_still_verifies(self, archive):
        path, members = archive
        fetch = local_file_fetcher(path)
        zipped = read_central_directory(fetch, path.stat().st_size)
        got = dict(fetch_members(fetch, zipped.entries,
                                total_bytes=path.stat().st_size))
        assert {entry.name: payload for entry, payload in got.items()} == \
            members

    def test_one_span_covers_a_small_archive(self, archive):
        path, _members = archive
        zipped = read_central_directory(
            local_file_fetcher(path), path.stat().st_size)
        spans = plan_spans(zipped.entries)
        assert len(spans) == 1

    def test_a_small_cap_splits_into_several_spans(self, archive):
        path, members = archive
        zipped = read_central_directory(
            local_file_fetcher(path), path.stat().st_size)
        spans = plan_spans(zipped.entries, max_span_bytes=1200)
        assert len(spans) > 1
        covered = [entry.name for _s, _e, group in spans for entry in group]
        assert sorted(covered) == sorted(members)

    def test_coalescing_makes_fewer_requests_than_member_by_member(self,
                                                                   archive):
        path, _members = archive
        counted = []

        def counting(start, end):
            counted.append(1)
            return local_file_fetcher(path)(start, end)

        zipped = read_central_directory(
            local_file_fetcher(path), path.stat().st_size)
        counted.clear()
        list(fetch_members(counting, zipped.entries,
                          total_bytes=path.stat().st_size))
        coalesced = len(counted)
        counted.clear()
        for entry in zipped.entries:
            fetch_member(counting, entry, total_bytes=path.stat().st_size)
        one_at_a_time = len(counted)
        assert coalesced < one_at_a_time


class TestSelectingWantedDirectories:
    def test_it_groups_by_whole_path_segment(self, archive):
        path, _members = archive
        zipped = read_central_directory(
            local_file_fetcher(path), path.stat().st_size)
        picked = select(zipped.entries, ["3001", "3003"])
        assert sorted(entry.name for entry in picked["3001"]) == [
            "photos/3001/a.jpg", "photos/3001/b.jpg", "renders/3001/d.jpeg"]
        assert [entry.name for entry in picked["3003"]] == [
            "photos/3003/c.jpg"]

    def test_a_longer_similar_name_is_not_absorbed(self, archive):
        """``3001`` must not sweep in ``30011``."""
        path, _members = archive
        zipped = read_central_directory(
            local_file_fetcher(path), path.stat().st_size)
        picked = select(zipped.entries, ["3001"])
        assert all("30011" not in entry.name for entry in picked["3001"])

    def test_a_wanted_name_that_matched_nothing_comes_back_empty(self,
                                                                 archive):
        path, _members = archive
        zipped = read_central_directory(
            local_file_fetcher(path), path.stat().st_size)
        picked = select(zipped.entries, ["9999"])
        assert picked == {"9999": ()}
