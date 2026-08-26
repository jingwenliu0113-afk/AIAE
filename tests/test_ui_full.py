"""The full interface, driven over a real socket.

Every request here goes through ``http.client`` against a server bound to
``127.0.0.1:0``, so the transport guards are the real ones rather than mocked:
the loopback check, the origin screening, the form key, the bounded body, the
content-type allowlist and the connection-closing behaviour on a refusal that
skipped the body.

The two-page interface has its own 195 tests and is untouched. What is checked
here is what the full version adds -- the upload, the corrections and their
provenance, the three method entries, the build-step pages -- and, in
particular, that adding them did not lose any of the refusals the two-page
version had.

No weights are loaded. The ``final_H2`` entry is exercised through its
refusals: disabled, and unavailable. The one real decode is a separate,
explicitly labelled smoke in ``scripts/35_full_ui.py``'s documentation and in
the round's report, not here.
"""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import socket
import threading
from urllib.parse import urlencode

import numpy as np
import pytest

from src.data.bricks import Brick, format_bricks
from src.delivery import pipeline
from src.ui import server_full
from src.ui.corrections import MAX_ITEMS, adopt
from src.ui.full import (METHOD_PIPELINE, METHOD_PROJECT, METHOD_RAG,
                         PHOTO_MULTI, PHOTO_SINGLE, RECOGNISE_CV,
                         RECOGNISE_LEARNED)
from src.ui.server import CSRF_FIELD
from src.ui.state import SessionStore
from src.ui.upload import MAX_UPLOAD_BYTES


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

STACK = [Brick(2, 4, 0, 0, 0), Brick(2, 4, 0, 0, 1)]

#: Three bricks, not two, so the build-step pages have a middle step to walk.
#: A fixture that cannot exercise "previous and next together" would have made
#: that test skip, and a skip in the private tree contradicts the release
#: gate's own contract.
SMALL = [Brick(1, 1, 0, 0, 0), Brick(1, 1, 0, 0, 1), Brick(1, 1, 0, 0, 2)]


def row(sid, oid, caption, bricks):
    return {"split": "train", "role": "control", "variant": "exact",
            "object_id": oid, "structure_id": sid, "caption": caption,
            "bricks_txt": format_bricks(bricks)}


@pytest.fixture
def frozen_split(tmp_path, monkeypatch):
    manifest = tmp_path / "object_splits.json"
    manifest.write_text(json.dumps({
        "meta": {"fixture": True}, "counts": {"train": 2},
        "objects": {"o-car": "train", "o-tower": "train"},
        "structures": {"s-car": "o-car", "s-tower": "o-tower"},
    }, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(pipeline, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(pipeline, "FROZEN_SPLIT_MANIFEST_SHA256",
                        hashlib.sha256(manifest.read_bytes()).hexdigest())
    return manifest


@pytest.fixture
def catalogue(tmp_path, frozen_split):
    path = tmp_path / "fixture_train.jsonl"
    path.write_text(
        json.dumps(row("s-car", "o-car", "a compact red car", STACK)) + "\n"
        + json.dumps(row("s-tower", "o-tower", "a tiny tower", SMALL)) + "\n",
        encoding="utf-8")
    return path


class Live:
    """A running server plus a client that speaks to it."""

    def __init__(self, server):
        self.server = server
        self.port = server.port
        self.thread = threading.Thread(target=server.serve_forever,
                                       daemon=True)
        self.thread.start()

    @property
    def key(self) -> str:
        return self.server.csrf_key

    def get(self, path, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port,
                                                timeout=20)
        try:
            connection.request("GET", path, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.read(), dict(response.getheaders())
        finally:
            connection.close()

    def post_form(self, path, data, *, headers=None, with_key=True):
        body = dict(data)
        if with_key:
            body.setdefault(CSRF_FIELD, self.key)
        payload = urlencode(body, doseq=True).encode("utf-8")
        base = {"Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(payload))}
        base.update(headers or {})
        connection = http.client.HTTPConnection("127.0.0.1", self.port,
                                                timeout=60)
        try:
            connection.request("POST", path, body=payload, headers=base)
            response = connection.getresponse()
            return response.status, response.read(), dict(
                response.getheaders())
        finally:
            connection.close()

    def post_multipart(self, path, *, image=None, fields=None, boundary="BND",
                       headers=None, with_key=True, filename="shot.png",
                       part_type="image/png"):
        pieces = []
        values = dict(fields or {})
        if with_key:
            values.setdefault(CSRF_FIELD, self.key)
        for name, value in values.items():
            pieces.append(
                f"--{boundary}\r\nContent-Disposition: form-data; "
                f'name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8"))
        if image is not None:
            pieces.append(
                (f"--{boundary}\r\nContent-Disposition: form-data; "
                 f'name="photo"; filename="{filename}"\r\n'
                 f"Content-Type: {part_type}\r\n\r\n").encode("utf-8")
                + image + b"\r\n")
        pieces.append(f"--{boundary}--\r\n".encode("utf-8"))
        payload = b"".join(pieces)
        base = {"Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(payload))}
        base.update(headers or {})
        connection = http.client.HTTPConnection("127.0.0.1", self.port,
                                                timeout=60)
        try:
            connection.request("POST", path, body=payload, headers=base)
            response = connection.getresponse()
            return response.status, response.read(), dict(
                response.getheaders())
        finally:
            connection.close()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def live(catalogue):
    server = server_full.create_server(
        port=0, catalog=catalogue, index_dir=None, checkpoint=None,
        sessions=SessionStore())
    server.allow_project_model = False
    running = Live(server)
    yield running
    running.close()


def png(width=600, height=300, blocks=((0xC9, 0x1A, 0x09),
                                        (0x00, 0x55, 0xBF))):
    from PIL import Image

    image = np.full((height, width, 3), 242, dtype=np.uint8)
    for index, colour in enumerate(blocks):
        left = 30 + index * 220
        image[90:210, left:left + 170] = colour
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


# --------------------------------------------------------------------------
# the start page
# --------------------------------------------------------------------------

class TestTheStartPage:
    def test_it_renders_with_a_form_key(self, live):
        status, body, _headers = live.get("/")
        assert status == 200
        text = body.decode()
        assert live.key in text
        assert "multipart/form-data" in text

    def test_it_offers_all_three_methods(self, live):
        _status, body, _headers = live.get("/")
        text = body.decode()
        for method in (METHOD_RAG, METHOD_PIPELINE, METHOD_PROJECT):
            assert f'value="{method}"' in text

    def test_the_learned_method_is_disabled_without_a_checkpoint(self, live):
        _status, body, _headers = live.get("/")
        text = body.decode()
        assert RECOGNISE_LEARNED in text
        assert "沒有可用的 checkpoint" in text

    def test_the_security_headers_are_still_there(self, live):
        _status, _body, headers = live.get("/")
        policy = headers["Content-Security-Policy"]
        assert "frame-ancestors 'none'" in policy
        assert "default-src 'none'" in policy
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["Referrer-Policy"] == "no-referrer"

    def test_an_unknown_path_is_a_readable_404(self, live):
        status, body, _headers = live.get("/nowhere")
        assert status == 404
        assert "沒有這個頁面" in body.decode()

    def test_reset_clears_and_says_so(self, live):
        status, body, _headers = live.get("/reset")
        assert status == 200
        assert "已清除" in body.decode()


# --------------------------------------------------------------------------
# the transport guards, still in place
# --------------------------------------------------------------------------

class TestTheGuardsSurvivedTheNewPages:
    def test_a_structured_external_origin_is_refused(self, live):
        status, body, _headers = live.post_form(
            "/result", {"caption": "x", "qty_2x4": "2",
                        "method": METHOD_PIPELINE},
            headers={"Origin": "https://example.com"})
        assert status == 403
        assert "其他來源" in body.decode()

    def test_another_port_on_this_machine_is_external(self, live):
        status, _body, _headers = live.post_form(
            "/result", {"caption": "x", "qty_2x4": "2"},
            headers={"Origin": f"http://127.0.0.1:{live.port + 1}"})
        assert status == 403

    def test_an_opaque_origin_with_the_right_key_proceeds(self, live,
                                                          catalogue):
        status, _body, _headers = live.post_form(
            "/result", {"caption": "a tiny tower", "qty_1x1": "2",
                        "method": METHOD_PIPELINE, "top_n": "1"},
            headers={"Origin": "null"})
        assert status == 200

    def test_an_opaque_origin_without_a_key_is_refused(self, live):
        status, body, _headers = live.post_form(
            "/result", {"caption": "x", "qty_2x4": "2"},
            headers={"Origin": "null"}, with_key=False)
        assert status == 403
        assert "表單金鑰" in body.decode()

    def test_a_missing_key_is_refused(self, live):
        status, body, _headers = live.post_form(
            "/result", {"caption": "x", "qty_2x4": "2"}, with_key=False)
        assert status == 403
        assert CSRF_FIELD in body.decode()

    def test_a_wrong_key_is_refused(self, live):
        status, body, _headers = live.post_form(
            "/result", {"caption": "x", "qty_2x4": "2",
                        CSRF_FIELD: "not-the-key"}, with_key=False)
        assert status == 403
        assert "不符" in body.decode()

    def test_a_duplicated_key_is_refused(self, live):
        status, _body, _headers = live.post_form(
            "/result", {"caption": "x", "qty_2x4": "2",
                        CSRF_FIELD: [live.key, live.key]}, with_key=False)
        assert status == 403

    def test_a_non_loopback_host_header_is_refused(self, live):
        status, body, _headers = live.get("/", headers={"Host": "example.com"})
        assert status == 403
        assert "只服務本機" in body.decode()

    def test_the_multipart_route_refuses_a_missing_key(self, live):
        status, body, _headers = live.post_multipart(
            "/photo", image=png(), fields={"photo_mode": PHOTO_MULTI},
            with_key=False)
        assert status == 403
        assert CSRF_FIELD in body.decode()

    def test_the_multipart_route_refuses_an_external_origin(self, live):
        status, _body, _headers = live.post_multipart(
            "/photo", image=png(), fields={"photo_mode": PHOTO_MULTI},
            headers={"Origin": "https://elsewhere.test"})
        assert status == 403

    def test_a_form_body_on_the_photo_route_is_refused(self, live):
        status, body, _headers = live.post_form(
            "/photo", {"photo_mode": PHOTO_MULTI})
        assert status == 415
        assert "檔案上傳表單" in body.decode()

    def test_chunked_transfer_is_refused(self, live):
        connection = http.client.HTTPConnection("127.0.0.1", live.port,
                                                timeout=20)
        try:
            connection.putrequest("POST", "/photo")
            connection.putheader("Content-Type",
                                 "multipart/form-data; boundary=BND")
            connection.putheader("Transfer-Encoding", "chunked")
            connection.endheaders()
            connection.send(b"0\r\n\r\n")
            response = connection.getresponse()
            assert response.status == 411
            assert "分塊傳輸" in response.read().decode()
        finally:
            connection.close()

    def test_an_oversized_multipart_is_refused_without_being_read(self, live):
        connection = http.client.HTTPConnection("127.0.0.1", live.port,
                                                timeout=20)
        try:
            claimed = MAX_UPLOAD_BYTES * 4
            connection.putrequest("POST", "/photo")
            connection.putheader("Content-Type",
                                 "multipart/form-data; boundary=BND")
            connection.putheader("Content-Length", str(claimed))
            connection.endheaders()
            connection.send(b"x" * 64)
            response = connection.getresponse()
            assert response.status == 413
            assert response.getheader("Connection") == "close"
        finally:
            connection.close()

    def test_a_refusal_that_skipped_the_body_closes_the_connection(self, live):
        """A refusal on a keep-alive connection must not leave bytes behind.

        The body here is a whole second HTTP request. If the connection stayed
        open, the server would parse it as one and answer twice.
        """
        smuggled = (b"POST /result HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                    b"Content-Length: 0\r\n\r\n")
        raw = (b"POST /photo HTTP/1.1\r\nHost: 127.0.0.1\r\n"
               b"Content-Type: text/plain\r\n"
               + f"Content-Length: {len(smuggled)}\r\n\r\n".encode()
               + smuggled)
        with socket.create_connection(("127.0.0.1", live.port),
                                      timeout=20) as sock:
            sock.sendall(raw)
            received = b""
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                received += chunk
        assert received.count(b"HTTP/1.") == 1, \
            "the refused body was parsed as a second request"


# --------------------------------------------------------------------------
# uploading a photograph
# --------------------------------------------------------------------------

class TestUploading:
    def test_a_png_is_analysed_and_the_photo_page_appears(self, live):
        status, body, _headers = live.post_multipart(
            "/photo", image=png(), fields={"photo_mode": PHOTO_MULTI,
                                            "recognise": RECOGNISE_CV})
        assert status == 200
        text = body.decode()
        assert "辨識完成" in text
        assert "人工修正" in text
        assert "shot.png" in text

    def test_the_boxes_are_an_overlay_over_the_untouched_image(self, live):
        payload = png()
        status, body, _headers = live.post_multipart(
            "/photo", image=payload, fields={"photo_mode": PHOTO_MULTI})
        assert status == 200
        text = body.decode()
        assert "<svg viewBox=\"0 0 600 300\"" in text
        handle = text.split('/photo/')[1].split('/image')[0]
        status, served, headers = live.get(f"/photo/{handle}/image")
        assert status == 200
        assert served == payload, "the image is served exactly as uploaded"
        assert headers["Content-Type"].startswith("image/")

    def test_a_single_brick_photo_reports_one_item(self, live):
        status, body, _headers = live.post_multipart(
            "/photo", image=png(blocks=((0xC9, 0x1A, 0x09),)),
            fields={"photo_mode": PHOTO_SINGLE})
        assert status == 200
        assert "找到 1 個項目" in body.decode()

    def test_no_file_is_refused_by_name(self, live):
        status, body, _headers = live.post_multipart(
            "/photo", image=None, fields={"photo_mode": PHOTO_MULTI})
        assert status == 400
        assert "沒有收到照片檔" in body.decode()

    def test_a_non_image_is_refused(self, live):
        status, body, _headers = live.post_multipart(
            "/photo", image=b"this is definitely not a picture" * 40,
            fields={"photo_mode": PHOTO_MULTI})
        assert status == 400
        assert "拒絕" in body.decode()

    def test_a_gif_is_refused_because_the_format_is_an_allowlist(self, live):
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), (10, 20, 30)).save(buffer, format="GIF")
        status, body, _headers = live.post_multipart(
            "/photo", image=buffer.getvalue(),
            fields={"photo_mode": PHOTO_MULTI}, filename="a.gif",
            part_type="image/gif")
        assert status == 400
        assert "PNG" in body.decode() or "JPEG" in body.decode()

    def test_a_declared_bomb_is_refused(self, live):
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (9000, 9000), (255, 255, 255)).save(buffer,
                                                             format="PNG")
        status, body, _headers = live.post_multipart(
            "/photo", image=buffer.getvalue(),
            fields={"photo_mode": PHOTO_MULTI})
        assert status == 400
        text = body.decode()
        assert "像素" in text or "pixels" in text

    def test_an_unexpected_field_is_refused_not_ignored(self, live):
        status, body, _headers = live.post_multipart(
            "/photo", image=png(), fields={"photo_mode": PHOTO_MULTI,
                                            "surprise": "value"})
        assert status == 400
        assert "未預期的欄位" in body.decode()

    def test_a_file_in_an_unexpected_field_is_refused(self, live):
        boundary = "BND"
        payload = (
            f"--{boundary}\r\nContent-Disposition: form-data; "
            f'name="{CSRF_FIELD}"\r\n\r\n{live.key}\r\n'
            f"--{boundary}\r\nContent-Disposition: form-data; "
            'name="other"; filename="x.png"\r\n'
            "Content-Type: image/png\r\n\r\n").encode() + png() + \
            f"\r\n--{boundary}--\r\n".encode()
        connection = http.client.HTTPConnection("127.0.0.1", live.port,
                                                timeout=30)
        try:
            connection.request(
                "POST", "/photo", body=payload,
                headers={"Content-Type":
                         f"multipart/form-data; boundary={boundary}",
                         "Content-Length": str(len(payload))})
            response = connection.getresponse()
            assert response.status == 400
            assert "不接受該欄位的檔案" in response.read().decode()
        finally:
            connection.close()

    def test_the_learned_method_without_a_checkpoint_is_refused_by_name(
            self, live):
        status, body, _headers = live.post_multipart(
            "/photo", image=png(), fields={"photo_mode": PHOTO_MULTI,
                                            "recognise": RECOGNISE_LEARNED})
        assert status == 400
        assert "checkpoint" in body.decode()

    def test_an_expired_photo_handle_is_a_readable_message(self, live):
        status, body, _headers = live.get("/photo/aaaaaaaaaaaa/image")
        assert status == 404
        assert "不在記憶體裡" in body.decode()

    def test_a_malformed_handle_is_a_plain_404(self, live):
        status, _body, _headers = live.get("/photo/..%2Fetc/image")
        assert status == 404


# --------------------------------------------------------------------------
# corrections
# --------------------------------------------------------------------------

def upload(live, **kw):
    status, body, _headers = live.post_multipart(
        "/photo", image=png(), fields={"photo_mode": PHOTO_MULTI, **kw})
    assert status == 200
    text = body.decode()
    handle = text.split('/photo/')[1].split('/image')[0]
    return handle, text


class TestCorrections:
    def test_setting_a_part_and_count_builds_the_adopted_inventory(self, live):
        handle, _text = upload(live)
        status, body, _headers = live.post_form(
            f"/photo/{handle}/correct",
            {"part_0": "2x4", "count_0": "3", "colour_0": "red",
             "part_1": "1x2", "count_1": "1", "colour_1": "blue"})
        assert status == 200
        text = body.decode()
        assert "2x4:3" in text or "修正後" in text
        assert "value=\"2x4:3,1x2:1\"" in text or "2x4:3" in text

    def test_the_model_prediction_survives_a_correction(self, live):
        handle, _text = upload(live)
        status, body, _headers = live.post_form(
            f"/photo/{handle}/correct", {"part_0": "2x6", "count_0": "1"})
        assert status == 200
        text = body.decode()
        assert "model+operator" in text
        session = live.server.sessions.photo(handle)
        assert session.items[0].predicted_part != "2x6"
        assert session.items[0].adopted_part == "2x6"

    def test_deleting_an_item_removes_it_from_the_stock(self, live):
        handle, _text = upload(live)
        live.post_form(f"/photo/{handle}/correct",
                       {"part_0": "2x4", "count_0": "2",
                        "part_1": "1x2", "count_1": "5"})
        status, _body, _headers = live.post_form(
            f"/photo/{handle}/correct",
            {"part_0": "2x4", "count_0": "2", "delete_1": "on"})
        assert status == 200
        adopted = adopt(live.server.sessions.photo(handle).items)
        assert adopted.parts == {"2x4": 2}
        assert adopted.as_dict()["deleted_items"] == 1

    def test_editing_a_box_is_applied_and_recorded(self, live):
        handle, _text = upload(live)
        status, _body, _headers = live.post_form(
            f"/photo/{handle}/correct",
            {"part_0": "2x4", "count_0": "1", "box_0": "5,6,120,130"})
        assert status == 200
        item = live.server.sessions.photo(handle).items[0]
        assert item.adopted_box == (5, 6, 120, 130)
        assert "box" in item.changed_fields
        assert item.box != item.adopted_box

    def test_a_box_outside_the_image_is_refused(self, live):
        handle, _text = upload(live)
        status, body, _headers = live.post_form(
            f"/photo/{handle}/correct", {"box_0": "0,0,9999,9999"})
        assert status == 400
        assert "超出圖片範圍" in body.decode()

    def test_a_box_with_no_area_is_refused(self, live):
        handle, _text = upload(live)
        status, body, _headers = live.post_form(
            f"/photo/{handle}/correct", {"box_0": "10,10,10,10"})
        assert status == 400
        assert "沒有面積" in body.decode()

    def test_a_malformed_box_is_refused(self, live):
        handle, _text = upload(live)
        status, body, _headers = live.post_form(
            f"/photo/{handle}/correct", {"box_0": "10,20"})
        assert status == 400
        assert "四個半形整數" in body.decode()

    def test_a_part_outside_the_eight_is_refused(self, live):
        handle, _text = upload(live)
        status, body, _headers = live.post_form(
            f"/photo/{handle}/correct", {"part_0": "2x8"})
        assert status == 400
        assert "2x8" in body.decode()

    def test_a_colour_outside_the_palette_is_refused(self, live):
        handle, _text = upload(live)
        status, body, _headers = live.post_form(
            f"/photo/{handle}/correct", {"colour_0": "chartreuse"})
        assert status == 400
        assert "chartreuse" in body.decode()

    def test_a_non_ascii_count_is_refused(self, live):
        handle, _text = upload(live)
        status, body, _headers = live.post_form(
            f"/photo/{handle}/correct", {"count_0": "٣"})
        assert status == 400
        assert "半形" in body.decode()

    def test_adding_an_item_is_recorded_as_operator_only(self, live):
        handle, _text = upload(live)
        status, body, _headers = live.post_form(
            f"/photo/{handle}/correct",
            {"add_part": "1x8", "add_count": "2", "add_colour": "yellow"})
        assert status == 200
        items = live.server.sessions.photo(handle).items
        added = items[-1]
        assert added.added_by_operator
        assert added.source == "operator"
        assert added.predicted_part == "unknown"
        assert adopt(items).parts.get("1x8") == 2

    def test_rotation_spellings_are_one_item(self, live):
        handle, _text = upload(live)
        live.post_form(f"/photo/{handle}/correct",
                       {"part_0": "2x4", "count_0": "1",
                        "add_part": "4x2", "add_count": "2"})
        adopted = adopt(live.server.sessions.photo(handle).items)
        assert adopted.parts.get("2x4") == 3
        assert "4x2" not in adopted.parts

    def test_an_unresolved_item_is_excluded_and_reported(self, live):
        handle, _text = upload(live)
        status, body, _headers = live.post_form(
            f"/photo/{handle}/correct", {"part_0": "2x4", "count_0": "2"})
        assert status == 200
        adopted = adopt(live.server.sessions.photo(handle).items)
        assert adopted.parts == {"2x4": 2}
        assert adopted.unresolved
        assert "不計入庫存" in body.decode()

    def test_the_before_and_after_inventory_are_both_shown(self, live):
        handle, _text = upload(live)
        status, body, _headers = live.post_form(
            f"/photo/{handle}/correct", {"part_0": "2x4", "count_0": "2"})
        text = body.decode()
        assert "修正前" in text and "修正後" in text

    def test_correcting_an_expired_handle_is_a_readable_message(self, live):
        status, body, _headers = live.post_form(
            "/photo/aaaaaaaaaaaa/correct", {"part_0": "2x4"})
        assert status == 404
        assert "不在記憶體裡" in body.decode()


# --------------------------------------------------------------------------
# running a method
# --------------------------------------------------------------------------

class TestInapplicableFieldsAreNamed:
    def test_a_time_limit_on_retrieval_is_refused(self, live):
        status, body, _headers = live.post_form(
            "/result", {"caption": "a car", "qty_2x4": "2",
                        "method": METHOD_RAG, "time_limit": "2"})
        assert status == 400
        assert "time limit" in body.decode()

    def test_a_placement_gate_on_the_pipeline_is_refused(self, live):
        status, body, _headers = live.post_form(
            "/result", {"caption": "a car", "qty_2x4": "2",
                        "method": METHOD_PIPELINE, "placement": "on"})
        assert status == 400
        assert "placement gate" in body.decode()

    def test_a_time_limit_on_the_project_model_is_refused(self, live):
        status, body, _headers = live.post_form(
            "/result", {"caption": "a car", "qty_2x4": "2",
                        "method": METHOD_PROJECT, "time_limit": "2"})
        assert status == 400
        assert "time limit" in body.decode()

    def test_both_inventory_inputs_at_once_are_refused(self, live):
        status, body, _headers = live.post_form(
            "/result", {"caption": "a car", "qty_2x4": "2",
                        "inventory_spec": "1x2:2",
                        "method": METHOD_PIPELINE})
        assert status == 400
        assert "只用其中一種" in body.decode()

    def test_an_empty_inventory_is_refused(self, live):
        status, body, _headers = live.post_form(
            "/result", {"caption": "a car", "method": METHOD_PIPELINE})
        assert status == 400
        assert "庫存不可空白" in body.decode()

    def test_an_empty_caption_is_refused(self, live):
        status, body, _headers = live.post_form(
            "/result", {"qty_2x4": "2", "method": METHOD_PIPELINE})
        assert status == 400
        assert "文字需求不可空白" in body.decode()

    def test_both_rotation_spellings_in_a_stock_string_are_refused(self, live):
        status, body, _headers = live.post_form(
            "/result", {"caption": "a car", "inventory_spec": "2x4:2,4x2:1",
                        "method": METHOD_PIPELINE})
        assert status == 400
        assert "same part" in body.decode() or "give it once" in body.decode()

    def test_a_non_ascii_quantity_is_a_named_400(self, live):
        status, body, _headers = live.post_form(
            "/result", {"caption": "a car", "qty_2x4": "٣",
                        "method": METHOD_PIPELINE})
        assert status == 400
        assert "半形" in body.decode()

    def test_a_quantity_over_the_world_capacity_is_refused(self, live):
        status, body, _headers = live.post_form(
            "/result", {"caption": "a car", "qty_2x4": "8001",
                        "method": METHOD_PIPELINE})
        assert status == 400
        assert "8000" in body.decode()


class TestTheFormsDefaultsDoNotRefuseTheDefaultMethod:
    """A real browser walk-through found both of these.

    The form pre-filled ``time_limit``, so the *default* method refused every
    first submission; and ``seed`` sat inside the CP-SAT-only fieldset, so
    choosing the project model silently dropped it -- the exact silent
    discard the design forbids.
    """

    def test_a_submission_with_only_the_defaults_is_not_refused(self, live):
        _status, page, _headers = live.get("/")
        text = page.decode()
        assert 'name="time_limit" min="0"\n' in text or 'value=""' in text
        status, body, _headers = live.post_form(
            "/result", {"caption": "a tiny tower", "qty_1x1": "4",
                        "method": METHOD_PIPELINE, "top_n": "2"})
        assert status == 200
        assert "沒有執行" not in body.decode()

    def test_the_blank_form_leaves_the_method_only_fields_empty(self):
        from src.ui.render_full import blank_form

        form = blank_form()
        assert form["time_limit"] == ""
        assert form["seed"] == ""

    def test_the_seed_is_in_its_own_fieldset_not_the_cpsat_one(self, live):
        _status, page, _headers = live.get("/")
        text = page.decode()
        cpsat = text.index('id="cpsat"')
        seedbox = text.index('id="seedbox"')
        seed_input = text.index('name="seed"')
        assert seedbox < seed_input, "the seed must live inside seedbox"
        assert not cpsat < seed_input < seedbox, \
            "the seed must not be inside the CP-SAT fieldset"

    def test_a_seed_on_retrieval_is_refused_by_name(self, live):
        status, body, _headers = live.post_form(
            "/result", {"caption": "a car", "qty_2x4": "2",
                        "method": METHOD_RAG, "seed": "1"})
        assert status == 400
        assert "seed" in body.decode()

    def test_a_seed_is_accepted_by_the_pipeline(self, live):
        status, _body, _headers = live.post_form(
            "/result", {"caption": "a tiny tower", "qty_1x1": "4",
                        "method": METHOD_PIPELINE, "seed": "3",
                        "top_n": "2", "time_limit": "3"})
        assert status == 200

    def test_the_decode_cap_is_not_tied_to_the_inventory_total(self):
        """Tying them made the cap fire exactly when the stock ran out, so the
        run recorded ``max_bricks`` instead of the gate's own
        ``inventory_exhausted`` -- the one the scorer accepts."""
        import inspect

        from src.ui import requests as requests_module

        source = inspect.getsource(requests_module.execute)
        assert "max_bricks=MAX_DECODE_BRICKS" in source
        assert "sum(request.inventory.values())" not in source


class TestRetrievalWithoutAnIndex:
    def test_it_is_refused_by_name_rather_than_failing_late(self, live):
        status, body, _headers = live.post_form(
            "/result", {"caption": "a car", "qty_2x4": "2",
                        "method": METHOD_RAG})
        assert status == 400
        assert "RAG 索引" in body.decode()


class TestTheProjectModelEntry:
    def test_it_is_refused_when_the_run_disabled_it(self, live):
        assert live.server.allow_project_model is False
        status, body, _headers = live.post_form(
            "/result", {"caption": "a tower", "qty_2x4": "3",
                        "method": METHOD_PROJECT})
        assert status == 400
        assert "沒有開放正式模型入口" in body.decode()

    def test_a_missing_pointer_is_a_readable_refusal(self, live, tmp_path):
        live.server.allow_project_model = True
        live.server.project_root = tmp_path
        status, body, _headers = live.post_form(
            "/result", {"caption": "a tower", "qty_2x4": "3",
                        "method": METHOD_PROJECT})
        assert status == 400
        text = body.decode()
        assert "project_model.json" in text
        assert "not published" in text or "private research tree" in text


class TestTheFPipelinePath:
    def test_a_buildable_request_produces_a_result_page(self, live):
        status, body, _headers = live.post_form(
            "/result", {"caption": "a tiny tower", "qty_1x1": "4",
                        "method": METHOD_PIPELINE, "top_n": "2",
                        "time_limit": "3"})
        assert status == 200
        text = body.decode()
        assert "方法與 provenance" in text
        assert "F-pipeline 各候選" in text

    def test_a_deliverable_result_offers_the_preview_and_the_ldraw(self, live):
        status, body, _headers = live.post_form(
            "/result", {"caption": "a tiny tower", "qty_1x1": "4",
                        "method": METHOD_PIPELINE, "top_n": "2",
                        "time_limit": "3"})
        text = body.decode()
        assert "通過靜態交付檢查的結果" in text
        handle = text.split("/artifact/")[1].split("/preview.png")[0]
        status, image, headers = live.get(f"/artifact/{handle}/preview.png")
        assert status == 200 and image[:8] == b"\x89PNG\r\n\x1a\n"
        status, ldraw, headers = live.get(f"/artifact/{handle}/model.ldr")
        assert status == 200
        assert headers["Content-Disposition"].startswith("attachment")
        assert ldraw.decode().count("0 STEP") >= 1

    def test_an_unbuildable_request_says_so_and_offers_nothing(self, live):
        status, body, _headers = live.post_form(
            "/result", {"caption": "something impossible here",
                        "inventory_spec": "1x8:1",
                        "method": METHOD_PIPELINE, "top_n": "1",
                        "time_limit": "1"})
        assert status == 200
        text = body.decode()
        assert "沒有可交付結果" in text
        assert "/artifact/" not in text

    def test_an_artifact_handle_that_never_existed_is_a_message(self, live):
        status, body, _headers = live.get("/artifact/aaaaaaaaaaaa/model.ldr")
        assert status == 404
        assert "不在記憶體裡" in body.decode()


# --------------------------------------------------------------------------
# the build-step pages
# --------------------------------------------------------------------------

def deliverable(live):
    status, body, _headers = live.post_form(
        "/result", {"caption": "a tiny tower", "qty_1x1": "4",
                    "method": METHOD_PIPELINE, "top_n": "2",
                    "time_limit": "3"})
    text = body.decode()
    assert "/steps/" in text, "the fixture must produce a build order"
    return text.split("/steps/")[1].split("/1")[0], text


class TestTheColourBlockRenders:
    """The path a real browser walk-through crashed on.

    Rendering the colour block reads the assignment object directly, and the
    template was reading a name that only exists in its serialised form. No
    test covered a deliverable result *with* a colour stock, so nothing caught
    it until a person clicked through.
    """

    def test_a_deliverable_result_with_a_colour_stock_renders(self, live):
        status, body, _headers = live.post_form(
            "/result", {"caption": "a tiny tower", "qty_1x1": "4",
                        "method": METHOD_PIPELINE, "top_n": "2",
                        "time_limit": "3",
                        "colour_stock": "1x1:red:2,1x1:blue:2"})
        assert status == 200
        text = body.decode()
        assert "確定性配色" in text
        assert "偏好顏色用到" in text
        assert "/artifact/" in text

    def test_a_colour_stock_too_short_is_a_named_problem_not_a_crash(self,
                                                                     live):
        status, body, _headers = live.post_form(
            "/result", {"caption": "a tiny tower", "qty_1x1": "4",
                        "method": METHOD_PIPELINE, "top_n": "2",
                        "time_limit": "3", "colour_stock": "1x1:red:1"})
        assert status == 200
        text = body.decode()
        assert "沒有配色" in text
        assert "needs" in text
        # the structure, preview and download are still produced
        assert "/artifact/" in text

    def test_the_assignment_counts_add_up(self):
        from src.colour.assign import assign, parse_colour_stock
        from src.data.bricks import parse_bricks

        bricks = parse_bricks("2x4 (0,0,0)\n2x4 (0,0,1)\n2x4 (0,0,2)")
        got = assign(bricks, parse_colour_stock("2x4:blue:1,2x4:red:2"),
                     preferences=["blue"])
        assert got.preferred_count == 1
        assert got.non_preferred_count == 2
        assert got.preferred_count + got.non_preferred_count == len(bricks)


class TestTheStepPages:
    def test_the_first_step_renders_with_navigation(self, live):
        handle, _text = deliverable(live)
        status, body, _headers = live.get(f"/steps/{handle}/1")
        assert status == 200
        text = body.decode()
        assert "組裝步驟 1" in text
        assert "下一步" in text
        assert "上一步" not in text

    def test_the_middle_steps_have_both_directions(self, live):
        handle, _text = deliverable(live)
        plan = live.server.sessions.result(handle).artifacts.plan
        assert plan.n_steps >= 3, "the fixture must have a middle step"
        status, body, _headers = live.get(f"/steps/{handle}/2")
        text = body.decode()
        assert "上一步" in text and "下一步" in text

    def test_the_last_step_has_no_next(self, live):
        handle, _text = deliverable(live)
        plan = live.server.sessions.result(handle).artifacts.plan
        status, body, _headers = live.get(f"/steps/{handle}/{plan.n_steps}")
        text = body.decode()
        assert "下一步" not in text
        assert f"組裝步驟 {plan.n_steps}" in text

    def test_a_step_image_is_a_png(self, live):
        handle, _text = deliverable(live)
        status, image, headers = live.get(f"/steps/{handle}/1/image")
        assert status == 200
        assert image[:8] == b"\x89PNG\r\n\x1a\n"
        assert headers["Content-Type"] == "image/png"

    def test_a_step_out_of_range_is_refused_by_name(self, live):
        handle, _text = deliverable(live)
        status, body, _headers = live.get(f"/steps/{handle}/999")
        assert status == 404
        assert "沒有這個步驟" in body.decode()

    def test_a_non_numeric_step_is_a_plain_404(self, live):
        handle, _text = deliverable(live)
        status, _body, _headers = live.get(f"/steps/{handle}/abc")
        assert status == 404

    def test_the_cumulative_parts_list_is_shown(self, live):
        handle, _text = deliverable(live)
        status, body, _headers = live.get(f"/steps/{handle}/1")
        text = body.decode()
        assert "累積零件表" in text
        assert "這一步加入" in text

    def test_every_step_states_it_is_not_a_physics_claim(self, live):
        handle, _text = deliverable(live)
        status, body, _headers = live.get(f"/steps/{handle}/1")
        assert "不是物理支撐" in body.decode()

    def test_the_result_page_can_be_revisited(self, live):
        handle, _text = deliverable(live)
        status, body, _headers = live.get(f"/result/{handle}")
        assert status == 200
        assert "方法與 provenance" in body.decode()


# --------------------------------------------------------------------------
# the store's own bounds
# --------------------------------------------------------------------------

class TestTheStoreIsBounded:
    def test_old_photographs_are_evicted(self):
        store = SessionStore(photo_limit=2)
        first = store.put_photo(image=b"a" * 10, media_type="image/png",
                                filename="a.png", width=10, height=10,
                                mode=PHOTO_SINGLE)
        store.put_photo(image=b"b" * 10, media_type="image/png",
                        filename="b.png", width=10, height=10,
                        mode=PHOTO_SINGLE)
        store.put_photo(image=b"c" * 10, media_type="image/png",
                        filename="c.png", width=10, height=10,
                        mode=PHOTO_SINGLE)
        with pytest.raises(KeyError):
            store.photo(first.handle)

    def test_the_byte_budget_is_enforced_as_well_as_the_count(self):
        store = SessionStore(photo_limit=8, total_image_bytes=100)
        first = store.put_photo(image=b"x" * 80, media_type="image/png",
                                filename="a.png", width=10, height=10,
                                mode=PHOTO_SINGLE)
        store.put_photo(image=b"y" * 80, media_type="image/png",
                        filename="b.png", width=10, height=10,
                        mode=PHOTO_SINGLE)
        with pytest.raises(KeyError):
            store.photo(first.handle)

    def test_clear_removes_everything(self):
        store = SessionStore()
        store.put_photo(image=b"x", media_type="image/png", filename="a.png",
                        width=1, height=1, mode=PHOTO_SINGLE)
        store.put_result("x")
        store.clear()
        assert store.counts() == {"photos": 0, "results": 0,
                                 "image_bytes": 0}


class TestNoWeightsWereLoaded:
    def test_the_ui_package_reaches_no_generation_entry_point_by_default(
            self, live):
        """The two-page interface's guarantee, still true of the new pages.

        The project-model entry is the one deliberate exception, and it is the
        only place that imports a loader -- inside a function, after the
        pointer has been verified.
        """
        import src.ui.full as full
        import src.ui.render_full as render_full
        import src.ui.requests as requests
        import src.ui.server_full as server_module

        for module in (full, render_full, requests, server_module):
            source = open(module.__file__, encoding="utf-8").read()
            top_level = [line for line in source.splitlines()
                         if line.startswith(("import ", "from "))]
            joined = "\n".join(top_level)
            assert "brickgpt" not in joined
            assert "training.lora" not in joined

    def test_the_upload_route_writes_nothing_to_disk(self, live, tmp_path):
        before = set(p.name for p in tmp_path.iterdir())
        live.post_multipart("/photo", image=png(),
                            fields={"photo_mode": PHOTO_MULTI})
        assert set(p.name for p in tmp_path.iterdir()) == before
