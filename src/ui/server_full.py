"""The full interface's HTTP layer, on the existing one's guards.

:class:`~src.ui.server.UiHandler` already implements every transport-level
refusal this project decided on: loopback-only binding plus a ``Host`` check,
structured external ``Origin`` refused before its body is read, an opaque
``null`` origin allowed through only to the form-key check, a per-process
unguessable form key compared with ``hmac.compare_digest``, a bounded body, a
strict content type, ``Connection: close`` on any refusal that skipped the body
so a keep-alive connection cannot be poisoned, ``frame-ancestors 'none'``, and a
defect reported as a sentence with the traceback going to the console.

So this module **subclasses** it rather than re-implementing any of that.  The
guards here are the same code, not a copy of it -- which is the only way to be
sure the full interface has not quietly lost one of them.  What is added:

* one more accepted content type, ``multipart/form-data``, with its own
  bounded parser in :mod:`src.ui.upload`;
* routes for the photograph, the corrections, the result and the build steps;
* an in-process session store, because a four-page flow has state and a
  two-page one did not.

The additions keep the boundaries: no request can name a path on disk, nothing
is written outside a temporary directory that is removed before the response is
sent, and every uploaded byte lives in a bounded in-memory store that does not
survive the process.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from urllib.parse import urlsplit

from src.ui import app as ui_app
from src.ui import full as full_module
from src.ui import render_full
from src.ui.server import CSRF_FIELD, UiHandler, UiServer, is_loopback_host
from src.ui.state import SessionStore, StateError
from src.ui.upload import (MAX_UPLOAD_BYTES, MULTIPART_MEDIA_TYPE, UploadError,
                           parse_multipart)

#: A multipart body may be larger than a form body, and only on the one route
#: that takes a photograph.
MAX_MULTIPART_BYTES = MAX_UPLOAD_BYTES + 64 * 1024

#: The fields the upload route accepts.  Anything else is refused by name.
PHOTO_IMAGE_FIELDS = ("photo",)
PHOTO_TEXT_FIELDS = (CSRF_FIELD, "photo_mode", "recognise")

#: Assembly steps a page may ask for.  The planner has its own much larger
#: cap; this one keeps a hand-typed URL from doing arithmetic on nothing.
MAX_STEP_NUMBER = 4096


class FullServer(UiServer):
    """The two-page server plus the state and configuration a flow needs."""

    def __init__(self, address, handler, *, catalog: Path,
                 delivery: ModuleType, store: ui_app.ResultStore,
                 sessions: SessionStore, index_dir: Path | None = None,
                 checkpoint: Path | None = None, device: str | None = None,
                 project_root: Path | None = None,
                 allow_project_model: bool = True) -> None:
        self.sessions = sessions
        self.index_dir = Path(index_dir) if index_dir else None
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.device = device
        self.project_root = Path(project_root) if project_root else None
        self.allow_project_model = bool(allow_project_model)
        super().__init__(address, handler, catalog=catalog, delivery=delivery,
                         store=store)

    @property
    def checkpoint_present(self) -> bool:
        return bool(self.checkpoint and self.checkpoint.is_dir())


class FullHandler(UiHandler):
    """Routing only.  Every refusal is inherited."""

    server_version = "BrickAgainFullUI/1.0"

    # -- refusals and pages use the full interface's own templates --------
    def _refuse_page(self, status, heading, detail, advice) -> None:
        self._html(status, render_full.render_error(
            heading=heading, detail=detail, advice=advice))

    def _start(self, *, form=None, error=None, notice=None,
               photo_handle=None, status=200) -> None:
        self._html(status, render_full.render_start(
            csrf_token=self.server.csrf_key, form=form, error=error,
            notice=notice, photo_handle=photo_handle,
            checkpoint_present=self.server.checkpoint_present,
            checkpoint_display=(str(self.server.checkpoint)
                                if self.server.checkpoint else "")))

    # -- reading a multipart body -----------------------------------------
    def _read_multipart(self):
        """Read and parse a multipart body, or refuse and return ``None``.

        The order matters: the transfer encoding, the content type and the
        declared length are all checked *before* anything is read, so an
        oversized or chunked upload costs a refusal rather than memory. Every
        refusal here skips the body, and the shared ``_send`` therefore closes
        the connection.
        """
        if self.headers.get("Transfer-Encoding"):
            self._refuse_page(
                411, "不接受分塊傳輸",
                "本介面只讀取帶 Content-Length 的請求本體。",
                "請從第一步的表單送出。")
            return None
        media_type = (self.headers.get("Content-Type") or "").split(";")[0]
        if media_type.strip().lower() != MULTIPART_MEDIA_TYPE:
            self._refuse_page(
                415, "這個路徑只接受檔案上傳表單",
                f"這個請求的 Content-Type 是 {media_type!r}，"
                f"本路徑只處理 {MULTIPART_MEDIA_TYPE}。",
                "請從第一步的照片表單送出。")
            return None
        raw_length = (self.headers.get("Content-Length") or "").strip()
        if not raw_length.isascii() or not raw_length.isdigit():
            self._refuse_page(
                411, "缺少 Content-Length",
                "沒有長度、或長度不是十進位整數的請求本體不會被讀取。",
                "請從第一步的照片表單送出。")
            return None
        length = (int(raw_length) if len(raw_length) <= 12
                  else MAX_MULTIPART_BYTES + 1)
        if length > MAX_MULTIPART_BYTES:
            self._refuse_page(
                413, "上傳的內容過大",
                f"本體 {raw_length} 位元組超過上限 {MAX_MULTIPART_BYTES}。"
                "本介面拒絕讀取，而不是先讀進來再說。",
                "請改用較小的照片。")
            return None
        raw = self.rfile.read(length) if length else b""
        self._body_consumed = True
        try:
            return parse_multipart(
                raw, self.headers.get("Content-Type") or "",
                image_fields=PHOTO_IMAGE_FIELDS,
                text_fields=PHOTO_TEXT_FIELDS,
                max_bytes=MAX_MULTIPART_BYTES)
        except UploadError as exc:
            self._refuse_page(
                400, "這次上傳被拒絕", str(exc),
                "請確認是 PNG 或 JPEG，且沒有超過大小與像素上限。")
            return None

    # -- routing ----------------------------------------------------------
    def do_GET(self) -> None:                    # noqa: N802 - stdlib name
        if not self._guard():
            return
        path = urlsplit(self.path).path
        try:
            parts = [piece for piece in path.strip("/").split("/") if piece]
            if path == "/":
                self._start()
            elif path == "/reset":
                self.server.sessions.clear()
                self._start(notice="已清除本次執行的所有照片、修正與結果。")
            elif len(parts) == 3 and parts[0] == "photo" \
                    and parts[2] == "image":
                self._serve_photo(parts[1])
            elif len(parts) == 3 and parts[0] == "artifact":
                self._artifact(path)
            elif len(parts) == 2 and parts[0] == "result":
                self._show_result(parts[1])
            elif len(parts) == 3 and parts[0] == "steps":
                self._steps_page(parts[1], parts[2])
            elif len(parts) == 4 and parts[0] == "steps" \
                    and parts[3] == "image":
                self._step_image(parts[1], parts[2])
            else:
                self._not_found()
        except Exception:                        # noqa: BLE001 - see _defect
            self._defect()

    do_HEAD = do_GET

    def do_POST(self) -> None:                   # noqa: N802 - stdlib name
        if not self._guard():
            return
        path = urlsplit(self.path).path
        parts = [piece for piece in path.strip("/").split("/") if piece]
        try:
            if not self._same_origin():
                return
            if path == "/photo":
                body = self._read_multipart()
                if body is None:
                    return
                if not self._form_key_ok(body.fields):
                    return
                self._analyse(body)
                return
            fields = self._read_form()
            if fields is None:
                return
            if not self._form_key_ok(fields):
                return
            if len(parts) == 3 and parts[0] == "photo" \
                    and parts[2] == "correct":
                self._correct(parts[1], fields)
            elif path == "/result":
                self._run(fields)
            else:
                self._not_found()
        except Exception:                        # noqa: BLE001 - see _defect
            self._defect()

    # -- handlers ---------------------------------------------------------
    def _analyse(self, body) -> None:
        from src.ui.corrections import adopt

        try:
            image = body.one_image("photo")
            if image is None:
                raise ui_app.UiError("沒有收到照片檔；請選擇一張 PNG 或 JPEG。")
            mode = (body.fields.get("photo_mode") or [""])[0].strip()
            method = (body.fields.get("recognise") or [""])[0].strip() \
                or full_module.RECOGNISE_CV
            analysis = full_module.analyse_photo(
                image.data, mode=mode or full_module.PHOTO_SINGLE,
                method=method, checkpoint=self.server.checkpoint,
                device=self.server.device)
        except (ui_app.UiError, UploadError) as exc:
            self._start(error=str(exc), status=400)
            return
        session = self.server.sessions.put_photo(
            image=image.data, media_type=(image.media_type
                                          or "application/octet-stream"),
            filename=image.filename, width=analysis.width,
            height=analysis.height, mode=analysis.mode,
            items=analysis.items, diagnostics=analysis.diagnostics,
            method=analysis.method)
        baseline = adopt(analysis.items)
        self._html(200, render_full.render_photo(
            csrf_token=self.server.csrf_key, handle=session.handle,
            filename=session.filename, analysis=analysis,
            items=session.items, before=baseline, corrected=baseline))

    def _correct(self, handle: str, fields) -> None:
        from src.ui.corrections import adopt

        if not _handle_ok(handle):
            self._not_found()
            return
        try:
            session = self.server.sessions.photo(handle)
        except StateError:
            self._expired()
            return
        baseline = adopt(session.items)
        try:
            items = _apply_form(session, fields)
        except ui_app.UiError as exc:
            analysis = _analysis_of(session)
            self._html(400, render_full.render_photo(
                csrf_token=self.server.csrf_key, handle=handle,
                filename=session.filename, analysis=analysis,
                items=session.items, before=baseline, corrected=baseline,
                error=str(exc)))
            return
        updated = self.server.sessions.update_photo(handle, items=items)
        self._html(200, render_full.render_photo(
            csrf_token=self.server.csrf_key, handle=handle,
            filename=updated.filename, analysis=_analysis_of(updated),
            items=updated.items, before=baseline, corrected=adopt(items)))

    def _serve_photo(self, handle: str) -> None:
        if not _handle_ok(handle):
            self._not_found()
            return
        try:
            session = self.server.sessions.photo(handle)
        except StateError:
            self._expired()
            return
        self._send(200, session.image, session.media_type or "image/png")

    def _run(self, fields) -> None:
        from src.ui.requests import RequestError, build_request, execute

        try:
            request = build_request(fields, server=self.server)
            result, finished, spec, origin = execute(request,
                                                     server=self.server)
        except (ui_app.UiError, RequestError, *ui_app.REFUSALS) as exc:
            self._start(form=render_full.form_state(fields), error=str(exc),
                        status=400)
            return
        session = self.server.sessions.put_result(
            request.method, payload=result.as_dict(), report=result.report,
            artifacts=finished, extra={"spec": spec, "origin": origin,
                                       "result": result})
        self._html(200, render_full.render_result(
            result=result, handle=session.handle, inventory_spec=spec,
            inventory_origin=origin, finished=finished,
            not_ready_reason=_not_ready(result)))

    def _show_result(self, handle: str) -> None:
        if not _handle_ok(handle):
            self._not_found()
            return
        try:
            session = self.server.sessions.result(handle)
        except StateError:
            self._expired()
            return
        result = session.extra["result"]
        self._html(200, render_full.render_result(
            result=result, handle=handle,
            inventory_spec=session.extra["spec"],
            inventory_origin=session.extra["origin"],
            finished=session.artifacts, not_ready_reason=_not_ready(result)))

    def _artifact(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) != 3 or not _handle_ok(parts[1]):
            self._not_found()
            return
        _, handle, name = parts
        try:
            session = self.server.sessions.result(handle)
        except StateError:
            self._expired()
            return
        finished = session.artifacts
        if finished is None:
            self._refuse_page(
                404, "這份結果沒有可下載的輸出",
                "只有通過靜態交付檢查的結果才會產生預覽與 LDraw。",
                "請回到第一步調整庫存或需求。")
            return
        if name == "preview.png":
            self._send(200, finished.preview, finished.preview_media_type)
        elif name == "model.ldr":
            self._send(200, finished.ldraw.encode("utf-8"),
                       "text/plain; charset=utf-8",
                       filename="brickagain.ldr")
        else:
            self._not_found()

    def _steps_page(self, handle: str, raw_number: str) -> None:
        from src.assembly.order import step_descriptions

        session, number = self._step_context(handle, raw_number)
        if session is None:
            return
        plan = session.artifacts.plan
        self._html(200, render_full.render_steps(
            handle=handle, plan=plan, number=number,
            descriptions=step_descriptions(plan)))

    def _step_image(self, handle: str, raw_number: str) -> None:
        session, number = self._step_context(handle, raw_number)
        if session is None:
            return
        images = session.artifacts.step_previews
        if not 1 <= number <= len(images):
            self._not_found()
            return
        self._send(200, images[number - 1], "image/png")

    def _step_context(self, handle: str, raw_number: str):
        if not _handle_ok(handle):
            self._not_found()
            return None, 0
        if not raw_number.isascii() or not raw_number.isdigit() \
                or len(raw_number) > 5:
            self._not_found()
            return None, 0
        number = int(raw_number)
        try:
            session = self.server.sessions.result(handle)
        except StateError:
            self._expired()
            return None, 0
        finished = session.artifacts
        if finished is None or finished.plan is None:
            self._refuse_page(
                404, "這份結果沒有組裝步驟",
                (finished.plan_problem if finished
                 and finished.plan_problem else
                 "只有通過靜態交付檢查且能排出合法順序的結構才有組裝步驟。"),
                "請回到第一步調整庫存或需求。")
            return None, 0
        if not 1 <= number <= min(finished.plan.n_steps, MAX_STEP_NUMBER):
            self._refuse_page(
                404, "沒有這個步驟",
                f"這件作品有 {finished.plan.n_steps} 個步驟，"
                f"沒有第 {number} 步。",
                "請用頁面上的上一步／下一步按鈕。")
            return None, 0
        return session, number

    def _expired(self) -> None:
        self._refuse_page(
            404, "這份資料已經不在記憶體裡",
            "照片、修正與結果只存在於本次執行的記憶體中，"
            "不寫入專案目錄，也不會保留到下一次啟動；"
            "太舊的內容會被較新的取代。",
            "請回到第一步重新上傳或重新送出。")


def _handle_ok(handle: str) -> bool:
    from src.ui.server import _HANDLE_ALPHABET

    return (bool(handle) and len(handle) <= 64
            and set(handle) <= _HANDLE_ALPHABET)


def _analysis_of(session):
    """Rebuild the display record for a stored photograph session."""
    return full_module.PhotoAnalysis(
        mode=session.mode, method=session.method, items=session.items,
        diagnostics=session.diagnostics, width=session.width,
        height=session.height)


def _apply_form(session, fields):
    """Read the correction form into edits, then apply them."""
    edits: dict[int, dict] = {}
    for item in session.items:
        index = item.index
        changes: dict = {}
        part = _one(fields, f"part_{index}").strip()
        if part:
            changes["part"] = part
        count = _one(fields, f"count_{index}").strip()
        if count:
            if not count.isascii() or not count.isdigit() or len(count) > 9:
                raise ui_app.UiError(
                    f"項目 {index} 的數量只接受半形 0-9，且位數不超過 9")
            changes["count"] = int(count)
        colour = _one(fields, f"colour_{index}").strip()
        if colour:
            changes["colour"] = colour
        box = _one(fields, f"box_{index}").strip()
        if box:
            pieces = [piece.strip() for piece in box.split(",")]
            if len(pieces) != 4 or not all(
                    piece.isascii() and piece.lstrip("-").isdigit()
                    and len(piece) <= 9 for piece in pieces):
                raise ui_app.UiError(
                    f"項目 {index} 的框必須是四個半形整數，例如 10,20,120,90")
            proposed = tuple(int(piece) for piece in pieces)
            if proposed != tuple(item.box):
                changes["box"] = proposed
        if _one(fields, f"delete_{index}"):
            changes["delete"] = True
        elif item.deleted:
            changes["delete"] = False
        if changes:
            edits[index] = changes
    items = full_module.apply_corrections(
        session.items, edits, width=session.width, height=session.height)

    add_part = _one(fields, "add_part").strip()
    if add_part:
        raw_count = _one(fields, "add_count").strip() or "1"
        if not raw_count.isascii() or not raw_count.isdigit() \
                or len(raw_count) > 9:
            raise ui_app.UiError("新增項目的數量只接受半形 0-9")
        items = full_module.add_correction(
            items, part=add_part, count=int(raw_count),
            colour_id=_one(fields, "add_colour").strip() or None,
            width=session.width, height=session.height)
    return items


def _one(fields, name: str) -> str:
    values = fields.get(name) or []
    if len(values) > 1:
        raise ui_app.UiError(f"欄位 {name} 被送出 {len(values)} 次；請只填一次")
    return values[0] if values else ""


def _not_ready(result) -> str:
    if result.ready:
        return ""
    report = result.report
    if report is None:
        return f"執行狀態是 {result.status}，沒有選中任何結果。"
    failed = [name for name, value in report["checks"].items()
              if value is False]
    if failed:
        return ("選出的結構沒有通過靜態交付檢查，未通過的是："
                + "、".join(sorted(failed)) + "。")
    return f"執行狀態是 {result.status}。"


def create_server(*, host: str = "127.0.0.1", port: int = 8766,
                  catalog=None, delivery=None, index_dir=None,
                  checkpoint=None, device=None, project_root=None,
                  sessions=None, store=None) -> FullServer:
    """Build the full server without starting it.  ``port=0`` picks a free one.

    Refuses a non-loopback bind address by name, exactly as the two-page
    server does and for the same reason: publishing, deploying or exposing
    this interface is outside what it is for, and there is no flag to widen it.
    """
    if not is_loopback_host(host):
        raise ui_app.UiError(
            f"host {host!r} is not a loopback address; this interface binds "
            "loopback only and is not for publishing or deployment")
    module = delivery or ui_app.load_delivery()
    target = (Path(catalog) if catalog is not None
              else ui_app.default_catalog(module))
    return FullServer(
        (host, port), FullHandler, catalog=target, delivery=module,
        store=store or ui_app.ResultStore(),
        sessions=sessions or SessionStore(), index_dir=index_dir,
        checkpoint=checkpoint, device=device, project_root=project_root)


def serve(*, host: str = "127.0.0.1", port: int = 8766, catalog=None,
          index_dir=None, checkpoint=None, device=None) -> int:
    """Run until interrupted.  Returns a process exit code."""
    server = create_server(host=host, port=port, catalog=catalog,
                           index_dir=index_dir, checkpoint=checkpoint,
                           device=device)
    bound_host, bound_port = server.server_address[:2]
    print("BrickAgain 完整介面")
    print(f"  網址      : http://{bound_host}:{bound_port}/")
    print(f"  目錄檔    : {server.catalog}"
          + ("" if server.catalog.is_file() else "  (不存在：送出後會被拒絕)"))
    print(f"  RAG 索引  : {server.index_dir or '(未提供，RAG 會被拒絕)'}")
    print(f"  視覺模型  : {server.checkpoint or '(未提供，只用 CV baseline)'}")
    print("  邊界      : 本機、離線、只綁 loopback；照片與結果只存在記憶體")
    print("  正式模型  : final_H2 由 runs/project_model.json 驗證後載入；"
          "不重訓、不調參、不重選")
    print("  停止      : Ctrl-C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。照片、預覽與 LDraw 只存在記憶體中，沒有留下任何檔案。")
    finally:
        server.server_close()
    return 0
