"""The HTTP layer: loopback only, and thin on purpose.

This module moves bytes.  Every decision it could be tempted to make -- what a
valid submission is, which fields apply, whether a result may be delivered --
already belongs to :mod:`src.ui.app` and, behind it, to
``scripts/27_delivery.py``.  What is left here is routing, the refusals that
are properly the transport's business, and the rule that a defect is reported
as a sentence rather than as a traceback.

The transport's own refusals:

* **Loopback only.**  The socket binds a loopback address and nothing else,
  and a request whose ``Host`` header names anything but loopback is refused.
  The second check is not redundant: without it a name that resolves to
  127.0.0.1 lets any page in the browser drive this server.
* **Origin screening, and a form key.**  Binding loopback keeps the *network*
  out; it does nothing about the browser.  Any page the operator has open can
  post a form to ``http://127.0.0.1:8765/result``, and the browser will send
  it.  A structured external ``Origin`` is therefore refused before its body
  is read.  An absent or opaque (``null``) origin proceeds to the second,
  authoritative check: the unguessable per-process key page one puts in its
  form.  This distinction matters because a real top-level browser submission
  has been observed to serialise its origin as ``null``; rejecting that value
  before reading the key locks out the operator without adding protection.
  A cross-origin page can cause a post but cannot read the key out of a page
  it is not allowed to read.
* **A bounded body.**  A submission larger than
  :data:`MAX_BODY_BYTES` is refused rather than read.
* **One content type.**  Only ``application/x-www-form-urlencoded``.
* **Opaque artifact handles.**  A handle is matched against a strict pattern
  and looked up in an in-memory map, so no request can name a path on disk.

One consequence of refusing early deserves stating, because getting it wrong
is silent: on a keep-alive connection, answering a request whose body was
never read leaves those bytes in the socket, and the next request on that
connection starts parsing mid-body.  A refusal that skips the body therefore
closes the connection, and :meth:`UiHandler._send` does that from the one
place every response goes through rather than at each refusal site.
"""

from __future__ import annotations

import hmac
import secrets
import socket
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from urllib.parse import parse_qs, urlsplit

from src.ui import app as ui_app
from src.ui import render

#: A form submission has no business being larger than this.
MAX_BODY_BYTES = 64 * 1024

FORM_MEDIA_TYPE = "application/x-www-form-urlencoded"

#: The hidden field page one carries and ``/result`` requires.
CSRF_FIELD = "csrf_token"

#: The only addresses this server will bind.  There is no flag to widen it:
#: a demonstration that measures nothing has nothing to serve to a network.
LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")

#: Host header values a browser may legitimately send for a loopback server.
LOOPBACK_HEADER_HOSTS = frozenset(
    {"127.0.0.1", "localhost", "::1", "[::1]", "localhost.localdomain"})

#: Schemes a structured ``Origin`` may use.  The opaque spelling ``null`` is
#: handled separately: it carries no trustworthy same-origin evidence, but it
#: may proceed to the independent form-key check rather than being trusted.
ORIGIN_SCHEMES = ("http", "https")

_HANDLE_ALPHABET = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")


def is_loopback_host(host: str) -> bool:
    """Whether a bind address is loopback. Anything unresolved is not."""
    if host in LOOPBACK_HOSTS:
        return True
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        address = info[4][0]
        if address not in ("127.0.0.1", "::1") and not address.startswith(
                "127."):
            return False
    return bool(infos)


def _split_host(raw: str) -> str:
    """``127.0.0.1:8765`` -> ``127.0.0.1``; ``[::1]:8765`` -> ``[::1]``."""
    host = raw.strip()
    if host.startswith("["):
        closing = host.find("]")
        return host[:closing + 1] if closing != -1 else ""
    return host.rsplit(":", 1)[0] if ":" in host else host


def _header_host_ok(raw: str | None) -> bool:
    """Whether a ``Host`` header names loopback. Absent is refused."""
    if not raw:
        return False
    return _split_host(raw).lower() in LOOPBACK_HEADER_HOSTS


def origin_verdict(raw: str | None, port: int) -> str:
    """Classify an ``Origin`` as absent, opaque, same or external.

    A structured origin uses scheme, host *and* port, with the host allowed to
    be any loopback spelling.  ``null`` is deliberately *not* called same: it
    provides no origin evidence.  It is called opaque so the caller can still
    require the independent form key.  A different port on the same machine
    remains external, which is the case a host-only check would wave through.
    """
    if raw is None or not raw.strip():
        return "absent"
    value = raw.strip()
    if value.lower() == "null":
        return "opaque"
    parts = urlsplit(value)
    if parts.scheme.lower() not in ORIGIN_SCHEMES or not parts.netloc:
        return "external"
    if _split_host(parts.netloc).lower() not in LOOPBACK_HEADER_HOSTS:
        return "external"
    try:
        seen = parts.port
    except ValueError:
        return "external"
    default = 443 if parts.scheme.lower() == "https" else 80
    return "same" if (seen if seen is not None else default) == port \
        else "external"


def _handle_ok(handle: str) -> bool:
    return (bool(handle) and len(handle) <= 64
            and set(handle) <= _HANDLE_ALPHABET)


class UiServer(ThreadingHTTPServer):
    """The server, carrying the one piece of configuration the UI needs."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, *, catalog: Path,
                 delivery: ModuleType, store: ui_app.ResultStore) -> None:
        self.catalog = Path(catalog)
        self.delivery = delivery
        self.store = store
        # New every time the process starts, so a key left in a stale tab
        # stops working rather than continuing to authorise submissions.
        self.csrf_key = secrets.token_urlsafe(32)
        super().__init__(address, handler)

    @property
    def port(self) -> int:
        return int(self.server_address[1])


class UiHandler(BaseHTTPRequestHandler):
    server_version = "BrickAgainUI/1.0"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    #: Whether this request's body has been taken off the socket. A response
    #: sent while it has not must close the connection; see the module
    #: docstring.
    _body_consumed = False

    # -- plumbing ---------------------------------------------------------
    def handle_one_request(self):
        self._body_consumed = False
        super().handle_one_request()

    def log_message(self, fmt, *args):          # noqa: D102 - quieter default
        sys.stderr.write("[ui] %s %s\n" % (self.address_string(), fmt % args))

    def _send(self, status: int, body: bytes, media_type: str, *,
              filename: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        # Everything the pages need is inline and same-origin; nothing here
        # may reach the network, and the policy says so rather than trusting
        # that no future edit adds a remote asset.
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; "
            "script-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; "
            "frame-ancestors 'none'")
        if filename:
            self.send_header("Content-Disposition",
                             f'attachment; filename="{filename}"')
        # The one place that knows whether the body was read is the one place
        # that decides whether this connection may carry another request.
        if self.headers.get("Content-Length") and not self._body_consumed:
            self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _html(self, status: int, html: str) -> None:
        self._send(status, html.encode("utf-8"), "text/html; charset=utf-8")

    def _refuse_page(self, status: int, heading: str, detail: str,
                     advice: str) -> None:
        self._html(status, render.render_error(
            heading=heading, detail=detail, advice=advice))

    def _guard(self) -> bool:
        """Refuse a non-loopback ``Host``. True means the request may proceed."""
        if _header_host_ok(self.headers.get("Host")):
            return True
        self._refuse_page(
            403, "只服務本機",
            "這個 Host 標頭不是 loopback。此介面只在本機使用，"
            "不對外提供服務，也不接受由其他名稱轉入的請求。",
            "請直接以 http://127.0.0.1:<port>/ 開啟。")
        return False

    # -- routes -----------------------------------------------------------
    def do_GET(self) -> None:                    # noqa: N802 - stdlib name
        if not self._guard():
            return
        path = urlsplit(self.path).path
        try:
            if path == "/":
                self._page_one()
            elif path.startswith("/artifact/"):
                self._artifact(path)
            else:
                self._not_found()
        except Exception:                        # noqa: BLE001 - see _defect
            self._defect()

    do_HEAD = do_GET

    def do_POST(self) -> None:                   # noqa: N802 - stdlib name
        if not self._guard():
            return
        if urlsplit(self.path).path != "/result":
            self._not_found()
            return
        try:
            if not self._same_origin():
                return
            fields = self._read_form()
            if fields is None:
                return
            if not self._form_key_ok(fields):
                return
            self._result(fields)
        except Exception:                        # noqa: BLE001 - see _defect
            self._defect()

    # -- the two cross-site checks ----------------------------------------
    def _same_origin(self) -> bool:
        """Refuse a structured external origin before reading its body.

        Absent and opaque origins are not trusted as same-origin.  They merely
        proceed to :meth:`_form_key_ok`, whose unpredictable key remains the
        authority for every accepted submission.
        """
        verdict = origin_verdict(self.headers.get("Origin"), self.server.port)
        if verdict in {"absent", "opaque", "same"}:
            return True
        self._refuse_page(
            403, "這個送出來自其他來源",
            f"Origin 標頭是 {(self.headers.get('Origin') or '')[:120]!r}，"
            "不是這個介面自己的來源。此介面只接受由它自己送出的表單，"
            "不接受任何跨來源的請求。",
            "請在 http://127.0.0.1:<port>/ 開啟第一頁後再送出。")
        return False

    def _form_key_ok(self, fields: dict[str, list[str]]) -> bool:
        """Refuse a submission that does not carry page one's own key."""
        supplied = fields.get(CSRF_FIELD) or []
        expected = self.server.csrf_key
        if len(supplied) != 1 or not supplied[0]:
            self._refuse_page(
                403, "這個送出沒有帶第一頁的表單金鑰",
                f"缺少 {CSRF_FIELD}，或它被送出了 {len(supplied)} 次。"
                "每一次送出都必須帶著第一頁發出的那一把金鑰；"
                "沒有金鑰的請求一律拒絕，不會被當成一般送出處理。",
                "請重新開啟第一頁再送出。伺服器重新啟動後舊金鑰即失效。")
            return False
        if not hmac.compare_digest(supplied[0], expected):
            self._refuse_page(
                403, "表單金鑰不符",
                "送出的金鑰不是這個行程發出的那一把。可能是頁面停留太久、"
                "伺服器重新啟動過，或這個請求根本不是由第一頁送出的。",
                "請重新開啟第一頁再送出。")
            return False
        return True

    # -- handlers ---------------------------------------------------------
    def _page_one(self, *, form=None, error=None, status=200) -> None:
        catalog = self.server.catalog
        self._html(status, render.render_page_one(
            csrf_token=self.server.csrf_key,
            form=form, error=error, catalog_present=catalog.is_file(),
            catalog_display=str(catalog)))

    def _read_form(self) -> dict[str, list[str]] | None:
        if self.headers.get("Transfer-Encoding"):
            self._refuse_page(
                411, "不接受分塊傳輸",
                "本介面只讀取帶 Content-Length 的表單本體。",
                "請從第一頁的表單送出。")
            return None
        media_type = (self.headers.get("Content-Type") or "").split(";")[0]
        if media_type.strip().lower() != FORM_MEDIA_TYPE:
            self._refuse_page(
                415, "只接受表單送出",
                f"這個請求的 Content-Type 是 {media_type!r}，"
                f"本介面只處理 {FORM_MEDIA_TYPE}。",
                "請從第一頁的表單送出。")
            return None
        raw_length = (self.headers.get("Content-Length") or "").strip()
        if not raw_length.isascii() or not raw_length.isdigit():
            self._refuse_page(
                411, "缺少 Content-Length",
                "沒有長度、或長度不是十進位整數的請求本體不會被讀取。",
                "請從第一頁的表單送出。")
            return None
        length = int(raw_length) if len(raw_length) <= 12 else MAX_BODY_BYTES + 1
        if length > MAX_BODY_BYTES:
            self._refuse_page(
                413, "送出的內容過大",
                f"本體 {raw_length} 位元組超過上限 {MAX_BODY_BYTES}。"
                "本介面拒絕讀取，而不是先讀進來再說。",
                "請縮短文字需求或庫存字串後重試。")
            return None
        raw = self.rfile.read(length) if length else b""
        self._body_consumed = True
        return parse_qs(raw.decode("utf-8", errors="replace"),
                        keep_blank_values=True)

    def _result(self, fields: dict[str, list[str]]) -> None:
        try:
            request = ui_app.parse_form(fields)
            result = ui_app.run_request(
                request, catalog=self.server.catalog,
                delivery=self.server.delivery, store=self.server.store)
        except ui_app.REFUSALS as exc:
            # A refusal returns to page one with the message and everything
            # that was typed, because an error page that discards the form is
            # an error page that gets retyped.
            self._page_one(form=render.form_state(fields), error=str(exc),
                           status=400)
            return
        self._html(200, render.render_page_two(result))

    def _artifact(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) != 3 or not _handle_ok(parts[1]):
            self._not_found()
            return
        _, handle, name = parts
        artifacts = self.server.store.get(handle)
        if artifacts is None:
            self._refuse_page(
                404, "這份輸出已經不在記憶體裡",
                "預覽與 LDraw 只存在於本次執行的記憶體中，"
                "不寫入專案目錄，也不會保留到下一次啟動。",
                "請回到第一頁重新送出同一組需求與庫存。")
            return
        if name == "preview.png":
            self._send(200, artifacts.preview, artifacts.preview_media_type)
        elif name == "model.ldr":
            self._send(200, artifacts.ldraw.encode("utf-8"),
                       "text/plain; charset=utf-8",
                       filename="brickagain.ldr")
        else:
            self._not_found()

    def _not_found(self) -> None:
        self._refuse_page(
            404, "沒有這個頁面",
            f"{urlsplit(self.path).path!r} 不是本介面的路徑。"
            "介面只有兩頁，加上預覽與 LDraw 兩個輸出。",
            "請回到第一頁重新開始。")

    def _defect(self) -> None:
        """An unexpected failure. The reader gets a sentence, stderr gets it all.

        A traceback in the page would be both unreadable and a disclosure of
        local paths, so it goes to the console the operator started the server
        from and nowhere else.
        """
        traceback.print_exc()
        try:
            self._refuse_page(
                500, "介面內部發生未預期的錯誤",
                "這次沒有產生任何結果、預覽或下載。"
                "詳細診斷已寫到啟動這個伺服器的終端機，不顯示在頁面上。",
                "請回到第一頁重試；若持續發生，請把終端機輸出附在回報中。")
        except Exception:                        # noqa: BLE001
            pass


def create_server(*, host: str = "127.0.0.1", port: int = 8765,
                  catalog: str | Path | None = None,
                  delivery: ModuleType | None = None,
                  store: ui_app.ResultStore | None = None) -> UiServer:
    """Build the server without starting it. ``port=0`` picks a free port.

    Refuses a non-loopback bind address by name.  There is deliberately no
    override: publishing, deploying or exposing this interface is outside what
    it is for.
    """
    if not is_loopback_host(host):
        raise ui_app.UiError(
            f"host {host!r} is not a loopback address; this interface binds "
            "loopback only and is not for publishing or deployment")
    module = delivery or ui_app.load_delivery()
    target = (Path(catalog) if catalog is not None
              else ui_app.default_catalog(module))
    return UiServer((host, port), UiHandler, catalog=target, delivery=module,
                    store=store or ui_app.ResultStore())


def serve(*, host: str = "127.0.0.1", port: int = 8765,
          catalog: str | Path | None = None) -> int:
    """Run until interrupted. Returns a process exit code."""
    server = create_server(host=host, port=port, catalog=catalog)
    bound_host, bound_port = server.server_address[:2]
    catalogue = server.catalog
    print("BrickAgain 最小兩頁式介面")
    print(f"  網址      : http://{bound_host}:{bound_port}/")
    print(f"  目錄檔    : {catalogue}"
          + ("" if catalogue.is_file() else "  (不存在：送出後會被拒絕)"))
    print("  邊界      : CPU、離線、只綁 loopback；不載入權重、"
          "不使用 GPU、不執行正式評估")
    print("  跨站防護  : 外部網址來源拒絕；所有送出都必須帶第一頁的表單金鑰")
    print("  停止      : Ctrl-C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。預覽與 LDraw 只存在記憶體中，沒有留下任何檔案。")
    finally:
        server.server_close()
    return 0
