"""HTTP 服务：静态页面 + 业务 API + 健康检查（仅依赖 Python 标准库）。

路由
----
GET  /healthz              健康响应（不依赖业务状态，永远 200）
GET  /                     复核页面
GET  /static/app.js        页面脚本
GET  /api/state            当前配置 / 日志 / 轨迹 / 修订号
POST /api/config           建立二维位置与速度初值、噪声与滞后长度
POST /api/observations     按接收顺序录入一条观测
POST /api/reset            清空日志与轨迹（重新建档）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from .kf import KalmanError
from .store import Store, StoreError

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
_MAX_BODY = 64 * 1024


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _static_response(handler: BaseHTTPRequestHandler, path: str, ctype: str) -> None:
    try:
        with open(path, "rb") as fh:
            body = fh.read()
    except OSError:
        _json_response(handler, 404, {"error": "not found"})
        return
    handler.send_response(200)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(store: Store):
    class Handler(BaseHTTPRequestHandler):
        server_version = "NavReview/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:  # 安静：验收输出更清晰
            if os.environ.get("HTTP_LOG"):
                sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        # -- GET --------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
            if route == "/healthz":
                _json_response(self, 200, {"status": "ok", "service": "nav-review"})
                return
            if route == "/api/state":
                _json_response(self, 200, store.state())
                return
            if route == "/" or route == "/index.html":
                _static_response(
                    self, os.path.join(STATIC_DIR, "index.html"), "text/html; charset=utf-8"
                )
                return
            if route == "/static/app.js":
                _static_response(
                    self, os.path.join(STATIC_DIR, "app.js"), "application/javascript; charset=utf-8"
                )
                return
            _json_response(self, 404, {"error": "not found", "path": route})

        # -- POST -------------------------------------------------------

        def do_POST(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length > _MAX_BODY:
                _json_response(self, 413, {"error": "请求体过大"})
                return
            raw = self.rfile.read(length) if length else b""
            payload: Optional[Any]
            if raw:
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    _json_response(self, 400, {"error": "请求体不是合法 JSON"})
                    return
            else:
                payload = {}

            try:
                if route == "/api/config":
                    config = Store.parse_config(payload)
                    _json_response(self, 200, store.initialize_config(config))
                    return
                if route == "/api/observations":
                    _json_response(self, 200, store.submit_observation(payload))
                    return
                if route == "/api/reset":
                    _json_response(self, 200, store.reset())
                    return
            except KalmanError as exc:
                # 非法噪声/不可逆创新等：422 且明确说明失败原因，轨迹保持不变
                _json_response(self, 422, {"error": str(exc), "kind": "kalman_error"})
                return
            except StoreError as exc:
                _json_response(self, 400, {"error": str(exc), "kind": "store_error"})
                return
            _json_response(self, 404, {"error": "not found", "path": route})

    return Handler


def build_server(port: int, state_path: str) -> ThreadingHTTPServer:
    store = Store(path=state_path or None)
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(store))
    server.store = store  # type: ignore[attr-defined]
    return server


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="导航轨迹固定滞后复核服务")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument(
        "--state",
        default=os.environ.get("STATE_FILE", "/data/state.json"),
        help="持久化状态文件路径",
    )
    args = parser.parse_args(argv)
    server = build_server(args.port, args.state)
    print(f"nav-review listening on 0.0.0.0:{args.port} state={args.state}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
