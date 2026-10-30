# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pass-through ASGI audit. Buffer timestamps/raw SSE in RAM; persist after response.
Server send timestamps are explicitly different from official client receipt times.
"""

import asyncio, hashlib, json, time
from pathlib import Path


class RequestAuditMiddleware:
    def __init__(self, app, output):
        self.app = app
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.sequence = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST" or scope.get("path") != "/v1/completions":
            return await self.app(scope, receive, send)
        self.sequence += 1
        seq = self.sequence
        parts = []
        chunks = []
        status = None
        start = time.monotonic_ns()

        async def recv():
            msg = await receive()
            if msg["type"] == "http.request":
                parts.append(msg.get("body", b""))
            return msg

        async def emit(msg):
            nonlocal status
            if msg["type"] == "http.response.start":
                status = msg["status"]
            if msg["type"] == "http.response.body" and msg.get("body"):
                chunks.append((time.monotonic_ns(), msg["body"]))
            await send(msg)

        try:
            await self.app(scope, recv, emit)
        finally:
            raw = b"".join(parts)
            headers = dict(scope.get("headers", []))
            rid = headers.get(b"x-request-id", b"").decode(errors="replace")

            def save():
                body = json.loads(raw) if raw else None
                record = dict(
                    sequence=seq,
                    request_id=rid,
                    http_status=status,
                    start_monotonic_ns=start,
                    body_sha256=hashlib.sha256(raw).hexdigest(),
                    body=body,
                    timestamp_origin="server ASGI send before socket; official client receipt ITLs saved separately",
                )
                (self.output / f"request_{seq:02d}.json").write_text(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                with (self.output / f"sse_{seq:02d}.jsonl").open("w") as out:
                    for stamp, data in chunks:
                        out.write(
                            json.dumps(
                                dict(monotonic_ns=stamp, body=data.decode("utf-8")), ensure_ascii=False
                            )
                            + "\n"
                        )

            await asyncio.to_thread(save)
