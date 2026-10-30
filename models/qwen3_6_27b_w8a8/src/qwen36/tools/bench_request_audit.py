# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Optional request-only audit; bench client, sampling, and SSE stream unchanged."""

import asyncio, hashlib, json
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
        sequence = self.sequence
        parts = []
        status = None

        async def audited_receive():
            message = await receive()
            if message["type"] == "http.request":
                parts.append(message.get("body", b""))
            return message

        async def audited_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, audited_receive, audited_send)
        finally:
            raw = b"".join(parts)
            headers = dict(scope.get("headers", []))
            rid = headers.get(b"x-request-id", b"").decode(errors="replace")

            def save():
                body = json.loads(raw) if raw else None
                prompt = json.dumps(
                    body.get("prompt") if body else None, ensure_ascii=False, separators=(",", ":")
                ).encode()
                result = dict(
                    sequence=sequence,
                    request_id=rid,
                    http_status=status,
                    prompt_sha256=hashlib.sha256(prompt).hexdigest(),
                    body_sha256=hashlib.sha256(raw).hexdigest(),
                    body=body,
                )
                (self.output / f"request_{sequence:02d}.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n"
                )

            await asyncio.to_thread(save)
