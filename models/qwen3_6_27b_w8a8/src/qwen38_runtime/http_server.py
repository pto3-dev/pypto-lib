# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native HTTP server with benchmark-only local lifecycle diagnostics."""

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stats-dir", type=Path, required=True)
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    from qwen38_runtime.engine_exit_observer import install_engine_exit_observer

    install_engine_exit_observer(options.stats_dir)
    from vllm.entrypoints.openai import api_server

    original = api_server.build_async_engine_client
    original_build_app = api_server.build_app
    audit_path = os.environ.get("QWEN36_BENCH_REQUEST_AUDIT")
    if audit_path:
        from qwen38_runtime.request_audit import RequestAuditMiddleware

        def audited_build_app(*args, **kwargs):
            app = original_build_app(*args, **kwargs)
            if app.middleware_stack is None:
                app.add_middleware(RequestAuditMiddleware, output=audit_path)
            else:
                app.middleware_stack = RequestAuditMiddleware(app.middleware_stack, output=audit_path)
            return app

        api_server.build_app = audited_build_app

    @asynccontextmanager
    async def observed_client(*args, **kwargs):
        async with original(*args, **kwargs) as engine:
            options.stats_dir.mkdir(parents=True, exist_ok=True)
            control_path = Path("/tmp") / f"qwen36-control-{os.getpid()}.sock"
            closed = False
            lock = asyncio.Lock()

            async def collect_and_close():
                nonlocal closed
                async with lock:
                    if closed:
                        return
                    stats = await engine.collective_rpc("benchmark_mpmd_stats", timeout=180)
                    (options.stats_dir / "worker_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
                    result = await engine.collective_rpc("benchmark_mpmd_close", timeout=300)
                    (options.stats_dir / "worker_close.json").write_text(json.dumps(result, indent=2) + "\n")
                    closed = True

            async def control(reader, writer):
                try:
                    message = json.loads(await reader.readline())
                    if message not in ({"command": "collect-and-close"}, {"command": "validate-and-close"}):
                        raise ValueError("Unsupported lifecycle command")
                    if message["command"] == "validate-and-close":
                        before = await engine.collective_rpc("benchmark_mpmd_stats", timeout=180)
                        (options.stats_dir / "worker_stats_before_validation.json").write_text(
                            json.dumps(before, indent=2) + "\n"
                        )
                        pair = await engine.collective_rpc("benchmark_retired_pair", timeout=300)
                        (options.stats_dir / "retired_pair.json").write_text(
                            json.dumps(pair, indent=2) + "\n"
                        )
                    await collect_and_close()
                    reply = {"ok": True, "mpmd_closed": closed}
                except Exception as error:
                    reply = {"ok": False, "error": f"{type(error).__name__}: {error}"}
                    (options.stats_dir / "lifecycle_error.json").write_text(
                        json.dumps(reply, indent=2) + "\n"
                    )
                writer.write((json.dumps(reply) + "\n").encode())
                await writer.drain()
                writer.close()
                await writer.wait_closed()

            server = await asyncio.start_unix_server(control, path=str(control_path))
            os.chmod(control_path, 0o600)
            (options.stats_dir / "lifecycle_control.json").write_text(
                json.dumps({"socket": str(control_path)}, indent=2) + "\n"
            )
            try:
                yield engine
            finally:
                server.close()
                await server.wait_closed()
                control_path.unlink(missing_ok=True)
                # The launcher collects/drains while the engine is still live,
                # before SIGTERM can cancel the API's asynchronous context.

    api_server.build_async_engine_client = observed_client
    from vllm.entrypoints.cli.main import main as cli_main

    try:
        cli_main()
    finally:
        api_server.build_async_engine_client = original
        api_server.build_app = original_build_app


if __name__ == "__main__":
    main()
