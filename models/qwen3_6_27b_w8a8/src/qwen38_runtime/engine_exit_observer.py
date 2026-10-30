# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Record native engine exit status; no model hooks or per-step instrumentation."""

import json
import multiprocessing.connection
from pathlib import Path
from threading import Thread
import time


def install_engine_exit_observer(output: Path):
    from vllm.v1.engine.core_client import MPClient

    original = MPClient.start_engine_core_monitor

    def start(self):
        manager = self.resources.engine_manager
        processes = tuple(manager.processes) if manager is not None else ()
        if processes:
            output.mkdir(parents=True, exist_ok=True)
            (output / "engine_processes.json").write_text(
                json.dumps(
                    [
                        dict(pid=p.pid, name=p.name, cgroup=Path(f"/proc/{p.pid}/cgroup").read_text())
                        for p in processes
                    ],
                    indent=2,
                )
                + "\n"
            )

            def observe():
                sentinels = [p.sentinel for p in processes]
                ready = multiprocessing.connection.wait(sentinels)
                records = []
                for p in processes:
                    if p.sentinel not in ready:
                        continue
                    pid, name = p.pid, p.name
                    code = p.exitcode
                    for _ in range(50):
                        if code is not None:
                            break
                        time.sleep(0.01)
                        try:
                            code = p.exitcode
                        except ValueError:
                            break
                    records.append(dict(pid=pid, name=name, exitcode=code, wall_time=time.time()))
                (output / "engine_exit.json").write_text(json.dumps(records, indent=2) + "\n")

            Thread(target=observe, daemon=True, name="RecoveryEngineExitObserver").start()
        return original(self)

    MPClient.start_engine_core_monitor = start
