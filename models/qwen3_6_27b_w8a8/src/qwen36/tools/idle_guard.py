# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded per-device idle checks, without querying unrelated devices."""

import concurrent.futures
import json
import re
import subprocess
import time


def _query(card, kind):
    return subprocess.run(
        ["npu-smi", "info", "-t", kind, "-i", str(card), "-c", "0"],
        capture_output=True,
        text=True,
        check=True,
        timeout=12,
    ).stdout


def wait_for_idle(devices, timeout=120, stable_seconds=0):
    start = time.monotonic()
    stable = 0
    free_since = None
    previous = None
    last_report = 0
    while time.monotonic() - start < timeout:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                keys = [(card, kind) for card in devices for kind in ("health", "usages", "proc-mem")]
                futures = [pool.submit(_query, *key) for key in keys]
                output = dict(zip(keys, [f.result() for f in futures]))
            state = {}
            for card in devices:
                usage = output[card, "usages"]
                value = lambda name: int(re.search(re.escape(name) + r"\s*:\s*(\d+)", usage).group(1))
                state[card] = dict(
                    healthy=bool(re.search(r"Health Status\s*:\s*OK", output[card, "health"])),
                    owned="No process in device." not in output[card, "proc-mem"],
                    util=value("NPU Utilization(%)"),
                    hbm_bw=value("HBM Bandwidth Usage Rate(%)"),
                    hbm_percent=value("HBM Usage Rate(%)"),
                )
            free = all(
                v["healthy"]
                and not v["owned"]
                and v["util"] == 0
                and v["hbm_bw"] == 0
                and v["hbm_percent"] <= 10
                for v in state.values()
            )
            stable = stable + 1 if free else 0
            now = time.monotonic()
            if free:
                if free_since is None:
                    free_since = now
            else:
                free_since = None
            if state != previous or now - last_report >= 30:
                print("TARGET_DEVICE_IDLE " + json.dumps(state), flush=True)
                previous = state
                last_report = now
            if stable >= 2 and free_since is not None and now - free_since >= stable_seconds:
                return list(devices)
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            AttributeError,
            ValueError,
        ) as error:
            stable = 0
            free_since = None
            print("TARGET_DEVICE_QUERY_RETRY " + str(error), flush=True)
        time.sleep(min(1, max(0, timeout - (time.monotonic() - start))))
    raise RuntimeError("No healthy idle target devices before the bounded deadline")


def wait_for_free_group(candidates=(0, 1, 2, 3, 4, 5, 6, 7), timeout=1800):
    until = time.monotonic() + timeout
    previous = None
    while time.monotonic() < until:
        free = []

        # Check every candidate's health and load before selecting a group.
        # An idle-looking but unhealthy first card must not hide free cards4–7.
        def eligible(card):
            try:
                if "No process in device." not in _query(card, "proc-mem"):
                    return False
                if not re.search(r"Health Status\s*:\s*OK", _query(card, "health")):
                    return False
                usage = _query(card, "usages")
                values = [
                    int(re.search(re.escape(name) + r"\s*:\s*(\d+)", usage).group(1))
                    for name in ("NPU Utilization(%)", "HBM Bandwidth Usage Rate(%)", "HBM Usage Rate(%)")
                ]
                return values[0] == 0 and values[1] == 0 and values[2] <= 10
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, AttributeError, ValueError):
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            free = [card for card, okay in zip(candidates, pool.map(eligible, candidates)) if okay]
        current = tuple(free)
        if current != previous:
            print("FREE_CARD_CANDIDATES " + json.dumps(free), flush=True)
            previous = current
        if len(free) >= 4:
            try:
                return wait_for_idle(
                    sorted(free[:4]), timeout=min(25, max(1, until - time.monotonic())), stable_seconds=5
                )
            except RuntimeError:
                pass
        time.sleep(min(2, max(0, until - time.monotonic())))
    raise RuntimeError("No healthy idle four-card group before the bounded deadline")
