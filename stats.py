# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
# 
# Modified by TernaInfer Authors, 2026

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class PhaseStats:
    name: str
    tokens: int
    time: float

    def show(self) -> str:
        tps = self.tokens / self.time
        return (
            f"[{self.name}] "
            f"\ttokens: {self.tokens}"
            f"\t total time: {self.time:.3f}s"
            f"\t {tps:.1f} tokens per second"
        )


class Stats:
    """
    Generation stats, split by phases.
    """

    def __init__(self):
        self.phases = []
        self.current = None

    def end_phase(self, tokens: int, now: Optional[float] = None):
        """Terminate the current phase."""
        if self.current is None:
            return
        if now is None:
            now = time.time()
        cname, ctime = self.current
        stats = PhaseStats(
            name=cname,
            tokens=tokens,
            time=now - ctime,
        )
        self.phases.append(stats)
        self.current=None

    def phase(self, name: str, tokens: int = 0):
        """
        Start a new phase, and terminate the current one,
        if one is ongoing.
        """
        now = time.time()
        # self.end_phase(0, now)
        self.current = (name, now)