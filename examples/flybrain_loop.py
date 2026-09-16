#!/usr/bin/env python3
"""Runnable example: FlyHarness.step with a third-party flybrain Model (FakeFlyBrain by default).

Wiring flybrain is using the harness, not a harness feature. Does not download
MaleCNS. Use --real only when files are already in ~/fly-data.

    python examples/flybrain_loop.py
    python examples/flybrain_loop.py --real
    python -m fly_harness.demo.flybrain_loop
"""

from fly_harness.demo.flybrain_loop import main

if __name__ == "__main__":
    main()
