#!/usr/bin/env python3
"""Runnable example: FlyHarness.step via local biorouter to flybrain.malecns.

Wiring flybrain is using the harness, not a harness feature. Default run uses
an in-thread FakeFlyBrain listed as flybrain.malecns. Does not download MaleCNS.

    python examples/biorouter_flybrain_loop.py
    python examples/biorouter_flybrain_loop.py --real
    python examples/biorouter_flybrain_loop.py --url http://127.0.0.1:8765
    python -m fly_harness.demo.biorouter_flybrain_loop
"""

from fly_harness.demo.biorouter_flybrain_loop import main

if __name__ == "__main__":
    main()
