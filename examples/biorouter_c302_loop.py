#!/usr/bin/env python3
"""Runnable example: FlyHarness.step via local biorouter to c302.celegans.

Wiring a worm sim is using the harness, not a harness feature. Default run
uses FakeC302 (302 hermaphrodite neurons). Does not start OpenWorm Docker.

    python examples/biorouter_c302_loop.py
    python examples/biorouter_c302_loop.py --real
    python examples/biorouter_c302_loop.py --url http://127.0.0.1:8765
    python -m fly_harness.demo.biorouter_c302_loop
"""

from fly_harness.demo.biorouter_c302_loop import main

if __name__ == "__main__":
    main()
