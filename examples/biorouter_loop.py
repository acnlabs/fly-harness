#!/usr/bin/env python3
"""Runnable example: local biorouter → HttpModelBackend → FlyHarness.step.

Does not load FlyWire or MaleCNS. Default run uses an in-thread stdlib server.

    python examples/biorouter_loop.py
    python examples/biorouter_loop.py --url http://127.0.0.1:8765
    python -m fly_harness.demo.biorouter_loop
"""

from fly_harness.demo.biorouter_loop import main

if __name__ == "__main__":
    main()
