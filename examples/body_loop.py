#!/usr/bin/env python3
"""Runnable example: FlyGym obs → FlyHarness.step → biorouter → flybrain.malecns → body.

Wiring FlyGym / flybrain is using the harness, not a harness feature. Default
run uses StubFlyGymEnv + FakeFlyBrain. Does not download MaleCNS.

    python examples/body_loop.py
    python examples/body_loop.py --real
    python examples/body_loop.py --try-flygym
    python examples/body_loop.py --url http://127.0.0.1:8765
    python -m fly_harness.demo.body_loop
"""

from fly_harness.demo.body_loop import main

if __name__ == "__main__":
    main()
