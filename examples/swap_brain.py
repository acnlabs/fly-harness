#!/usr/bin/env python3
"""Runnable example: same FlyGym body, same loop, swap the Model.

    python examples/swap_brain.py
    python examples/swap_brain.py --real
    python examples/swap_brain.py --try-flygym
    python examples/swap_brain.py --url http://127.0.0.1:8765
    python -m fly_harness.demo.swap_brain
"""

from fly_harness.demo.swap_brain import main

if __name__ == "__main__":
    main()
