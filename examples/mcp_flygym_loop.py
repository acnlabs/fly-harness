#!/usr/bin/env python3
"""Runnable example: optional MCP + FlyGym extras composed through FlyHarness.step.

Uses the 24-neuron reflex fixture. Does not load FlyWire or MaleCNS (~166k).

    python examples/mcp_flygym_loop.py
    python examples/mcp_flygym_loop.py --serve   # optional FastMCP stdio
    python -m fly_harness.demo.composed
"""

from fly_harness.demo.composed import main

if __name__ == "__main__":
    main()
