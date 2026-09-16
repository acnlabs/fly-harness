"""Optional **biorouter** process: local HTTP routing for deployed bio-sim models.

Stdlib only. Install extra ``fly-harness[biorouter]`` (no extra packages) or
run ``biorouter`` / ``python -m fly_harness.router``. OpenRouter routes existing
LLMs; biorouter routes existing deployed biological simulation models. Not the
harness, not FlyWire dumps, not a marketplace.
"""

from fly_harness.router.registry import (
    DEFAULT_FAKE_GAIN_MODEL_ID,
    DEFAULT_FAKE_MODEL_ID,
    DEFAULT_FAKE_N_NEURONS,
    FLYBRAIN_MODEL_ID,
    create_default_backends,
    create_registry,
    try_register_flybrain,
)
from fly_harness.router.server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    RouterHTTPServer,
    main,
    make_server,
    running_router,
    serve,
)

__all__ = [
    "DEFAULT_FAKE_GAIN_MODEL_ID",
    "DEFAULT_FAKE_MODEL_ID",
    "DEFAULT_FAKE_N_NEURONS",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "FLYBRAIN_MODEL_ID",
    "RouterHTTPServer",
    "create_default_backends",
    "create_registry",
    "main",
    "make_server",
    "running_router",
    "serve",
    "try_register_flybrain",
]
