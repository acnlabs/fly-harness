"""Optional local HTTP ModelBackend router process.

Stdlib only. Install extra ``fly-harness[router]`` (no extra packages) or just
run ``fly-harness-router``. Not OpenRouter-the-company and not a marketplace.
"""

from fly_harness.router.registry import (
    DEFAULT_FAKE_GAIN_MODEL_ID,
    DEFAULT_FAKE_MODEL_ID,
    DEFAULT_FAKE_N_NEURONS,
    create_default_backends,
    create_registry,
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
    "RouterHTTPServer",
    "create_default_backends",
    "create_registry",
    "main",
    "make_server",
    "running_router",
    "serve",
]
