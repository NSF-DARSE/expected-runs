"""Configuration for the TrackMan Data API client.

Loads the four OAuth secrets from a repo-root .env file and exposes the base
URLs. Fails loudly, naming any missing secret, before any network call is made.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# .env lives at the repo root (one level up from this file's package dir).
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

# Production endpoints. Token URL is the client_credentials authority declared
# in trackman_api/swagger.json (login.trackman.com -- NOT login.trackmanbaseball.com,
# which is the older password-grant authority from the Quick Start Guide PDF).
TOKEN_URL = "https://login.trackman.com/connect/token"
DATA_BASE = "https://dataapi.trackmanbaseball.com/api/v1"

_REQUIRED = (
    "TRACKMAN_CLIENT_ID",
    "TRACKMAN_CLIENT_SECRET",
)


@dataclass(frozen=True)
class TrackManConfig:
    client_id: str
    client_secret: str
    token_url: str = TOKEN_URL
    data_base: str = DATA_BASE


def load_config() -> TrackManConfig:
    """Read secrets from the environment, raising if any are missing.

    Raises:
        RuntimeError: if one or more required secrets are unset/empty. The
            message names exactly which ones so it is obvious what to fill in.
    """
    missing = [name for name in _REQUIRED if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "Missing TrackMan secret(s): "
            + ", ".join(missing)
            + f". Copy .env.example to .env at {REPO_ROOT} and fill them in."
        )
    return TrackManConfig(
        client_id=os.environ["TRACKMAN_CLIENT_ID"],
        client_secret=os.environ["TRACKMAN_CLIENT_SECRET"],
    )
