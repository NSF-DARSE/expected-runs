"""OAuth 2.0 auth for the TrackMan Data API.

The Data API uses the client_credentials grant (per trackman_api/swagger.json,
which supersedes the password-grant flow described in the Quick Start Guide
v2.5 PDF): a client identifier + secret from the portal's "Data integration
clients" page, exchanged at https://login.trackman.com/connect/token. There is
no refresh token in this flow -- renewal is simply requesting a new token, so
long-running jobs call get_token() again when Token.expired turns true.

No retry loop here by design: a failed call raises with the response body so
auth failures and IP-whitelist blocks are immediately legible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from config import TrackManConfig

_TIMEOUT = 30  # seconds


@dataclass
class Token:
    access_token: str
    # Absolute epoch seconds at which the access token expires.
    expires_at: float

    @property
    def expired(self) -> bool:
        # 30s safety margin so we never use a token about to lapse mid-request.
        return time.time() >= self.expires_at - 30


def get_token(config: TrackManConfig) -> Token:
    """Obtain an access token via the client_credentials grant."""
    resp = requests.post(
        config.token_url,
        data={
            "grant_type": "client_credentials",
            "client_id": config.client_id,
            "client_secret": config.client_secret,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=_TIMEOUT,
    )
    if not resp.ok:
        raise RuntimeError(
            f"TrackMan token request failed ({resp.status_code}): {resp.text}"
        )
    payload = resp.json()
    return Token(
        access_token=payload["access_token"],
        expires_at=time.time() + float(payload.get("expires_in", 3600)),
    )
