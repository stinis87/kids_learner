"""One-time interactive login for the Ring camera tool.

Prompts for the user's Ring username/password (and 2FA code if required) and
persists the resulting OAuth token so the app never has to store credentials.
"""

from __future__ import annotations
import asyncio
import getpass
import logging

from ring_doorbell import Auth, Requires2FAError

from reachy_mini_conversation_app.ring_client import token_cache_path, write_token_cache


logger = logging.getLogger(__name__)


async def async_run_ring_login(instance_path: str | None = None) -> None:
    """Interactively authenticate with Ring and write the token cache file."""
    cache_path = token_cache_path(instance_path)

    def save_token(token: dict[str, object]) -> None:
        write_token_cache(cache_path, token)

    username = input("Ring username (email): ").strip()
    password = getpass.getpass("Ring password: ")

    auth = Auth("reachy-mini-conversation-app", None, save_token)
    try:
        await auth.async_fetch_token(username, password)
    except Requires2FAError:
        otp_code = input("Ring 2FA code: ").strip()
        await auth.async_fetch_token(username, password, otp_code)

    await auth.async_close()
    print(f"Ring login saved to {cache_path}")


def run_ring_login(instance_path: str | None = None) -> None:
    """Run the interactive Ring login synchronously, for the `--ring-login` CLI flag."""
    asyncio.run(async_run_ring_login(instance_path))
