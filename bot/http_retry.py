"""
HTTP retry helpers for external APIs.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Iterable

import httpx


RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def _should_retry_exception(exc: Exception) -> bool:
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
            httpx.PoolTimeout,
            httpx.WriteError,
            httpx.ReadError,
            httpx.NetworkError,
            httpx.ProtocolError,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return False


async def _retry_sleep(attempt: int, base_backoff_seconds: float) -> None:
    delay = base_backoff_seconds * (2 ** max(0, attempt - 1))
    jitter = random.uniform(0.0, max(0.05, 0.25 * delay))
    await asyncio.sleep(delay + jitter)


async def get_json(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: float = 15,
    attempts: int = 3,
    base_backoff_seconds: float = 0.5,
    retry_status_codes: Iterable[int] | None = None,
    no_retry_status_codes: Iterable[int] | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict | list:
    retry_codes = set(retry_status_codes or RETRYABLE_STATUS_CODES)
    no_retry_codes = set(no_retry_status_codes or [])
    own_client = client is None
    use_client = client or httpx.AsyncClient(timeout=timeout)
    try:
        for attempt in range(1, max(1, attempts) + 1):
            try:
                resp = await use_client.get(url, params=params, headers=headers, timeout=timeout)
                if resp.status_code >= 400:
                    if resp.status_code in no_retry_codes:
                        resp.raise_for_status()
                    if resp.status_code not in retry_codes:
                        resp.raise_for_status()
                    raise httpx.HTTPStatusError(
                        f"retryable status {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                return resp.json()
            except Exception as exc:
                if attempt >= max(1, attempts) or not _should_retry_exception(exc):
                    raise
                await _retry_sleep(attempt, base_backoff_seconds)
    finally:
        if own_client:
            await use_client.aclose()


async def get_bytes(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: float = 20,
    attempts: int = 3,
    base_backoff_seconds: float = 0.5,
    retry_status_codes: Iterable[int] | None = None,
    no_retry_status_codes: Iterable[int] | None = None,
    client: httpx.AsyncClient | None = None,
) -> bytes:
    retry_codes = set(retry_status_codes or RETRYABLE_STATUS_CODES)
    no_retry_codes = set(no_retry_status_codes or [])
    own_client = client is None
    use_client = client or httpx.AsyncClient(timeout=timeout)
    try:
        for attempt in range(1, max(1, attempts) + 1):
            try:
                resp = await use_client.get(url, params=params, headers=headers, timeout=timeout)
                if resp.status_code >= 400:
                    if resp.status_code in no_retry_codes:
                        resp.raise_for_status()
                    if resp.status_code not in retry_codes:
                        resp.raise_for_status()
                    raise httpx.HTTPStatusError(
                        f"retryable status {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                return resp.content
            except Exception as exc:
                if attempt >= max(1, attempts) or not _should_retry_exception(exc):
                    raise
                await _retry_sleep(attempt, base_backoff_seconds)
    finally:
        if own_client:
            await use_client.aclose()
