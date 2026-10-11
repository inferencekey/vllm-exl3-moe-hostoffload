"""Helper calls survive a dropped keep-alive connection; anything else still fails."""
import asyncio

import pytest

aiohttp = pytest.importorskip("aiohttp")
from hostoffload.harness import loadgen  # noqa: E402


def test_retry_unmeasured_recovers_from_a_disconnect():
    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise aiohttp.ServerDisconnectedError()
        return "ok"
    assert asyncio.run(loadgen.retry_unmeasured(flaky)) == "ok" and len(calls) == 2


def test_retry_unmeasured_gives_up_and_does_not_hide_other_errors():
    async def always_drops():
        raise aiohttp.ServerDisconnectedError()

    async def bug():
        raise KeyError("tokens")
    with pytest.raises(aiohttp.ServerDisconnectedError):
        asyncio.run(loadgen.retry_unmeasured(always_drops, attempts=2))
    with pytest.raises(KeyError):
        asyncio.run(loadgen.retry_unmeasured(bug))
