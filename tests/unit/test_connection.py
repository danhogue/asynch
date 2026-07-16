from unittest.mock import AsyncMock

import pytest

from asynch.connection import Connection


@pytest.mark.asyncio
async def test_refresh_reconnects_after_failed_ping():
    connection = Connection()
    connection._opened = True
    connection.ping = AsyncMock(side_effect=ConnectionError("stale connection"))
    connection._connection.disconnect = AsyncMock()
    connection._connection.connect = AsyncMock()

    await connection._refresh()

    connection._connection.disconnect.assert_awaited_once()
    connection._connection.connect.assert_awaited_once()
    assert connection.opened is True
    assert connection.closed is False
