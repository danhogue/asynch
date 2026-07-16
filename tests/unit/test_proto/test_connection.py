import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from asynch.errors import OperationalError
from asynch.proto.connection import Connection
from asynch.proto.streams.buffered import BufferedWriter


async def wait_forever(*args, **kwargs):
    await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_ping_honors_sync_request_timeout():
    connection = Connection(sync_request_timeout=0.01)
    connection.reader = Mock()
    connection.reader.reader.at_eof.return_value = False
    connection.reader.read_varint = AsyncMock(side_effect=wait_forever)
    connection.writer = Mock()
    connection.writer.write_varint = AsyncMock()
    connection.writer.flush = AsyncMock()
    connection.disconnect = AsyncMock()

    assert await asyncio.wait_for(connection.ping(), timeout=1.0) is False
    connection.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_force_connect_recovers_from_operational_error():
    connection = Connection()
    connection.connected = True
    connection.reader = Mock()
    connection.reader.reader.at_eof.return_value = False
    connection.reader.read_varint = AsyncMock(side_effect=OperationalError("remote closed"))
    connection.writer = Mock()
    connection.writer.write_varint = AsyncMock()
    connection.writer.flush = AsyncMock()
    connection.connect = AsyncMock()

    await connection.force_connect()

    connection.connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_connect_honors_connect_timeout_and_closes_partial_transport():
    connection = Connection(connect_timeout=0.01)
    stream_writer = Mock()
    stream_writer.wait_closed = AsyncMock()

    async def start_connection(*args, **kwargs):
        connection.writer = BufferedWriter(stream_writer)
        await wait_forever()

    connection._init_connection = AsyncMock(side_effect=start_connection)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(connection.connect(), timeout=1.0)

    stream_writer.close.assert_called_once()
    assert connection.writer is None
    assert connection.connected is False


@pytest.mark.asyncio
async def test_disconnect_honors_sync_request_timeout():
    connection = Connection(sync_request_timeout=0.01)
    writer = Mock()
    writer.close = AsyncMock(side_effect=asyncio.TimeoutError)
    connection.connected = True
    connection.writer = writer

    await connection.disconnect()

    writer.close.assert_awaited_once_with(timeout=0.01)
    assert connection.connected is False


@pytest.mark.asyncio
async def test_buffered_writer_aborts_when_wait_closed_times_out():
    stream_writer = Mock()
    stream_writer.wait_closed = AsyncMock(side_effect=wait_forever)
    buffered_writer = BufferedWriter(stream_writer)

    with pytest.raises(asyncio.TimeoutError):
        await buffered_writer.close(timeout=0.01)

    stream_writer.close.assert_called_once()
    stream_writer.transport.abort.assert_called_once()
