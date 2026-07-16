import asyncio

import pytest

from asynch.errors import AsynchPoolError
from asynch.pool import Pool


class FakeConnection:
    def __init__(self, refresh=None):
        self._refresh_impl = refresh
        self.refresh_calls = 0
        self.close_calls = 0
        self.opened = True
        self.closed = False

    async def _refresh(self):
        self.refresh_calls += 1
        if self._refresh_impl:
            await self._refresh_impl()

    async def close(self):
        if self.closed:
            return
        self.close_calls += 1
        self.opened = False
        self.closed = True


async def use_connection(pool):
    async with pool.connection() as connection:
        return connection


@pytest.mark.asyncio
async def test_slow_health_check_does_not_block_healthy_connection():
    health_check_started = asyncio.Event()
    finish_health_check = asyncio.Event()

    async def slow_health_check():
        health_check_started.set()
        await finish_health_check.wait()

    slow = FakeConnection(slow_health_check)
    healthy = FakeConnection()
    pool = Pool(minsize=0, maxsize=2)
    pool._free_connections.extend([slow, healthy])

    slow_task = asyncio.create_task(use_connection(pool))
    await health_check_started.wait()
    healthy_task = asyncio.create_task(use_connection(pool))

    try:
        assert await asyncio.wait_for(asyncio.shield(healthy_task), timeout=1.0) is healthy
    finally:
        finish_health_check.set()
        await asyncio.gather(slow_task, healthy_task)

    assert slow_task.result() is slow
    assert pool.acquired_connections == 0
    assert pool.free_connections == 2


@pytest.mark.asyncio
async def test_timed_out_health_check_tries_next_connection():
    async def timeout():
        raise asyncio.TimeoutError

    stale = FakeConnection(timeout)
    healthy = FakeConnection()
    pool = Pool(minsize=0, maxsize=2)
    pool._free_connections.extend([stale, healthy])

    assert await use_connection(pool) is healthy

    assert stale.close_calls == 1
    assert pool.acquired_connections == 0
    assert pool.free_connections == 1


@pytest.mark.asyncio
async def test_cancelled_health_check_discards_reserved_connection():
    health_check_started = asyncio.Event()
    never_finish = asyncio.Event()

    async def stuck_health_check():
        health_check_started.set()
        await never_finish.wait()

    stuck = FakeConnection(stuck_health_check)
    healthy = FakeConnection()
    pool = Pool(minsize=0, maxsize=1)
    pool._free_connections.append(stuck)

    task = asyncio.create_task(use_connection(pool))
    await health_check_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert stuck.close_calls == 1
    assert pool.acquired_connections == 0
    assert pool.free_connections == 0

    pool._free_connections.append(healthy)
    assert await asyncio.wait_for(use_connection(pool), timeout=1.0) is healthy


@pytest.mark.asyncio
async def test_cancelled_after_health_check_discards_reserved_connection():
    health_check_started = asyncio.Event()
    finish_health_check = asyncio.Event()

    async def health_check():
        health_check_started.set()
        await finish_health_check.wait()

    connection = FakeConnection(health_check)
    pool = Pool(minsize=0, maxsize=1)
    pool._free_connections.append(connection)

    task = asyncio.create_task(use_connection(pool))
    await health_check_started.wait()
    await pool._lock.acquire()
    finish_health_check.set()
    await asyncio.sleep(0)
    task.cancel()
    pool._lock.release()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert connection.close_calls == 1
    assert pool.acquired_connections == 0
    assert pool.free_connections == 0


@pytest.mark.asyncio
async def test_cancelled_startup_can_resume_from_created_connections():
    second_connection_started = asyncio.Event()
    never_finish = asyncio.Event()
    first = FakeConnection()
    second = FakeConnection()
    pool = Pool(minsize=2, maxsize=2)
    open_calls = 0

    async def open_connection():
        nonlocal open_calls
        open_calls += 1
        if open_calls == 1:
            return first
        second_connection_started.set()
        await never_finish.wait()

    pool._open_connection = open_connection
    startup_task = asyncio.create_task(pool.startup())
    await second_connection_started.wait()
    assert pool.free_connections == 1
    startup_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await startup_task

    assert pool.free_connections == 1
    assert not pool._connection_reservations

    async def finish_startup():
        return second

    pool._open_connection = finish_startup
    await pool.startup()

    assert pool.opened
    assert pool.free_connections == 2


@pytest.mark.asyncio
async def test_startup_waits_for_pending_acquisition():
    connection_creation_started = asyncio.Event()
    never_finish = asyncio.Event()
    replacement = FakeConnection()
    pool = Pool(minsize=1, maxsize=1)
    open_calls = 0

    async def open_connection():
        nonlocal open_calls
        open_calls += 1
        if open_calls > 1:
            return replacement
        connection_creation_started.set()
        await never_finish.wait()

    pool._open_connection = open_connection
    acquisition_task = asyncio.create_task(use_connection(pool))
    await connection_creation_started.wait()
    startup_task = asyncio.create_task(pool.startup())
    await asyncio.sleep(0)

    assert not startup_task.done()

    acquisition_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await acquisition_task
    await startup_task

    assert pool.opened
    assert pool.free_connections == 1
    assert pool.acquired_connections == 0
    assert not pool._connection_reservations


@pytest.mark.asyncio
async def test_cancelled_acquisition_restores_minsize():
    first_connection_acquired = asyncio.Event()
    release_first_connection = asyncio.Event()
    second_connection_started = asyncio.Event()
    never_finish = asyncio.Event()
    first = FakeConnection()
    replacement = FakeConnection()
    pool = Pool(minsize=1, maxsize=2)
    pool._opened = True
    pool._free_connections.append(first)
    open_calls = 0

    async def hold_invalid_connection():
        async with pool.connection() as connection:
            first_connection_acquired.set()
            await release_first_connection.wait()
            await connection.close()

    async def open_connection():
        nonlocal open_calls
        open_calls += 1
        if open_calls > 1:
            return replacement
        second_connection_started.set()
        await never_finish.wait()

    pool._open_connection = open_connection
    first_task = asyncio.create_task(hold_invalid_connection())
    await first_connection_acquired.wait()
    second_task = asyncio.create_task(use_connection(pool))
    await second_connection_started.wait()

    release_first_connection.set()
    await first_task
    second_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second_task

    assert pool.opened
    assert pool.free_connections == 1
    assert pool.acquired_connections == 0
    assert not pool._connection_reservations


@pytest.mark.asyncio
async def test_shutdown_waits_for_pending_connection_creation():
    connection_creation_started = asyncio.Event()
    finish_connection_creation = asyncio.Event()
    connection = FakeConnection()
    pool = Pool(minsize=0, maxsize=1)

    async def open_connection():
        connection_creation_started.set()
        await finish_connection_creation.wait()
        return connection

    pool._open_connection = open_connection
    task = asyncio.create_task(use_connection(pool))
    await connection_creation_started.wait()

    shutdown_task = asyncio.create_task(pool.shutdown())
    await asyncio.sleep(0)

    assert not shutdown_task.done()

    finish_connection_creation.set()
    with pytest.raises(AsynchPoolError, match="closed while creating"):
        await task
    await shutdown_task

    assert connection.close_calls == 1
    assert pool.acquired_connections == 0
    assert pool.free_connections == 0


@pytest.mark.asyncio
async def test_waiting_acquisition_does_not_reopen_shutdown_pool():
    release_connection = asyncio.Event()
    first_connection_acquired = asyncio.Event()
    connection = FakeConnection()
    pool = Pool(minsize=0, maxsize=1)
    pool._free_connections.append(connection)

    async def hold_connection():
        async with pool.connection():
            first_connection_acquired.set()
            await release_connection.wait()

    first_task = asyncio.create_task(hold_connection())
    await first_connection_acquired.wait()
    waiting_task = asyncio.create_task(use_connection(pool))
    await asyncio.sleep(0)

    await pool.shutdown()
    release_connection.set()
    await first_task

    with pytest.raises(AsynchPoolError, match="closed"):
        await waiting_task

    assert pool.acquired_connections == 0
    assert pool.free_connections == 0


@pytest.mark.asyncio
async def test_pending_acquisition_counts_toward_pool_capacity():
    connection_creation_started = asyncio.Event()
    finish_connection_creation = asyncio.Event()
    connection = FakeConnection()
    pool = Pool(minsize=1, maxsize=1)
    open_calls = 0

    async def open_connection():
        nonlocal open_calls
        open_calls += 1
        connection_creation_started.set()
        await finish_connection_creation.wait()
        return connection

    pool._open_connection = open_connection
    task = asyncio.create_task(use_connection(pool))
    await connection_creation_started.wait()

    await pool._ensure_minsize_connections(strict=True)
    assert open_calls == 1

    finish_connection_creation.set()
    assert await task is connection


@pytest.mark.asyncio
async def test_connection_is_not_health_checked_on_release():
    connection = FakeConnection()
    pool = Pool(minsize=0, maxsize=1)
    pool._free_connections.append(connection)

    assert await use_connection(pool) is connection

    assert connection.refresh_calls == 1
    assert pool.acquired_connections == 0
    assert pool.free_connections == 1
