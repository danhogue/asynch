import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Optional

from asynch.connection import Connection
from asynch.errors import AsynchPoolError, OperationalError
from asynch.proto import constants
from asynch.proto.models.enums import PoolStatus

logger = logging.getLogger(__name__)


class Pool:
    def __init__(
        self,
        minsize: int = constants.POOL_MIN_SIZE,
        maxsize: int = constants.POOL_MAX_SIZE,
        **kwargs,
    ):
        if maxsize < 1:
            raise ValueError("maxsize is expected to be greater than zero")
        if minsize < 0:
            raise ValueError("minsize is expected to be greater or equal to zero")
        if minsize > maxsize:
            raise ValueError("minsize is greater than maxsize")
        self._maxsize = maxsize
        self._minsize = minsize
        self._connection_kwargs = kwargs
        self._sem = asyncio.Semaphore(maxsize)
        self._lock = asyncio.Lock()
        self._fill_lock = asyncio.Lock()
        self._connection_reservations: set[object] = set()
        self._pending_acquisitions: set[asyncio.Future] = set()
        self._startup_waiter: Optional[asyncio.Future] = None
        self._acquired_connections: deque[Connection] = deque(maxlen=maxsize)
        self._free_connections: deque[Connection] = deque(maxlen=maxsize)
        self._opened: bool = False
        self._closed: bool = False

    async def __aenter__(self) -> "Pool":
        await self.startup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.shutdown()

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        status = self.status
        return (
            f"<{cls_name}(minsize={self._minsize}, maxsize={self._maxsize})"
            f" object at 0x{id(self):x}; status: {status}>"
        )

    @property
    def opened(self) -> bool:
        """Returns True if the pool is opened.

        :returns: the pool open status
        :rtype: bool
        """

        return self._opened

    @property
    def closed(self) -> bool:
        """Return True if the pool is closed.

        :returns: the pool close status
        :rtype: bool
        """

        return self._closed

    @property
    def status(self) -> str:
        """Return the status of the pool.

        :raise AsynchPoolError: an unresolved pool state.
        :return: the Pool object status
        :rtype: str (PoolStatus StrEnum)
        """

        if not (self._opened or self._closed):
            return PoolStatus.created
        if self._opened and not self._closed:
            return PoolStatus.opened
        if self._closed and not self._opened:
            return PoolStatus.closed
        raise AsynchPoolError(f"{self} is in an unknown state")

    @property
    def acquired_connections(self) -> int:
        """Return the number of connections acquired from the pool.

        A connection is acquired when the `pool.connection()` is invoked.

        :return: the number of connections requested from the pool
        :rtype: int
        """

        return len(self._acquired_connections)

    @property
    def free_connections(self) -> int:
        """Return the number of free connections in the pool.

        :return: the number of free connections in the pool
        :rtype: int
        """

        return len(self._free_connections)

    @property
    def _pool_size(self) -> int:
        """Return the number of connections associated with the pool.

        This number is the sum of the acquired and free connections.
        So this sum may be interpreted as the current size of the pool
        or the number of connections associated with the pool and so on.

        :return: the number of connections related to the pool
        :rtype: int
        """

        return self.acquired_connections + self.free_connections

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def minsize(self) -> int:
        return self._minsize

    async def _begin_acquisition(self) -> asyncio.Future:
        while True:
            async with self._lock:
                if self._closed:
                    raise AsynchPoolError(f"{self} is closed")
                if self._startup_waiter is None:
                    done = asyncio.get_running_loop().create_future()
                    self._pending_acquisitions.add(done)
                    return done
                startup_waiter = self._startup_waiter
            await asyncio.shield(startup_waiter)

    async def _finish_acquisition(self, done: asyncio.Future) -> None:
        async with self._lock:
            self._pending_acquisitions.discard(done)
            if not done.done():
                done.set_result(None)

    async def _restore_minsize(self) -> None:
        async with self._lock:
            should_fill = self._opened and not self._closed
        if should_fill:
            with suppress(Exception):
                await self._ensure_minsize_connections(strict=True)

    async def _close_connection(self, conn: Connection) -> None:
        with suppress(Exception):
            await conn.close()

    async def _open_connection(self) -> Connection:
        conn = Connection(**self._connection_kwargs)
        try:
            await conn.connect()
            await conn.ping()
        except asyncio.CancelledError:
            await self._close_connection(conn)
            raise
        except Exception as e:
            await self._close_connection(conn)
            msg = f"failed to create a {conn} for {self}"
            raise AsynchPoolError(msg) from e
        return conn

    async def _reserve_connection_creations(
        self, n: int, *, allow_closed: bool = False
    ) -> list[object]:
        async with self._lock:
            if self._closed and not allow_closed:
                raise AsynchPoolError(f"{self} is closed")
            size = self._pool_size + len(self._connection_reservations)
            if size + n > self._maxsize:
                msg = (
                    f"{self} has the {size} connections or pending connections, "
                    f"adding {n} will exceed its maxsize ({self.maxsize})"
                )
                raise AsynchPoolError(msg)
            reservations = [object() for _ in range(n)]
            self._connection_reservations.update(reservations)
            return reservations

    async def _cancel_connection_creation(
        self, reservation: object, conn: Optional[Connection] = None
    ) -> None:
        async with self._lock:
            self._connection_reservations.discard(reservation)
            if conn is not None:
                with suppress(ValueError):
                    self._free_connections.remove(conn)
                with suppress(ValueError):
                    self._acquired_connections.remove(conn)

    async def _store_connection(
        self, conn: Connection, reservation: object, connections: deque[Connection]
    ) -> bool:
        async with self._lock:
            if reservation not in self._connection_reservations:
                return False
            self._connection_reservations.remove(reservation)
            connections.append(conn)
            return True

    async def _create_connection(self, reservation: Optional[object] = None) -> None:
        if reservation is None:
            reservation = (await self._reserve_connection_creations(1))[0]

        conn = None
        try:
            conn = await self._open_connection()
            if await self._store_connection(conn, reservation, self._free_connections):
                return
        except asyncio.CancelledError:
            await self._cancel_connection_creation(reservation, conn)
            if conn:
                await self._close_connection(conn)
            raise
        except Exception:
            await self._cancel_connection_creation(reservation, conn)
            if conn:
                await self._close_connection(conn)
            raise

        await self._close_connection(conn)
        raise AsynchPoolError(f"{self} was closed while creating a connection")

    def _pop_connection(self) -> Connection:
        if not self._free_connections:
            raise AsynchPoolError(f"no free connection in {self}")
        return self._free_connections.popleft()

    async def _discard_acquired_connection(self, conn: Connection) -> None:
        async with self._lock:
            with suppress(ValueError):
                self._acquired_connections.remove(conn)
        await self._close_connection(conn)

    async def _get_fresh_connection(self) -> Optional[Connection]:
        while True:
            async with self._lock:
                if not self._free_connections:
                    return None
                conn = self._pop_connection()
                self._acquired_connections.append(conn)

            try:
                await conn._refresh()
                async with self._lock:
                    acquired = conn in self._acquired_connections
            except asyncio.CancelledError:
                await self._discard_acquired_connection(conn)
                raise
            except (
                ConnectionError,
                OSError,
                asyncio.TimeoutError,
                RuntimeError,
                OperationalError,
            ):
                await self._discard_acquired_connection(conn)
                continue
            except Exception:
                await self._discard_acquired_connection(conn)
                raise

            if acquired:
                return conn
            await self._close_connection(conn)
            raise AsynchPoolError(f"{self} was closed while acquiring a connection")

    async def _acquire_connection(self) -> Connection:
        done = await self._begin_acquisition()
        failed = False
        try:
            if conn := await self._get_fresh_connection():
                return conn

            reservation = (await self._reserve_connection_creations(1))[0]
            conn = None
            try:
                conn = await self._open_connection()
                if await self._store_connection(conn, reservation, self._acquired_connections):
                    return conn
            except asyncio.CancelledError:
                await self._cancel_connection_creation(reservation, conn)
                if conn:
                    await self._close_connection(conn)
                raise
            except Exception:
                await self._cancel_connection_creation(reservation, conn)
                if conn:
                    await self._close_connection(conn)
                raise

            await self._close_connection(conn)
            raise AsynchPoolError(f"{self} was closed while creating a connection")
        except BaseException:
            failed = True
            raise
        finally:
            await self._finish_acquisition(done)
            if failed:
                await self._restore_minsize()

    async def _release_connection(self, conn: Connection) -> None:
        async with self._lock:
            if conn not in self._acquired_connections:
                raise AsynchPoolError(f"the connection {conn} does not belong to {self}")

            self._acquired_connections.remove(conn)
            if conn.opened and not conn.closed:
                self._free_connections.append(conn)
                return

        raise AsynchPoolError(f"the {conn} is invalidated")

    async def _init_connections(
        self, n: int, *, strict: bool = False, allow_closed: bool = False
    ) -> None:
        if n < 0:
            msg = f"cannot create a negative number ({n}) of connections for {self}"
            raise ValueError(msg)
        if not n:
            return

        reservations = await self._reserve_connection_creations(n, allow_closed=allow_closed)
        # it is possible that the `_create_connection` may not create `n` connections
        tasks: list[asyncio.Task] = [
            asyncio.create_task(self._create_connection(reservation))
            for reservation in reservations
        ]
        # that is why possible exceptions from the `_create_connection` are also gathered
        if strict and any(
            i
            for i in await asyncio.gather(*tasks, return_exceptions=True)
            if isinstance(i, Exception)
        ):
            msg = f"failed to create the {n} connection(s) for the {self}"
            raise AsynchPoolError(msg)

    async def _ensure_minsize_connections(self, *, strict: bool = False) -> None:
        async with self._fill_lock:
            async with self._lock:
                if self._closed:
                    return
                gap = self.minsize - (self._pool_size + len(self._connection_reservations))
            if gap > 0:
                await self._init_connections(gap, strict=strict)

    def _reset_for_new_loop(self) -> None:
        """Recreate asyncio primitives and discard connections when the event loop changes.

        asyncio.Lock and asyncio.Semaphore bind to the first event loop that awaits them
        (Python 3.10+ _LoopBoundMixin).  When the same Pool singleton is reused across
        tests that each create a fresh event loop, the primitives raise
        "bound to a different event loop".  Recreating them (and discarding the stale
        connections, which are also loop-bound) restores a usable state.
        """
        self._sem = asyncio.Semaphore(self._maxsize)
        self._lock = asyncio.Lock()
        self._fill_lock = asyncio.Lock()
        self._connection_reservations.clear()
        self._pending_acquisitions.clear()
        self._startup_waiter = None
        self._free_connections.clear()
        self._acquired_connections.clear()
        self._opened = False
        self._closed = False

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[Connection]:
        """Get a connection from the pool.

        If requested more connections than the pool can provide,
        the pool gets blocked until a connection comes back.

        :raises AsynchPoolError: if a connection cannot be acquired or released

        :return: a free connection from the pool
        :rtype: Connection
        """

        running = asyncio.get_running_loop()
        if getattr(self._lock, "_loop", None) not in (None, running):
            self._reset_for_new_loop()

        async with self._sem:
            conn = await self._acquire_connection()
            try:
                yield conn
            finally:
                try:
                    await self._release_connection(conn)
                except AsynchPoolError as e:
                    logger.warning(e)
                await self._ensure_minsize_connections(strict=True)

    async def startup(self) -> "Pool":
        """Initialise the pool.

        When entering the context,
        the pool get filled with connections
        up to the pool `minsize` value.

        :return: a pool object with `minsize` opened connections
        :rtype: Pool
        """

        running = asyncio.get_running_loop()
        if getattr(self._lock, "_loop", None) not in (None, running):
            self._reset_for_new_loop()

        async with self._fill_lock:
            async with self._lock:
                if self._opened:
                    return self
                startup_waiter = asyncio.get_running_loop().create_future()
                self._startup_waiter = startup_waiter
                pending_acquisitions = tuple(self._pending_acquisitions)

            try:
                await asyncio.gather(*(asyncio.shield(done) for done in pending_acquisitions))
                async with self._lock:
                    gap = max(0, self.minsize - self._pool_size)
                # If we cannot create the minsize connections here,
                # the Pool does not meet the minsize requirement.
                await self._init_connections(gap, strict=True, allow_closed=True)
                async with self._lock:
                    self._opened = True
                    if self._closed:
                        self._closed = False
            finally:
                async with self._lock:
                    if self._startup_waiter is startup_waiter:
                        self._startup_waiter = None
                        if not startup_waiter.done():
                            startup_waiter.set_result(None)
        return self

    async def shutdown(self) -> None:
        """Close the pool.

        This method closes consequently free connections first.
        Then it does the same for the acquired connections.
        Then the pool is marked closed.
        """

        async with self._fill_lock:
            async with self._lock:
                connections = list(self._free_connections) + list(self._acquired_connections)
                pending_acquisitions = tuple(self._pending_acquisitions)
                self._connection_reservations.clear()
                self._free_connections.clear()
                self._acquired_connections.clear()
                self._opened = False
                self._closed = True

            await asyncio.gather(
                *(self._close_connection(conn) for conn in connections),
                *(asyncio.shield(done) for done in pending_acquisitions),
            )
