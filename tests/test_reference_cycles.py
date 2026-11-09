import asyncio
import gc
import typing
import unittest

import tenacity
from tenacity import retry, stop_after_attempt, wait_none


@retry(stop=stop_after_attempt(3), wait=wait_none(), reraise=True)
def _succeeds() -> int:
    return 1


class _FailsOnce:
    def __init__(self) -> None:
        self.calls = 0

    def check(self) -> None:
        self.calls += 1
        if self.calls % 2:
            raise ValueError("fails once")


_sync_fails_once = _FailsOnce()
_async_fails_once = _FailsOnce()


@retry(stop=stop_after_attempt(3), wait=wait_none(), reraise=True)
def _succeeds_after_one_retry() -> int:
    _sync_fails_once.check()
    return 1


@retry(stop=stop_after_attempt(3), wait=wait_none(), reraise=True)
async def _async_succeeds() -> int:
    return 1


@retry(stop=stop_after_attempt(3), wait=wait_none(), reraise=True)
async def _async_succeeds_after_one_retry() -> int:
    _async_fails_once.check()
    return 1


@retry(stop=stop_after_attempt(2), wait=wait_none(), reraise=True)
def _always_fails_reraise() -> None:
    raise ValueError("always fails")


@retry(stop=stop_after_attempt(2), wait=wait_none())
def _always_fails() -> None:
    raise ValueError("always fails")


@retry(stop=stop_after_attempt(2), wait=wait_none(), reraise=True)
async def _async_always_fails_reraise() -> None:
    raise ValueError("always fails")


def _swallow(call: typing.Callable[[], object]) -> typing.Callable[[], None]:
    def run() -> None:
        try:
            call()
        except (ValueError, tenacity.RetryError):
            pass

    return run


async def _swallow_async() -> None:
    try:
        await _async_always_fails_reraise()
    except ValueError:
        pass


def _retrying_objects_in_cyclic_garbage(call: typing.Callable[[], object]) -> int:
    call()
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    gc.set_debug(gc.DEBUG_SAVEALL)
    try:
        call()
        gc.collect()
        # type(), not isinstance(): the garbage can hold dead weakref proxies
        return sum(issubclass(type(obj), tenacity.BaseRetrying) for obj in gc.garbage)
    finally:
        gc.set_debug(0)
        gc.garbage.clear()
        if was_enabled:
            gc.enable()
        gc.collect()


class TestNoReferenceCycles(unittest.TestCase):
    def test_successful_call(self) -> None:
        self.assertEqual(_retrying_objects_in_cyclic_garbage(_succeeds), 0)

    def test_call_that_retries_once(self) -> None:
        self.assertEqual(
            _retrying_objects_in_cyclic_garbage(_succeeds_after_one_retry), 0
        )

    def test_async_successful_call(self) -> None:
        self.assertEqual(
            _retrying_objects_in_cyclic_garbage(lambda: asyncio.run(_async_succeeds())),
            0,
        )

    def test_async_call_that_retries_once(self) -> None:
        self.assertEqual(
            _retrying_objects_in_cyclic_garbage(
                lambda: asyncio.run(_async_succeeds_after_one_retry())
            ),
            0,
        )

    def test_all_attempts_fail_reraise(self) -> None:
        self.assertEqual(
            _retrying_objects_in_cyclic_garbage(_swallow(_always_fails_reraise)), 0
        )

    def test_all_attempts_fail(self) -> None:
        self.assertEqual(
            _retrying_objects_in_cyclic_garbage(_swallow(_always_fails)), 0
        )

    def test_async_all_attempts_fail(self) -> None:
        self.assertEqual(
            _retrying_objects_in_cyclic_garbage(
                lambda: asyncio.run(_swallow_async())
            ),
            0,
        )

    def test_exhausted_retry_error_keeps_the_last_attempt(self) -> None:
        with self.assertRaises(tenacity.RetryError) as caught:
            _always_fails()
        self.assertIsInstance(caught.exception.last_attempt.exception(), ValueError)
