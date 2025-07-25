"""
Async helper utilities
"""

import asyncio
from typing import Any, Coroutine


def run_async(coro: Coroutine) -> Any:
    """Run async function in sync environment"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)
