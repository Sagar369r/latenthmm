"""
MT5 live tick feed via MetaApi.
Delivers ticks to RingBuffer on the asyncio loop — no cross-thread writes.
"""
from __future__ import annotations
import asyncio
import logging
import time
from typing import Callable, Optional
import os

logger = logging.getLogger(__name__)

try:
    from metaapi_cloud_sdk import MetaApi
    from metaapi_cloud_sdk.clients.metaapi.synchronizationListener import \
        SynchronizationListener
    _METAAPI_AVAILABLE = True
except ImportError:
    _METAAPI_AVAILABLE = False
    logger.error(
        "metaapi-cloud-sdk not installed. Run: pip install metaapi-cloud-sdk"
    )
    # Dummy removed per strict architectural requirements; failure will be explicit if SDK is missing.


class QuoteListener(SynchronizationListener):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def on_symbol_price_updated(self, instance_index: str, symbol: str, price) -> None:
        self.callback(instance_index, symbol, price)


class MT5Feed:
    """
    Connects to MT5 via MetaApi, streams live ticks, invokes on_tick callback.

    All callbacks are invoked on the asyncio event loop — safe to call
    ring_buffer.push() directly, no locks needed beyond threading.Lock
    already in RingBuffer.
    """

    def __init__(
        self,
        symbol: str,
        on_tick: Callable[[float, float, float, int], None],
        account_id: str = "",
        token: str = "",
    ):
        self.symbol     = symbol.upper()
        self._on_tick   = on_tick
        self._token     = token or os.getenv("METAAPI_TOKEN", "")
        self._account_id = account_id or os.getenv("METAAPI_ACCOUNT_ID", "")
        self._api       = None
        self._account   = None
        self._connection = None
        self._listener  = None
        self._last_bid  = 0.0
        self._last_ask  = 0.0
        self._last_time = 0.0
        self._running   = False

    async def connect(self) -> None:
        if not _METAAPI_AVAILABLE:
            raise RuntimeError("metaapi-cloud-sdk is required.")
        if not self._token or not self._account_id:
            raise RuntimeError(
                "METAAPI_TOKEN and METAAPI_ACCOUNT_ID env vars must be set."
            )
        logger.info("Connecting to MetaApi (MT5)...")
        self._api = MetaApi(self._token)
        self._account = await self._api.metatrader_account_api.get_account(
            self._account_id
        )
        if self._account.state not in ("DEPLOYED", "DEPLOYING"):
            await self._account.deploy()
        await self._account.wait_connected()
        self._connection = self._account.get_rpc_connection()
        await self._connection.connect()
        await self._connection.wait_synchronized()
        logger.info(f"Connected to MT5 account. Subscribing to {self.symbol}...")
        self._running = True
        
        self._listener = QuoteListener(self.on_symbol_price)
        self._connection.add_synchronization_listener(self._listener)
        
        # Subscribe to price stream
        await self._connection.subscribe_to_market_data(
            self.symbol,
            [{"type": "quotes"}],
        )
        logger.info(f"MT5 Feed live for {self.symbol}")

    def on_symbol_price(self, account_id: str, symbol: str, quotes) -> None:
        """Called by MetaApi on each tick — runs on asyncio loop."""
        if symbol != self.symbol or not self._running:
            return
        try:
            bid = float(quotes.bid)
            ask = float(quotes.ask)
            now = time.time()
            delta_t = max(now - self._last_time, 1e-6) if self._last_time > 0 else 1e-6
            mid = (bid + ask) / 2.0
            last_mid = (self._last_bid + self._last_ask) / 2.0
            sign = 1 if mid >= last_mid else -1
            self._last_bid  = bid
            self._last_ask  = ask
            self._last_time = now
            self._on_tick(bid, ask, delta_t, sign)
        except Exception as e:
            logger.error(f"MT5Feed tick error: {e}")

    async def disconnect(self) -> None:
        self._running = False
        if self._connection:
            await self._connection.close()
        if self._api:
            self._api.close()
        logger.info("MT5Feed disconnected.")
