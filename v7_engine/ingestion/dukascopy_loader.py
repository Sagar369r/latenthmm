"""
Dukascopy Historical Tick Loader.
Downloads real CFD tick data (bid/ask/timestamp) directly into RAM.
No CSV writes. Pure byte-stream → NumPy arrays.

Supports: EURUSD, GBPUSD, USDJPY, US30, DE40, and all Dukascopy FX/CFD pairs.

Usage:
    from v7_engine.ingestion.dukascopy_loader import DukascopyLoader
    loader = DukascopyLoader()
    ticks = loader.load("EURUSD", "2024-01-01", "2024-06-01")
    # ticks = {"timestamp_ns", "bid", "ask", "delta_t", "sign"}
"""
from __future__ import annotations

import io
import struct
import lzma
import logging
import datetime
from v7_engine.config import (
    INGEST_PRICE_NORM, INGEST_TIME_NORM_US, INGEST_TIME_NORM_NS, 
    INGEST_RETRY_BOUNDS, INGEST_TIMEOUT_LOCKS, INGEST_CHUNK_LIMITS, 
    INGEST_HISTORY_MONTHS, INGEST_HISTORY_DAYS_MARGIN, INGEST_PRICE_NORM_SMALL, 
    INGEST_TIME_NORM_MS
)
from typing import Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)

# Dukascopy stores data in per-hour BI5 files (LZMA-compressed binary).
_DUKA_URL = "https://datafeed.dukascopy.com/datafeed/{symbol}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"

# BI5 record: 4-byte uint (ms offset), 4-byte uint ask, 4-byte uint bid,
#             4-byte float ask_vol, 4-byte float bid_vol  → 20 bytes total
_BI5_RECORD_SIZE = 20
_BI5_FMT         = ">IIIff"   # big-endian: uint, uint, uint, float, float

# Pip scale per instrument (Dukascopy stores prices as integers × pip_scale)
_PIP_SCALE: dict[str, float] = {
    "EURUSD": 1e5, "GBPUSD": 1e5, "USDJPY": 1e3, "USDCHF": 1e5,
    "AUDUSD": 1e5, "NZDUSD": 1e5, "USDCAD": 1e5, "EURGBP": 1e5,
    "EURJPY": 1e3, "US30":   1.0, "US100":  1.0, "US500":  1.0,
    "DE40":   1.0, "UK100":  1.0, "JP225":  1.0, "XAUUSD": 1e2,
}


class DukascopyLoader:
    """
    Downloads Dukascopy BI5 tick files and converts them to the internal
    TickRecord array format used by the V7 pipeline.

    All data stays in RAM. No disk writes.
    """

    def __init__(
        self,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        session: Optional[requests.Session] = None,
    ):
        self._timeout    = timeout_s
        self._max_retries = max_retries
        self._session    = session or self._make_session()

    # ── public API ─────────────────────────────────────────────────────────────

    def load(
        self,
        symbol: str,
        start: str,
        end: str,
    ) -> dict[str, np.ndarray]:
        """
        Load all ticks for *symbol* in [start, end) into memory.

        Parameters
        ----------
        symbol : e.g. "EURUSD", "US30", "DE40"
        start  : ISO date string "YYYY-MM-DD"
        end    : ISO date string "YYYY-MM-DD"

        Returns
        -------
        dict with keys: timestamp_ns, bid, ask, delta_t, sign
        """
        symbol = symbol.upper()
        pip_scale = _PIP_SCALE.get(symbol, 1e5)

        import os
        import polars as pl
        from pathlib import Path
        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{symbol}_{start}_{end}.parquet"
        npz_file = cache_dir / f"{symbol}_{start}_{end}.npz"

        if cache_file.exists():
            logger.info(f"Loading {symbol} from Parquet cache: {cache_file} (Polars Multithreading)")
            df = pl.read_parquet(cache_file)
            return {
                "timestamp_ns": df["timestamp_ns"].to_numpy(),
                "bid":          df["bid"].to_numpy(),
                "ask":          df["ask"].to_numpy(),
                "delta_t":      df["delta_t"].to_numpy(),
                "sign":         df["sign"].to_numpy(),
            }
        elif npz_file.exists():
            logger.info(f"Migrating {symbol} from old .npz cache to Parquet...")
            data = np.load(npz_file)
            df = pl.DataFrame({
                "timestamp_ns": data["timestamp_ns"],
                "bid":          data["bid"],
                "ask":          data["ask"],
                "delta_t":      data["delta_t"],
                "sign":         data["sign"],
            })
            df.write_parquet(cache_file, compression="zstd")
            logger.info(f"Successfully migrated to Parquet: {cache_file}")
            return {
                "timestamp_ns": data["timestamp_ns"],
                "bid":          data["bid"],
                "ask":          data["ask"],
                "delta_t":      data["delta_t"],
                "sign":         data["sign"],
            }

        t_start = datetime.date.fromisoformat(start)
        t_end   = datetime.date.fromisoformat(end)

        all_ts_arrays  = []
        all_bid_arrays = []
        all_ask_arrays = []

        current = t_start
        total_days = (t_end - t_start).days
        total_hours = total_days * 24

        try:
            from tqdm import tqdm
            pbar = tqdm(total=total_hours, desc=f"Downloading {symbol}", unit="hr")
        except ImportError:
            pbar = None

        tasks = []
        current_day = t_start
        while current_day < t_end:
            for hour in range(24):
                tasks.append((current_day, hour))
            current_day += datetime.timedelta(days=1)
            
        def _fetch_task(date_obj, hr):
            records = self._fetch_hour(symbol, date_obj, hr, pip_scale)
            if records is None:
                return None
            ms_arr, bid_arr, ask_arr = records
            day_start_ms = int(
                datetime.datetime(date_obj.year, date_obj.month, date_obj.day,
                                  tzinfo=datetime.timezone.utc).timestamp() * 1000
            )
            return (day_start_ms + ms_arr.astype(np.int64)) * 1_000_000, bid_arr, ask_arr

        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Dukascopy servers handle concurrent requests well. Use 16 workers to speed up.
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(_fetch_task, d, h) for d, h in tasks]
            for fut in as_completed(futures):
                if pbar:
                    pbar.update(1)
                res = fut.result()
                if res is not None:
                    ts_arr, b_arr, a_arr = res
                    all_ts_arrays.append(ts_arr)
                    all_bid_arrays.append(b_arr)
                    all_ask_arrays.append(a_arr)

        if pbar:
            pbar.close()

        if not all_ts_arrays:
            raise RuntimeError(
                f"No tick data downloaded for {symbol} [{start}, {end}). "
                "Check symbol name and date range."
            )

        ts_ns  = np.concatenate(all_ts_arrays)
        bids   = np.concatenate(all_bid_arrays)
        asks   = np.concatenate(all_ask_arrays)

        # Sort by time (Dukascopy hours can occasionally be mis-ordered)
        sort_idx = np.argsort(ts_ns)
        ts_ns    = ts_ns[sort_idx]
        bids     = bids[sort_idx]
        asks     = asks[sort_idx]

        # Compute delta_t in seconds
        dt = np.diff(ts_ns, prepend=ts_ns[0]) / 1e9
        if len(dt) > 1:
            dt[0] = max(dt[1], 1e-6)
        else:
            dt[0] = 1e-6
        dt     = np.clip(dt, 1e-6, 60.0)

        # Micro-price / Mid-price approx
        mid = (bids + asks) / 2.0
        sign = np.sign(np.diff(mid, prepend=mid[0])).astype(np.int8)
        
        # Build Polars DataFrame for highly compressed Parquet storage
        import polars as pl
        df = pl.DataFrame({
            'timestamp_ns': ts_ns,
            'bid': bids,
            'ask': asks,
            'delta_t': dt,
            'sign': sign
        })
        
        logger.info(f"Saving {symbol} to Parquet cache: {cache_file}")
        df.write_parquet(cache_file, compression="zstd")
        
        return {
            "timestamp_ns": ts_ns,
            "bid":          bids,
            "ask":          asks,
            "delta_t":      dt,
            "sign":         sign,
        }

    # ── private helpers ────────────────────────────────────────────────────────

    def _fetch_hour(
        self,
        symbol: str,
        date: datetime.date,
        hour: int,
        pip_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Download and decode one BI5 file. Returns (ms_arr, bid_arr, ask_arr)."""
        url = _DUKA_URL.format(
            symbol=symbol,
            year=date.year,
            month=date.month - 1,   # Dukascopy months are 0-based!
            day=date.day,
            hour=hour,
        )
        for attempt in range(self._max_retries):
            try:
                resp = self._session.get(url, timeout=self._timeout)
                if resp.status_code == 404:
                    return None   # No data for this hour (weekend, holiday)
                if resp.status_code != 200:
                    logger.debug("HTTP %d for %s", resp.status_code, url)
                    return None
                return self._decode_bi5(resp.content, pip_scale)
            except requests.RequestException as exc:
                if attempt < self._max_retries - 1:
                    logger.debug("Retry %d for %s: %s", attempt + 1, url, exc)
                else:
                    logger.warning("Failed after %d attempts: %s", self._max_retries, url)
                    return None
        return None

    @staticmethod
    def _decode_bi5(data: bytes, pip_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Decompress LZMA BI5 and parse binary tick records."""
        if not data:
            return None
        try:
            raw = lzma.decompress(data)
        except lzma.LZMAError:
            return None

        n_records = len(raw) // _BI5_RECORD_SIZE
        if n_records == 0:
            return None

        # Vectorized parsing with NumPy
        dt = np.dtype([('ms', '>u4'), ('ask_raw', '>u4'), ('bid_raw', '>u4'), ('avol', '>f4'), ('bvol', '>f4')])
        arr = np.frombuffer(raw[: n_records * _BI5_RECORD_SIZE], dtype=dt)
        
        bid = arr['bid_raw'] / pip_scale
        ask = arr['ask_raw'] / pip_scale
        ms = arr['ms']
        
        mask = (bid > 0) & (ask > 0) & (ask >= bid)
        
        valid_ms = ms[mask]
        valid_bid = bid[mask]
        valid_ask = ask[mask]
        
        return valid_ms, valid_bid, valid_ask

    @staticmethod
    def _make_session() -> requests.Session:
        s = requests.Session()
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; V7-Engine/1.0)",
            "Accept-Encoding": "gzip, deflate",
        })
        return s
