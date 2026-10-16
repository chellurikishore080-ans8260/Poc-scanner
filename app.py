"""
Daily POC Pullback Scanner  (Streamlit)

Rules, ported from the Pine Script "Daily POC Pullback Strategy [Kishore Signals]":
  1. prevPOC  = Point of Control of the PREVIOUS day (price level with the most traded volume).
  2. BUY      = first test of prevPOC today, coming from above:
                previous candle's low is above prevPOC (price came from above)
                AND this candle's low <= prevPOC AND this candle closes >= prevPOC
                AND no earlier BUY today.
  3. EXIT     = this candle closes below prevPOC AND the previous candle's low was >= prevPOC.

Differences from the Pine code (on purpose):
  * Pine's "POC" is really the day's VWAP. Here the default POC is a real volume-profile POC.
    Switch "POC method" to "Daily VWAP" to get the Pine behaviour.
  * Signals are checked on CLOSED candles only, so they do not repaint.
"""
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import ccxt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Daily POC Pullback Scanner", page_icon="🎯", layout="wide")

DAY_MS = 86_400_000
TF_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000}
STABLES = {"USDT", "USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "USDD", "PYUSD",
           "EUR", "GBP", "AEUR", "USD1", "USDE", "XUSD", "RLUSD"}
SOURCE_ORDER = ["binance", "okx", "kucoin", "hyperliquid"]


# ─────────────────────────────────────────────────────────────
# 1. HELPERS
# ─────────────────────────────────────────────────────────────
def fmt_ist(ms):
    return pd.to_datetime(ms, unit="ms", utc=True).tz_convert("Asia/Kolkata").strftime("%d %b %H:%M")


def with_retry(fn, tries=3):
    """Retry network errors; give up at once on region blocks."""
    for i in range(tries):
        try:
            return fn()
        except (ccxt.NetworkError, requests.RequestException) as e:
            msg = str(e).lower()
            if "restricted" in msg or "451" in msg or i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def to_frame(rows):
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["ts"] = df["ts"].astype("int64")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    return df


def day_bounds(now_ms, shift_ms):
    """Start of the current day and of the previous day. shift_ms = 0 for UTC, 19_800_000 for IST."""
    day_start = ((now_ms + shift_ms) // DAY_MS) * DAY_MS - shift_ms
    return day_start, day_start - DAY_MS


# ─────────────────────────────────────────────────────────────
# 2. DATA SOURCES  (Binance / OKX / KuCoin via ccxt, Hyperliquid via its public API, Demo)
# ─────────────────────────────────────────────────────────────
class CcxtSource:
    def __init__(self, ex_id):
        self.id = ex_id
        self.label = ex_id.capitalize()
        self.markets = None
        self._local = threading.local()

    def ex(self):
        e = getattr(self._local, "ex", None)
        if e is None:
            e = getattr(ccxt, self.id)({"enableRateLimit": True, "timeout": 20000})
            if self.markets:
                try:
                    e.set_markets(self.markets)
                except Exception:
                    pass
            self._local.ex = e
        return e

    def list_symbols(self, top_n, min_vol):
        ex = self.ex()
        ex.load_markets()
        self.markets = ex.markets
        tickers = ex.fetch_tickers()
        rows = []
        for sym, t in tickers.items():
            m = ex.markets.get(sym)
            if not m or not m.get("spot") or m.get("quote") != "USDT" or m.get("active") is False:
                continue
            if m["base"] in STABLES:
                continue
            vol = t.get("quoteVolume")
            if vol is None and t.get("baseVolume") and t.get("last"):
                vol = t["baseVolume"] * t["last"]
            vol = float(vol or 0)
            if vol >= min_vol:
                rows.append({"display": sym, "ref": sym, "vol": vol})
        rows.sort(key=lambda r: -r["vol"])
        return rows[:top_n]

    def candles(self, sym, tf, start_ms, end_ms):
        ex, step, out, since, guard = self.ex(), TF_MS[tf], [], int(start_ms), 0
        while since < end_ms and guard < 40:
            guard += 1
            chunk = with_retry(lambda: ex.fetch_ohlcv(sym, tf, since=since, limit=1000))
            if not chunk:
                break
            out.extend(chunk)
            nxt = chunk[-1][0] + step
            if nxt <= since or nxt >= end_ms:
                break
            since = nxt
        if not out:
            return to_frame([])
        df = to_frame(out)
        return df[(df.ts >= start_ms) & (df.ts < end_ms)].reset_index(drop=True)


class HyperliquidSource:
    id, label = "hyperliquid", "Hyperliquid"
    URL = "https://api.hyperliquid.xyz/info"

    def _post(self, body):
        def call():
            r = requests.post(self.URL, json=body, timeout=20)
            r.raise_for_status()
            return r.json()
        return with_retry(call)

    def list_symbols(self, top_n, min_
