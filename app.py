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

    def list_symbols(self, top_n, min_vol):
        meta, ctxs = self._post({"type": "metaAndAssetCtxs"})
        rows = []
        for u, c in zip(meta["universe"], ctxs):
            if u.get("isDelisted"):
                continue
            vol = float(c.get("dayNtlVlm") or 0)
            if vol >= min_vol:
                rows.append({"display": f"{u['name']}/USDC", "ref": u["name"], "vol": vol})
        rows.sort(key=lambda r: -r["vol"])
        return rows[:top_n]

    def candles(self, coin, tf, start_ms, end_ms):
        step, out, since, guard = TF_MS[tf], [], int(start_ms), 0
        while since < end_ms and guard < 10:
            guard += 1
            data = self._post({"type": "candleSnapshot",
                               "req": {"coin": coin, "interval": tf, "startTime": since, "endTime": int(end_ms)}})
            if not data:
                break
            out.extend([[d["t"], d["o"], d["h"], d["l"], d["c"], d["v"]] for d in data])
            nxt = data[-1]["t"] + step
            if nxt <= since or nxt >= end_ms:
                break
            since = nxt
        if not out:
            return to_frame([])
        df = to_frame(out)
        return df[(df.ts >= start_ms) & (df.ts < end_ms)].reset_index(drop=True)


class DemoSource:
    """Synthetic candles so the dashboard can be tried without internet. NOT real market data."""
    id, label = "demo", "Demo (synthetic)"
    NAMES = [f"SYN{i:02d}/USDT" for i in range(1, 41)]

    def list_symbols(self, top_n, min_vol):
        return [{"display": n, "ref": n, "vol": 5e7 / (i + 1)} for i, n in enumerate(self.NAMES)][:top_n]

    def _minutes(self, sym, end_ms):
        now = int(end_ms)
        day_start = (now // DAY_MS) * DAY_MS
        start = day_start - 3 * DAY_MS
        n = (now - start) // 60_000 + 1
        rng = np.random.default_rng(zlib.crc32(sym.encode()))
        base = 20 + rng.random() * 400
        kind = zlib.crc32(sym.encode()) % 4
        elapsed = max(90, (now - day_start) // 60_000)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = x[i - 1] * 0.995 + rng.normal(0, 0.0004)          # mean-reverting around the base
        today0 = (day_start - start) // 60_000
        t = np.arange(n) - today0
        today = t >= 0
        if kind == 0:      # starts above, drifts down onto the level, bounces
            tgt = np.interp(t, [0, elapsed - 60, elapsed - 25, elapsed], [0.03, 0.015, -0.001, 0.004])
            x = np.where(today, tgt + rng.normal(0, 0.0002, n), x)
        elif kind == 1:    # above then closes below the level
            tgt = np.interp(t, [0, elapsed - 20, elapsed], [0.025, 0.002, -0.012])
            x = np.where(today, tgt + rng.normal(0, 0.0006, n), x)
        elif kind == 2:    # staying above, slowly approaching
            tgt = np.interp(t, [0, elapsed], [0.04, 0.008])
            x = np.where(today, tgt + rng.normal(0, 0.0006, n), x)
        close = base * (1 + x)
        openp = np.r_[close[0], close[:-1]]
        spread = np.abs(rng.normal(0, 0.0008, n)) + 0.0003
        high = np.maximum(openp, close) * (1 + spread)
        low = np.minimum(openp, close) * (1 - spread)
        vol = rng.lognormal(3, 0.6, n) * (1 + 8 * np.exp(-(x / 0.002) ** 2))     # heavy volume node near the base price
        ts = start + np.arange(n) * 60_000
        return pd.DataFrame({"ts": ts, "open": openp, "high": high, "low": low, "close": close, "volume": vol})

    def candles(self, sym, tf, start_ms, end_ms):
        end_ms = int(min(end_ms, time.time() * 1000))
        m = self._minutes(sym, end_ms)
        m = m[(m.ts >= start_ms) & (m.ts < end_ms)]
        if tf == "1m" or m.empty:
            return m.reset_index(drop=True)
        step = TF_MS[tf]
        g = (m.ts // step) * step
        agg = m.groupby(g).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                               close=("close", "last"), volume=("volume", "sum")).reset_index(names="ts")
        return agg[(agg.ts >= start_ms) & (agg.ts < end_ms)].reset_index(drop=True)


def make_source(sid):
    if sid == "hyperliquid":
        return HyperliquidSource()
    if sid == "demo":
        return DemoSource()
    return CcxtSource(sid)


def pick_source(choice, top_n, min_vol):
    order = SOURCE_ORDER if choice == "Auto" else [choice.lower().split()[0]]
    errors = {}
    for sid in order:
        src = make_source(sid)
        try:
            syms = src.list_symbols(top_n, min_vol)
            if syms:
                return src, syms, errors
            errors[sid] = "no symbols returned"
        except Exception as e:
            errors[sid] = f"{type(e).__name__}: {str(e)[:110]}"
    return None, [], errors


# ─────────────────────────────────────────────────────────────
# 3. VOLUME PROFILE + POC
# ─────────────────────────────────────────────────────────────
def volume_profile(df, bins=100):
    """Spread each candle's volume evenly over the price bins its high-low range covers.
    Returns (bin edges, volume per bin)."""
    lo, hi = float(df["low"].min()), float(df["high"].max())
    if hi <= lo:
        hi = lo * 1.0001 + 1e-12
    edges = np.linspace(lo, hi, bins + 1)
    lows = df["low"].values[:, None]
    highs = df["high"].values[:, None]
    vols = df["volume"].values
    overlap = np.clip(np.minimum(highs, edges[None, 1:]) - np.maximum(lows, edges[None, :-1]), 0, None)
    rng = highs - lows
    weight = np.divide(overlap, rng, out=np.zeros_like(overlap), where=rng > 0)
    prof = (weight * vols[:, None]).sum(axis=0)
    flat = rng[:, 0] <= 0                      # candles with no range: all volume goes to the close's bin
    if flat.any():
        idx = np.clip(np.searchsorted(edges, df["close"].values[flat], side="right") - 1, 0, bins - 1)
        np.add.at(prof, idx, vols[flat])
    return edges, prof


def value_area(prof, edges, poc_idx, pct=0.70):
    total, lo, hi, acc = prof.sum(), poc_idx, poc_idx, prof[poc_idx]
    while acc < total * pct and (lo > 0 or hi < len(prof) - 1):
        up = prof[hi + 1] if hi < len(prof) - 1 else -1.0
        dn = prof[lo - 1] if lo > 0 else -1.0
        if up >= dn:
            hi += 1
            acc += prof[hi]
        else:
            lo -= 1
            acc += prof[lo]
    return float(edges[lo]), float(edges[hi + 1])


def compute_poc(df, bins, method):
    if method == "Daily VWAP (same as the Pine code)":
        hlc3 = (df["high"] + df["low"] + df["close"]) / 3
        v = df["volume"].sum()
        poc = float((hlc3 * df["volume"]).sum() / v) if v > 0 else float(df["close"].iloc[-1])
        edges, prof = volume_profile(df, bins)
        return {"poc": poc, "vah": None, "val": None, "edges": edges, "prof": prof}
    edges, prof = volume_profile(df, bins)
    i = int(np.argmax(prof))
    val, vah = value_area(prof, edges, i)
    return {"poc": float((edges[i] + edges[i + 1]) / 2), "vah": vah, "val": val, "edges": edges, "prof": prof}


# ─────────────────────────────────────────────────────────────
# 4. THE PINE RULES, REPLAYED OVER TODAY'S CLOSED CANDLES
# ─────────────────────────────────────────────────────────────
def evaluate_day(sig, prev_poc, day_start_ms):
    """Returns (events, tested_today). events = [(row index, ts, 'BUY' | 'EXIT')]."""
    ts, low, high, close = sig["ts"].values, sig["low"].values, sig["high"].values, sig["close"].values
    tested, events = False, []
    for i in range(1, len(sig)):
        if ts[i] < day_start_ms:
            continue
        came_from_above = high[i - 1] > prev_poc and low[i - 1] > prev_poc
        hits = low[i] <= prev_poc and close[i] >= prev_poc
        first_test = (not tested) and came_from_above and hits
        exit_long = close[i] < prev_poc and low[i - 1] >= prev_poc
        if first_test:
            tested = True
            events.append((i, int(ts[i]), "BUY"))
        if exit_long:
            events.append((i, int(ts[i]), "EXIT"))
    return events, tested


def scan_symbol(src, item, p, cache):
    ref, now_ms = item["ref"], p["now_ms"]
    key = (src.id, ref, p["prev_start"], p["profile_tf"], p["bins"], p["method"])
    poc = cache.get(key)
    if poc is None:
        prev = src.candles(ref, p["profile_tf"], p["prev_start"], p["day_start"])
        expected = DAY_MS // TF_MS[p["profile_tf"]]
        if len(prev) < int(expected * 0.9):
            return {"skip": f"Previous day has only {len(prev)} of {expected} candles"}
        poc = compute_poc(prev, p["bins"], p["method"])
        cache[key] = poc
    sig = src.candles(ref, p["tf"], p["prev_start"], now_ms)
    sig = sig[sig.ts + TF_MS[p["tf"]] <= now_ms].reset_index(drop=True)      # closed candles only
    if (sig.ts >= p["day_start"]).sum() == 0:
        return {"skip": "No closed candle yet today"}
    events, tested = evaluate_day(sig, poc["poc"], p["day_start"])
    price = float(sig["close"].iloc[-1])
    return {"symbol": item["display"], "price": price, "poc": poc["poc"], "vah": poc["vah"], "val": poc["val"],
            "dist": (price / poc["poc"] - 1) * 100, "tested": tested, "events": events, "n": len(sig),
            "sig": sig, "edges": poc["edges"], "prof": poc["prof"], "vol": item["vol"]}


def run_scan(src, items, p, cache, on_progress):
    results, skipped = [], []
    with ThreadPoolExecutor(max_workers=p["workers"]) as pool:
        futures = {pool.submit(scan_symbol, src, it, p, cache): it for it in items}
        for done, f in enumerate(as_completed(futures), 1):
            it = futures[f]
            try:
                r = f.result()
                if "skip" in r:
                    skipped.append({"Coin": it["display"], "Reason": r["skip"]})
                else:
                    results.append(r)
            except Exception as e:
                skipped.append({"Coin": it["display"], "Reason": f"{type(e).__name__}: {str(e)[:100]}"})
            on_progress(done, len(items))
    return results, skipped


def status_of(r, k, watch_pct):
    n = r["n"]
    recent = [e for e in r["events"] if (n - 1) - e[0] < k]
    if recent:
        i, ts, kind = recent[-1]
        return kind, (n - 1) - i, ts
    if (not r["tested"]) and r["price"] > r["poc"] and r["dist"] <= watch_pct:
        return "WATCH", None, None
    return "", None, None


LABEL = {"BUY": "🟢 POC BUY", "EXIT": "🔴 POC EXIT", "WATCH": "👀 Approaching", "": "—"}


def table_rows(results, k, watch_pct):
    rows = []
    for r in results:
        kind, ago, ts = status_of(r, k, watch_pct)
        rows.append({
            "Coin": r["symbol"], "Signal": LABEL[kind], "_kind": kind,
            "Price": r["price"], "Prev-day POC": r["poc"], "To POC (%)": r["dist"],
            "Candles ago": ago, "Signal time (IST)": fmt_ist(ts) if ts else "",
            "Prev VAL": r["val"], "Prev VAH": r["vah"],
            "First test done today": "yes" if r["tested"] else "no", "24h vol ($M)": r["vol"] / 1e6,
        })
    return pd.DataFrame(rows)


COLS = {
    "Price": st.column_config.NumberColumn(format="%.6g"),
    "Prev-day POC": st.column_config.NumberColumn(format="%.6g"),
    "Prev VAL": st.column_config.NumberColumn(format="%.6g"),
    "Prev VAH": st.column_config.NumberColumn(format="%.6g"),
    "To POC (%)": st.column_config.NumberColumn(format="%.2f", help="Price vs previous-day POC. Positive = price is above it."),
    "24h vol ($M)": st.column_config.NumberColumn(format="%.1f"),
    "Candles ago": st.column_config.NumberColumn(format="%d", help="0 = the latest closed candle"),
}


def show_table(df):
    show = df.drop(columns=["_kind"])
    try:
        st.dataframe(show, hide_index=True, width="stretch", column_config=COLS)
    except TypeError:
        st.dataframe(show, hide_index=True, use_container_width=True, column_config=COLS)


def make_chart(r):
    c = r["sig"]
    t = pd.to_datetime(c["ts"], unit="ms", utc=True).dt.tz_convert("Asia/Kolkata")
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.8, 0.2], horizontal_spacing=0.01)
    fig.add_trace(go.Candlestick(x=t, open=c["open"], high=c["high"], low=c["low"], close=c["close"],
                                 name="Price", showlegend=False), row=1, col=1)
    if r["vah"] is not None:
        fig.add_hrect(y0=r["val"], y1=r["vah"], fillcolor="orange", opacity=0.10, line_width=0, row=1, col=1)
    fig.add_hline(y=r["poc"], line=dict(color="orange", width=2), row=1, col=1,
                  annotation_text=f"Prev-day POC {r['poc']:.6g}", annotation_position="top left")
    for i, ts, kind in r["events"]:
        x = t.iloc[i]
        if kind == "BUY":
            fig.add_trace(go.Scatter(x=[x], y=[c["low"].iloc[i]], mode="markers+text", text=["BUY"], textposition="bottom center",
                                     marker=dict(symbol="triangle-up", size=12, color="#2ecc71"), showlegend=False), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=[x], y=[c["high"].iloc[i]], mode="markers+text", text=["EXIT"], textposition="top center",
                                     marker=dict(symbol="triangle-down", size=12, color="#e74c3c"), showlegend=False), row=1, col=1)
    edges, prof = r["edges"], r["prof"]
    mids = (edges[:-1] + edges[1:]) / 2
    colors = ["orange" if abs(m - r["poc"]) <= (edges[1] - edges[0]) / 2 + 1e-12 else "#6c7a89" for m in mids]
    fig.add_trace(go.Bar(x=prof, y=mids, orientation="h", width=np.diff(edges) * 0.92, marker_color=colors,
                         name="Previous-day volume profile", showlegend=False), row=1, col=2)
    fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    return fig


@st.cache_resource
def poc_cache():
    return {}


# ─────────────────────────────────────────────────────────────
# 5. DASHBOARD
# ─────────────────────────────────────────────────────────────
st.title("🎯 Daily POC Pullback Scanner")
st.caption("Finds coins that are testing yesterday's Point of Control from above. Signals are read on closed candles only.")

with st.expander("Rules used", expanded=False):
    st.markdown(
        "- **Previous-day POC**: the price level with the most traded volume yesterday, from a volume profile built on 1-minute candles.\n"
        "- **🟢 POC BUY**: the previous candle's low was above the POC, this candle's low touches or goes below it, and it closes at or above it. Only the first such test of the day counts.\n"
        "- **🔴 POC EXIT**: this candle closes below the POC after the previous candle's low was at or above it.\n"
        "- **👀 Approaching**: price is above the POC, within the set distance, and the first test has not happened today."
    )

sb = st.sidebar
sb.header("Scan settings")
source_choice = sb.selectbox("Data source", ["Auto", "Binance", "OKX", "KuCoin", "Hyperliquid", "Demo (synthetic)"],
                             help="Auto tries Binance, then OKX, KuCoin, Hyperliquid. Binance blocks some cloud servers.")
top_n = sb.slider("Coins to scan (top by 24h volume)", 10, 300, 100, step=10)
min_vol_m = sb.number_input("Minimum 24h volume ($M)", 0.0, 1000.0, 2.0, step=1.0)
tf = sb.selectbox("Signal timeframe", ["5m", "15m", "30m", "1h"], index=1)
method = sb.selectbox("POC method", ["Volume profile (true POC)", "Daily VWAP (same as the Pine code)"])
bins = sb.slider("Profile price bins", 24, 300, 100, step=4,
                 help="More bins = a sharper POC. 100 is a good start. (The Pine code had 24 but never used it.)")
profile_tf = sb.selectbox("Profile candles", ["1m", "5m"], index=0, help="1m is the most accurate. 5m is faster.")
day_label = sb.selectbox("Day starts at", ["00:00 UTC (Binance / TradingView crypto)", "00:00 IST"])
k = sb.slider("Show signals from the last N closed candles", 1, 10, 2)
watch_pct = sb.slider("Approaching: within % above POC", 0.1, 5.0, 1.0, step=0.1)
workers = sb.slider("Parallel workers", 1, 8, 4, help="Lower this if the exchange starts rate-limiting.")
auto = sb.checkbox("Auto-refresh")
auto_min = sb.number_input("Refresh every (minutes)", 1, 60, 5) if auto else 5
run = sb.button("Scan now", type="primary", width="stretch")

if run or st.session_state.pop("auto_scan", False):
    now_ms = int(time.time() * 1000)
    shift = 19_800_000 if day_label == "00:00 IST" else 0
    day_start, prev_start = day_bounds(now_ms, shift)
    cache = poc_cache()
    for key in [x for x in cache if x[2] < prev_start]:
        cache.pop(key, None)
    with st.status("Scanning…", expanded=True) as status:
        st.write("Choosing a data source and listing coins…")
        src, items, errs = pick_source(source_choice, top_n, min_vol_m * 1e6)
        if src is None:
            status.update(label="No data source worked", state="error")
            st.error("Could not reach any data source.")
            st.json(errs)
            st.stop()
        if errs:
            st.write("Skipped sources: " + "; ".join(f"{a} ({b})" for a, b in errs.items()))
        st.write(f"Using **{src.label}** with {len(items)} coins.")
        bar = st.progress(0.0)
        params = {"now_ms": now_ms, "day_start": day_start, "prev_start": prev_start, "tf": tf, "bins": bins,
                  "method": method, "profile_tf": profile_tf, "workers": workers}
        results, skipped = run_scan(src, items, params, cache, lambda d, n: bar.progress(d / n))
        status.update(label=f"Done: {len(results)} coins scanned", state="complete", expanded=False)
    st.session_state["scan"] = {"results": results, "skipped": skipped, "source": src.label, "tf": tf,
                                "day_start": day_start, "at": now_ms, "method": method}

scan = st.session_state.get("scan")
if not scan:
    st.info("Set your options in the sidebar and press **Scan now**. Try **Demo (synthetic)** first to see the layout without internet.")
    st.stop()

if scan["source"].startswith("Demo"):
    st.warning("Demo data is synthetic and not real market data. Do not trade from it.")

df = table_rows(scan["results"], k, watch_pct)
n_buy = int((df["_kind"] == "BUY").sum()) if len(df) else 0
n_exit = int((df["_kind"] == "EXIT").sum()) if len(df) else 0
n_watch = int((df["_kind"] == "WATCH").sum()) if len(df) else 0
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("🟢 POC BUY", n_buy)
m2.metric("🔴 POC EXIT", n_exit)
m3.metric("👀 Approaching", n_watch)
m4.metric("Coins scanned", len(scan["results"]))
m5.metric("Skipped", len(scan["skipped"]))
st.caption(f"Source: {scan['source']} · {scan['tf']} candles · day began {fmt_ist(scan['day_start'])} IST · "
           f"scanned {fmt_ist(scan['at'])} IST · POC method: {scan['method']}")

tab_sig, tab_watch, tab_all, tab_chart = st.tabs(["Signals", "Approaching", "All coins", "Chart"])
with tab_sig:
    sig_df = df[df["_kind"].isin(["BUY", "EXIT"])].sort_values(["_kind", "Candles ago"]) if len(df) else df
    if len(sig_df):
        show_table(sig_df)
    else:
        st.info(f"No coin has a BUY or EXIT in the last {k} closed candles.")
with tab_watch:
    w = df[df["_kind"] == "WATCH"].sort_values("To POC (%)") if len(df) else df
    if len(w):
        show_table(w)
    else:
        st.info("No coin is approaching its POC from above right now.")
with tab_all:
    if len(df):
        show_table(df.sort_values("Coin"))
    if scan["skipped"]:
        with st.expander(f"Skipped coins ({len(scan['skipped'])})"):
            st.dataframe(pd.DataFrame(scan["skipped"]), hide_index=True)
with tab_chart:
    if scan["results"]:
        order = sorted(scan["results"], key=lambda r: (status_of(r, k, watch_pct)[0] == "", r["symbol"]))
        names = [r["symbol"] for r in order]
        pick = st.selectbox("Coin", names)
        r = next(x for x in order if x["symbol"] == pick)
        st.plotly_chart(make_chart(r), width="stretch")
        st.caption("Orange line = previous-day POC. Shaded band = previous-day value area (70% of volume). "
                   "Bars on the right = previous-day volume profile, orange bar = POC.")

if auto:
    time.sleep(int(auto_min) * 60)
    st.session_state["auto_scan"] = True
    st.rerun()
