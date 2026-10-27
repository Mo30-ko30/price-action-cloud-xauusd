import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SYMBOL = "XAUUSD"
API_BASE = "https://biquote.io/api"
STATE_FILE = Path("state.json")
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Mirrors the supplied MQ4 defaults.
TREND_FAST_EMA = 50
TREND_SLOW_EMA = 200
ENTRY_FAST_EMA = 20
ENTRY_SLOW_EMA = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
SR_LOOKBACK = 20
NEAR_LEVEL_ATR = 0.35
STOP_ATR_MULTIPLIER = 1.50
MINIMUM_RISK_ATR = 0.80
TARGET1_R = 1.0
TARGET2_R = 2.0
MIN_SIGNAL_SCORE = 5
USE_ICHIMOKU_FILTER = True
USE_RSI_FILTER = True
USE_SUPPORT_RESISTANCE = True


def fetch_bars(interval: str, limit: int = 1000) -> pd.DataFrame:
    url = f"{API_BASE}/{SYMBOL}/ohlc"
    response = requests.get(url, params={"interval": interval, "limit": limit}, timeout=25)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload.get("bars"), list):
        raise RuntimeError(f"Unexpected candle response for {interval}: {payload}")
    df = pd.DataFrame(payload["bars"])
    required = {"openTime", "open", "high", "low", "close", "isOpen"}
    if df.empty or not required.issubset(df.columns):
        raise RuntimeError(f"Missing OHLC fields for {interval}")
    df["openTime"] = pd.to_datetime(df["openTime"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["openTime", "open", "high", "low", "close"])
    df = df.sort_values("openTime").drop_duplicates("openTime").set_index("openTime")
    df = df.loc[~df["isOpen"].astype(bool)]
    if len(df) < 250:
        raise RuntimeError(f"Not enough closed bars for {interval}: {len(df)}")
    return df


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.where(avg_loss != 0, 100)
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50)
    return out


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def ichimoku_lines(df: pd.DataFrame):
    high, low = df["high"], df["low"]
    tenkan = (high.rolling(TENKAN_PERIOD).max() + low.rolling(TENKAN_PERIOD).min()) / 2
    kijun = (high.rolling(KIJUN_PERIOD).max() + low.rolling(KIJUN_PERIOD).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(KIJUN_PERIOD)
    span_b = ((high.rolling(SENKOU_B_PERIOD).max() + low.rolling(SENKOU_B_PERIOD).min()) / 2).shift(KIJUN_PERIOD)
    return tenkan, kijun, span_a, span_b


def candle_patterns(df: pd.DataFrame):
    # df is chronological; -1 is the most recent CLOSED candle.
    o1, c1 = float(df["open"].iloc[-1]), float(df["close"].iloc[-1])
    o2, c2 = float(df["open"].iloc[-2]), float(df["close"].iloc[-2])
    h1, l1 = float(df["high"].iloc[-1]), float(df["low"].iloc[-1])

    bull_engulf = c2 < o2 and c1 > o1 and o1 <= c2 and c1 >= o2
    bear_engulf = c2 > o2 and c1 < o1 and o1 >= c2 and c1 <= o2

    body = abs(c1 - o1)
    candle_range = h1 - l1
    if candle_range <= 0:
        return False, False
    lower_wick = min(o1, c1) - l1
    upper_wick = h1 - max(o1, c1)
    bull_pin = lower_wick >= body * 2 and lower_wick > upper_wick and c1 > l1 + candle_range * 0.55
    bear_pin = upper_wick >= body * 2 and upper_wick > lower_wick and c1 < l1 + candle_range * 0.45
    return (bull_engulf or bull_pin), (bear_engulf or bear_pin)


def calculate_signal(h1: pd.DataFrame, m15: pd.DataFrame):
    h1 = h1.copy()
    m15 = m15.copy()

    h1["ema_fast"] = ema(h1["close"], TREND_FAST_EMA)
    h1["ema_slow"] = ema(h1["close"], TREND_SLOW_EMA)
    _, _, h1_span_a, h1_span_b = ichimoku_lines(h1)

    h = h1.iloc[-1]  # latest CLOSED H1 bar
    h_close = float(h["close"])
    fast_h1 = float(h["ema_fast"])
    slow_h1 = float(h["ema_slow"])
    trend = 0
    if np.isfinite(fast_h1) and np.isfinite(slow_h1):
        bullish = fast_h1 > slow_h1 and h_close > fast_h1
        bearish = fast_h1 < slow_h1 and h_close < fast_h1
        if USE_ICHIMOKU_FILTER:
            a, b = h1_span_a.iloc[-1], h1_span_b.iloc[-1]
            if not (np.isfinite(a) and np.isfinite(b)):
                bullish = bearish = False
            else:
                bullish = bullish and h_close > max(float(a), float(b))
                bearish = bearish and h_close < min(float(a), float(b))
        if bullish:
            trend = 1
        elif bearish:
            trend = -1

    m15["ema_fast"] = ema(m15["close"], ENTRY_FAST_EMA)
    m15["ema_slow"] = ema(m15["close"], ENTRY_SLOW_EMA)
    m15["rsi"] = rsi(m15["close"], RSI_PERIOD)
    m15["atr"] = atr(m15, ATR_PERIOD)
    tenkan, kijun, _, _ = ichimoku_lines(m15)
    m = m15.iloc[-1]
    close1 = float(m["close"])
    low1 = float(m["low"])
    high1 = float(m["high"])
    atr1 = float(m["atr"])
    rsi1 = float(m["rsi"])
    ema_fast = float(m["ema_fast"])
    ema_slow = float(m["ema_slow"])
    signal_time = m15.index[-1]

    if not all(np.isfinite(x) for x in [atr1, rsi1, ema_fast, ema_slow]) or atr1 <= 0:
        return {"status": "WAIT", "reason": "indicator warm-up", "signal_time": signal_time.isoformat()}

    # Match MQ4 support/resistance lookback: 20 candles starting two bars back.
    sr_window = m15.iloc[-(SR_LOOKBACK + 1):-1]
    support = float(sr_window["low"].min())
    resistance = float(sr_window["high"].max())
    bull_candle, bear_candle = candle_patterns(m15)

    near_support = support > 0 and close1 >= support and close1 - support <= atr1 * NEAR_LEVEL_ATR
    near_resistance = resistance > 0 and close1 <= resistance and resistance - close1 <= atr1 * NEAR_LEVEL_ATR

    buy_score = 2 if trend == 1 else 0
    sell_score = 2 if trend == -1 else 0
    if ema_fast > ema_slow:
        buy_score += 1
    if ema_fast < ema_slow:
        sell_score += 1
    if bull_candle:
        buy_score += 2
    if bear_candle:
        sell_score += 2
    if USE_RSI_FILTER:
        if 50 <= rsi1 < 70:
            buy_score += 1
        if 30 < rsi1 <= 50:
            sell_score += 1
    if USE_SUPPORT_RESISTANCE:
        if near_support:
            buy_score += 1
        if near_resistance:
            sell_score += 1
    if USE_ICHIMOKU_FILTER:
        t, k = tenkan.iloc[-1], kijun.iloc[-1]
        if np.isfinite(t) and np.isfinite(k):
            if t > k:
                buy_score += 1
            if t < k:
                sell_score += 1

    buy_signal = trend == 1 and buy_score >= MIN_SIGNAL_SCORE and buy_score > sell_score and bull_candle
    sell_signal = trend == -1 and sell_score >= MIN_SIGNAL_SCORE and sell_score > buy_score and bear_candle
    result = {
        "symbol": SYMBOL,
        "signal_time": signal_time.isoformat(),
        "close": close1,
        "trend": "BULLISH" if trend == 1 else "BEARISH" if trend == -1 else "NEUTRAL",
        "buy_score": buy_score,
        "sell_score": sell_score,
        "rsi": rsi1,
        "atr": atr1,
        "status": "BUY" if buy_signal else "SELL" if sell_signal else "WAIT",
    }
    if not (buy_signal or sell_signal):
        return result

    is_buy = buy_signal
    entry = close1
    if is_buy:
        structural_stop = support - atr1 * 0.15 if support > 0 else 0
        stop = entry - atr1 * STOP_ATR_MULTIPLIER
        if 0 < structural_stop < entry:
            stop = min(stop, structural_stop)
    else:
        structural_stop = resistance + atr1 * 0.15 if resistance > 0 else 0
        stop = entry + atr1 * STOP_ATR_MULTIPLIER
        if structural_stop > entry:
            stop = max(stop, structural_stop)
    risk = abs(entry - stop)
    if risk < atr1 * MINIMUM_RISK_ATR:
        risk = atr1 * MINIMUM_RISK_ATR
        stop = entry - risk if is_buy else entry + risk
    target1 = entry + risk * TARGET1_R if is_buy else entry - risk * TARGET1_R
    target2 = entry + risk * TARGET2_R if is_buy else entry - risk * TARGET2_R
    result.update({
        "entry": entry, "sl": stop, "tp1": target1, "tp2": target2,
        "direction": "BUY" if is_buy else "SELL",
    })
    return result


def send_telegram(message: str):
    if DRY_RUN:
        print("DRY_RUN Telegram message:\n" + message)
        return
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in GitHub Actions Secrets.")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=20)
    r.raise_for_status()
    payload = r.json()
    if not payload.get("ok"):
        raise RuntimeError(f"Telegram API error: {payload}")


def load_state():
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main():
    print("Fetching XAUUSD closed candles...")
    h1 = fetch_bars("1h", 1000)
    m15 = fetch_bars("15m", 1000)
    result = calculate_signal(h1, m15)
    candle_id = result["signal_time"]
    state = load_state()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    # Don't repeat the same closed M15 candle across scheduled workflow runs.
    if state.get("last_processed_m15") == candle_id:
        print("This M15 candle was already processed; no duplicate notification.")
        return

    state["last_processed_m15"] = candle_id
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

    if result["status"] in ("BUY", "SELL"):
        msg = (
            f"Price Action Signal Pro — {SYMBOL}\n"
            f"Signal: {result['status']}\n"
            f"Time (UTC): {result['signal_time']}\n"
            f"Trend H1: {result['trend']}\n"
            f"Entry (M15 close): {result['entry']:.2f}\n"
            f"SL: {result['sl']:.2f}\n"
            f"TP1: {result['tp1']:.2f}\n"
            f"TP2: {result['tp2']:.2f}\n"
            f"Score — Buy: {result['buy_score']} | Sell: {result['sell_score']}\n"
            f"RSI: {result['rsi']:.1f} | ATR: {result['atr']:.2f}\n"
            "Signal only — no trade was placed.\n"
            "Warning: indicative external feed; verify against your Alpari MT4 chart before acting."
        )
        send_telegram(msg)
    else:
        print("No BUY/SELL signal on the latest closed M15 candle.")


if __name__ == "__main__":
    main()
