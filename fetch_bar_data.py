# Download OHLCV data from OKX.

import csv
import os
from datetime import datetime, timezone
import ccxt


SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "LTC/USDT",
    "BCH/USDT",
    "ETC/USDT",
    "XRP/USDT",
    "EOS/USDT",
    "ADA/USDT",
    "LINK/USDT",
]

TIMEFRAMES = ["1m", "1h"]
PAGE_LIMIT = 300
DATA_DIR = "data"
START_ISO = "2021-01-01T00:00:00Z"
END_ISO = "2025-12-31T23:59:59Z"


def build_exchange(exchange_id: str):
    if exchange_id == "binance":
        return ccxt.binance(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
                "timeout": 20000,
            }
        )
    if exchange_id == "okx":
        return ccxt.okx(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
                "timeout": 20000,
            }
        )
    raise ValueError(f"Unsupported exchange: {exchange_id}")


def get_exchange():
    for exchange_id in ["binance", "okx"]:
        exchange = build_exchange(exchange_id)
        try:
            exchange.load_markets()
            exchange.fetch_ohlcv("BTC/USDT", timeframe="1m", limit=1)
            print(f"Using exchange: {exchange_id}")
            return exchange, exchange_id
        except Exception as exc:
            print(f"{exchange_id} unavailable: {exc}")
    raise RuntimeError("Both Binance and OKX are unavailable.")


def timeframe_to_ms(timeframe: str) -> int:
    if timeframe == "1m":
        return 60 * 1000
    if timeframe == "1h":
        return 60 * 60 * 1000
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def fetch_ohlcv_range(exchange, symbol: str, timeframe: str, start_ms: int, end_ms: int):
    all_rows = []
    since = start_ms
    step_ms = timeframe_to_ms(timeframe)

    while since <= end_ms:
        page = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=PAGE_LIMIT)
        if not page:
            break
        filtered = [r for r in page if start_ms <= int(r[0]) <= end_ms]
        all_rows.extend(filtered)

        last_ts = int(page[-1][0])
        # Move to next bar to avoid duplicate boundary row.
        next_since = last_ts + step_ms
        if next_since <= since:
            break
        since = next_since

        print(f"Fetching {symbol} {timeframe}: got {len(all_rows)} rows so far...")

    # Deduplicate and sort by timestamp.
    by_ts = {int(r[0]): r for r in all_rows}
    sorted_rows = [by_ts[ts] for ts in sorted(by_ts.keys())]
    return sorted_rows


def save_ohlcv(symbol: str, timeframe: str, rows, exchange_id: str):
    tf_dir = os.path.join(DATA_DIR, timeframe)
    os.makedirs(tf_dir, exist_ok=True)
    safe_symbol = symbol.replace("/", "_")
    out_path = os.path.join(tf_dir, f"{safe_symbol}_{exchange_id}.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "datetime_utc", "open", "high", "low", "close", "volume"])
        for row in rows:
            ts = int(row[0])
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            writer.writerow([ts, dt, row[1], row[2], row[3], row[4], row[5]])

    print(f"Saved: {out_path} ({len(rows)} rows)")


def main():
    exchange, exchange_id = get_exchange()
    start_ms = exchange.parse8601(START_ISO)
    end_ms = exchange.parse8601(END_ISO)

    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            try:
                print(f"Start downloading: {symbol} {timeframe} ({START_ISO} -> {END_ISO})")
                rows = fetch_ohlcv_range(exchange, symbol, timeframe, start_ms, end_ms)
                save_ohlcv(symbol, timeframe, rows, exchange_id)
            except Exception as exc:
                print(f"Failed: {symbol} {timeframe} -> {exc}")


if __name__ == "__main__":
    main()
