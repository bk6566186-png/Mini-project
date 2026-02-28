"""
Crypto Screener — Python Script

Features:
- Collects coin data from CoinGecko (market cap in INR), Binance, and CoinDCX (via CCXT when available).
- Filters coins with market cap >= 2000 crore INR (20,000,000,000 INR). If fewer than requested_count (default 100), the script gradually lowers threshold until it gathers enough coins.
- Computes simple technical signals (14-period RSI on 1h candles from Binance when available) and basic rules to tag: BUY / HOLD / AVOID.
- Saves results to a timestamped CSV file. Each run produces fresh live data.

Usage:
- Install dependencies: pip install requests pandas ccxt
- Run: python crypto_screener.py

Note: This script provides algorithmic signals for educational purposes only. This is NOT financial advice.

"""

import time
import math
import requests
import ccxt
import pandas as pd
from datetime import datetime

# ------------------------- CONFIG -------------------------
COINGECKO_API = 'https://api.coingecko.com/api/v3'
VS_CURRENCY = 'inr'  # get market cap in INR
INITIAL_MARKETCAP_THRESHOLD = 20_000_000_000  # 2000 crore INR = 20,000,000,000
REQUESTED_COUNT = 100
BINANCE = ccxt.binance({'enableRateLimit': True})
COINDCX = None
try:
    COINDCX = ccxt.coindcx({'enableRateLimit': True})
except Exception:
    COINDCX = None

# ------------------------- HELPERS -------------------------

def fetch_coingecko_markets(per_page=250, page=1):
    url = f"{COINGECKO_API}/coins/markets"
    params = {
        'vs_currency': VS_CURRENCY,
        'order': 'market_cap_desc',
        'per_page': per_page,
        'page': page,
        'price_change_percentage': '24h,7d'
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def gather_coins(min_marketcap, needed):
    # Try pages until we collect enough or reach reasonable limit
    coins = []
    page = 1
    while len(coins) < needed and page <= 5:
        items = fetch_coingecko_markets(per_page=250, page=page)
        for c in items:
            try:
                mc = c.get('market_cap') or 0
                if mc >= min_marketcap:
                    coins.append({
                        'id': c['id'],
                        'symbol': c['symbol'],
                        'name': c['name'],
                        'market_cap': mc,
                        'current_price': c.get('current_price'),
                        'market_cap_rank': c.get('market_cap_rank'),
                        'price_change_percentage_24h': c.get('price_change_percentage_24h'),
                        'total_volume': c.get('total_volume')
                    })
            except Exception:
                continue
        page += 1
        time.sleep(1)
    return coins


def rsi_from_prices(prices, period=14):
    # prices: list of close prices (oldest -> newest)
    if len(prices) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        if diff > 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(diff))
    # first average
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    # smooth
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def fetch_binance_symbol_price(symbol):
    # Convert coingecko symbol to potential Binance market symbol(s)
    # Commonly symbol/USDT
    try:
        pair = f"{symbol.upper()}/USDT"
        ticker = BINANCE.fetch_ticker(pair)
        return ticker
    except Exception:
        # try /BUSD
        try:
            pair = f"{symbol.upper()}/BUSD"
            ticker = BINANCE.fetch_ticker(pair)
            return ticker
        except Exception:
            return None


def fetch_binance_ohlcv(symbol, timeframe='1h', limit=200):
    try:
        pair = f"{symbol.upper()}/USDT"
        ohlcv = BINANCE.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
        # ohlcv rows: [timestamp, open, high, low, close, volume]
        closes = [row[4] for row in ohlcv]
        return closes
    except Exception:
        return None

# ------------------------- SIGNAL RULES -------------------------

def decide_tag(rsi, pct24h, market_cap, volume):
    # A simple heuristic system — tweak as needed
    # rsi: numeric or None
    # pct24h: 24h percent change
    # market_cap: INR
    # volume: last 24h volume in INR equivalent (approx)
    tag = 'UNKNOWN'
    reasons = []
    if market_cap is not None and market_cap >= 100_000_000_000:  # >=100 billion INR
        reasons.append('Strong market cap')
    if rsi is not None:
        if rsi < 30:
            reasons.append('RSI oversold')
            tag = 'BUY'
        elif rsi < 50:
            if pct24h is not None and pct24h > 0:
                tag = 'HOLD'
            else:
                tag = 'WATCH'
        elif rsi < 70:
            tag = 'HOLD'
        else:
            reasons.append('RSI overbought')
            tag = 'AVOID'
    else:
        # no RSI
        if pct24h is not None:
            if pct24h < -15:
                tag = 'AVOID'
            elif pct24h > 10:
                tag = 'HOLD'
            else:
                tag = 'WATCH'
        else:
            tag = 'WATCH'

    # Volume and market cap sanity checks
    if volume is not None and volume < 1_000_000:  # very low volume
        tag = 'AVOID' if tag != 'BUY' else tag
        reasons.append('Low volume')
    return tag, '; '.join(reasons)

# ------------------------- MAIN RUN -------------------------

def run_screener(requested_count=REQUESTED_COUNT, initial_threshold=INITIAL_MARKETCAP_THRESHOLD):
    threshold = initial_threshold
    coins = gather_coins(threshold, requested_count)
    # If less than needed, relax threshold by half iteratively
    relax_steps = 0
    while len(coins) < requested_count and relax_steps < 6:
        threshold = int(threshold * 0.7)  # reduce to 70%
        coins = gather_coins(threshold, requested_count)
        relax_steps += 1
    print(f"Collected {len(coins)} coins with marketcap threshold {threshold}")

    results = []
    for c in coins[:requested_count]:
        symbol = c['symbol']
        name = c['name']
        market_cap = c['market_cap']
        price = c.get('current_price')
        pct24h = c.get('price_change_percentage_24h')
        total_vol = c.get('total_volume')

        # Binance data
        bin_ticker = fetch_binance_symbol_price(symbol)
        bin_price = bin_ticker['last'] if bin_ticker and 'last' in bin_ticker else None
        # OHLCV for RSI
        closes = fetch_binance_ohlcv(symbol)
        rsi = rsi_from_prices(closes) if closes else None

        # CoinDCX availability (via ccxt if configured)
        cd_ticker = None
        if COINDCX:
            try:
                # CoinDCX symbol mapping is not standardized; try symbol/INR
                pair = f"{symbol.upper()}/INR"
                cd_ticker = COINDCX.fetch_ticker(pair)
            except Exception:
                cd_ticker = None

        volume_est = total_vol or (bin_ticker['baseVolume'] if bin_ticker and 'baseVolume' in bin_ticker else None)

        tag, reasons = decide_tag(rsi, pct24h, market_cap, volume_est)

        results.append({
            'timestamp': datetime.utcnow().isoformat(),
            'id': c['id'],
            'symbol': symbol.upper(),
            'name': name,
            'market_cap_inr': market_cap,
            'coingecko_price_inr': price,
            'binance_price_usdt': bin_price,
            '24h_change_pct': pct24h,
            'rsi_14_1h': rsi,
            'estimated_volume': volume_est,
            'suggestion': tag,
            'reason': reasons
        })
        time.sleep(0.2)  # be polite to APIs

    df = pd.DataFrame(results)
    timestr = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f'crypto_screener_{timestr}.csv'
    df.to_csv(fname, index=False)
    print(f"Saved {len(df)} rows to {fname}")
    return df, fname


if __name__ == '__main__':
    df, fname = run_screener()
    print(df.head(20))
