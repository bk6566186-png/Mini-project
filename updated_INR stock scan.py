"""
Advanced Value Stock Screener (NSE/BSE Live) - Fixed + MarketCap filter

Usage:
 pip install nsetools yfinance pandas numpy tabulate
 python value_stock_screener_fixed.py
"""

from nsetools import Nse
import yfinance as yf
import pandas as pd
import numpy as np
from tabulate import tabulate
import time

nse = Nse()

# Fetch all stock codes
all_stock_codes = nse.get_stock_codes()

# all_stock_codes usually a dict like {'SYMBOL': 'NAME', 'RELIANCE': 'Reliance Industries Limited', ...}
if isinstance(all_stock_codes, dict):
    all_tickers = list(all_stock_codes.keys())[1:]  # skip header key 'SYMBOL'
else:
    print("⚠️ Warning: NSE returned", type(all_stock_codes))
    all_tickers = list(all_stock_codes)

# Limit for demo (remove limit for full scan)
TICKERS = [s + ".NS" for s in all_tickers[:100]]
TOP_N = 10
DELAY_BETWEEN = 0.5

# Market cap threshold: 2000 crore INR = 2000 * 10^7 = 20,000,000,000 INR
MARKETCAP_THRESHOLD_INR = 2000 * 10**7

def safe_div(a, b, default=np.nan):
    try:
        return a / b
    except Exception:
        return default

def fetch_fx_inr_per_usd():
    """
    Try to get INR per 1 USD using yfinance ticker 'INR=X'
    Returns float (INR per USD) or None if cannot fetch.
    """
    try:
        fx = yf.Ticker("INR=X")
        info = fx.info
        rate = info.get('regularMarketPrice') or info.get('previousClose')
        if rate is None:
            # fallback to history
            hist = fx.history(period='1d')
            if not hist.empty:
                rate = hist['Close'].iloc[-1]
        return float(rate) if rate is not None else None
    except Exception:
        return None

# Cache fx rate once
FX_INR_PER_USD = None

def fetch_metrics(ticker):
    """
    Returns a dict with metrics including marketCap (raw), currency and market_cap_in_inr (if convertible).
    """
    global FX_INR_PER_USD
    t = yf.Ticker(ticker)

    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

    pe = info.get('trailingPE') or info.get('forwardPE')
    pb = info.get('priceToBook')
    dividend_yield = info.get('dividendYield')
    price = info.get('regularMarketPrice') or info.get('previousClose')
    market_cap = info.get('marketCap')
    market_cap_currency = info.get('currency')  # e.g., 'INR' or 'USD'

    # Try to compute market_cap_in_inr
    market_cap_in_inr = None
    try:
        if market_cap is not None:
            if market_cap_currency == 'INR' or ticker.endswith('.NS'):
                market_cap_in_inr = float(market_cap)
            elif market_cap_currency == 'USD':
                if FX_INR_PER_USD is None:
                    FX_INR_PER_USD = fetch_fx_inr_per_usd()
                if FX_INR_PER_USD:
                    market_cap_in_inr = float(market_cap) * float(FX_INR_PER_USD)
            else:
                # Unknown currency — attempt conversion if FX available (best-effort)
                if FX_INR_PER_USD is None:
                    FX_INR_PER_USD = fetch_fx_inr_per_usd()
                if FX_INR_PER_USD:
                    # This is a guess: treat market_cap as USD if unknown
                    market_cap_in_inr = float(market_cap) * float(FX_INR_PER_USD)
    except Exception:
        market_cap_in_inr = None

    # Earnings growth (year-over-year) best-effort
    earnings_growth = None
    try:
        earnings = t.earnings
        if earnings is not None and len(earnings) >= 2:
            last = float(earnings['Earnings'].iloc[-1])
            prev = float(earnings['Earnings'].iloc[-2])
            if prev != 0 and not np.isnan(prev):
                earnings_growth = (last - prev) / abs(prev)
    except Exception:
        earnings_growth = None

    return {
        'ticker': ticker,
        'longName': info.get('longName'),
        'price': price,
        'pe': pe,
        'pb': pb,
        'dividend_yield': dividend_yield,
        'earnings_growth': earnings_growth,
        'market_cap': market_cap,
        'market_cap_currency': market_cap_currency,
        'market_cap_in_inr': market_cap_in_inr
    }

def score_metrics(m):
    w_pe = 0.30
    w_pb = 0.20
    w_div = 0.25
    w_growth = 0.25

    pe = m.get('pe')
    if pe is None or pe <= 0 or np.isnan(pe) or pe > 200:
        pe_score = 0.0
    else:
        pe_score = 1.0 / pe

    pb = m.get('pb')
    if pb is None or pb <= 0 or np.isnan(pb) or pb > 50:
        pb_score = 0.0
    else:
        pb_score = 1.0 / pb

    dy = m.get('dividend_yield')
    if dy is None or np.isnan(dy) or dy <= 0:
        div_score = 0.0
    else:
        div_score = min(dy, 0.08)

    g = m.get('earnings_growth')
    if g is None or np.isnan(g):
        growth_score = 0.0
    else:
        growth_score = min(max(g, -0.5), 0.5)

    pe_score_n = pe_score * 10
    pb_score_n = pb_score * 10
    div_score_n = div_score * 12
    growth_score_n = (growth_score + 0.5) * 2

    combined = (w_pe * pe_score_n) + (w_pb * pb_score_n) + (w_div * div_score_n) + (w_growth * growth_score_n)

    suggestion = 'BUY' if combined > 2.5 else ('HOLD' if combined > 1.5 else 'AVOID')

    return {
        'pe_score': pe_score_n,
        'pb_score': pb_score_n,
        'div_score': div_score_n,
        'growth_score': growth_score_n,
        'combined_score': combined,
        'suggestion': suggestion
    }

def main():
    results = []
    skipped_due_to_marketcap = []
    print(f"🔎 Fetching and screening up to {len(TICKERS)} NSE stocks...\n")

    for i, tk in enumerate(TICKERS, 1):
        try:
            print(f"[{i}/{len(TICKERS)}] Checking {tk}...", end=' ')
            m = fetch_metrics(tk)

            # If we couldn't determine market cap in INR, we will skip (but still show a note)
            mc_in_inr = m.get('market_cap_in_inr')
            if mc_in_inr is None:
                print("SKIP (market cap unknown)")
                skipped_due_to_marketcap.append((tk, m.get('market_cap'), m.get('market_cap_currency')))
                time.sleep(DELAY_BETWEEN)
                continue

            if mc_in_inr < MARKETCAP_THRESHOLD_INR:
                print(f"SKIP (market cap {mc_in_inr:.0f} INR < threshold {MARKETCAP_THRESHOLD_INR:.0f})")
                skipped_due_to_marketcap.append((tk, mc_in_inr))
                time.sleep(DELAY_BETWEEN)
                continue

            s = score_metrics(m)
            results.append({**m, **s})
            print(s['suggestion'])
        except Exception as e:
            print("error:", str(e))
        time.sleep(DELAY_BETWEEN)

    if not results:
        print("\n⚠️ कोई स्टॉक threshold के ऊपर नहीं मिला या market cap अनिर्णीत था.")
    else:
        df = pd.DataFrame(results)
        df['combined_score'] = pd.to_numeric(df['combined_score'], errors='coerce').fillna(0)
        df = df.sort_values('combined_score', ascending=False)

        show_cols = ['ticker', 'longName', 'price', 'pe', 'pb', 'dividend_yield',
                     'earnings_growth', 'market_cap', 'market_cap_currency',
                     'market_cap_in_inr', 'combined_score', 'suggestion']
        out_df = df[show_cols].reset_index(drop=True)

        print('\nTop Value Picks:')
        print(tabulate(out_df.head(TOP_N), headers='keys', tablefmt='psql', showindex=False))

        out_df.to_csv('nse_value_stocks_filtered_by_marketcap.csv', index=False)
        print('\n✅ Data saved to nse_value_stocks_filtered_by_marketcap.csv')

    # optional: show skipped list summary
    if skipped_due_to_marketcap:
        print(f"\nNote: {len(skipped_due_to_marketcap)} tickers skipped due to market cap filter or unknown market cap.")
        # You could optionally save skipped list to CSV for review.

if __name__ == '__main__':
    main()
