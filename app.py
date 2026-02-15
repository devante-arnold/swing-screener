import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import List, Dict
import time
import json

# Page config
st.set_page_config(
    page_title="Swing Trade Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for exit checklist
if 'exit_checklist' not in st.session_state:
    st.session_state.exit_checklist = {
        'r1_hit': False,
        'profit_100': False,
        'stop_hit': False,
        'otm_10': False,
        'loss_50': False
    }

# Initialize scan results cache
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
    st.session_state.scan_timestamp = None
    st.session_state.market_regime = None
    st.session_state.vix_level = None

# Try to load cached results from file
import os
import pickle
from pathlib import Path

CACHE_FILE = Path("/tmp/swing_scanner_cache.pkl")

def load_cached_results():
    """Load cached scan results from file"""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'rb') as f:
                cached_data = pickle.load(f)
                # Check if cache is less than 7 days old
                cache_age = (datetime.utcnow() - cached_data['timestamp']).days
                if cache_age < 7:
                    return cached_data
        except:
            pass
    return None

def save_cached_results(results, timestamp, regime, vix):
    """Save scan results to file"""
    try:
        cache_data = {
            'results': results,
            'timestamp': timestamp,
            'regime': regime,
            'vix': vix
        }
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache_data, f)
    except:
        pass

# Load cached results on startup if available
if st.session_state.scan_results is None:
    cached = load_cached_results()
    if cached:
        st.session_state.scan_results = cached['results']
        st.session_state.scan_timestamp = cached['timestamp']
        st.session_state.market_regime = cached['regime']
        st.session_state.vix_level = cached['vix']

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .bullish {
        color: #00c853;
        font-weight: bold;
    }
    .bearish {
        color: #ff1744;
        font-weight: bold;
    }
    .entry-reminders {
        background-color: #FFFACD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# [Previous SwingScreener class code remains the same]
class SwingScreener:
    def __init__(self, min_market_cap=2e9, min_volume=500000, confluence_threshold=4):
        self.min_market_cap = min_market_cap
        self.min_volume = min_volume
        self.confluence_threshold = confluence_threshold
        
    def get_market_tickers(self) -> List[str]:
        sp500_major = [
            'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG', 'META', 'AVGO', 'ORCL', 'CSCO', 'ADBE',
            'CRM', 'AMD', 'INTC', 'IBM', 'QCOM', 'TXN', 'AMAT', 'MU', 'LRCX', 'KLAC',
            'SNPS', 'CDNS', 'MCHP', 'ADI', 'NXPI', 'PANW', 'PLTR', 'NOW', 'TEAM', 'WDAY',
            'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'ABNB',
            'GM', 'F', 'MAR', 'CMG', 'ORLY', 'YUM', 'DHI', 'LEN', 'DG', 'ROST',
            'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'BMY',
            'AMGN', 'GILD', 'CVS', 'CI', 'ISRG', 'REGN', 'VRTX', 'ZTS', 'HCA', 'BSX',
            'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SPGI', 'C',
            'SCHW', 'AXP', 'CB', 'PGR', 'MMC', 'ICE', 'CME', 'AON', 'USB', 'TFC',
            'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR', 'EA', 'TTWO', 'MTCH',
            'WMT', 'PG', 'COST', 'KO', 'PEP', 'PM', 'MO', 'MDLZ', 'CL', 'KMB',
            'GIS', 'K', 'HSY', 'SYY', 'KHC', 'STZ', 'TAP', 'CPB', 'CAG', 'SJM',
            'BA', 'CAT', 'UNP', 'HON', 'UPS', 'RTX', 'LMT', 'GE', 'MMM', 'DE',
            'GD', 'NOC', 'FDX', 'NSC', 'CSX', 'EMR', 'ETN', 'ITW', 'PH', 'CMI',
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL',
            'KMI', 'WMB', 'DVN', 'HES', 'FANG', 'BKR', 'TRGP', 'OKE', 'APA', 'MRO',
            'LIN', 'APD', 'SHW', 'ECL', 'NEM', 'FCX', 'DOW', 'DD', 'PPG', 'NUE',
            'NEE', 'SO', 'DUK', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'PEG', 'ED',
            'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'O', 'WELL', 'DLR', 'SBAC', 'AVB'
        ]
        
        nasdaq_major = [
            'INTU', 'BKNG', 'ADP', 'VRTX', 'SBUX', 'GILD', 'ADI', 'REGN', 'LRCX', 'MDLZ',
            'PANW', 'MU', 'PYPL', 'KLAC', 'SNPS', 'CDNS', 'MELI', 'ASML', 'ABNB', 'CHTR',
            'CTAS', 'MAR', 'ORLY', 'AZN', 'CRWD', 'FTNT', 'CSX', 'NXPI', 'PCAR', 'MRVL',
            'MNST', 'WDAY', 'ADSK', 'DASH', 'CPRT', 'PAYX', 'ROST', 'ODFL', 'KDP', 'FAST',
            'VRSK', 'CTSH', 'EA', 'GEHC', 'LULU', 'DDOG', 'IDXX', 'XEL', 'EXC', 'ON',
            'TEAM', 'ANSS', 'CSGP', 'ZS', 'DXCM', 'TTWO', 'BIIB', 'ILMN', 'WBD', 'MDB', 'ZM', 'MRNA'
        ]
        
        active_traders = ['SPY', 'QQQ', 'IWM', 'SMH', 'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP',
            'SOFI', 'PLTR', 'NIO', 'RIVN', 'COIN', 'RBLX', 'SNOW', 'NET',
            'SHOP', 'SQ', 'UBER', 'LYFT', 'PINS', 'SNAP', 'ROKU', 'ARKK', 'GME', 'AMC', 'PLUG']
        
        return list(set(sp500_major + nasdaq_major + active_traders))
    
    def calculate_pivot_points(self, high, low, close):
        pivot = (high + low + close) / 3
        return {
            'PP': pivot,
            'R1': (2 * pivot) - low,
            'R2': pivot + (high - low),
            'S1': (2 * pivot) - high,
            'S2': pivot - (high - low)
        }
    
    def calculate_vrvp_levels(self, df, days=30):
        try:
            recent = df.tail(days).copy()
            price_min = recent['Low'].min()
            price_max = recent['High'].max()
            bins = np.linspace(price_min, price_max, 50)
            
            volume_profile = []
            for i in range(len(bins) - 1):
                mask = (recent['Close'] >= bins[i]) & (recent['Close'] < bins[i+1])
                vol = recent.loc[mask, 'Volume'].sum()
                volume_profile.append((bins[i], vol))
            
            poc_price = max(volume_profile, key=lambda x: x[1])[0] if volume_profile else recent['Close'].iloc[-1]
            
            total_vol = sum(v for _, v in volume_profile)
            cumulative = 0
            val, vah = poc_price, poc_price
            
            for price, vol in sorted(volume_profile, key=lambda x: x[0]):
                cumulative += vol
                if cumulative < total_vol * 0.15:
                    val = price
                elif cumulative > total_vol * 0.85:
                    vah = price
                    break
            
            return {'POC': poc_price, 'VAH': vah, 'VAL': val}
        except:
            return {'POC': 0, 'VAH': 0, 'VAL': 0}
    
    def calculate_atr(self, df, period=14):
        high = df['High']
        low = df['Low']
        close = df['Close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else 0
    
    def check_confluence(self, ticker, data):
        try:
            if len(data) < 50:
                return None
            
            current_price = data['Close'].iloc[-1]
            
            weekly = data.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            if len(weekly) < 2:
                return None
            
            prev_week = weekly.iloc[-2]
            pivots = self.calculate_pivot_points(prev_week['High'], prev_week['Low'], prev_week['Close'])
            vrvp = self.calculate_vrvp_levels(data, days=30)
            
            ema_20 = data['Close'].ewm(span=20).mean().iloc[-1]
            ema_50 = data['Close'].ewm(span=50).mean().iloc[-1]
            
            prev_week_high = prev_week['High']
            prev_week_low = prev_week['Low']
            atr = self.calculate_atr(data)
            
            avg_volume_20 = data['Volume'].tail(20).mean()
            current_volume = data['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 0
            
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            
            score = 0
            factors = []
            confluence_breakdown = {
                'pivot': False,
                'vrvp': False,
                'ema_sma': False,
                'vpa': False,
                'prev_week': False,
                'rsi': False,
                'price_distance': False
            }
            
            # 1. Near pivot
            for level_name, level_price in pivots.items():
                if abs(current_price - level_price) / current_price < 0.02:
                    score += 1
                    factors.append(f"Near {level_name}")
                    confluence_breakdown['pivot'] = True
                    break
            
            # 2. Near VRVP
            for vrvp_name, vrvp_price in vrvp.items():
                if vrvp_price > 0 and abs(current_price - vrvp_price) / current_price < 0.02:
                    score += 1
                    factors.append(f"VRVP {vrvp_name}")
                    confluence_breakdown['vrvp'] = True
                    break
            
            # 3. EMA/SMA alignment
            if current_price > ema_20 > ema_50:
                score += 1
                factors.append("Bullish EMA/SMA")
                confluence_breakdown['ema_sma'] = True
            elif current_price < ema_20 < ema_50:
                score += 1
                factors.append("Bearish EMA/SMA")
                confluence_breakdown['ema_sma'] = True
            
            # 4. VPA confluence (Volume spike)
            if volume_ratio > 1.5:
                score += 1
                factors.append(f"VPA {volume_ratio:.1f}x")
                confluence_breakdown['vpa'] = True
            
            # 5. Previous week H/L
            if abs(current_price - prev_week_high) / current_price < 0.03:
                score += 1
                factors.append("Prev week high")
                confluence_breakdown['prev_week'] = True
            elif abs(current_price - prev_week_low) / current_price < 0.03:
                score += 1
                factors.append("Prev week low")
                confluence_breakdown['prev_week'] = True
            
            # 6. RSI
            if 30 < current_rsi < 70:
                score += 1
                factors.append(f"RSI {current_rsi:.0f}")
                confluence_breakdown['rsi'] = True
            
            # 7. Price distance from 20 EMA
            price_distance = abs(current_price - ema_20) / ema_20
            if price_distance < 0.03:
                score += 1
                factors.append("Near 20 EMA")
                confluence_breakdown['price_distance'] = True
            
            setup_type = "Range-bound"
            if current_price > pivots['R1'] and current_price > ema_20:
                setup_type = "Bullish breakout"
            elif current_price < pivots['S1'] and current_price < ema_20:
                setup_type = "Bearish breakdown"
            elif current_price > ema_20 and abs(current_price - vrvp['VAL']) / current_price < 0.02:
                setup_type = "Bullish reversal"
            elif current_price < ema_20 and abs(current_price - vrvp['VAH']) / current_price < 0.02:
                setup_type = "Bearish reversal"
            
            if "Bullish" in setup_type:
                target = pivots['R2'] if current_price < pivots['R2'] else vrvp['VAH']
                stop = current_price - (1.5 * atr)  # Changed to 1.5x ATR
                option_type = "CALL"
                strike = round(current_price * 1.02)
            else:
                target = pivots['S2'] if current_price > pivots['S2'] else vrvp['VAL']
                stop = current_price + (1.5 * atr)  # Changed to 1.5x ATR
                option_type = "PUT"
                strike = round(current_price * 0.98)
            
            if score < self.confluence_threshold:
                return None
            
            return {
                'ticker': ticker,
                'price': current_price,
                'score': score,
                'setup_type': setup_type,
                'factors': factors,
                'option_type': option_type,
                'strike': strike,
                'target': target,
                'stop': stop,
                'atr': atr,
                'rsi': current_rsi,
                'volume_ratio': volume_ratio,
                'confluence_breakdown': confluence_breakdown
            }
        except:
            return None
    
    def scan_market(self, progress_callback=None):
        tickers = self.get_market_tickers()
        results = []
        
        for i, ticker in enumerate(tickers):
            try:
                if progress_callback:
                    progress_callback(i + 1, len(tickers), ticker)
                
                stock = yf.Ticker(ticker)
                data = stock.history(period='6mo', interval='1d')
                
                if data.empty or len(data) < 50:
                    continue
                
                info = stock.info
                market_cap = info.get('marketCap', 0)
                avg_volume = data['Volume'].tail(20).mean()
                
                if market_cap < self.min_market_cap or avg_volume < self.min_volume:
                    continue
                
                result = self.check_confluence(ticker, data)
                if result:
                    results.append(result)
            except:
                continue
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results


def main():
    st.markdown('<p class="main-header">📊 Swing Trade Screener</p>', unsafe_allow_html=True)
    st.markdown("**AI-powered confluence scanner with interactive checklists**")
    
    # Disclaimer
    st.warning("⚠️ **This is not financial advice. This is only the display of my personal trading tools.**")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    confluence_threshold = st.sidebar.slider(
        "Minimum Confluence Score",
        min_value=3,
        max_value=7,
        value=4,
        help="Minimum number of confluence factors required"
    )
    
    min_market_cap = st.sidebar.selectbox(
        "Minimum Market Cap",
        options=[500e6, 1e9, 2e9, 5e9, 10e9],
        index=2,
        format_func=lambda x: f"${x/1e9:.1f}B"
    )
    
    min_volume = st.sidebar.selectbox(
        "Minimum Daily Volume",
        options=[100000, 250000, 500000, 1000000],
        index=2,
        format_func=lambda x: f"{x/1000:.0f}K shares"
    )
    
    slack_webhook = st.sidebar.text_input(
        "Slack Webhook URL (optional)",
        type="password",
        help="Get webhook from api.slack.com/apps"
    )
    
    # Exit Reminders Checklist in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚨 Exit Reminders")
    
    st.session_state.exit_checklist['r1_hit'] = st.sidebar.checkbox(
        "R1 hit → begin to scale out",
        value=st.session_state.exit_checklist['r1_hit']
    )
    st.session_state.exit_checklist['profit_100'] = st.sidebar.checkbox(
        "P/L: Up 100% → Consider profit take",
        value=st.session_state.exit_checklist['profit_100']
    )
    st.session_state.exit_checklist['stop_hit'] = st.sidebar.checkbox(
        "Stop hit (1.5x ATR) → Exit 100%",
        value=st.session_state.exit_checklist['stop_hit']
    )
    st.session_state.exit_checklist['otm_10'] = st.sidebar.checkbox(
        "Moneyness: >10% OTM at day 14 → Exit 100%",
        value=st.session_state.exit_checklist['otm_10']
    )
    st.session_state.exit_checklist['loss_50'] = st.sidebar.checkbox(
        "P/L: Down 50% → Exit 100%",
        value=st.session_state.exit_checklist['loss_50']
    )
    
    # Main area - Scan button
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        scan_button = st.button("🚀 Run Market Scan", type="primary", use_container_width=True)
    
    with col2:
        if st.session_state.scan_timestamp:
            # Convert UTC to user's local time using JavaScript
            utc_timestamp = st.session_state.scan_timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')
            st.components.v1.html(f"""
                <div style="margin-top: 8px; font-size: 0.9em; color: #666;">
                    Last scan: <span id="local-time"></span>
                </div>
                <script>
                    const utcDate = new Date('{utc_timestamp}');
                    const localTime = utcDate.toLocaleTimeString('en-US', {{
                        hour: 'numeric',
                        minute: '2-digit',
                        hour12: true
                    }});
                    document.getElementById('local-time').textContent = localTime;
                </script>
            """, height=30)
    
    # Run scan or display cached results
    if scan_button:
        screener = SwingScreener(
            min_market_cap=min_market_cap,
            min_volume=min_volume,
            confluence_threshold=confluence_threshold
        )
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total, ticker):
            progress = current / total
            progress_bar.progress(progress)
            status_text.text(f"Scanning {ticker}... ({current}/{total})")
        
        with st.spinner("Initializing scanner..."):
            results = screener.scan_market(progress_callback=update_progress)
        
        progress_bar.empty()
        status_text.empty()
        
        # Get market regime and VIX
        try:
            spy = yf.Ticker('SPY')
            spy_data = spy.history(period='3mo')
            spy_price = spy_data['Close'].iloc[-1]
            spy_ma50 = spy_data['Close'].rolling(50).mean().iloc[-1]
            market_bullish = spy_price > spy_ma50
            
            vix = yf.Ticker('^VIX')
            vix_data = vix.history(period='5d')
            current_vix = vix_data['Close'].iloc[-1]
        except:
            market_bullish = True
            current_vix = None
        
        # Cache results with UTC timestamp
        st.session_state.scan_results = results
        st.session_state.scan_timestamp = datetime.utcnow()  # Store in UTC
        st.session_state.market_regime = market_bullish
        st.session_state.vix_level = current_vix
        
        # Save to persistent file cache
        save_cached_results(results, datetime.utcnow(), market_bullish, current_vix)
    
    # Display results (from cache or fresh scan)
    if st.session_state.scan_results is not None:
        results = st.session_state.scan_results
        market_bullish = st.session_state.market_regime
        current_vix = st.session_state.vix_level
        
        if results:
            st.success(f"✅ Found {len(results)} high-quality setups!")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Setups", len(results))
            with col2:
                bullish = sum(1 for r in results if "Bullish" in r['setup_type'])
                st.metric("Bullish", bullish)
            with col3:
                bearish = len(results) - bullish
                st.metric("Bearish", bearish)
            with col4:
                avg_score = sum(r['score'] for r in results) / len(results)
                st.metric("Avg Score", f"{avg_score:.1f}/7")
            
            # Market Regime & VIX Display
            regime_color = "🟢" if market_bullish else "🔴"
            regime_text = "BULLISH" if market_bullish else "BEARISH"
            vix_text = f"{current_vix:.1f}" if current_vix else "N/A"
            
            st.info(f"{regime_color} **Market Regime:** {regime_text} (SPY vs 50MA) | **VIX:** {vix_text}")
            
            st.markdown("---")
            
            # VIX Guide Table (moved to top)
            st.subheader("📊 VIX Volatility Guide")
            vix_data = {
                "VIX Level": ["10-15", "15-20", "20-25", "25-30", "30-40", "40+"],
                "Market State": ["Low volatility", "Normal", "Elevated", "High", "Very High", "Extreme"],
                "What It Means": [
                    "Calm market, normal position sizes",
                    "Typical market conditions",
                    "Start being cautious",
                    "⚠️ Warning appears - reduce size 50%",
                    "Major uncertainty, consider sitting out",
                    "Crisis mode (like COVID crash)"
                ]
            }
            vix_df = pd.DataFrame(vix_data)
            st.table(vix_df)
            
            st.markdown("---")
            
            # Entry Reminders Section (Yellow Box)
            st.markdown("""
            <div class="entry-reminders">
                <h4>📋 Entry Reminders</h4>
                <ul>
                    <li><b>Market regime aligned</b> <span title="Only take calls in bullish market (SPY > 50MA), puts in bearish market (SPY < 50MA)">ⓘ</span></li>
                    <li><b>DTE: >45 days</b></li>
                    <li><b>Strike: ATM - 2% OTM (max 3%)</b></li>
                    <li><b>IV < 50%</b> (or spiking with your direction)</li>
                    <li><b>No earnings in next 7 days</b> (unless intentional)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Filters
            st.subheader("🔍 Filter Results")
            col1, col2 = st.columns(2)
            
            with col1:
                setup_filter = st.multiselect(
                    "Setup Type",
                    options=list(set(r['setup_type'] for r in results)),
                    default=list(set(r['setup_type'] for r in results))
                )
            
            with col2:
                min_score_filter = st.slider("Minimum Score", 3, 7, 4)
            
            filtered_results = [
                r for r in results 
                if r['setup_type'] in setup_filter and r['score'] >= min_score_filter
            ]
            
            st.info(f"Showing {len(filtered_results)} of {len(results)} setups")
            
            # Results with confluence checklist
            for i, setup in enumerate(filtered_results[:50], 1):
                # Create unique key for each position
                position_key = f"{setup['ticker']}_{i}"
                
                with st.expander(
                    f"{'🟢' if 'Bullish' in setup['setup_type'] else '🔴'} "
                    f"**{i}. {setup['ticker']}** - ${setup['price']:.2f} "
                    f"(Score: {setup['score']}/7)",
                    expanded=(i <= 10)
                ):
                    # TradingView Chart (bigger, full width)
                    tradingview_html = f"""
                    <!-- TradingView Widget BEGIN -->
                    <div class="tradingview-widget-container" style="height:600px; width:100%;">
                      <div class="tradingview-widget-container__widget" style="height:100%; width:100%;"></div>
                      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
                      {{
                      "width": "100%",
                      "height": "600",
                      "symbol": "{setup['ticker']}",
                      "interval": "D",
                      "timezone": "America/New_York",
                      "theme": "light",
                      "style": "1",
                      "locale": "en",
                      "enable_publishing": false,
                      "hide_top_toolbar": false,
                      "hide_legend": false,
                      "allow_symbol_change": false,
                      "save_image": false,
                      "calendar": false,
                      "hide_volume": false,
                      "support_host": "https://www.tradingview.com"
                      }}
                      </script>
                    </div>
                    <!-- TradingView Widget END -->
                    """
                    st.components.v1.html(tradingview_html, height=620)
                    
                    # Main content (remove the separator, put content right below chart)
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Setup:** {setup['setup_type']}")
                        st.markdown(f"**Entry:** {setup['option_type']} ${setup['strike']}, 45-120 DTE")
                        st.markdown(f"**Target:** ${setup['target']:.2f}")
                        st.markdown(f"**Stop:** ${setup['stop']:.2f}")
                        # Determine RSI status and color
                        if setup['rsi'] < 30:
                            rsi_status = '<span style="color: green;">(Oversold)</span>'
                        elif setup['rsi'] > 70:
                            rsi_status = '<span style="color: red;">(Overbought)</span>'
                        else:
                            rsi_status = '<span style="color: black;">(Neutral)</span>'
                        
                        st.markdown(f"**RSI:** {setup['rsi']:.1f} {rsi_status} | **Volume:** {setup['volume_ratio']:.1f}x | **ATR:** ${setup['atr']:.2f}", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**Confluence Checklist:**")
                        cb = setup['confluence_breakdown']
                        st.checkbox("Near Pivot Level", value=cb['pivot'], disabled=True, key=f"{position_key}_pivot")
                        st.checkbox("Near VRVP Level", value=cb['vrvp'], disabled=True, key=f"{position_key}_vrvp")
                        st.checkbox("EMA/SMA Alignment", value=cb['ema_sma'], disabled=True, key=f"{position_key}_ema")
                        st.checkbox("VPA Confluence", value=cb['vpa'], disabled=True, key=f"{position_key}_vpa")
                        st.checkbox("Previous Week H/L", value=cb['prev_week'], disabled=True, key=f"{position_key}_prev")
                        st.checkbox("RSI Confirmation", value=cb['rsi'], disabled=True, key=f"{position_key}_rsi")
                        st.checkbox("Price Distance from 20 EMA", value=cb['price_distance'], disabled=True, key=f"{position_key}_dist")
            
            # Download CSV
            df = pd.DataFrame(filtered_results)
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"swing_setups_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No setups found meeting your criteria. Try lowering the confluence threshold.")
    
    # Info section
    with st.expander("ℹ️ How It Works"):
        st.markdown("""
        ### 7 Confluence Factors Analyzed:
        
        1. **Near Pivot Level** - Weekly R1, R2, S1, S2 (within 2%)
        2. **Near VRVP Level** - Volume profile POC, VAH, VAL (within 2%)
        3. **EMA/SMA Alignment** - Bullish/bearish trend confirmation
        4. **VPA Confluence** - Volume spike >1.5x average
        5. **Previous Week H/L** - Support/resistance (within 3%)
        6. **RSI Confirmation** - Neutral zone 30-70
        7. **Price Distance from 20 EMA** - Proximity check (<3%)
        
        **Minimum 4 factors required** for a setup to qualify.
        
        ### Setup Types:
        - **Bullish Breakout** - Price breaking above R1 with bullish trend
        - **Bearish Breakdown** - Price breaking below S1 with bearish trend
        - **Bullish Reversal** - Price bouncing at VAL support
        - **Bearish Reversal** - Price rejecting at VAH resistance
        """)


if __name__ == "__main__":
    main()
