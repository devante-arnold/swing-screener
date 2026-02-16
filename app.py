import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import requests
from typing import List, Dict
import time
import pickle
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Swing Trade Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File paths for persistence
CACHE_FILE = Path("/tmp/swing_scanner_cache.pkl")
POSITIONS_FILE = Path("/tmp/active_positions.pkl")

# Initialize session state
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
    st.session_state.scan_timestamp = None
    st.session_state.market_regime = None
    st.session_state.vix_level = None

if 'active_positions' not in st.session_state:
    st.session_state.active_positions = []

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .bullish { color: #00c853; font-weight: bold; }
    .bearish { color: #ff1744; font-weight: bold; }
    .neutral { color: #000000; font-weight: bold; }
    .entry-reminders {
        background-color: #FFFACD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .position-card {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .checkpoint-safe { color: #00c853; }
    .checkpoint-warning { color: #ff9800; }
    .checkpoint-danger { color: #ff1744; }
</style>
""", unsafe_allow_html=True)

# Persistence functions
def load_cached_results():
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'rb') as f:
                cached_data = pickle.load(f)
                cache_age = (datetime.utcnow() - cached_data['timestamp']).days
                if cache_age < 7:
                    return cached_data
        except:
            pass
    return None

def save_cached_results(results, timestamp, regime, vix):
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

def load_positions():
    if POSITIONS_FILE.exists():
        try:
            with open(POSITIONS_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    return []

def save_positions(positions):
    try:
        with open(POSITIONS_FILE, 'wb') as f:
            pickle.dump(positions, f)
    except:
        pass

# Load positions on startup
if not st.session_state.active_positions:
    st.session_state.active_positions = load_positions()

# Load cached scan results on startup
if st.session_state.scan_results is None:
    cached = load_cached_results()
    if cached:
        st.session_state.scan_results = cached['results']
        st.session_state.scan_timestamp = cached['timestamp']
        st.session_state.market_regime = cached['regime']
        st.session_state.vix_level = cached['vix']

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
            
            # RSI calculation
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            
            # RSI momentum (last 3 periods)
            rsi_prev = rsi.iloc[-4:-1].mean() if len(rsi) >= 4 else current_rsi
            if current_rsi > rsi_prev + 2:
                rsi_momentum = "Rising"
                rsi_color = "bullish"
            elif current_rsi < rsi_prev - 2:
                rsi_momentum = "Falling"
                rsi_color = "bearish"
            else:
                rsi_momentum = "Flat"
                rsi_color = "neutral"
            
            # Check for extreme RSI
            if current_rsi < 30:
                rsi_status = "**Extremely Oversold**"
                rsi_color = "bullish"
            elif current_rsi > 70:
                rsi_status = "**Extremely Overbought**"
                rsi_color = "bearish"
            else:
                rsi_status = rsi_momentum
            
            price_distance = abs(current_price - ema_20) / ema_20
            
            score = 0
            factors = []
            confluence_breakdown = {
                'pivot': {'hit': False, 'level': '', 'price': 0, 'color': ''},
                'vrvp': {'hit': False, 'level': '', 'price': 0},
                'ema_sma': {'hit': False, 'direction': '', 'color': ''},
                'prev_week': {'hit': False, 'level': '', 'distance': 0},
                'rsi': {'hit': False, 'value': current_rsi, 'status': rsi_status, 'color': rsi_color},
                'price_distance': {'hit': False, 'distance': price_distance}
            }
            
            # 1. Near pivot
            closest_pivot = None
            min_distance = float('inf')
            for level_name, level_price in pivots.items():
                distance = abs(current_price - level_price) / current_price
                if distance < min_distance:
                    min_distance = distance
                    closest_pivot = (level_name, level_price)
            
            if closest_pivot and min_distance < 0.02:
                score += 1
                level_name, level_price = closest_pivot
                
                # Determine color
                if level_name in ['R1', 'R2']:
                    pivot_color = 'bearish'  # Resistance
                elif level_name in ['S1', 'S2']:
                    pivot_color = 'bullish'  # Support
                else:
                    pivot_color = 'neutral'  # PP
                
                factors.append(f"Near {level_name}")
                confluence_breakdown['pivot'] = {
                    'hit': True,
                    'level': level_name,
                    'price': level_price,
                    'color': pivot_color
                }
            
            # 2. Near VRVP
            closest_vrvp = None
            min_vrvp_distance = float('inf')
            for vrvp_name, vrvp_price in vrvp.items():
                if vrvp_price > 0:
                    distance = abs(current_price - vrvp_price) / current_price
                    if distance < min_vrvp_distance:
                        min_vrvp_distance = distance
                        closest_vrvp = (vrvp_name, vrvp_price)
            
            if closest_vrvp and min_vrvp_distance < 0.02:
                score += 1
                vrvp_name, vrvp_price = closest_vrvp
                factors.append(f"VRVP {vrvp_name}")
                confluence_breakdown['vrvp'] = {
                    'hit': True,
                    'level': vrvp_name,
                    'price': vrvp_price
                }
            
            # 3. EMA/SMA alignment
            if current_price > ema_20 > ema_50:
                score += 1
                factors.append("Bullish EMA/SMA")
                confluence_breakdown['ema_sma'] = {
                    'hit': True,
                    'direction': 'Bullish',
                    'color': 'bullish'
                }
            elif current_price < ema_20 < ema_50:
                score += 1
                factors.append("Bearish EMA/SMA")
                confluence_breakdown['ema_sma'] = {
                    'hit': True,
                    'direction': 'Bearish',
                    'color': 'bearish'
                }
            
            # 4. Previous week H/L
            dist_to_high = abs(current_price - prev_week_high) / current_price
            dist_to_low = abs(current_price - prev_week_low) / current_price
            
            if dist_to_high < 0.03:
                score += 1
                factors.append("Prev week high")
                confluence_breakdown['prev_week'] = {
                    'hit': True,
                    'level': 'High',
                    'distance': dist_to_high * 100
                }
            elif dist_to_low < 0.03:
                score += 1
                factors.append("Prev week low")
                confluence_breakdown['prev_week'] = {
                    'hit': True,
                    'level': 'Low',
                    'distance': dist_to_low * 100
                }
            
            # 5. RSI confirmation (30-70 neutral zone)
            if 30 < current_rsi < 70:
                score += 1
                factors.append(f"RSI {current_rsi:.0f}")
                confluence_breakdown['rsi']['hit'] = True
            
            # 6. Price distance from 20 EMA
            if price_distance < 0.03:
                score += 1
                factors.append("Near 20 EMA")
                confluence_breakdown['price_distance'] = {
                    'hit': True,
                    'distance': price_distance * 100
                }
            
            # Determine setup type
            setup_type = "Range-bound"
            if current_price > pivots['R1'] and current_price > ema_20:
                setup_type = "Bullish breakout"
            elif current_price < pivots['S1'] and current_price < ema_20:
                setup_type = "Bearish breakdown"
            elif current_price > ema_20 and abs(current_price - vrvp['VAL']) / current_price < 0.02:
                setup_type = "Bullish reversal"
            elif current_price < ema_20 and abs(current_price - vrvp['VAH']) / current_price < 0.02:
                setup_type = "Bearish reversal"
            
            # Calculate targets and stops
            if "Bullish" in setup_type:
                target_r1 = pivots['R1'] if current_price < pivots['R1'] else pivots['R2']
                target_r2 = pivots['R2']
                stop = current_price - (1.5 * atr)
                option_type = "CALL"
                
                # Strike selection based on score
                if score == 6:
                    strike = round(current_price)  # ATM
                elif score == 5:
                    strike = round(current_price * 1.01)  # 1% OTM
                else:
                    strike = round(current_price * 1.02)  # 2% OTM
            else:
                target_r1 = pivots['S1'] if current_price > pivots['S1'] else pivots['S2']
                target_r2 = pivots['S2']
                stop = current_price + (1.5 * atr)
                option_type = "PUT"
                
                # Strike selection based on score
                if score == 6:
                    strike = round(current_price)  # ATM
                elif score == 5:
                    strike = round(current_price * 0.99)  # 1% OTM
                else:
                    strike = round(current_price * 0.98)  # 2% OTM
            
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
                'target_r1': target_r1,
                'target_r2': target_r2,
                'stop': stop,
                'atr': atr,
                'rsi': current_rsi,
                'rsi_status': rsi_status,
                'rsi_color': rsi_color,
                'confluence_breakdown': confluence_breakdown,
                'ema_20': ema_20,
                'ema_50': ema_50
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


def calculate_time_checkpoints(dte_at_entry, setup_type):
    """Calculate time-based exit checkpoints"""
    if "breakout" in setup_type.lower() or "breakdown" in setup_type.lower():
        quick_exit = 10
        max_hold = 15
    else:  # Reversal
        quick_exit = 10
        max_hold = 14
    
    theta_warning = 25
    
    return {
        'quick_exit': quick_exit,
        'max_hold': max_hold,
        'theta_warning': theta_warning
    }


def get_current_stock_price(ticker):
    """Fetch current stock price"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d')
        if not data.empty:
            return data['Close'].iloc[-1]
    except:
        pass
    return None


def render_position_card(pos, index):
    """Render simple position card in sidebar - just ticker and expiration"""
    days_held = (datetime.now() - datetime.fromisoformat(pos['entry_date'])).days
    dte_remaining = pos['dte_at_entry'] - days_held
    
    # Calculate expiration date
    entry_dt = datetime.fromisoformat(pos['entry_date'])
    expiration_date = entry_dt + timedelta(days=pos['dte_at_entry'])
    exp_str = expiration_date.strftime('%m/%d/%y')
    
    # Simple display - just ticker and expiration
    emoji = "🟢" if "Bullish" in pos['setup_type'] else "🔴"
    
    if st.button(f"{emoji} {pos['ticker']} - Exp {exp_str}", key=f"pos_btn_{index}", use_container_width=True):
        st.session_state[f'show_details_{index}'] = True
        st.rerun()


def render_position_details_sidebar(pos, index):
    """Render detailed position view in sidebar"""
    
    days_held = (datetime.now() - datetime.fromisoformat(pos['entry_date'])).days
    dte_remaining = pos['dte_at_entry'] - days_held
    checkpoints = calculate_time_checkpoints(pos['dte_at_entry'], pos['setup_type'])
    
    # Fetch current price
    current_price = get_current_stock_price(pos['ticker'])
    if not current_price:
        current_price = pos['entry_price']
        st.sidebar.warning("⚠️ Using entry price")
    
    # Calculate moneyness and changes
    if pos['option_type'] == 'CALL':
        moneyness_pct = ((pos['strike'] - current_price) / pos['strike']) * 100
    else:
        moneyness_pct = ((current_price - pos['strike']) / pos['strike']) * 100
    
    stock_change = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
    dist_to_r1 = ((pos['target_r1'] - current_price) / current_price) * 100
    dist_to_r2 = ((pos['target_r2'] - current_price) / current_price) * 100
    dist_to_stop = ((pos['stop'] - current_price) / current_price) * 100
    
    # Header
    st.sidebar.markdown(f"### 📊 {pos['ticker']} ${pos['strike']}{pos['option_type'][0]}")
    
    # Current status
    st.sidebar.markdown("**Current Status:**")
    st.sidebar.markdown(f"Price: **${current_price:.2f}** ({stock_change:+.1f}%)")
    st.sidebar.markdown(f"Days: **{days_held}** / DTE: **{dte_remaining}d**")
    st.sidebar.markdown(f"Moneyness: **{abs(moneyness_pct):.1f}% {'OTM' if moneyness_pct > 0 else 'ITM'}**")
    
    # Status indicator
    if days_held < checkpoints['quick_exit']:
        st.sidebar.success(f"🟢 Safe Zone (Day {days_held})")
    elif days_held < checkpoints['max_hold']:
        st.sidebar.warning(f"⚠️ Decision Point (Day {days_held})")
    else:
        st.sidebar.error(f"🔴 MAX HOLD! (Day {days_held})")
    
    st.sidebar.markdown("---")
    
    # Targets
    st.sidebar.markdown("**Targets & Stop:**")
    st.sidebar.markdown(f"R1: **${pos['target_r1']:.2f}** ({abs(dist_to_r1):.1f}% away)")
    st.sidebar.markdown(f"R2: **${pos['target_r2']:.2f}** ({abs(dist_to_r2):.1f}% away)")
    st.sidebar.markdown(f"Stop: **${pos['stop']:.2f}** ({abs(dist_to_stop):.1f}% away)")
    
    st.sidebar.markdown("---")
    
    # Time checkpoints
    st.sidebar.markdown("**⏰ Time Checkpoints:**")
    if days_held < checkpoints['quick_exit']:
        st.sidebar.markdown(f"✅ Day {days_held} - Safe")
        st.sidebar.markdown(f"⚠️ Day {checkpoints['quick_exit']} - Decision")
        st.sidebar.markdown(f"🔴 Day {checkpoints['max_hold']} - Max hold")
    elif days_held < checkpoints['max_hold']:
        st.sidebar.markdown(f"⚠️ Day {days_held} - **DECISION ZONE**")
        st.sidebar.markdown(f"🔴 Day {checkpoints['max_hold']} - Max hold")
    else:
        st.sidebar.markdown(f"🔴 Day {days_held} - **EXIT NOW**")
    st.sidebar.markdown(f"💀 Day {checkpoints['theta_warning']} - Never hold")
    
    st.sidebar.markdown("---")
    
    # Scaling plan
    st.sidebar.markdown("**📈 Scaling Plan:**")
    st.sidebar.markdown(f"R1 hit → Exit **75%** ({pos['contracts'] * 0.75:.0f} contracts)")
    st.sidebar.markdown(f"R2 hit → Exit **25%** ({pos['contracts'] * 0.25:.0f} contracts)")
    st.sidebar.markdown(f"Stop hit → Exit **100%**")
    
    st.sidebar.markdown("---")
    
    # Exit checklist
    st.sidebar.markdown("**🚨 Exit Checklist:**")
    st.sidebar.checkbox("R1 hit → Exit 75%", key=f"r1_{index}")
    st.sidebar.checkbox("Up 100% → Profit take", key=f"profit_{index}")
    st.sidebar.checkbox("Stop hit → Exit 100%", key=f"stop_{index}")
    st.sidebar.checkbox(">10% OTM day 14 → Exit", key=f"otm_{index}")
    st.sidebar.checkbox("Down 50% → Exit 100%", key=f"loss_{index}")
    
    st.sidebar.markdown("---")
    
    # Actions
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("✖️ Close", key=f"close_{index}", use_container_width=True):
            st.session_state[f'show_details_{index}'] = False
            st.rerun()
    with col2:
        if st.button("🗑️ Remove", key=f"remove_{index}", use_container_width=True):
            st.session_state.active_positions.pop(index)
            save_positions(st.session_state.active_positions)
            st.session_state[f'show_details_{index}'] = False
            st.rerun()


def render_position_details(pos, index):
    """Render detailed position view - full screen"""
    
    # Header with back and remove buttons
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown(f"## 📊 {pos['ticker']} ${pos['strike']} {pos['option_type']}")
    with col2:
        if st.button("❌ Remove Position", key=f"remove_{index}", type="secondary"):
            st.session_state.active_positions.pop(index)
            save_positions(st.session_state.active_positions)
            st.session_state[f'show_details_{index}'] = False
            st.success(f"Removed {pos['ticker']}")
            time.sleep(1)
            st.rerun()
    with col3:
        if st.button("⬅️ Back", key=f"back_{index}", type="primary"):
            st.session_state[f'show_details_{index}'] = False
            st.rerun()
    
    st.markdown("---")
    
    days_held = (datetime.now() - datetime.fromisoformat(pos['entry_date'])).days
    dte_remaining = pos['dte_at_entry'] - days_held
    checkpoints = calculate_time_checkpoints(pos['dte_at_entry'], pos['setup_type'])
    
    # Fetch current price
    current_price = get_current_stock_price(pos['ticker'])
    if not current_price:
        current_price = pos['entry_price']
        st.warning("⚠️ Unable to fetch current price. Using entry price.")
    
    # Calculate moneyness
    if pos['option_type'] == 'CALL':
        moneyness_pct = ((pos['strike'] - current_price) / pos['strike']) * 100
    else:
        moneyness_pct = ((current_price - pos['strike']) / pos['strike']) * 100
    
    stock_change = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
    
    # Key metrics at top
    st.markdown("### 📈 Current Status")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Stock Price", f"${current_price:.2f}", f"{stock_change:+.1f}%")
    with col2:
        st.metric("Days Held", f"{days_held} days", f"{dte_remaining}d DTE left")
    with col3:
        moneyness_label = "OTM" if moneyness_pct > 0 else "ITM"
        st.metric("Moneyness", f"{abs(moneyness_pct):.1f}% {moneyness_label}")
    with col4:
        if days_held < checkpoints['quick_exit']:
            status = "🟢 On Track"
        elif days_held < checkpoints['max_hold']:
            status = "⚠️ Decision Zone"
        else:
            status = "🔴 Max Hold!"
        st.metric("Status", status)
    
    st.markdown("---")
    
    # Position info
    st.markdown("### 📋 Position Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Entry Date:** {pos['entry_date'][:10]}")
        st.markdown(f"**Entry Price:** ${pos['entry_price']:.2f}")
        st.markdown(f"**Strike:** ${pos['strike']}")
        st.markdown(f"**Contracts:** {pos['contracts']}")
    with col2:
        st.markdown(f"**Setup Type:** {pos['setup_type']}")
        st.markdown(f"**DTE at Entry:** {pos['dte_at_entry']} days")
        st.markdown(f"**Target R1:** ${pos['target_r1']:.2f}")
        st.markdown(f"**Target R2:** ${pos['target_r2']:.2f}")
    with col3:
        dist_to_r1 = ((pos['target_r1'] - current_price) / current_price) * 100
        dist_to_r2 = ((pos['target_r2'] - current_price) / current_price) * 100
        dist_to_stop = ((pos['stop'] - current_price) / current_price) * 100
        
        st.markdown(f"**To R1:** {abs(dist_to_r1):.1f}% away")
        st.markdown(f"**To R2:** {abs(dist_to_r2):.1f}% away")
        st.markdown(f"**To Stop:** {abs(dist_to_stop):.1f}% away")
        st.markdown(f"**Stop Loss:** ${pos['stop']:.2f}")
    
    st.markdown("---")
    
    # Time exit strategy
    st.markdown("### ⏰ Time Exit Strategy")
    st.markdown(f"**Setup Type:** {pos['setup_type']}")
    st.markdown(f"**Based on {pos['dte_at_entry']} DTE entry:**")
    
    st.markdown("")
    
    # Checkpoint display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if days_held < checkpoints['quick_exit']:
            st.markdown(f"<div style='padding:1rem; background:#d4edda; border-left:4px solid #28a745; border-radius:4px;'><b>✅ Day {days_held} (Today)</b><br>Safe Zone - Continue monitoring</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='padding:1rem; background:#f8f9fa; border-left:4px solid #6c757d; border-radius:4px;'><b>Day 0-{checkpoints['quick_exit']}</b><br>Safe Zone (passed)</div>", unsafe_allow_html=True)
    
    with col2:
        if days_held >= checkpoints['quick_exit'] and days_held < checkpoints['max_hold']:
            st.markdown(f"<div style='padding:1rem; background:#fff3cd; border-left:4px solid #ffc107; border-radius:4px;'><b>⚠️ Day {days_held} (Today)</b><br>Decision Point - Exit if no R1 progress</div>", unsafe_allow_html=True)
        elif days_held < checkpoints['quick_exit']:
            st.markdown(f"<div style='padding:1rem; background:#f8f9fa; border-left:4px solid #ffc107; border-radius:4px;'><b>Day {checkpoints['quick_exit']}</b><br>Decision Point (upcoming)</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='padding:1rem; background:#f8f9fa; border-left:4px solid #6c757d; border-radius:4px;'><b>Day {checkpoints['quick_exit']}</b><br>Decision Point (passed)</div>", unsafe_allow_html=True)
    
    with col3:
        if days_held >= checkpoints['max_hold']:
            st.markdown(f"<div style='padding:1rem; background:#f8d7da; border-left:4px solid #dc3545; border-radius:4px;'><b>🔴 Day {days_held} (Today)</b><br>MAXIMUM HOLD - EXIT NOW!</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='padding:1rem; background:#f8f9fa; border-left:4px solid #dc3545; border-radius:4px;'><b>Day {checkpoints['max_hold']}</b><br>Maximum Hold - Exit by this day</div>", unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown(f"<div style='padding:1rem; background:#343a40; color:white; border-left:4px solid #000; border-radius:4px;'><b>💀 Day {checkpoints['theta_warning']}</b><br>Theta Death Zone - Never hold this long</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Scaling plan
    st.markdown("### 📈 Scaling Exit Plan")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **If R1 (${pos['target_r1']:.2f}) hit:**
        - ✅ Sell 75% ({pos['contracts'] * 0.75:.1f} contracts)
        - ✅ Lock in majority of profit
        - ✅ Let 25% run to R2
        """)
    with col2:
        st.markdown(f"""
        **If R2 (${pos['target_r2']:.2f}) hit:**
        - ✅ Sell remaining 25%
        - ✅ Trade complete
        
        **If Stop (${pos['stop']:.2f}) hit:**
        - ❌ Exit 100% immediately
        - ❌ Move on to next setup
        """)
    
    st.markdown("---")
    
    # Exit checklist
    st.markdown("### 🚨 Exit Checklist")
    st.markdown("*Check these daily to know when to exit:*")
    
    col1, col2 = st.columns(2)
    with col1:
        st.checkbox(f"☐ R1 hit → Exit 75% of position", key=f"r1_{index}")
        st.checkbox(f"☐ P/L: Up 100% → Consider profit take", key=f"profit_{index}")
        st.checkbox(f"☐ Stop hit (${pos['stop']:.2f}) → Exit 100%", key=f"stop_{index}")
    with col2:
        st.checkbox(f"☐ Moneyness: >10% OTM at day 14 → Exit 100%", key=f"otm_{index}")
        st.checkbox(f"☐ P/L: Down 50% → Exit 100%", key=f"loss_{index}")
    
    st.markdown("---")
    
    # Adjustment rules
    st.markdown("### 🔧 Adjustment Rules")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **⚠️ Stock moved but option didn't:**
        - If stock up 2%+ but option still down
        - → Exit immediately (IV crush/theta)
        
        **✅ Early profit (before targets):**
        - If up 50%+ before day 7
        - → Take it (don't be greedy)
        """)
    with col2:
        st.markdown("""
        **🔄 Stopped on gap but setup valid:**
        - If gapped through stop
        - But setup still intact
        - → Can re-enter when stabilizes
        
        **📊 Stock at target but option flat:**
        - If stock reaches R1
        - But option hasn't gained
        - → Exit anyway (something wrong)
        """)
    
    st.markdown("---")
    
    # Bottom buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Positions List", key=f"back_bottom_{index}", use_container_width=True, type="primary"):
            st.session_state[f'show_details_{index}'] = False
            st.rerun()
    with col2:
        if st.button("❌ Remove This Position", key=f"remove_bottom_{index}", use_container_width=True, type="secondary"):
            st.session_state.active_positions.pop(index)
            save_positions(st.session_state.active_positions)
            st.session_state[f'show_details_{index}'] = False
            st.success(f"Removed {pos['ticker']}")
            time.sleep(1)
            st.rerun()


def main():
    st.markdown('<p class="main-header">📊 Swing Trade Screener</p>', unsafe_allow_html=True)
    st.markdown("**AI-powered confluence scanner with position tracker**")
    
    # Disclaimer
    st.warning("⚠️ **This is not financial advice. This is only the display of my personal trading tools.**")
    
    # Market status notice (using Eastern Time)
    eastern = timezone(timedelta(hours=-5))
    current_eastern = datetime.now(eastern)
    current_day = current_eastern.weekday()
    current_hour = current_eastern.hour
    
    is_weekend = current_day >= 5
    is_before_open = current_hour < 9 or (current_hour == 9 and current_eastern.minute < 30)
    is_after_close = current_hour >= 16
    
    if is_weekend:
        st.info("🗓️ **Weekend Mode:** Markets are closed. Scanning is disabled. Showing cached results from the last scan.")
    elif is_before_open:
        st.info(f"⏰ **Pre-Market:** Markets open at 9:30 AM ET. Currently {current_eastern.strftime('%I:%M %p')} ET. Showing cached results.")
    elif is_after_close:
        st.info("🌙 **After Hours:** Markets closed at 4:00 PM ET. Showing cached results. Scan available tomorrow at 9:30 AM ET.")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    confluence_threshold = st.sidebar.slider(
        "Minimum Confluence Score",
        min_value=3,
        max_value=6,
        value=4,
        help="Minimum number of confluence factors required (now 6 total)"
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
    
    # Active Positions Section
    st.sidebar.markdown("---")
    
    # Initialize folder state
    if 'positions_folder_open' not in st.session_state:
        st.session_state.positions_folder_open = True
    
    # Folder header with toggle
    col1, col2 = st.sidebar.columns([4, 1])
    with col1:
        st.sidebar.subheader(f"📂 Active Positions ({len(st.session_state.active_positions)}/5)")
    with col2:
        # Toggle button
        if st.session_state.positions_folder_open:
            if st.button("▼", key="close_folder", help="Close folder"):
                st.session_state.positions_folder_open = False
                # Close any open position details
                for i in range(len(st.session_state.active_positions)):
                    st.session_state[f'show_details_{i}'] = False
                st.rerun()
        else:
            if st.button("▶", key="open_folder", help="Open folder"):
                st.session_state.positions_folder_open = True
                st.rerun()
    
    # Check if showing details for any position
    showing_details = None
    for i in range(len(st.session_state.active_positions)):
        if st.session_state.get(f'show_details_{i}', False):
            showing_details = i
            break
    
    # Only show contents if folder is open
    if st.session_state.positions_folder_open:
        if len(st.session_state.active_positions) == 0:
            st.sidebar.info("No active positions. Add setups from scan results below.")
        else:
            # Show position list
            for i, pos in enumerate(st.session_state.active_positions):
                render_position_card(pos, i)
            
            # If a position is selected, show it with indent/indicator
            if showing_details is not None:
                st.sidebar.markdown("---")
                # Show selected position with return indicator
                pos = st.session_state.active_positions[showing_details]
                entry_dt = datetime.fromisoformat(pos['entry_date'])
                expiration_date = entry_dt + timedelta(days=pos['dte_at_entry'])
                exp_str = expiration_date.strftime('%m/%d/%y')
                emoji = "🟢" if "Bullish" in pos['setup_type'] else "🔴"
                
                st.sidebar.markdown(f"**↳ {emoji} {pos['ticker']} - Exp {exp_str}**")
                st.sidebar.markdown("---")
                render_position_details_sidebar(pos, showing_details)
    else:
        # Folder closed - show nothing
        st.sidebar.caption("Click ▶ to open positions folder")
    
    # Main area - Scan button
    col1, col2, col3 = st.columns([2, 2, 1])
    
    markets_closed = is_weekend or is_before_open or is_after_close
    
    with col1:
        if markets_closed:
            st.button("🚀 Run Market Scan", type="primary", use_container_width=True, disabled=True)
            if is_weekend:
                st.caption("⏸️ Markets closed (Weekend). Scan disabled until Monday 9:30 AM ET.")
            elif is_before_open:
                st.caption(f"⏸️ Markets closed. Opens at 9:30 AM ET (currently {current_eastern.strftime('%I:%M %p')} ET).")
            else:
                st.caption("⏸️ Markets closed. Scan available tomorrow at 9:30 AM ET.")
        else:
            scan_button = st.button("🚀 Run Market Scan", type="primary", use_container_width=True)
    
    with col2:
        if st.session_state.scan_timestamp:
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
                    const localDate = utcDate.toLocaleDateString('en-US', {{
                        month: 'short',
                        day: 'numeric'
                    }});
                    document.getElementById('local-time').textContent = localDate + ' ' + localTime;
                </script>
            """, height=30)
    
    # Run scan
    if not markets_closed and 'scan_button' in locals() and scan_button:
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
        
        # Cache results
        st.session_state.scan_results = results
        st.session_state.scan_timestamp = datetime.utcnow()
        st.session_state.market_regime = market_bullish
        st.session_state.vix_level = current_vix
        
        save_cached_results(results, datetime.utcnow(), market_bullish, current_vix)
    
    # Display results
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
                st.metric("Avg Score", f"{avg_score:.1f}/6")
            
            # Market Regime & VIX
            regime_color = "🟢" if market_bullish else "🔴"
            regime_text = "BULLISH" if market_bullish else "BEARISH"
            vix_text = f"{current_vix:.1f}" if current_vix else "N/A"
            
            st.info(f"{regime_color} **Market Regime:** {regime_text} (SPY vs 50MA) | **VIX:** {vix_text}")
            
            st.markdown("---")
            
            # VIX Guide
            st.subheader("📊 VIX Volatility Guide")
            vix_data = {
                "VIX Level": ["10-15", "15-20", "20-25", "25-30", "30-40", "40+"],
                "Market State": ["Low volatility", "Normal", "Elevated", "High", "Very High", "Extreme"],
                "What It Means": [
                    "Calm market, normal position sizes",
                    "Typical market conditions",
                    "Start being cautious",
                    "⚠️ Warning - reduce size 50%",
                    "Major uncertainty, consider sitting out",
                    "Crisis mode"
                ]
            }
            vix_df = pd.DataFrame(vix_data)
            st.table(vix_df)
            
            st.markdown("---")
            
            # Entry Reminders
            st.markdown("""
            <div class="entry-reminders">
                <h4>📋 Entry Reminders</h4>
                <ul>
                    <li><b>Market regime aligned</b> <span title="Only take calls in bullish market (SPY > 50MA), puts in bearish market (SPY < 50MA)">ⓘ</span></li>
                    <li><b>DTE: 45-120 days</b></li>
                    <li><b>Strike: ATM (6/6), 1% OTM (5/6), 2% OTM (4/6)</b></li>
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
                min_score_filter = st.slider("Minimum Score", 3, 6, 4)
            
            filtered_results = [
                r for r in results 
                if r['setup_type'] in setup_filter and r['score'] >= min_score_filter
            ]
            
            st.info(f"Showing {len(filtered_results)} of {len(results)} setups")
            
            # Results - ALL TABS CLOSED by default
            for i, setup in enumerate(filtered_results[:50], 1):
                position_key = f"{setup['ticker']}_{i}"
                
                with st.expander(
                    f"{'🟢' if 'Bullish' in setup['setup_type'] else '🔴'} "
                    f"**{i}. {setup['ticker']}** - ${setup['price']:.2f} "
                    f"(Score: {setup['score']}/6)",
                    expanded=False  # ALL CLOSED
                ):
                    # TradingView Chart
                    tradingview_html = f"""
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
                    """
                    st.components.v1.html(tradingview_html, height=620)
                    
                    # Main content
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Setup:** {setup['setup_type']}")
                        
                        # Strike display based on score
                        if setup['score'] == 6:
                            strike_text = f"${setup['strike']} ATM"
                        elif setup['score'] == 5:
                            strike_text = f"${setup['strike']} (1% OTM)"
                        else:
                            strike_text = f"${setup['strike']} (2% OTM)"
                        
                        st.markdown(f"**Entry:** {setup['option_type']} {strike_text}, 45-120 DTE")
                        st.markdown(f"**Target R1:** ${setup['target_r1']:.2f} (75% exit)")
                        st.markdown(f"**Target R2:** ${setup['target_r2']:.2f} (25% exit)")
                        st.markdown(f"**Stop:** ${setup['stop']:.2f}")
                        
                        # RSI with color and bold
                        rsi_class = setup['rsi_color']
                        st.markdown(f"**RSI:** {setup['rsi']:.1f} <span class='{rsi_class}'>{setup['rsi_status']}</span> | **ATR:** ${setup['atr']:.2f}", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**Confluence Checklist:**")
                        cb = setup['confluence_breakdown']
                        
                        # Near Pivot Level (with color)
                        if cb['pivot']['hit']:
                            pivot_class = cb['pivot']['color']
                            st.markdown(f"☑ <span class='{pivot_class}'>Near Pivot: {cb['pivot']['level']} ${cb['pivot']['price']:.2f}</span>", unsafe_allow_html=True)
                        else:
                            st.markdown("☐ Near Pivot Level")
                        
                        # Near VRVP Level (with level name)
                        if cb['vrvp']['hit']:
                            st.markdown(f"☑ Near VRVP: {cb['vrvp']['level']} ${cb['vrvp']['price']:.2f}")
                        else:
                            st.markdown("☐ Near VRVP Level")
                        
                        # EMA/SMA Alignment (with color)
                        if cb['ema_sma']['hit']:
                            ema_class = cb['ema_sma']['color']
                            st.markdown(f"☑ <span class='{ema_class}'>EMA/SMA: 20/50 ({cb['ema_sma']['direction']})</span>", unsafe_allow_html=True)
                        else:
                            st.markdown("☐ EMA/SMA Alignment")
                        
                        # Previous Week H/L (with distance)
                        if cb['prev_week']['hit']:
                            st.markdown(f"☑ Prev Week {cb['prev_week']['level']}: {cb['prev_week']['distance']:.1f}% away")
                        else:
                            st.markdown("☐ Previous Week H/L")
                        
                        # RSI Confirmation
                        if cb['rsi']['hit']:
                            st.markdown(f"☑ RSI Confirmation: {cb['rsi']['value']:.1f}")
                        else:
                            st.markdown("☐ RSI Confirmation")
                        
                        # Price Distance from 20 EMA
                        if cb['price_distance']['hit']:
                            st.markdown(f"☑ Price from 20 EMA: {cb['price_distance']['distance']:.1f}% away")
                        else:
                            st.markdown("☐ Price Distance from 20 EMA")
                    
                    # Add to Positions button
                    st.markdown("---")
                    
                    # Check if this position is being added
                    add_key = f"adding_{position_key}"
                    if add_key not in st.session_state:
                        st.session_state[add_key] = False
                    
                    if not st.session_state[add_key]:
                        if st.button(f"➕ Add to Positions", key=f"add_{position_key}", type="primary"):
                            if len(st.session_state.active_positions) >= 5:
                                st.error("⚠️ Maximum 5 positions reached. Remove a position before adding new ones.")
                            else:
                                st.session_state[add_key] = True
                                st.rerun()
                    else:
                        # Show add position form
                        st.markdown(f"### ➕ Add {setup['ticker']} to Active Positions")
                        
                        with st.form(key=f"form_{position_key}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                entry_date = st.date_input("Entry Date", value=datetime.now())
                                dte_input = st.number_input("DTE at Entry", min_value=30, max_value=120, value=60, step=5)
                            with col2:
                                contracts_input = st.number_input("Number of Contracts", min_value=1, max_value=10, value=1)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                cancel_btn = st.form_submit_button("Cancel")
                            with col2:
                                submit_btn = st.form_submit_button("Add Position", type="primary")
                            
                            if cancel_btn:
                                st.session_state[add_key] = False
                                st.rerun()
                            
                            if submit_btn:
                                new_position = {
                                    'ticker': setup['ticker'],
                                    'strike': setup['strike'],
                                    'option_type': setup['option_type'],
                                    'entry_date': entry_date.isoformat(),
                                    'entry_price': setup['price'],
                                    'dte_at_entry': dte_input,
                                    'contracts': contracts_input,
                                    'setup_type': setup['setup_type'],
                                    'target_r1': setup['target_r1'],
                                    'target_r2': setup['target_r2'],
                                    'stop': setup['stop']
                                }
                                st.session_state.active_positions.append(new_position)
                                save_positions(st.session_state.active_positions)
                                st.session_state[add_key] = False
                                st.success(f"✅ Added {setup['ticker']} to Active Positions!")
                                time.sleep(1)
                                st.rerun()
            
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
        ### 6 Confluence Factors Analyzed:
        
        1. **Near Pivot Level** - Weekly R1, R2, S1, S2, PP (within 2%)
        2. **Near VRVP Level** - Volume profile POC, VAH, VAL (within 2%)
        3. **EMA/SMA Alignment** - 20/50 EMAs bullish/bearish trend
        4. **Previous Week H/L** - Support/resistance (within 3%)
        5. **RSI Confirmation** - Momentum direction + extreme levels
        6. **Price Distance from 20 EMA** - Proximity check (<3%)
        
        **VPA Confluence removed** - Requires human discretion
        
        **Minimum 4 factors required** for a setup to qualify.
        
        ### Strike Selection (NEW):
        - **6/6 score:** ATM strike (best probability)
        - **5/6 score:** 1% OTM strike
        - **4/6 score:** 2% OTM strike
        
        ### Setup Types:
        - **Bullish Breakout** - Price breaking above R1
        - **Bearish Breakdown** - Price breaking below S1
        - **Bullish Reversal** - Price bouncing at VAL
        - **Bearish Reversal** - Price rejecting at VAH
        """)


if __name__ == "__main__":
    main()
