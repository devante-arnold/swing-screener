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

# Helper function for neon glow icons
def get_neon_icon(option_type, size="1.1em"):
    """Get neon glow bull or bear emoji"""
    if option_type == 'CALL':
        return f'<span style="color: #00ff00; text-shadow: 0 0 8px #00ff00, 0 0 12px #00ff00; font-size: {size};">🐂</span>'
    else:
        return f'<span style="color: #ff0000; text-shadow: 0 0 8px #ff0000, 0 0 12px #ff0000; font-size: {size};">🐻</span>'

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

if 'manual_search_result' not in st.session_state:
    st.session_state.manual_search_result = None

if 'manual_search_ticker' not in st.session_state:
    st.session_state.manual_search_ticker = None

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
        """
        Get comprehensive ticker list from S&P 500, NASDAQ-100, and active traders
        Total: ~700+ stocks
        """
        try:
            # Get S&P 500 tickers from Wikipedia (most reliable free source)
            import pandas as pd
            sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            sp500_table = pd.read_html(sp500_url)[0]
            sp500_tickers = sp500_table['Symbol'].tolist()
            
            # Clean tickers (remove dots that Wikipedia sometimes adds)
            sp500_tickers = [t.replace('.', '-') for t in sp500_tickers]
            
        except:
            # Fallback to major S&P 500 stocks if Wikipedia fails
            sp500_tickers = [
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
        
        try:
            # Get NASDAQ-100 tickers
            nasdaq100_url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
            nasdaq100_table = pd.read_html(nasdaq100_url)[4]  # Table 4 contains the tickers
            nasdaq100_tickers = nasdaq100_table['Ticker'].tolist()
            nasdaq100_tickers = [t.replace('.', '-') for t in nasdaq100_tickers]
        except:
            # Fallback NASDAQ major stocks
            nasdaq100_tickers = [
                'INTU', 'BKNG', 'ADP', 'VRTX', 'SBUX', 'GILD', 'ADI', 'REGN', 'LRCX', 'MDLZ',
                'PANW', 'MU', 'PYPL', 'KLAC', 'SNPS', 'CDNS', 'MELI', 'ASML', 'ABNB', 'CHTR',
                'CTAS', 'MAR', 'ORLY', 'AZN', 'CRWD', 'FTNT', 'CSX', 'NXPI', 'PCAR', 'MRVL',
                'MNST', 'WDAY', 'ADSK', 'DASH', 'CPRT', 'PAYX', 'ROST', 'ODFL', 'KDP', 'FAST',
                'VRSK', 'CTSH', 'EA', 'GEHC', 'LULU', 'DDOG', 'IDXX', 'XEL', 'EXC', 'ON',
                'TEAM', 'ANSS', 'CSGP', 'ZS', 'DXCM', 'TTWO', 'BIIB', 'ILMN', 'WBD', 'MDB', 'ZM', 'MRNA'
            ]
        
        # Additional active traders and popular options stocks
        active_traders = [
            'SPY', 'QQQ', 'IWM', 'SMH', 'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP',
            'SOFI', 'PLTR', 'NIO', 'RIVN', 'COIN', 'RBLX', 'SNOW', 'NET',
            'SHOP', 'SQ', 'UBER', 'LYFT', 'PINS', 'SNAP', 'ROKU', 'ARKK', 'GME', 'AMC', 'PLUG',
            'HOOD', 'LCID', 'UPST', 'AFRM', 'CVNA', 'W', 'BYND', 'SPCE', 'DKNG', 'TLRY'
        ]
        
        # Combine and deduplicate
        all_tickers = list(set(sp500_tickers + nasdaq100_tickers + active_traders))
        
        # Remove any problematic tickers
        exclude = ['BRK.B', 'BF.B']  # Tickers with dots that cause issues
        all_tickers = [t for t in all_tickers if t not in exclude]
        
        return all_tickers
    
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
            
            # POC = Point of Control (price with highest volume)
            poc_price = max(volume_profile, key=lambda x: x[1])[0] if volume_profile else recent['Close'].iloc[-1]
            
            # Calculate Value Area (70% of volume around POC)
            total_vol = sum(v for _, v in volume_profile)
            sorted_profile = sorted(volume_profile, key=lambda x: x[1], reverse=True)
            
            # Find prices that account for 70% of volume
            value_area_vol = 0
            value_area_prices = []
            target_vol = total_vol * 0.70
            
            for price, vol in sorted_profile:
                if value_area_vol < target_vol:
                    value_area_prices.append(price)
                    value_area_vol += vol
                else:
                    break
            
            # VAL = lowest price in value area, VAH = highest
            if value_area_prices:
                val = min(value_area_prices)
                vah = max(value_area_prices)
            else:
                val = price_min
                vah = price_max
            
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
            
            # ─────────────────────────────────────────────────
            # STEP 1: DETERMINE DIRECTIONAL BIAS FIRST
            # Needed before pivot scoring so we score directionally
            # ─────────────────────────────────────────────────

            # Distances to key levels (signed: positive = level above price)
            dist_to_r1  = (pivots['R1'] - current_price) / current_price
            dist_to_r2  = (pivots['R2'] - current_price) / current_price
            dist_to_s1  = (current_price - pivots['S1']) / current_price
            dist_to_s2  = (current_price - pivots['S2']) / current_price
            dist_to_val = (current_price - vrvp['VAL']) / current_price
            dist_to_vah = (vrvp['VAH'] - current_price) / current_price
            dist_to_pwh = (prev_week_high - current_price) / current_price
            dist_to_pwl = (current_price - prev_week_low) / current_price

            # Determine broad direction from EMA alignment and PP position
            price_above_pp  = current_price > pivots['PP']
            ema_bullish     = current_price > ema_20 > ema_50
            ema_bearish     = current_price < ema_20 < ema_50

            is_bullish_bias = price_above_pp and (ema_bullish or current_price > ema_20)
            is_bearish_bias = not price_above_pp and (ema_bearish or current_price < ema_20)

            # ─────────────────────────────────────────────────
            # STEP 2: SETUP TYPE DETECTION
            # Detect price APPROACHING a level, not already past it
            # ─────────────────────────────────────────────────

            setup_type = None

            # Bullish breakout: approaching R1 from below (within 5%)
            # dist_to_r1 is positive when R1 is above current price
            if 0 < dist_to_r1 < 0.05 and current_price > ema_20:
                setup_type = "Bullish breakout"

            # Bullish reversal: bouncing off support (S1, VAL, or prev week low)
            # dist_to_s1 is positive when current price is above S1
            elif (0 < dist_to_s1 < 0.03 or
                  0 < dist_to_val < 0.03 or
                  0 < dist_to_pwl < 0.03) and current_price > ema_50:
                setup_type = "Bullish reversal"

            # Bearish breakdown: approaching S1 from above (within 5%)
            # When price approaching S1 from above, dist_to_s1 is small positive
            elif 0 < dist_to_s1 < 0.05 and current_price < ema_20:
                setup_type = "Bearish breakdown"

            # Bearish reversal: rejecting resistance (R1, VAH, or prev week high)
            # dist_to_r1 is positive when R1 is above, dist_to_vah positive when VAH above
            elif (0 < dist_to_r1 < 0.03 or
                  0 < dist_to_vah < 0.03 or
                  0 < dist_to_pwh < 0.03) and current_price < ema_50:
                setup_type = "Bearish reversal"
            
            # ADDITIONAL: Catch more setups that meet EMA/PP criteria even if not at specific levels
            # This prevents missing good setups just because they're not near a pivot
            elif price_above_pp and current_price > ema_20 > ema_50:
                # Strong bullish alignment but not near a specific level
                # Check if at least approaching something within broader range
                if dist_to_r1 < 0.10 or dist_to_s1 < 0.10:
                    setup_type = "Bullish breakout"
            elif not price_above_pp and current_price < ema_20 < ema_50:
                # Strong bearish alignment
                if dist_to_s1 < 0.10 or dist_to_r1 < 0.10:
                    setup_type = "Bearish breakdown"

            # Skip if no valid setup found
            if setup_type is None:
                return None

            is_bullish_setup = "Bullish" in setup_type

            # ─────────────────────────────────────────────────
            # STEP 3: DIRECTION-AWARE CONFLUENCE SCORING
            # ─────────────────────────────────────────────────

            score   = 0
            factors = []
            confluence_breakdown = {
                'pivot':          {'hit': False, 'level': '', 'price': 0, 'color': '', 'reason': ''},
                'pp_side':        {'hit': False, 'above': price_above_pp},
                'vrvp':           {'hit': False, 'level': '', 'price': 0},
                'ema_sma':        {'hit': False, 'direction': '', 'color': ''},
                'prev_week':      {'hit': False, 'level': '', 'distance': 0},
                'rsi':            {'hit': False, 'value': current_rsi, 'status': rsi_status, 'color': rsi_color},
                'price_distance': {'hit': False, 'distance': price_distance}
            }

            # ── FACTOR 1: PIVOT LEVEL (direction-aware) ──────
            # Bullish: reward S1/S2 (support below), penalise R1 nearby (wall above)
            # Bearish: reward R1/R2 (resistance above), penalise S1 nearby (floor below)

            if is_bullish_setup:
                # Good: near support level (S1 or S2 below us)
                if 0 < dist_to_s1 < 0.02:
                    score += 1
                    factors.append("At S1 support")
                    confluence_breakdown['pivot'] = {
                        'hit': True, 'level': 'S1',
                        'price': pivots['S1'], 'color': 'bullish',
                        'reason': 'Support below — bounce zone'
                    }
                elif 0 < dist_to_s2 < 0.02:
                    score += 1
                    factors.append("At S2 support")
                    confluence_breakdown['pivot'] = {
                        'hit': True, 'level': 'S2',
                        'price': pivots['S2'], 'color': 'bullish',
                        'reason': 'Strong support below — bounce zone'
                    }
                # Good: approaching R1 (breakout target) with room to run
                elif 0 < dist_to_r1 < 0.03:
                    # Only counts if there is ROOM between entry and R1 (at least 1%)
                    if dist_to_r1 > 0.01:
                        score += 1
                        factors.append(f"Approaching R1 ({dist_to_r1*100:.1f}% away)")
                        confluence_breakdown['pivot'] = {
                            'hit': True, 'level': 'R1',
                            'price': pivots['R1'], 'color': 'neutral',
                            'reason': f'Breakout target {dist_to_r1*100:.1f}% above'
                        }
                    # If R1 is less than 1% away it's a wall, not a target — skip
                # PP: score only if price is clearly above PP (bullish side)
                elif price_above_pp and abs(current_price - pivots['PP']) / current_price > 0.01:
                    score += 1
                    factors.append("Above PP (bullish side)")
                    confluence_breakdown['pivot'] = {
                        'hit': True, 'level': 'PP',
                        'price': pivots['PP'], 'color': 'bullish',
                        'reason': 'Price above PP — bullish bias confirmed'
                    }

            else:  # Bearish setup
                # Good: near resistance level (R1 or R2 above us)
                if 0 < dist_to_r1 < 0.02:
                    score += 1
                    factors.append("At R1 resistance")
                    confluence_breakdown['pivot'] = {
                        'hit': True, 'level': 'R1',
                        'price': pivots['R1'], 'color': 'bearish',
                        'reason': 'Resistance above — rejection zone'
                    }
                elif 0 < dist_to_r2 < 0.02:
                    score += 1
                    factors.append("At R2 resistance")
                    confluence_breakdown['pivot'] = {
                        'hit': True, 'level': 'R2',
                        'price': pivots['R2'], 'color': 'bearish',
                        'reason': 'Strong resistance above — rejection zone'
                    }
                # Good: approaching S1 (breakdown target) with room to fall
                elif 0 < dist_to_s1 < 0.03:
                    if dist_to_s1 > 0.01:
                        score += 1
                        factors.append(f"Approaching S1 ({dist_to_s1*100:.1f}% away)")
                        confluence_breakdown['pivot'] = {
                            'hit': True, 'level': 'S1',
                            'price': pivots['S1'], 'color': 'neutral',
                            'reason': f'Breakdown target {dist_to_s1*100:.1f}% below'
                        }
                # PP: score only if price is clearly below PP (bearish side)
                elif not price_above_pp and abs(current_price - pivots['PP']) / current_price > 0.01:
                    score += 1
                    factors.append("Below PP (bearish side)")
                    confluence_breakdown['pivot'] = {
                        'hit': True, 'level': 'PP',
                        'price': pivots['PP'], 'color': 'bearish',
                        'reason': 'Price below PP — bearish bias confirmed'
                    }

            # ── FACTOR 2: VRVP (direction-aware) ─────────────
            # Bullish: near VAL (volume support below) = good
            # Bearish: near VAH (volume resistance above) = good

            if is_bullish_setup:
                if 0 < dist_to_val < 0.02:
                    score += 1
                    factors.append("At VRVP VAL support")
                    confluence_breakdown['vrvp'] = {
                        'hit': True, 'level': 'VAL',
                        'price': vrvp['VAL']
                    }
                elif abs(current_price - vrvp['POC']) / current_price < 0.01:
                    score += 1
                    factors.append("At VRVP POC")
                    confluence_breakdown['vrvp'] = {
                        'hit': True, 'level': 'POC',
                        'price': vrvp['POC']
                    }
            else:
                if 0 < dist_to_vah < 0.02:
                    score += 1
                    factors.append("At VRVP VAH resistance")
                    confluence_breakdown['vrvp'] = {
                        'hit': True, 'level': 'VAH',
                        'price': vrvp['VAH']
                    }
                elif abs(current_price - vrvp['POC']) / current_price < 0.01:
                    score += 1
                    factors.append("At VRVP POC")
                    confluence_breakdown['vrvp'] = {
                        'hit': True, 'level': 'POC',
                        'price': vrvp['POC']
                    }

            # ── FACTOR 3: EMA ALIGNMENT (direction-aware) ────
            if is_bullish_setup and ema_bullish:
                score += 1
                factors.append("Bullish EMA stack")
                confluence_breakdown['ema_sma'] = {
                    'hit': True, 'direction': 'Bullish', 'color': 'bullish'
                }
            elif not is_bullish_setup and ema_bearish:
                score += 1
                factors.append("Bearish EMA stack")
                confluence_breakdown['ema_sma'] = {
                    'hit': True, 'direction': 'Bearish', 'color': 'bearish'
                }
            # Partial credit: price at least on correct side of EMA20
            elif is_bullish_setup and current_price > ema_20:
                score += 1
                factors.append("Above EMA20")
                confluence_breakdown['ema_sma'] = {
                    'hit': True, 'direction': 'Bullish (partial)', 'color': 'bullish'
                }
            elif not is_bullish_setup and current_price < ema_20:
                score += 1
                factors.append("Below EMA20")
                confluence_breakdown['ema_sma'] = {
                    'hit': True, 'direction': 'Bearish (partial)', 'color': 'bearish'
                }

            # ── FACTOR 4: PREVIOUS WEEK H/L (direction-aware) ─
            if is_bullish_setup and 0 < dist_to_pwl < 0.03:
                # Bouncing off prev week low = bullish support
                score += 1
                factors.append(f"At prev week low ({dist_to_pwl*100:.1f}%)")
                confluence_breakdown['prev_week'] = {
                    'hit': True, 'level': 'Low', 'distance': dist_to_pwl * 100
                }
            elif is_bullish_setup and 0 < dist_to_pwh < 0.03:
                # Approaching prev week high = breakout target
                if dist_to_pwh > 0.01:
                    score += 1
                    factors.append(f"Near prev week high ({dist_to_pwh*100:.1f}%)")
                    confluence_breakdown['prev_week'] = {
                        'hit': True, 'level': 'High', 'distance': dist_to_pwh * 100
                    }
            elif not is_bullish_setup and 0 < dist_to_pwh < 0.03:
                # Rejecting off prev week high = bearish resistance
                score += 1
                factors.append(f"At prev week high ({dist_to_pwh*100:.1f}%)")
                confluence_breakdown['prev_week'] = {
                    'hit': True, 'level': 'High', 'distance': dist_to_pwh * 100
                }
            elif not is_bullish_setup and 0 < dist_to_pwl < 0.03:
                # Approaching prev week low = breakdown target
                if dist_to_pwl > 0.01:
                    score += 1
                    factors.append(f"Near prev week low ({dist_to_pwl*100:.1f}%)")
                    confluence_breakdown['prev_week'] = {
                        'hit': True, 'level': 'Low', 'distance': dist_to_pwl * 100
                    }

            # ── FACTOR 5: RSI (direction-aware) ──────────────
            # Bullish: RSI oversold/rising = good. Overbought = bad (don't score)
            # Bearish: RSI overbought/falling = good. Oversold = bad (don't score)
            if is_bullish_setup:
                if current_rsi < 45 or (30 < current_rsi < 60 and rsi_momentum == "Rising"):
                    score += 1
                    factors.append(f"RSI {current_rsi:.0f} (bullish)")
                    confluence_breakdown['rsi']['hit'] = True
            else:
                if current_rsi > 55 or (40 < current_rsi < 70 and rsi_momentum == "Falling"):
                    score += 1
                    factors.append(f"RSI {current_rsi:.0f} (bearish)")
                    confluence_breakdown['rsi']['hit'] = True

            # ── FACTOR 6: PRICE NEAR EMA20 ───────────────────
            # Only scores if price is on the CORRECT side of EMA20
            if is_bullish_setup and price_distance < 0.03 and current_price >= ema_20:
                score += 1
                factors.append(f"Near EMA20 ({price_distance*100:.1f}%)")
                confluence_breakdown['price_distance'] = {
                    'hit': True, 'distance': price_distance * 100
                }
            elif not is_bullish_setup and price_distance < 0.03 and current_price <= ema_20:
                score += 1
                factors.append(f"Near EMA20 ({price_distance*100:.1f}%)")
                confluence_breakdown['price_distance'] = {
                    'hit': True, 'distance': price_distance * 100
                }
            
            # ─────────────────────────────────────────────────
            # TARGETS, STOP, STRIKE
            # ─────────────────────────────────────────────────
            
            if is_bullish_setup:
                option_type = "CALL"
                
                # Stop: below nearest support, at least 1x ATR away
                support_level = max(
                    p for p in [pivots['S1'], vrvp['VAL'], prev_week_low]
                    if p < current_price and p > 0
                ) if any(p < current_price and p > 0 for p in [pivots['S1'], vrvp['VAL'], prev_week_low]) else current_price - (1.5 * atr)
                
                stop = min(support_level - (0.5 * atr), current_price - (1.5 * atr))
                
                # Targets: next resistance levels ABOVE current price
                resistance_levels = sorted([
                    p for p in [pivots['R1'], pivots['R2'], vrvp['VAH'], prev_week_high]
                    if p > current_price
                ])
                
                if len(resistance_levels) >= 2:
                    target_r1 = resistance_levels[0]
                    target_r2 = resistance_levels[1]
                elif len(resistance_levels) == 1:
                    target_r1 = resistance_levels[0]
                    target_r2 = target_r1 + (target_r1 - current_price)
                else:
                    target_r1 = current_price + (2 * atr)
                    target_r2 = current_price + (3 * atr)
                
                # Strike selection
                if score == 6:
                    suggested_strike = current_price
                elif score == 5:
                    suggested_strike = current_price * 1.01
                else:
                    suggested_strike = current_price * 1.02
                
            else:  # Bearish
                option_type = "PUT"
                
                # Stop: above nearest resistance, at least 1x ATR away
                resistance_level = min(
                    p for p in [pivots['R1'], vrvp['VAH'], prev_week_high]
                    if p > current_price and p > 0
                ) if any(p > current_price and p > 0 for p in [pivots['R1'], vrvp['VAH'], prev_week_high]) else current_price + (1.5 * atr)
                
                stop = max(resistance_level + (0.5 * atr), current_price + (1.5 * atr))
                
                # Targets: next support levels BELOW current price
                support_levels = sorted([
                    p for p in [pivots['S1'], pivots['S2'], vrvp['VAL'], prev_week_low]
                    if p < current_price and p > 0
                ], reverse=True)
                
                if len(support_levels) >= 2:
                    target_r1 = support_levels[0]
                    target_r2 = support_levels[1]
                elif len(support_levels) == 1:
                    target_r1 = support_levels[0]
                    target_r2 = target_r1 - (current_price - target_r1)
                else:
                    target_r1 = current_price - (2 * atr)
                    target_r2 = current_price - (3 * atr)
                
                # Strike selection
                if score == 6:
                    suggested_strike = current_price
                elif score == 5:
                    suggested_strike = current_price * 0.99
                else:
                    suggested_strike = current_price * 0.98
            
            # ─────────────────────────────────────────────────
            # STRIKE ROUNDING to realistic increments
            # ─────────────────────────────────────────────────
            if current_price >= 100:
                strike = round(suggested_strike / 5) * 5
            elif current_price >= 20:
                strike = round(suggested_strike)
            else:
                strike = round(suggested_strike * 2) / 2
            
            # ─────────────────────────────────────────────────
            # VALIDATE targets are in right direction
            # ─────────────────────────────────────────────────
            if is_bullish_setup:
                if target_r1 <= current_price or target_r2 <= current_price:
                    return None
                if stop >= current_price:
                    return None
            else:
                if target_r1 >= current_price or target_r2 >= current_price:
                    return None
                if stop <= current_price:
                    return None
            
            # ─────────────────────────────────────────────────
            # RISK/REWARD CALCULATION
            # Filter out setups with R/R < 1.5:1
            # ─────────────────────────────────────────────────
            risk        = abs(current_price - stop)
            reward_r1   = abs(target_r1 - current_price)
            reward_r2   = abs(target_r2 - current_price)
            rr_ratio_r1 = reward_r1 / risk if risk > 0 else 0
            rr_ratio_r2 = reward_r2 / risk if risk > 0 else 0
            
            # Must have at least 1.5:1 R/R to R1
            if rr_ratio_r1 < 1.5:
                return None
            
            # ─────────────────────────────────────────────────
            # DTE RECOMMENDATION based on setup type
            # ─────────────────────────────────────────────────
            if "breakout" in setup_type.lower() or "breakdown" in setup_type.lower():
                recommended_dte = 60   # Breakouts need less time
                dte_min         = 45
                dte_max         = 90
            else:
                recommended_dte = 75   # Reversals need more time to develop
                dte_min         = 60
                dte_max         = 120
            
            if score < self.confluence_threshold:
                return None
            
            return {
                'ticker':           ticker,
                'price':            current_price,
                'score':            score,
                'setup_type':       setup_type,
                'factors':          factors,
                'option_type':      option_type,
                'strike':           strike,
                'target_r1':        target_r1,
                'target_r2':        target_r2,
                'stop':             stop,
                'risk':             risk,
                'reward_r1':        reward_r1,
                'reward_r2':        reward_r2,
                'rr_ratio_r1':      rr_ratio_r1,
                'rr_ratio_r2':      rr_ratio_r2,
                'recommended_dte':  recommended_dte,
                'dte_min':          dte_min,
                'dte_max':          dte_max,
                'atr':              atr,
                'rsi':              current_rsi,
                'rsi_status':       rsi_status,
                'rsi_color':        rsi_color,
                'confluence_breakdown': confluence_breakdown,
                'ema_20':           ema_20,
                'ema_50':           ema_50
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
    """
    Calculate time-based exit checkpoints based on theta decay zones
    
    Key DTE thresholds (based on theta decay science):
    - 45 DTE: Theta acceleration begins (~2%/day)
    - 30 DTE: Major theta impact (~3%/day) 
    - 21 DTE: Exponential decay zone (~4-5%/day)
    - 14 DTE: Death zone (avoid!)
    
    Args:
        dte_at_entry: Days to expiration when position opened
        setup_type: "Bullish breakout", "Bearish breakdown", "Reversal", etc.
    
    Returns:
        Dictionary with checkpoint days and DTE thresholds
    """
    
    if "breakout" in setup_type.lower() or "breakdown" in setup_type.lower():
        # Breakouts: Move fast or fail
        quick_exit_days = 7  # If no progress in 7 days, setup failed
        max_hold_dte = 30    # Exit when 30 DTE remaining (theta danger zone)
    else:  # Reversal setups
        # Reversals: Need time to develop
        quick_exit_days = 10  # Give more time for reversal to play out
        max_hold_dte = 25     # Can hold slightly longer (but not below 25 DTE)
    
    # Calculate max hold in days from entry
    max_hold_days = dte_at_entry - max_hold_dte
    
    # Safety checks
    if max_hold_days < quick_exit_days:
        # If entered with low DTE, adjust
        max_hold_days = quick_exit_days + 3
    
    if max_hold_days < 5:
        # Minimum 5 days to give trade a chance
        max_hold_days = 5
    
    # Theta warning = 21 DTE (universal danger zone)
    theta_warning_days = dte_at_entry - 21
    if theta_warning_days < max_hold_days:
        theta_warning_days = max_hold_days + 5
    
    # Calculate actual DTE thresholds for display
    quick_exit_dte = dte_at_entry - quick_exit_days
    max_hold_dte_actual = dte_at_entry - max_hold_days
    theta_warning_dte = dte_at_entry - theta_warning_days
    
    return {
        'quick_exit': quick_exit_days,
        'max_hold': max_hold_days,
        'theta_warning': theta_warning_days,
        'dte_thresholds': {
            'quick_exit_dte': quick_exit_dte,
            'max_hold_dte': max_hold_dte_actual,
            'theta_warning_dte': theta_warning_dte
        }
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


def render_position_details_sidebar(pos, index):
    """Render detailed position view in sidebar"""
    
    days_held = (datetime.now() - datetime.fromisoformat(pos['entry_date'])).days
    
    # Calculate DTE from actual expiration date if available
    if 'expiration_date' in pos:
        exp_dt = datetime.fromisoformat(pos['expiration_date'])
        dte_remaining = (exp_dt.date() - datetime.now().date()).days
    else:
        # Fallback to old calculation
        dte_remaining = pos['dte_at_entry'] - days_held
    
    checkpoints = calculate_time_checkpoints(pos['dte_at_entry'], pos['setup_type'])
    
    # Fetch current price
    current_price = get_current_stock_price(pos['ticker'])
    if not current_price:
        current_price = pos['entry_price']
        st.sidebar.warning("⚠️ Using entry price")
    
    # CORRECT moneyness calculation
    if pos['option_type'] == 'CALL':
        if pos['strike'] > current_price:
            # Call is OTM (strike above stock price)
            moneyness_pct = ((pos['strike'] - current_price) / current_price) * 100
            moneyness_status = "OTM"
        else:
            # Call is ITM (strike below stock price)
            moneyness_pct = ((current_price - pos['strike']) / current_price) * 100
            moneyness_status = "ITM"
    else:  # PUT
        if pos['strike'] < current_price:
            # Put is OTM (strike below stock price)
            moneyness_pct = ((current_price - pos['strike']) / current_price) * 100
            moneyness_status = "OTM"
        else:
            # Put is ITM (strike above stock price)
            moneyness_pct = ((pos['strike'] - current_price) / current_price) * 100
            moneyness_status = "ITM"
    
    stock_change = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
    dist_to_r1 = ((pos['target_r1'] - current_price) / current_price) * 100
    dist_to_r2 = ((pos['target_r2'] - current_price) / current_price) * 100
    dist_to_stop = ((pos['stop'] - current_price) / current_price) * 100
    
    # Header
    neon_icon = get_neon_icon(pos['option_type'], size="1.2em")
    st.sidebar.markdown(f"### 📊 {neon_icon} {pos['ticker']} ${pos['strike']}{pos['option_type'][0]}", unsafe_allow_html=True)
    
    # Current status
    st.sidebar.markdown("**Current Status:**")
    st.sidebar.markdown(f"Price: **${current_price:.2f}** ({stock_change:+.1f}%)")
    st.sidebar.markdown(f"Days: **{days_held}** / DTE: **{dte_remaining}d**")
    st.sidebar.markdown(f"Moneyness: **{moneyness_pct:.1f}% {moneyness_status}**")
    
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
        st.sidebar.markdown(f"✅ Day {days_held} ({dte_remaining} DTE) - Safe")
        st.sidebar.markdown(f"⚠️ Day {checkpoints['quick_exit']} ({checkpoints['dte_thresholds']['quick_exit_dte']} DTE) - Decision")
        st.sidebar.markdown(f"🔴 Day {checkpoints['max_hold']} ({checkpoints['dte_thresholds']['max_hold_dte']} DTE) - Max hold")
    elif days_held < checkpoints['max_hold']:
        st.sidebar.markdown(f"⚠️ Day {days_held} ({dte_remaining} DTE) - **DECISION ZONE**")
        st.sidebar.markdown(f"🔴 Day {checkpoints['max_hold']} ({checkpoints['dte_thresholds']['max_hold_dte']} DTE) - Max hold")
    else:
        st.sidebar.markdown(f"🔴 Day {days_held} ({dte_remaining} DTE) - **EXIT NOW**")
    st.sidebar.markdown(f"💀 Day {checkpoints['theta_warning']} ({checkpoints['dte_thresholds']['theta_warning_dte']} DTE) - Never hold")
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"*Theta decay: ~{3 if dte_remaining > 30 else 5 if dte_remaining > 21 else 8}%/day at {dte_remaining} DTE*")
    
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
        neon_icon = get_neon_icon(pos['option_type'], size="1.5em")
        option_color = "green" if pos['option_type'] == 'CALL' else "red"
        st.markdown(f"## 📊 {neon_icon} {pos['ticker']} <span style='color:{option_color}; font-weight:bold;'>${pos['strike']} {pos['option_type']}</span>", unsafe_allow_html=True)
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
    
    # Calculate DTE from actual expiration date if available
    if 'expiration_date' in pos:
        exp_dt = datetime.fromisoformat(pos['expiration_date'])
        dte_remaining = (exp_dt.date() - datetime.now().date()).days
    else:
        # Fallback to old calculation
        dte_remaining = pos['dte_at_entry'] - days_held
    
    checkpoints = calculate_time_checkpoints(pos['dte_at_entry'], pos['setup_type'])
    
    # Fetch current price
    current_price = get_current_stock_price(pos['ticker'])
    if not current_price:
        current_price = pos['entry_price']
        st.warning("⚠️ Unable to fetch current price. Using entry price.")
    
    # CORRECT moneyness calculation
    if pos['option_type'] == 'CALL':
        if pos['strike'] > current_price:
            # Call is OTM (strike above stock price)
            moneyness_pct = ((pos['strike'] - current_price) / current_price) * 100
            moneyness_status = "OTM"
        else:
            # Call is ITM (strike below stock price)
            moneyness_pct = ((current_price - pos['strike']) / current_price) * 100
            moneyness_status = "ITM"
    else:  # PUT
        if pos['strike'] < current_price:
            # Put is OTM (strike below stock price)
            moneyness_pct = ((current_price - pos['strike']) / current_price) * 100
            moneyness_status = "OTM"
        else:
            # Put is ITM (strike above stock price)
            moneyness_pct = ((pos['strike'] - current_price) / current_price) * 100
            moneyness_status = "ITM"
    
    stock_change = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
    
    # Key metrics at top
    st.markdown("### 📈 Current Status")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Stock Price", f"${current_price:.2f}", f"{stock_change:+.1f}%")
    with col2:
        st.metric("Days Held", f"{days_held} days", f"{dte_remaining}d DTE left")
    with col3:
        st.metric("Moneyness", f"{moneyness_pct:.1f}% {moneyness_status}")
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
        
        # R/R from saved data or recalculate
        rr1 = pos.get('rr_ratio_r1', abs(pos['target_r1'] - pos['entry_price']) / abs(pos['entry_price'] - pos['stop']) if pos['entry_price'] != pos['stop'] else 0)
        rr2 = pos.get('rr_ratio_r2', abs(pos['target_r2'] - pos['entry_price']) / abs(pos['entry_price'] - pos['stop']) if pos['entry_price'] != pos['stop'] else 0)
        rr1_color = "green" if rr1 >= 2 else "orange" if rr1 >= 1.5 else "red"
        rr2_color = "green" if rr2 >= 2 else "orange" if rr2 >= 1.5 else "red"
        
        st.markdown(f"**Target R1:** ${pos['target_r1']:.2f} &nbsp;<span style='color:{rr1_color}'>({rr1:.1f}:1 R/R)</span>", unsafe_allow_html=True)
        st.markdown(f"**Target R2:** ${pos['target_r2']:.2f} &nbsp;<span style='color:{rr2_color}'>({rr2:.1f}:1 R/R)</span>", unsafe_allow_html=True)
    with col3:
        dist_to_r1   = ((pos['target_r1'] - current_price) / current_price) * 100
        dist_to_r2   = ((pos['target_r2'] - current_price) / current_price) * 100
        dist_to_stop = ((pos['stop'] - current_price) / current_price) * 100
        
        # Direction-aware labels
        if "Bullish" in pos['setup_type']:
            r1_label   = f"+{abs(dist_to_r1):.1f}% away"
            r2_label   = f"+{abs(dist_to_r2):.1f}% away"
            stop_label = f"-{abs(dist_to_stop):.1f}% away"
        else:
            r1_label   = f"-{abs(dist_to_r1):.1f}% away"
            r2_label   = f"-{abs(dist_to_r2):.1f}% away"
            stop_label = f"+{abs(dist_to_stop):.1f}% away"
        
        st.markdown(f"**To R1:** {r1_label}")
        st.markdown(f"**To R2:** {r2_label}")
        st.markdown(f"**To Stop:** {stop_label}")
        st.markdown(f"**Stop Loss:** ${pos['stop']:.2f}")
    
    st.markdown("---")
    
    # Time exit strategy
    st.markdown("### ⏰ Time Exit Strategy")
    st.markdown(f"**Setup Type:** {pos['setup_type']}")
    st.markdown(f"**Entry:** {pos['dte_at_entry']} DTE | **Current:** {dte_remaining} DTE remaining")
    
    # Show theta decay rate at current DTE
    if dte_remaining > 45:
        theta_rate = "~1-2%"
        theta_status = "Low theta decay"
    elif dte_remaining > 30:
        theta_rate = "~2-3%"
        theta_status = "Moderate theta decay"
    elif dte_remaining > 21:
        theta_rate = "~3-4%"
        theta_status = "High theta decay"
    elif dte_remaining > 14:
        theta_rate = "~5-7%"
        theta_status = "⚠️ Rapid theta decay"
    else:
        theta_rate = "~8-10%"
        theta_status = "🔴 Extreme theta decay"
    
    st.info(f"📊 Current theta decay: **{theta_rate} per day** ({theta_status})")
    
    st.markdown("")
    
    # Checkpoint display with DTE
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if days_held < checkpoints['quick_exit']:
            st.markdown(f"<div style='padding:1rem; background:#1e4620; border-left:4px solid #28a745; border-radius:4px; color:#ffffff;'><b>✅ Day {days_held} (Today)</b><br>{dte_remaining} DTE left<br>Safe Zone - Continue monitoring</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='padding:1rem; background:#2a2a2a; border-left:4px solid #6c757d; border-radius:4px; color:#cccccc;'><b>Day 0-{checkpoints['quick_exit']}</b><br>({checkpoints['dte_thresholds']['quick_exit_dte']}+ DTE)<br>Safe Zone (passed)</div>", unsafe_allow_html=True)
    
    with col2:
        if days_held >= checkpoints['quick_exit'] and days_held < checkpoints['max_hold']:
            st.markdown(f"<div style='padding:1rem; background:#4a3800; border-left:4px solid #ffc107; border-radius:4px; color:#ffffff;'><b>⚠️ Day {days_held} (Today)</b><br>{dte_remaining} DTE left<br>Decision - Exit if no R1 progress</div>", unsafe_allow_html=True)
        elif days_held < checkpoints['quick_exit']:
            st.markdown(f"<div style='padding:1rem; background:#2a2a2a; border-left:4px solid #ffc107; border-radius:4px; color:#cccccc;'><b>Day {checkpoints['quick_exit']}</b><br>({checkpoints['dte_thresholds']['quick_exit_dte']} DTE)<br>Decision Point (upcoming)</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='padding:1rem; background:#2a2a2a; border-left:4px solid #6c757d; border-radius:4px; color:#cccccc;'><b>Day {checkpoints['quick_exit']}</b><br>({checkpoints['dte_thresholds']['quick_exit_dte']} DTE)<br>Decision Point (passed)</div>", unsafe_allow_html=True)
    
    with col3:
        if days_held >= checkpoints['max_hold']:
            st.markdown(f"<div style='padding:1rem; background:#4a1f1f; border-left:4px solid #dc3545; border-radius:4px; color:#ffffff;'><b>🔴 Day {days_held} (Today)</b><br>{dte_remaining} DTE left<br>MAXIMUM HOLD - EXIT NOW!</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='padding:1rem; background:#2a2a2a; border-left:4px solid #dc3545; border-radius:4px; color:#cccccc;'><b>Day {checkpoints['max_hold']}</b><br>({checkpoints['dte_thresholds']['max_hold_dte']} DTE)<br>Maximum Hold - Exit by this day</div>", unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown(f"<div style='padding:1rem; background:#1a1a1a; color:#ffffff; border-left:4px solid #000; border-radius:4px;'><b>💀 Day {checkpoints['theta_warning']} ({checkpoints['dte_thresholds']['theta_warning_dte']} DTE)</b><br>Theta Death Zone - Never hold this long (theta ~8-10%/day)</div>", unsafe_allow_html=True)
    
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
    
    # Market status notice (using Eastern Time with DST support)
    from datetime import timezone as tz
    import calendar
    
    # Simple DST check for US Eastern Time
    now_utc = datetime.now(tz.utc)
    year = now_utc.year
    
    # DST starts 2nd Sunday in March, ends 1st Sunday in November (US)
    march = datetime(year, 3, 1, tzinfo=tz.utc)
    march_second_sunday = march + timedelta(days=(13 - march.weekday()) % 7)
    november = datetime(year, 11, 1, tzinfo=tz.utc)
    november_first_sunday = november + timedelta(days=(6 - november.weekday()) % 7)
    
    # Determine if DST is active
    if march_second_sunday <= now_utc < november_first_sunday:
        eastern_offset = -4  # EDT (daylight time)
    else:
        eastern_offset = -5  # EST (standard time)
    
    eastern = timezone(timedelta(hours=eastern_offset))
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
    
    # ─────────────────────────────────────────────────
    # MANUAL STOCK SEARCH
    # ─────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Manual Stock Search")
    st.sidebar.caption("Search any ticker outside the top 50")
    
    manual_ticker = st.sidebar.text_input(
        "Enter Ticker Symbol",
        value="",
        placeholder="e.g. TSLA, F, GME...",
        key="manual_ticker_input"
    ).upper().strip()
    
    search_btn = st.sidebar.button(
        "🔍 Analyze Stock",
        key="manual_search_btn",
        use_container_width=True,
        type="primary"
    )
    
    # Run manual search
    if search_btn and manual_ticker:
        with st.sidebar.status(f"Analyzing {manual_ticker}...", expanded=False):
            try:
                screener_manual = SwingScreener(
                    min_market_cap=0,       # No market cap filter for manual search
                    min_volume=0,           # No volume filter for manual search
                    confluence_threshold=confluence_threshold
                )
                
                # Fetch data
                stock = yf.Ticker(manual_ticker)
                data = stock.history(period='6mo')
                
                if data.empty or len(data) < 50:
                    st.sidebar.error(f"❌ No data found for {manual_ticker}")
                    st.session_state.manual_search_result = None
                    st.session_state.manual_search_ticker = None
                else:
                    # Run confluence check
                    result = screener_manual.check_confluence(manual_ticker, data)
                    
                    if result:
                        st.session_state.manual_search_result = result
                        st.session_state.manual_search_ticker = manual_ticker
                        st.sidebar.success(f"✅ Found setup! Score: {result['score']}/6")
                    else:
                        # Still store basic data even if no setup — show analysis
                        st.session_state.manual_search_result = 'no_setup'
                        st.session_state.manual_search_ticker = manual_ticker
                        st.sidebar.warning(f"⚠️ {manual_ticker} — No qualifying setup found")
                
                st.rerun()
                
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")
                st.session_state.manual_search_result = None
    
    elif search_btn and not manual_ticker:
        st.sidebar.warning("Please enter a ticker symbol")
    
    # Active Positions Section
    st.sidebar.markdown("---")
    
    # Initialize folder state
    if 'positions_folder_open' not in st.session_state:
        st.session_state.positions_folder_open = True
    
    # Check if showing details for any position
    showing_details = None
    for i in range(len(st.session_state.active_positions)):
        if st.session_state.get(f'show_details_{i}', False):
            showing_details = i
            break
    
    # Folder header with toggle and selected position
    col1, col2 = st.sidebar.columns([4, 1])
    with col1:
        st.sidebar.subheader(f"📂 Active Positions ({len(st.session_state.active_positions)}/5)")
        
        # Show selected position under folder if one is selected
        if showing_details is not None:
            pos = st.session_state.active_positions[showing_details]
            if 'expiration_date' in pos:
                exp_dt = datetime.fromisoformat(pos['expiration_date'])
                exp_str = exp_dt.strftime('%m/%d/%y')
            else:
                entry_dt = datetime.fromisoformat(pos['entry_date'])
                expiration_date = entry_dt + timedelta(days=pos['dte_at_entry'])
                exp_str = expiration_date.strftime('%m/%d/%y')
            
            neon_icon = get_neon_icon(pos['option_type'], size="1em")
            st.sidebar.markdown(f"↳ {neon_icon} {pos['ticker']} ${pos['strike']}{pos['option_type'][0]} - Exp {exp_str}", unsafe_allow_html=True)
    
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
    
    # Only show contents if folder is open
    if st.session_state.positions_folder_open:
        if len(st.session_state.active_positions) == 0:
            st.sidebar.info("No active positions. Add setups from scan results below.")
        else:
            # Show clickable position list
            st.sidebar.markdown("**Click a position:**")
            for i, pos in enumerate(st.session_state.active_positions):
                if 'expiration_date' in pos:
                    exp_dt = datetime.fromisoformat(pos['expiration_date'])
                    exp_str = exp_dt.strftime('%m/%d/%y')
                else:
                    entry_dt = datetime.fromisoformat(pos['entry_date'])
                    expiration_date = entry_dt + timedelta(days=pos['dte_at_entry'])
                    exp_str = expiration_date.strftime('%m/%d/%y')
                
                # Use neon glow icon instead of emoji
                neon_icon = get_neon_icon(pos['option_type'], size="1.2em")
                button_label = f"{pos['ticker']} ${pos['strike']}{pos['option_type'][0]} - Exp {exp_str}"
                
                # Create button with HTML prefix for icon
                col1, col2 = st.sidebar.columns([0.15, 0.85])
                with col1:
                    st.markdown(neon_icon, unsafe_allow_html=True)
                with col2:
                    if st.button(button_label, key=f"pos_btn_{i}", use_container_width=True):
                        st.session_state[f'show_details_{i}'] = True
                        st.rerun()
    else:
        # Folder closed - show nothing
        st.sidebar.caption("Click ▶ to open positions folder")
    
    # Check if a position is selected for main area display
    showing_details = None
    for i in range(len(st.session_state.active_positions)):
        if st.session_state.get(f'show_details_{i}', False):
            showing_details = i
            break
    
    # If position selected, show in main area instead of scan results
    if showing_details is not None:
        render_position_details(st.session_state.active_positions[showing_details], showing_details)
        return  # Skip showing scan results
    
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
    
    # ─────────────────────────────────────────────────
    # MANUAL SEARCH RESULTS
    # ─────────────────────────────────────────────────
    if st.session_state.manual_search_result is not None:
        ticker = st.session_state.manual_search_ticker
        result = st.session_state.manual_search_result
        
        st.markdown("---")
        
        if result == 'no_setup':
            # No qualifying setup — but still allow adding to positions
            st.warning(f"### 🔍 Manual Search: {ticker}")
            st.markdown(f"**⚠️ This stock does not meet confluence criteria**")
            
            # Fetch basic data for manual add
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period='6mo')
                current_price = data['Close'].iloc[-1]
                
                st.markdown(f"**Current Price:** ${current_price:.2f}")
                st.markdown("")
                st.markdown("**Why it doesn't qualify:**")
                st.markdown("- Too far from key pivot/support/resistance levels")
                st.markdown("- EMAs not aligned for a directional trade")
                st.markdown("- RSI not confirming direction")
                st.markdown("- Risk/Reward below 1.5:1 minimum")
                st.markdown("- Not approaching any meaningful level")
                
                st.markdown("---")
                st.info("💡 **You can still add this to positions manually if you have your own analysis**")
                
                # Manual add form (no confluence data)
                position_key = f"manual_{ticker}"
                add_key = f"add_{position_key}"
                if add_key not in st.session_state:
                    st.session_state[add_key] = False
                
                max_positions = 5
                current_positions = len(st.session_state.active_positions)
                already_tracked = any(p['ticker'] == ticker for p in st.session_state.active_positions)
                
                if already_tracked:
                    st.info(f"📌 {ticker} is already in your active positions")
                elif current_positions >= max_positions:
                    st.warning(f"⚠️ Maximum {max_positions} positions reached")
                elif not st.session_state[add_key]:
                    if st.button(f"➕ Add {ticker} to Positions (Manual Entry)", key=f"add_manual_no_setup", type="secondary"):
                        st.session_state[add_key] = True
                        st.rerun()
                else:
                    st.markdown(f"### ➕ Manually Add {ticker} to Active Positions")
                    st.caption("⚠️ No confluence - you'll need to set all parameters manually")
                    
                    # Option type selector OUTSIDE form so it updates dynamically
                    option_type_input = st.selectbox(
                        "Option Type", 
                        ["CALL", "PUT"],
                        help="Choose CALL (bullish) or PUT (bearish)",
                        key=f"option_type_{position_key}"
                    )
                    
                    is_call = option_type_input == "CALL"
                    
                    # Direction-aware defaults and labels
                    if is_call:
                        default_target1 = float(round(current_price * 1.05, 2))
                        default_target2 = float(round(current_price * 1.10, 2))
                        default_stop = float(round(current_price * 0.95, 2))
                        target1_label = "🎯 Target R1 (First Resistance)"
                        target2_label = "🎯 Target R2 (Second Resistance)"
                        target1_help = "First resistance target ABOVE current price (75% exit)"
                        target2_help = "Second resistance target ABOVE current price (25% exit)"
                        stop_help = "Stop loss BELOW current price"
                        direction_info = f"📈 **CALL Setup:** Price must move UP from ${current_price:.2f}"
                    else:  # PUT
                        default_target1 = float(round(current_price * 0.95, 2))
                        default_target2 = float(round(current_price * 0.90, 2))
                        default_stop = float(round(current_price * 1.05, 2))
                        target1_label = "🎯 Target S1 (First Support)"
                        target2_label = "🎯 Target S2 (Second Support)"
                        target1_help = "First support target BELOW current price (75% exit)"
                        target2_help = "Second support target BELOW current price (25% exit)"
                        stop_help = "Stop loss ABOVE current price"
                        direction_info = f"📉 **PUT Setup:** Price must move DOWN from ${current_price:.2f}"
                    
                    st.info(direction_info)
                    
                    with st.form(key=f"form_{position_key}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            strike_input = st.number_input(
                                "Strike Price",
                                min_value=1.0,
                                value=float(round(current_price)),
                                step=0.50 if current_price < 20 else (1.0 if current_price < 100 else 5.0),
                                help="Enter your chosen strike price"
                            )
                            target_r1_input = st.number_input(
                                target1_label,
                                min_value=1.0,
                                value=default_target1,
                                step=0.50,
                                help=target1_help
                            )
                        with col2:
                            entry_date = st.date_input("Entry Date", value=datetime.now())
                            min_exp = datetime.now() + timedelta(days=45)
                            default_exp = datetime.now() + timedelta(days=60)
                            exp_date_input = st.date_input(
                                "Expiration Date",
                                value=default_exp,
                                min_value=min_exp
                            )
                            contracts_input = st.number_input("Contracts", min_value=1, max_value=10, value=1)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            target_r2_input = st.number_input(
                                target2_label,
                                min_value=1.0,
                                value=default_target2,
                                step=0.50,
                                help=target2_help
                            )
                        with col2:
                            stop_input = st.number_input(
                                "Stop Loss",
                                min_value=1.0,
                                value=default_stop,
                                step=0.50,
                                help=stop_help
                            )
                        
                        dte_calculated = (exp_date_input - entry_date).days
                        risk = abs(current_price - stop_input)
                        reward_r1 = abs(target_r1_input - current_price)
                        rr_ratio = reward_r1 / risk if risk > 0 else 0
                        
                        # Show current price for reference
                        st.caption(f"Stock: ${current_price:.2f} | DTE: {dte_calculated}d | R/R: {rr_ratio:.1f}:1")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            cancel_btn = st.form_submit_button("Cancel")
                        with col2:
                            submit_btn = st.form_submit_button("Add Position", type="primary")
                        
                        if cancel_btn:
                            st.session_state[add_key] = False
                            st.rerun()
                        
                        if submit_btn:
                            # Validate
                            if dte_calculated < 7:
                                st.error("⚠️ Expiration must be at least 7 days from entry")
                            elif dte_calculated > 365:
                                st.error("⚠️ Expiration must be within 1 year")
                            elif option_type_input == "CALL" and (target_r1_input <= current_price or target_r2_input <= current_price):
                                st.error("⚠️ For CALLS, targets must be above current price")
                            elif option_type_input == "CALL" and stop_input >= current_price:
                                st.error("⚠️ For CALLS, stop must be below current price")
                            elif option_type_input == "PUT" and (target_r1_input >= current_price or target_r2_input >= current_price):
                                st.error("⚠️ For PUTS, targets must be below current price")
                            elif option_type_input == "PUT" and stop_input <= current_price:
                                st.error("⚠️ For PUTS, stop must be above current price")
                            else:
                                reward_r2 = abs(target_r2_input - current_price)
                                rr_ratio_r2 = reward_r2 / risk if risk > 0 else 0
                                
                                new_position = {
                                    'ticker': ticker,
                                    'strike': strike_input,
                                    'option_type': option_type_input,
                                    'entry_date': entry_date.isoformat(),
                                    'expiration_date': exp_date_input.isoformat(),
                                    'entry_price': current_price,
                                    'dte_at_entry': dte_calculated,
                                    'contracts': contracts_input,
                                    'setup_type': f"Manual entry ({option_type_input})",
                                    'target_r1': target_r1_input,
                                    'target_r2': target_r2_input,
                                    'stop': stop_input,
                                    'risk': risk,
                                    'reward_r1': reward_r1,
                                    'reward_r2': reward_r2,
                                    'rr_ratio_r1': rr_ratio,
                                    'rr_ratio_r2': rr_ratio_r2,
                                    'manual_search': True,
                                    'no_confluence': True
                                }
                                st.session_state.active_positions.append(new_position)
                                save_positions(st.session_state.active_positions)
                                st.session_state[add_key] = False
                                st.success(f"✅ Added {ticker} ${strike_input} {option_type_input} (Manual Entry)")
                                time.sleep(1)
                                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error fetching {ticker} data: {str(e)}")
            
            if st.button("✖ Clear Search", key="clear_no_setup"):
                st.session_state.manual_search_result = None
                st.session_state.manual_search_ticker = None
                st.rerun()
        else:
            # Valid setup found — render full card
            setup  = result
            direction_emoji = "🟢" if "Bullish" in setup['setup_type'] else "🔴"
            
            st.markdown(f"### 🔍 Manual Search Result: {direction_emoji} **{ticker}**")
            
            col_clear, col_space = st.columns([1, 5])
            with col_clear:
                if st.button("✖ Clear Search", key="clear_manual_search"):
                    st.session_state.manual_search_result = None
                    st.session_state.manual_search_ticker = None
                    st.rerun()
            
            position_key = f"manual_{ticker}"
            add_key       = f"add_{position_key}"
            if add_key not in st.session_state:
                st.session_state[add_key] = False
            
            with st.expander(
                f"{direction_emoji} **{ticker}** — ${setup['price']:.2f} "
                f"| Score: {setup['score']}/6 "
                f"| {setup['setup_type']} "
                f"| R/R {setup['rr_ratio_r1']:.1f}:1",
                expanded=True   # Auto-open manual search result
            ):
                # TradingView chart
                tradingview_html = f"""
                <div class="tradingview-widget-container" style="height:600px; width:100%;">
                  <div class="tradingview-widget-container__widget" style="height:100%; width:100%;"></div>
                  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
                  {{
                  "width": "100%", "height": "600",
                  "symbol": "{ticker}",
                  "interval": "D", "timezone": "America/New_York",
                  "theme": "dark", "style": "1", "locale": "en",
                  "enable_publishing": false, "hide_top_toolbar": false,
                  "hide_legend": false, "allow_symbol_change": false,
                  "save_image": false, "calendar": false, "hide_volume": false,
                  "support_host": "https://www.tradingview.com"
                  }}
                  </script>
                </div>
                """
                st.components.v1.html(tradingview_html, height=620)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**Setup:** {setup['setup_type']}")
                    st.markdown(f"**Stock Price:** ${setup['price']:.2f}")
                    
                    if setup['score'] == 6:
                        strike_text = f"${setup['strike']} ATM"
                    elif setup['score'] == 5:
                        strike_text = f"${setup['strike']} (1% OTM)"
                    else:
                        strike_text = f"${setup['strike']} (2% OTM)"
                    
                    st.markdown(f"**Entry:** {setup['option_type']} {strike_text}")
                    st.markdown(f"**Recommended DTE:** {setup['recommended_dte']} days ({setup['dte_min']}-{setup['dte_max']} range)")
                    
                    st.markdown("---")
                    
                    rr1_color = "green" if setup['rr_ratio_r1'] >= 2 else "orange" if setup['rr_ratio_r1'] >= 1.5 else "red"
                    rr2_color = "green" if setup['rr_ratio_r2'] >= 2 else "orange" if setup['rr_ratio_r2'] >= 1.5 else "red"
                    
                    st.markdown(
                        f"**Target R1:** ${setup['target_r1']:.2f} "
                        f"(+${setup['reward_r1']:.2f} | "
                        f"<span style='color:{rr1_color}'>{setup['rr_ratio_r1']:.1f}:1 R/R</span>) → 75% exit",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"**Target R2:** ${setup['target_r2']:.2f} "
                        f"(+${setup['reward_r2']:.2f} | "
                        f"<span style='color:{rr2_color}'>{setup['rr_ratio_r2']:.1f}:1 R/R</span>) → 25% exit",
                        unsafe_allow_html=True
                    )
                    st.markdown(f"**Stop:** ${setup['stop']:.2f} (-${setup['risk']:.2f} | below structure)")
                    st.markdown(f"**RSI:** {setup['rsi']:.1f} <span class='{setup['rsi_color']}'>{setup['rsi_status']}</span> | **ATR:** ${setup['atr']:.2f}", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Confluence Checklist:**")
                    cb = setup['confluence_breakdown']
                    
                    # Pivot
                    if cb['pivot']['hit']:
                        st.markdown(f"✅ **Pivot:** {cb['pivot']['level']} @ ${cb['pivot']['price']:.2f}")
                        st.caption(f"   {cb['pivot'].get('reason','')}")
                    else:
                        st.markdown("⬜ Pivot level")
                    
                    # VRVP
                    if cb['vrvp']['hit']:
                        st.markdown(f"✅ **VRVP:** {cb['vrvp']['level']} @ ${cb['vrvp']['price']:.2f}")
                    else:
                        st.markdown("⬜ VRVP level")
                    
                    # EMA
                    if cb['ema_sma']['hit']:
                        st.markdown(f"✅ **EMA:** {cb['ema_sma']['direction']}")
                    else:
                        st.markdown("⬜ EMA alignment")
                    
                    # Prev week
                    if cb['prev_week']['hit']:
                        st.markdown(f"✅ **Prev Week:** {cb['prev_week']['level']} ({cb['prev_week']['distance']:.1f}%)")
                    else:
                        st.markdown("⬜ Prev week H/L")
                    
                    # RSI
                    if cb['rsi']['hit']:
                        st.markdown(f"✅ **RSI:** {cb['rsi']['value']:.0f} {cb['rsi']['status']}")
                    else:
                        st.markdown(f"⬜ RSI: {cb['rsi']['value']:.0f} {cb['rsi']['status']}")
                    
                    # EMA distance
                    if cb['price_distance']['hit']:
                        st.markdown(f"✅ **Near EMA20:** {cb['price_distance']['distance']:.1f}%")
                    else:
                        st.markdown("⬜ Near EMA20")
                    
                    st.markdown(f"**Score: {setup['score']}/6**")
                
                # ── ADD TO POSITIONS ──
                st.markdown("---")
                
                max_positions = 5
                current_positions = len(st.session_state.active_positions)
                already_tracked   = any(p['ticker'] == ticker for p in st.session_state.active_positions)
                
                if already_tracked:
                    st.info(f"📌 {ticker} is already in your active positions")
                elif current_positions >= max_positions:
                    st.warning(f"⚠️ Maximum {max_positions} positions reached")
                elif not st.session_state[add_key]:
                    if st.button(f"➕ Add {ticker} to Active Positions", key=f"add_btn_{position_key}", type="primary"):
                        st.session_state[add_key] = True
                        st.rerun()
                else:
                    st.markdown(f"### ➕ Add {ticker} to Active Positions")
                    
                    rr1_color = "green" if setup['rr_ratio_r1'] >= 2 else "orange"
                    st.markdown(
                        f"<div style='background:#1e1e2e; padding:12px; border-radius:8px; margin-bottom:12px;'>"
                        f"<b>{setup['option_type']} {ticker}</b> &nbsp;|&nbsp; "
                        f"Stock: <b>${setup['price']:.2f}</b> &nbsp;|&nbsp; "
                        f"R/R: <span style='color:{rr1_color}'><b>{setup['rr_ratio_r1']:.1f}:1</b></span> &nbsp;|&nbsp; "
                        f"Stop: <b>${setup['stop']:.2f}</b> &nbsp;|&nbsp; "
                        f"T1: <b>${setup['target_r1']:.2f}</b> &nbsp;|&nbsp; "
                        f"T2: <b>${setup['target_r2']:.2f}</b>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    
                    with st.form(key=f"form_{position_key}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            entry_date   = st.date_input("Entry Date", value=datetime.now())
                            strike_input = st.number_input(
                                "Strike Price",
                                min_value=1.0,
                                value=float(setup['strike']),
                                step=0.50 if setup['price'] < 20 else (1.0 if setup['price'] < 100 else 5.0),
                                help=f"Suggested: ${setup['strike']} ({setup['option_type']}, {setup['score']}/6 score)"
                            )
                        with col2:
                            min_exp     = datetime.now() + timedelta(days=setup['dte_min'])
                            default_exp = datetime.now() + timedelta(days=setup['recommended_dte'])
                            exp_date_input = st.date_input(
                                f"Expiration Date (Rec: {setup['recommended_dte']} DTE)",
                                value=default_exp,
                                min_value=min_exp,
                                help=f"Recommended {setup['dte_min']}-{setup['dte_max']} DTE"
                            )
                            contracts_input = st.number_input("Contracts", min_value=1, max_value=10, value=1)
                        
                        dte_calculated = (exp_date_input - entry_date).days
                        st.caption(f"DTE at Entry: {dte_calculated} days")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            cancel_btn = st.form_submit_button("Cancel")
                        with col2:
                            submit_btn = st.form_submit_button("Add Position", type="primary")
                        
                        if cancel_btn:
                            st.session_state[add_key] = False
                            st.rerun()
                        
                        if submit_btn:
                            if dte_calculated < 7:
                                st.error("⚠️ Expiration must be at least 7 days from entry")
                            elif dte_calculated > 365:
                                st.error("⚠️ Expiration must be within 1 year")
                            else:
                                new_position = {
                                    'ticker':          ticker,
                                    'strike':          strike_input,
                                    'option_type':     setup['option_type'],
                                    'entry_date':      entry_date.isoformat(),
                                    'expiration_date': exp_date_input.isoformat(),
                                    'entry_price':     setup['price'],
                                    'dte_at_entry':    dte_calculated,
                                    'contracts':       contracts_input,
                                    'setup_type':      setup['setup_type'],
                                    'target_r1':       setup['target_r1'],
                                    'target_r2':       setup['target_r2'],
                                    'stop':            setup['stop'],
                                    'risk':            setup['risk'],
                                    'reward_r1':       setup['reward_r1'],
                                    'reward_r2':       setup['reward_r2'],
                                    'rr_ratio_r1':     setup['rr_ratio_r1'],
                                    'rr_ratio_r2':     setup['rr_ratio_r2'],
                                    'manual_search':   True
                                }
                                st.session_state.active_positions.append(new_position)
                                save_positions(st.session_state.active_positions)
                                st.session_state[add_key] = False
                                st.success(f"✅ Added {ticker} ${strike_input} {setup['option_type']} | R/R {setup['rr_ratio_r1']:.1f}:1")
                                time.sleep(1)
                                st.rerun()
        
        st.markdown("---")
    
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
                        st.markdown(f"**Stock Price:** ${setup['price']:.2f}")
                        
                        # Strike display based on score
                        if setup['score'] == 6:
                            strike_text = f"${setup['strike']} ATM"
                        elif setup['score'] == 5:
                            strike_text = f"${setup['strike']} (1% OTM)"
                        else:
                            strike_text = f"${setup['strike']} (2% OTM)"
                        
                        st.markdown(f"**Entry:** {setup['option_type']} {strike_text}")
                        st.markdown(f"**Recommended DTE:** {setup['recommended_dte']} days ({setup['dte_min']}-{setup['dte_max']} range)")
                        
                        st.markdown("---")
                        
                        # Targets with R/R
                        rr1_color = "green" if setup['rr_ratio_r1'] >= 2 else "orange" if setup['rr_ratio_r1'] >= 1.5 else "red"
                        rr2_color = "green" if setup['rr_ratio_r2'] >= 2 else "orange" if setup['rr_ratio_r2'] >= 1.5 else "red"
                        
                        st.markdown(
                            f"**Target R1:** ${setup['target_r1']:.2f} "
                            f"(+${setup['reward_r1']:.2f} | "
                            f"<span style='color:{rr1_color}'>{setup['rr_ratio_r1']:.1f}:1 R/R</span>) "
                            f"→ 75% exit",
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f"**Target R2:** ${setup['target_r2']:.2f} "
                            f"(+${setup['reward_r2']:.2f} | "
                            f"<span style='color:{rr2_color}'>{setup['rr_ratio_r2']:.1f}:1 R/R</span>) "
                            f"→ 25% exit",
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f"**Stop:** ${setup['stop']:.2f} "
                            f"(-${setup['risk']:.2f} | below structure)"
                        )
                        
                        st.markdown(f"**RSI:** {setup['rsi']:.1f} <span class='{setup['rsi_color']}'>{setup['rsi_status']}</span> | **ATR:** ${setup['atr']:.2f}", unsafe_allow_html=True)
                    
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
                                
                                # Strike price input with suggestion
                                strike_input = st.number_input(
                                    "Strike Price",
                                    min_value=1.0,
                                    value=float(setup['strike']),
                                    step=0.50 if setup['price'] < 20 else (1.0 if setup['price'] < 100 else 5.0),
                                    help=f"Suggested: ${setup['strike']} ({setup['option_type']} based on {setup['score']}/6 score)"
                                )
                            with col2:
                                # Expiration date input
                                min_exp     = datetime.now() + timedelta(days=setup['dte_min'])
                                default_exp = datetime.now() + timedelta(days=setup['recommended_dte'])
                                
                                exp_date_input = st.date_input(
                                    f"Expiration Date (Rec: {setup['recommended_dte']} DTE)",
                                    value=default_exp,
                                    min_value=min_exp,
                                    help=f"Recommended {setup['dte_min']}-{setup['dte_max']} DTE for {setup['setup_type']}"
                                )
                                
                                contracts_input = st.number_input("Number of Contracts", min_value=1, max_value=10, value=1)
                            
                            # Calculate DTE from expiration date
                            dte_calculated = (exp_date_input - entry_date).days
                            st.caption(f"DTE at Entry: {dte_calculated} days")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                cancel_btn = st.form_submit_button("Cancel")
                            with col2:
                                submit_btn = st.form_submit_button("Add Position", type="primary")
                            
                            if cancel_btn:
                                st.session_state[add_key] = False
                                st.rerun()
                            
                            if submit_btn:
                                # Validation
                                if dte_calculated < 7:
                                    st.error("⚠️ Expiration must be at least 7 days from entry date")
                                elif dte_calculated > 365:
                                    st.error("⚠️ Expiration must be within 1 year")
                                else:
                                    new_position = {
                                        'ticker':           setup['ticker'],
                                        'strike':           strike_input,
                                        'option_type':      setup['option_type'],
                                        'entry_date':       entry_date.isoformat(),
                                        'expiration_date':  exp_date_input.isoformat(),
                                        'entry_price':      setup['price'],
                                        'dte_at_entry':     dte_calculated,
                                        'contracts':        contracts_input,
                                        'setup_type':       setup['setup_type'],
                                        'target_r1':        setup['target_r1'],
                                        'target_r2':        setup['target_r2'],
                                        'stop':             setup['stop'],
                                        'risk':             setup['risk'],
                                        'reward_r1':        setup['reward_r1'],
                                        'reward_r2':        setup['reward_r2'],
                                        'rr_ratio_r1':      setup['rr_ratio_r1'],
                                        'rr_ratio_r2':      setup['rr_ratio_r2'],
                                    }
                                    st.session_state.active_positions.append(new_position)
                                    save_positions(st.session_state.active_positions)
                                    st.session_state[add_key] = False
                                    st.success(f"✅ Added {setup['ticker']} ${strike_input} {setup['option_type']} | R/R {setup['rr_ratio_r1']:.1f}:1")
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
