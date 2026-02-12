import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import List, Dict
import time

st.set_page_config(page_title="Swing Trade Screener", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; margin-bottom: 1rem; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    .bullish { color: #00c853; font-weight: bold; }
    .bearish { color: #ff1744; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class SwingScreener:
    def __init__(self, min_market_cap=2e9, min_volume=500000, confluence_threshold=4):
        self.min_market_cap = min_market_cap
        self.min_volume = min_volume
        self.confluence_threshold = confluence_threshold
    
    def get_market_tickers(self):
        sp500 = ['AAPL','MSFT','NVDA','GOOGL','GOOG','META','AVGO','ORCL','CSCO','ADBE','CRM','AMD','INTC','IBM','QCOM','TXN','AMAT','MU','LRCX','KLAC','SNPS','CDNS','MCHP','ADI','NXPI','PANW','PLTR','NOW','TEAM','WDAY','AMZN','TSLA','HD','MCD','NKE','SBUX','LOW','TJX','BKNG','ABNB','GM','F','MAR','CMG','ORLY','YUM','DHI','LEN','DG','ROST','UNH','JNJ','LLY','ABBV','MRK','TMO','ABT','DHR','PFE','BMY','AMGN','GILD','CVS','CI','ISRG','REGN','VRTX','ZTS','HCA','BSX','JPM','V','MA','BAC','WFC','GS','MS','BLK','SPGI','C','SCHW','AXP','CB','PGR','MMC','ICE','CME','AON','USB','TFC','NFLX','DIS','CMCSA','T','VZ','TMUS','CHTR','EA','TTWO','MTCH','WMT','PG','COST','KO','PEP','PM','MO','MDLZ','CL','KMB','GIS','K','HSY','SYY','KHC','STZ','TAP','CPB','CAG','SJM','BA','CAT','UNP','HON','UPS','RTX','LMT','GE','MMM','DE','GD','NOC','FDX','NSC','CSX','EMR','ETN','ITW','PH','CMI','XOM','CVX','COP','SLB','EOG','MPC','PSX','VLO','OXY','HAL','KMI','WMB','DVN','HES','FANG','BKR','TRGP','OKE','APA','MRO','LIN','APD','SHW','ECL','NEM','FCX','DOW','DD','PPG','NUE','NEE','SO','DUK','D','AEP','EXC','SRE','XEL','PEG','ED','PLD','AMT','CCI','EQIX','PSA','O','WELL','DLR','SBAC','AVB']
        nasdaq = ['INTU','BKNG','ADP','VRTX','SBUX','GILD','ADI','REGN','LRCX','MDLZ','PANW','MU','PYPL','KLAC','SNPS','CDNS','MELI','ASML','ABNB','CHTR','CTAS','MAR','ORLY','AZN','CRWD','FTNT','CSX','NXPI','PCAR','MRVL','MNST','WDAY','ADSK','DASH','CPRT','AEP','PAYX','ROST','ODFL','KDP','FAST','VRSK','CTSH','EA','GEHC','LULU','DDOG','IDXX','XEL','EXC','CCEP','ON','TEAM','ANSS','FANG','CSGP','ZS','DXCM','TTWO','BIIB','ILMN','WBD','MDB','GFS','ZM','MRNA']
        active = ['SPY','QQQ','IWM','SMH','XLF','XLE','XLK','XLV','XLI','XLP','SOFI','PLTR','NIO','RIVN','LCID','COIN','RBLX','U','SNOW','NET','SHOP','SQ','UBER','LYFT','PINS','SNAP','TWLO','ROKU','Z','CVNA','ARKK','ARKG','ARKF','GME','AMC','BB','NOK','PLUG','RIOT','MARA']
        return list(set(sp500 + nasdaq + active))
    
    def calculate_pivot_points(self, high, low, close):
        pivot = (high + low + close) / 3
        return {'PP': pivot, 'R1': (2*pivot)-low, 'R2': pivot+(high-low), 'S1': (2*pivot)-high, 'S2': pivot-(high-low)}
    
    def calculate_vrvp_levels(self, df, days=30):
        try:
            recent = df.tail(days).copy()
            bins = np.linspace(recent['Low'].min(), recent['High'].max(), 50)
            volume_profile = []
            for i in range(len(bins)-1):
                mask = (recent['Close'] >= bins[i]) & (recent['Close'] < bins[i+1])
                volume_profile.append((bins[i], recent.loc[mask, 'Volume'].sum()))
            poc = max(volume_profile, key=lambda x: x[1])[0] if volume_profile else recent['Close'].iloc[-1]
            total_vol = sum(v for _, v in volume_profile)
            cumulative, val, vah = 0, poc, poc
            for price, vol in sorted(volume_profile, key=lambda x: x[0]):
                cumulative += vol
                if cumulative < total_vol * 0.15:
                    val = price
                elif cumulative > total_vol * 0.85:
                    vah = price
                    break
            return {'POC': poc, 'VAH': vah, 'VAL': val}
        except:
            return {'POC': 0, 'VAH': 0, 'VAL': 0}
    
    def calculate_atr(self, df, period=14):
        high, low, close = df['High'], df['Low'], df['Close'].shift(1)
        tr = pd.concat([high-low, abs(high-close), abs(low-close)], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        return atr if not pd.isna(atr) else 0
    
    def check_confluence(self, ticker, data):
        try:
            if len(data) < 50:
                return None
            current_price = data['Close'].iloc[-1]
            weekly = data.resample('W').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
            if len(weekly) < 2:
                return None
            prev_week = weekly.iloc[-2]
            pivots = self.calculate_pivot_points(prev_week['High'], prev_week['Low'], prev_week['Close'])
            vrvp = self.calculate_vrvp_levels(data, days=30)
            ema_20 = data['Close'].ewm(span=20).mean().iloc[-1]
            ema_50 = data['Close'].ewm(span=50).mean().iloc[-1]
            atr = self.calculate_atr(data)
            avg_volume = data['Volume'].tail(20).mean()
            volume_ratio = data['Volume'].iloc[-1] / avg_volume if avg_volume > 0 else 0
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = 100 - (100 / (1 + gain/loss))
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            score, factors = 0, []
            for name, price in pivots.items():
                if abs(current_price - price) / current_price < 0.02:
                    score += 1
                    factors.append(f"Near {name}")
                    break
            for name, price in vrvp.items():
                if price > 0 and abs(current_price - price) / current_price < 0.02:
                    score += 1
                    factors.append(f"VRVP {name}")
                    break
            if current_price > ema_20 > ema_50:
                score += 1
                factors.append("Bullish EMA")
            elif current_price < ema_20 < ema_50:
                score += 1
                factors.append("Bearish EMA")
            if volume_ratio > 1.5:
                score += 1
                factors.append(f"Volume {volume_ratio:.1f}x")
            if abs(current_price - prev_week['High']) / current_price < 0.03:
                score += 1
                factors.append("Prev week high")
            elif abs(current_price - prev_week['Low']) / current_price < 0.03:
                score += 1
                factors.append("Prev week low")
            if 30 < current_rsi < 70:
                score += 1
                factors.append(f"RSI {current_rsi:.0f}")
            setup_type = "Range-bound"
            if current_price > pivots['R1'] and current_price > ema_20:
                setup_type = "Bullish breakout"
            elif current_price < pivots['S1'] and current_price < ema_20:
                setup_type = "Bearish breakdown"
            if "Bullish" in setup_type:
                target = pivots['R2'] if current_price < pivots['R2'] else vrvp['VAH']
                stop = current_price - (2*atr)
                option_type, strike = "CALL", round(current_price * 1.02)
            else:
                target = pivots['S2'] if current_price > pivots['S2'] else vrvp['VAL']
                stop = current_price + (2*atr)
                option_type, strike = "PUT", round(current_price * 0.98)
            if score < self.confluence_threshold:
                return None
            return {'ticker':ticker,'price':current_price,'score':score,'setup_type':setup_type,'factors':factors,'option_type':option_type,'strike':strike,'target':target,'stop':stop,'atr':atr,'rsi':current_rsi,'volume_ratio':volume_ratio}
        except:
            return None
    
    def scan_market(self, progress_callback=None):
        tickers = self.get_market_tickers()
        results = []
        for i, ticker in enumerate(tickers):
            try:
                if progress_callback:
                    progress_callback(i+1, len(tickers), ticker)
                stock = yf.Ticker(ticker)
                data = stock.history(period='6mo', interval='1d')
                if data.empty or len(data) < 50:
                    continue
                info = stock.info
                if info.get('marketCap', 0) < self.min_market_cap or data['Volume'].tail(20).mean() < self.min_volume:
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
    st.markdown("**AI-powered confluence scanner for swing trading opportunities**")
    st.sidebar.header("⚙️ Settings")
    confluence_threshold = st.sidebar.slider("Minimum Confluence Score", 3, 6, 4)
    min_market_cap = st.sidebar.selectbox("Minimum Market Cap", [500e6,1e9,2e9,5e9,10e9], index=2, format_func=lambda x: f"${x/1e9:.1f}B")
    min_volume = st.sidebar.selectbox("Minimum Daily Volume", [100000,250000,500000,1000000], index=2, format_func=lambda x: f"{x/1000:.0f}K")
    slack_webhook = st.sidebar.text_input("Slack Webhook URL (optional)", type="password")
    scan_button = st.button("🚀 Run Market Scan", type="primary", use_container_width=True)
    
    if scan_button:
        screener = SwingScreener(min_market_cap, min_volume, confluence_threshold)
        progress_bar = st.progress(0)
        status_text = st.empty()
        def update_progress(current, total, ticker):
            progress_bar.progress(current/total)
            status_text.text(f"Scanning {ticker}... ({current}/{total})")
        with st.spinner("Initializing..."):
            results = screener.scan_market(update_progress)
        progress_bar.empty()
        status_text.empty()
        
        if results:
            st.success(f"✅ Found {len(results)} setups!")
            col1,col2,col3,col4 = st.columns(4)
            with col1:
                st.metric("Total", len(results))
            with col2:
                st.metric("Bullish", sum(1 for r in results if "Bullish" in r['setup_type']))
            with col3:
                st.metric("Bearish", sum(1 for r in results if "Bearish" in r['setup_type']))
            with col4:
                st.metric("Avg Score", f"{sum(r['score'] for r in results)/len(results):.1f}/6")
            st.markdown("---")
            st.subheader("🔍 Filter Results")
            col1,col2 = st.columns(2)
            with col1:
                setup_filter = st.multiselect("Setup Type", list(set(r['setup_type'] for r in results)), default=list(set(r['setup_type'] for r in results)))
            with col2:
                min_score = st.slider("Min Score", 3, 6, 4)
            filtered = [r for r in results if r['setup_type'] in setup_filter and r['score'] >= min_score]
            st.info(f"Showing {len(filtered)} of {len(results)} setups")
            for i, s in enumerate(filtered[:50], 1):
                with st.expander(f"{'🟢' if 'Bullish' in s['setup_type'] else '🔴'} **{i}. {s['ticker']}** - ${s['price']:.2f} (Score: {s['score']}/6)", expanded=(i<=10)):
                    col1,col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Setup:** {s['setup_type']}")
                        st.markdown(f"**Entry:** {s['option_type']} ${s['strike']}")
                        st.markdown(f"**Target:** ${s['target']:.2f}")
                        st.markdown(f"**Stop:** ${s['stop']:.2f}")
                    with col2:
                        st.markdown(f"**RSI:** {s['rsi']:.1f}")
                        st.markdown(f"**Volume:** {s['volume_ratio']:.1f}x")
                        st.markdown(f"**ATR:** ${s['atr']:.2f}")
                    st.markdown(f"**Factors:** {', '.join(s['factors'])}")
            csv = pd.DataFrame(filtered).to_csv(index=False)
            st.download_button("📥 Download CSV", csv, f"swing_setups_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
            if slack_webhook and st.button("📱 Send to Slack"):
                with st.spinner("Sending..."):
                    try:
                        for start in [0,10,20]:
                            batch = filtered[start:start+10]
                            if not batch:
                                break
                            blocks = [{"type":"header","text":{"type":"plain_text","text":f"📊 Setups {start+1}-{start+len(batch)}"}}]
                            for j, s in enumerate(batch, start+1):
                                emoji = "🟢" if "Bullish" in s['setup_type'] else "🔴"
                                blocks.append({"type":"section","text":{"type":"mrkdwn","text":f"{emoji} *{j}. {s['ticker']}* - ${s['price']:.2f}\n*Entry:* {s['option_type']} ${s['strike']}\n*Target:* ${s['target']:.2f} | *Stop:* ${s['stop']:.2f}"}})
                                blocks.append({"type":"divider"})
                            requests.post(slack_webhook, json={"blocks":blocks})
                            time.sleep(1)
                        st.success("✅ Sent!")
                    except Exception as e:
                        st.error(f"❌ Failed: {e}")
        else:
            st.warning("No setups found. Try lowering confluence threshold.")
    
    with st.expander("ℹ️ How It Works"):
        st.markdown("""
        ### Confluence Factors:
        1. **Pivot Levels** - R1, R2, S1, S2
        2. **VRVP** - POC, VAH, VAL
        3. **EMAs** - 9, 20, 50 alignment
        4. **Volume** - Spike vs 20-day avg
        5. **Previous Week H/L**
        6. **RSI** - Neutral zone (30-70)
        
        **Min 4 factors required**
        """)

if __name__ == "__main__":
    main()
