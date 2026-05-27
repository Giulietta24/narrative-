import streamlit as st
import yfinance as yf
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px

st.set_page_config(page_title="Narrative Sentiment Radar", layout="wide")

st.title("📈 Multi-Asset Narrative Sentiment Radar")
st.write("Cross-referencing real-time financial headlines with 30-day technical trends.")

# Initialize VADER
@st.cache_resource
def load_analyzer():
    return SentimentIntensityAnalyzer()

analyzer = load_analyzer()

# ------------------------------------
# FETCH REAL-TIME NEWS & SENTIMENT
# ------------------------------------
def get_ticker_sentiment(ticker_symbol):
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        news_list = ticker_obj.news
        
        if not news_list:
            return 0.0, "NEUTRAL", "No news available"
        
        scores = []
        headlines = []
        
        # Take the top 5 recent headlines
        for item in news_list[:5]:
            title = item.get('title', '')
            if title:
                headlines.append(title)
                vs = analyzer.polarity_scores(title)
                scores.append(vs['compound'])
                
        if not scores:
            return 0.0, "NEUTRAL", "No text found"
            
        avg_score = sum(scores) / len(scores)
        
        if avg_score >= 0.05:
            sentiment_label = "POSITIVE"
        elif avg_score <= -0.05:
            sentiment_label = "NEGATIVE"
        else:
            sentiment_label = "NEUTRAL"
            
        sample_headline = headlines[0] if headlines else "N/A"
        return round(avg_score, 2), sentiment_label, sample_headline
        
    except Exception:
        return 0.0, "NEUTRAL", "API Error fetching news"

# ------------------------------------
# FETCH PRICE HISTORY
# ------------------------------------
def get_price_change(ticker_symbol):
    try:
        df = yf.download(ticker_symbol, period="2mo", interval="1d", group_by="column", progress=False)
        if df is None or df.empty:
            return None
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        close = df["Close"]
        if len(close) < 31:
            return None
            
        last = float(close.iloc[-1])
        last_30 = float(close.iloc[-30])
        
        return round(((last - last_30) / last_30) * 100, 2)
    except Exception:
        return None

# ------------------------------------
# MAIN ENGINE RUN
# ------------------------------------
watch_list = ["IWM", "IJR", "FCIT.L", "SMEU.L", "AAPL", "NVDA", "TSLA"]

if st.button("🚀 Run Narrative Radar Scan"):
    results = []
    
    with st.spinner("Scanning global markets and calculating sentiment..."):
        for ticker in watch_list:
            price_change = get_price_change(ticker)
            if price_change is None:
                continue
                
            sentiment_score, sentiment_label, top_news = get_ticker_sentiment(ticker)
            
            # Formulate Advice Category
            if sentiment_score >= 0.05 and price_change > 0:
                action = "🚀 Momentum Leader (Buy/Hold)"
            elif sentiment_score >= 0.05 and price_change <= 0:
                action = "🔍 Bullish Divergence (Watch for Dip)"
            elif sentiment_score < -0.05 and price_change > 0:
                action = "⚠️ Bearish Divergence (Take Profit/Risk)"
            else:
                action = "📉 Value Trap / Weak (Avoid)"
                
            results.append({
                "Ticker": ticker,
                "30d Price Change (%)": price_change,
                "Sentiment Score (-1 to +1)": sentiment_score,
                "Narrative Status": sentiment_label,
                "Strategy / Action": action,
                "Latest Headline": top_news
            })
            
    if results:
        df_results = pd.DataFrame(results)
        
        # Display Summary Dashboard Metrics
        st.write("### 📌 Tactical Market Screener")
        st.dataframe(df_results, use_container_width=True)
        
        # Data Visualization Map
        st.write("### 🗺️ Narrative Matrix Visualization")
        fig = px.scatter(
            df_results, 
            x="Sentiment Score (-1 to +1)", 
            y="30d Price Change (%)",
            text="Ticker",
            color="Strategy / Action",
            title="Where do your Tickers sit?",
            labels={"x": "News Sentiment", "y": "30-Day Trend (%)"}
        )
        fig.update_traces(textposition='top center', marker=dict(size=12))
        # Add baseline crosshairs
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("Could not fetch data for any assets in the watch list.")