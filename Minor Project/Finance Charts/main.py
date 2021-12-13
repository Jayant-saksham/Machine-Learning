import streamlit as st 
import yfinance as yf
import pandas as pd 
st.title("Finance DashBoard")

companies = (
    'TSLA', 'AAPL', 'MSFT', 'BTC-USD', 'ETH-USD'
)
# Tesla, Apple, Microsoft, Bitcoin, Etherium
dropdown = st.multiselect("Pick your assets", companies)

start_date = st.date_input("Start", value = pd.to_datetime("2021-01-01"))
end_date = st.date_input("End", value = pd.to_datetime("today"))

def relative(df):
    rel = df.pct_change()
    # precent change 
    return rel

if len(dropdown) > 0:
    df =   (yf.download(dropdown, start_date, end_date)['Adj Close'])
    st.header(f"Returns of {dropdown}")
    st.line_chart(df)
