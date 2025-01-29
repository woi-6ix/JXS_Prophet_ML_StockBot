## Import Packages ##

import yfinance as yf
import streamlit as st
from prophet import Prophet     
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

## Defining Functions ##

def get_stock_data(ticker, start_date='1998-01-01'):                  
    stock_df = yf.download(ticker, start=start_date)                  
    stock_df['MA_50'] = stock_df['Close'].rolling(window=50).mean()   
    stock_df['MA_100'] = stock_df['Close'].rolling(window=100).mean()
    stock_df['MA_200'] = stock_df['Close'].rolling(window=200).mean()  
    return stock_df.dropna()

def prepare_prophet_stock_data(stock_df):          
    prophet_df = stock_df[['Close']].reset_index() 
    prophet_df.columns = ['ds', 'y']               
    return prophet_df

def create_candlestick_chart(stock_df):
    fig = go.Figure(data=[go.Candlestick(
        x=stock_df.index,
        open=stock_df['Open'],
        high=stock_df['High'],
        low=stock_df['Low'],
        close=stock_df['Close'],
        increasing_line_color='purple',
        decreasing_line_color='gray'
    )])
    fig.update_layout(
        title=f'{ticker} Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        plot_bgcolor='black',
        paper_bgcolor='black'
    )
    return fig

def main():
    st.set_page_config(
        page_title="JXS Prophet Stock Prediction App",
        page_icon="📈",
        layout="wide"
    )

    # Custom CSS for theme and centering
    st.markdown("""
    <style>
    .stApp {
        background-color: black;
        color: white;
    }
    h1, h2, h3, h4, h5, h6 {
        color: purple;
    }
    .stButton>button {
        background-color: purple;
        color: white;
        border-radius: 5px;
        border: 1px solid purple;
    }
    .stTextInput>div>div>input {
        background-color: black;
        color: white;
        border: 1px solid purple;
    }
    .stSlider>div>div>div>div {
        background-color: purple;
    }
    .centered-header {
        text-align: center !important;
    }
    .dataframe-container {
        display: flex;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("JXS Prophet Stock Prediction App")
    
    # Introductory content
    st.write("""
    ### How the Model Works
    [Keep existing content]
    """)
    
    st.warning("""
    ### Disclaimer
    [Keep existing content]
    """)
    
    # User inputs
    ticker = st.text_input('Enter Stock Ticker (e.g., XYZ):', 'XYZ')
    prediction_days = st.slider('Select number of days to predict (0-730):', 0, 730, 365)
    
    if st.button('Predict'):
        with st.spinner('Fetching data and making predictions...'):
            df = get_stock_data(ticker)
            
            # Centered table title and content
            st.markdown(f"<h3 class='centered-header'>{ticker} Raw Stock Data</h3>", unsafe_allow_html=True)
            st.write(f"<div style='text-align: center;'>Data from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}</div>", unsafe_allow_html=True)
            
            # Centered dataframe
            col1, col2, col3 = st.columns([1, 8, 1])
            with col2:
                st.dataframe(df.reset_index().rename(columns={'Date': 'Dates'}))
            
            # Candlestick chart
            st.markdown(f"<h3 class='centered-header'>{ticker} Candlestick Chart</h3>", unsafe_allow_html=True)
            st.plotly_chart(create_candlestick_chart(df), use_container_width=True)
            
            # Rest of prediction code remains the same...
            # [Keep existing Prophet code]
            
            # Centered combined table
            st.markdown("<h3 class='centered-header'>Combined Historical and Predicted Values</h3>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 8, 1])
            with col2:
                combined_df = pd.concat([...])  # Keep existing dataframe code
                st.dataframe(combined_df)
            
            st.success('Prediction completed!')

if __name__ == '__main__':
    main()
