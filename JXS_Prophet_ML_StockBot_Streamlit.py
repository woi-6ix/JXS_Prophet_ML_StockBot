## Import Packages ##

import yfinance as yf
import streamlit as st
from prophet import Prophet     
import pandas as pd
import matplotlib.pyplot as plt

## Defining Functions ##

# Downloading Data From yFinance #
def get_stock_data(ticker, start_date='1998-01-01'):                  
    
    stock_df = yf.download(ticker, start=start_date)                  
    stock_df['MA_50'] = stock_df['Close'].rolling(window=50).mean()   
    stock_df['MA_100'] = stock_df['Close'].rolling(window=100).mean()
    stock_df['MA_200'] = stock_df['Close'].rolling(window=200).mean()  
    stock_df = stock_df.dropna()                                      
    return stock_df

# Prepping Stock Data Frame into Prophet Data Frame #
def prepare_prophet_stock_data(stock_df):          
    prophet_df = stock_df[['Close']].reset_index() 
    prophet_df.columns = ['ds', 'y']               
    return prophet_df

# Main Function #
def main():
    st.title("JXS Prophet Stock Prediction App")
    ticker = st.text_input('Enter Stock Ticker (e.g., XYZ):', 'XYZ')
    
    if st.button('Predict'):
        with st.spinner('Fetching data and making predictions...'):
            # Get data and calculate moving averages
            df = get_stock_data(ticker)
            
            # Display Stock Data section
            st.subheader(f"{ticker} Raw Stock Data")
            st.write(f"Data from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            
            # Modified dataframe with renamed date column
            display_df = df.reset_index().rename(columns={'Date': 'Dates'})
            st.dataframe(display_df)
            
            # Prepare Prophet data
            prophet_df = prepare_prophet_stock_data(df)
            
            # Create and fit model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.5
            )
            model.fit(prophet_df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=365, freq='B')
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Create plots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Modified historical data plot title
            ax1.plot(df['Close'], label='Close Price')
            ax1.plot(df['MA_50'], label='50-day MA')
            ax1.plot(df['MA_100'], label='100-day MA')
            ax1.plot(df['MA_200'], label='200-day MA')
            ax1.set_title(f'{ticker} Historical Prices & Moving Averages')
            ax1.legend()
            
            # Prophet forecast plot
            model.plot(forecast, ax=ax2)
            ax2.set_title(f'{ticker} 365-Day Price Prediction')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Price')
            
            st.pyplot(fig)
            
            # Show components
            st.subheader('Forecast Components')
            components_fig = model.plot_components(forecast)
            st.pyplot(components_fig)
            
            st.success('Prediction completed!')

if __name__ == '__main__':
    main()
