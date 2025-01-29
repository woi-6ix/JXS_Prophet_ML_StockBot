## Import Packages ##

#import yfinance as yf
#import streamlit as st
#from prophet import Prophet     
#import pandas as pd
#import matplotlib.pyplot as plt

## Defining Functions ##

# Downloading Data From yFinance #
#def get_stock_data(ticker, start_date='1998-01-01'):                  
    
#    stock_df = yf.download(ticker, start=start_date)                  
#    stock_df['MA_50'] = stock_df['Close'].rolling(window=50).mean()   
#    stock_df['MA_100'] = stock_df['Close'].rolling(window=100).mean()
#    stock_df['MA_200'] = stock_df['Close'].rolling(window=200).mean()  
#    stock_df = stock_df.dropna()                                      
#    return stock_df

# Prepping Stock Data Frame into Prophet Data Frame #
#def prepare_prophet_stock_data(stock_df):          
#    prophet_df = stock_df[['Close']].reset_index() 
#    prophet_df.columns = ['ds', 'y']               
#    return prophet_df

# Main Function #
#def main():
#    st.title("JXS Prophet Stock Prediction App")
#    ticker = st.text_input('Enter Stock Ticker (e.g., XYZ):', 'XYZ')
    
#    if st.button('Predict'):
#        with st.spinner('Fetching data and making predictions...'):
            # Get data and calculate moving averages
#            df = get_stock_data(ticker)
            
            # Display Stock Data section
#            st.subheader(f"{ticker} Raw Stock Data")
#            st.write(f"Data from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            
            # Modified dataframe with renamed date column
#            display_df = df.reset_index().rename(columns={'Date': 'Dates'})
#            st.dataframe(display_df)
            
            # Prepare Prophet data
#            prophet_df = prepare_prophet_stock_data(df)
            
            # Create and fit model
#            model = Prophet(
#                yearly_seasonality=True,
#                weekly_seasonality=False,
#                daily_seasonality=False,
#                changepoint_prior_scale=0.5
#            )
#            model.fit(prophet_df)
            
            # Create future dataframe
#            future = model.make_future_dataframe(periods=365, freq='B')
            
            # Generate forecast
#            forecast = model.predict(future)
            
            # Create plots
#            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Modified historical data plot title
#            ax1.plot(df['Close'], label='Close Price')
#            ax1.plot(df['MA_50'], label='50-day MA')
#            ax1.plot(df['MA_100'], label='100-day MA')
#            ax1.plot(df['MA_200'], label='200-day MA')
#            ax1.set_title(f'{ticker} Historical Prices & Moving Averages')
#            ax1.legend()
            
            # Prophet forecast plot
#            model.plot(forecast, ax=ax2)
#            ax2.set_title(f'{ticker} 365-Day Price Prediction')
#            ax2.set_xlabel('Date')
#            ax2.set_ylabel('Price')
            
#            st.pyplot(fig)
            
            # Show components
#            st.subheader('Forecast Components')
#            components_fig = model.plot_components(forecast)
#            st.pyplot(components_fig)
            
#            st.success('Prediction completed!')

#if __name__ == '__main__':
#    main()



## Import Packages ##
import yfinance as yf
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objs as go
from datetime import date, timedelta

## Defining Functions ##
def get_stock_data(ticker, start_date='1998-01-01'):
    stock_df = yf.download(ticker, start=start_date)
    stock_df['MA_50'] = stock_df['Close'].rolling(window=50).mean()
    stock_df['MA_100'] = stock_df['Close'].rolling(window=100).mean()
    stock_df['MA_200'] = stock_df['Close'].rolling(window=200).mean()
    stock_df = stock_df.dropna()
    return stock_df

def prepare_prophet_stock_data(stock_df):
    prophet_df = stock_df[['Close']].reset_index()
    prophet_df.columns = ['ds', 'y']
    return prophet_df

def calculate_metrics(actual, predicted):
    return {
        'RMSE': round(mean_squared_error(actual, predicted, squared=False), 2),
        'MAE': round(mean_absolute_error(actual, predicted), 2),
        'R²': round(r2_score(actual, predicted), 2)
    }

# Main Function #
def main():
    st.title("JXS Prophet Stock Prediction App")
    ticker = st.text_input('Enter Stock Ticker (e.g., AAPL):', 'AAPL').upper()
    
    # Date range slider
    prediction_days = st.slider('Select prediction days:', 1, 730, 365)
    
    if st.button('Predict'):
        with st.spinner('Fetching data and making predictions...'):
            try:
                df = get_stock_data(ticker)
                
                # Display formatted stock data
                st.subheader(f"{ticker} Historical Stock Data")
                display_df = df.reset_index().rename(columns={
                    'Date': 'Date', 'Open': 'Open', 'High': 'High',
                    'Low': 'Low', 'Close': 'Close', 'Adj Close': 'Adj Close',
                    'Volume': 'Volume'
                })
                st.dataframe(display_df.style.format({
                    'Open': '{:.2f}', 'High': '{:.2f}', 'Low': '{:.2f}',
                    'Close': '{:.2f}', 'Adj Close': '{:.2f}', 'Volume': '{:,.0f}'
                }))

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
                future = model.make_future_dataframe(periods=prediction_days, freq='B')
                forecast = model.predict(future)
                
                # Merge actual and predicted values
                merged_df = pd.merge(prophet_df, forecast[['ds', 'yhat']], on='ds')
                metrics = calculate_metrics(merged_df['y'], merged_df['yhat'])
                
                # Create interactive plot
                st.subheader(f"{ticker} Price Prediction with Historical Data")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Historical Price'))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted Price'))
                fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Price',
                    xaxis_rangeslider_visible=True,
                    hovermode='x unified'
                )
                st.plotly_chart(fig)
                
                # Metrics display
                st.subheader("Model Performance Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("RMSE", metrics['RMSE'])
                col2.metric("MAE", metrics['MAE'])
                col3.metric("R² Score", metrics['R²'])
                
                # Trend analysis and news
                st.subheader("Trend Analysis & News")
                trend = "Bullish" if forecast['yhat'].iloc[-1] > df['Close'].iloc[-1] else "Bearish"
                analysis = f"""
                The model predicts a **{trend}** trend for {ticker} over the next {prediction_days} days. 
                With a Mean Absolute Error (MAE) of {metrics['MAE']}, the predictions suggest the stock price 
                might fluctuate within ±{metrics['MAE']} from the forecasted values. Recent technical indicators show:
                - 50-day MA: ${df['MA_50'].iloc[-1]:.2f}
                - 100-day MA: ${df['MA_100'].iloc[-1]:.2f}
                - 200-day MA: ${df['MA_200'].iloc[-1]:.2f}
                """
                st.write(analysis)
                
                # Display news articles
                try:
                    ticker_info = yf.Ticker(ticker)
                    news = ticker_info.news
                    if news:
                        st.write("**Recent News Articles:**")
                        for article in news[:3]:
                            st.write(f"- [{article['title']}]({article['link']}) ({article['publisher']})")
                except Exception as e:
                    st.warning("Could not load news articles")
                
                st.success('Prediction completed!')
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == '__main__':
    main()
