### Import Packages ##
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


## Defining Functions ##
# Downloading Data From yFinance #
@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(ticker, start_date="2010-01-01"):
    """
    Download daily stock data and add moving averages.
    Returns: tuple[pd.DataFrame, str | None]: dataframe plus an optional error message.
    """
    ticker = str(ticker).strip().upper()

    if not ticker:
        return pd.DataFrame(), "Please enter a ticker symbol."

    logging.getLogger("yfinance").setLevel(logging.WARNING)

    try:
        stock_df = yf.download(
            tickers=ticker,
            period="10y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if stock_df.empty:
            return pd.DataFrame(), f"No Yahoo Finance data was returned for '{ticker}'. You may be rate limited or the ticker may be unavailable."

        if isinstance(stock_df.columns, pd.MultiIndex):
            stock_df.columns = stock_df.columns.get_level_values(0)

        if "Close" not in stock_df.columns:
            return pd.DataFrame(), f"No closing price data found for '{ticker}'."

        stock_df = stock_df.dropna(subset=["Close"])

        stock_df["MA_50"] = stock_df["Close"].rolling(window=50, min_periods=1).mean()
        stock_df["MA_100"] = stock_df["Close"].rolling(window=100, min_periods=1).mean()
        stock_df["MA_200"] = stock_df["Close"].rolling(window=200, min_periods=1).mean()

        return stock_df, None

    except Exception as e:
        return pd.DataFrame(), f"Error downloading data for '{ticker}': {e}"
        
# Prepping Stock Data Frame into Prophet Data Frame #
def prepare_prophet_stock_data(stock_df):
    # Keep only the closing price and reset the date index into a column
    prophet_df = stock_df[["Close"]].copy().reset_index()

    # The first column after reset_index is the date column.
    # It might be called "Date", "Datetime", or "index", so rename it safely.
    date_column = prophet_df.columns[0]

    prophet_df = prophet_df.rename(
        columns={
            date_column: "ds",
            "Close": "y"
        }
    )

    # Convert columns into the exact format Prophet needs
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], errors="coerce")
    prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)
    prophet_df["y"] = pd.to_numeric(prophet_df["y"], errors="coerce")

    # Keep only valid Prophet rows
    prophet_df = prophet_df[["ds", "y"]].dropna()

    return prophet_df


def calculate_metrics(df, forecast):
    """Calculate in-sample forecast error metrics aligned by historical dates."""
    forecast_history = forecast.set_index("ds").reindex(df.index)
    valid = forecast_history["yhat"].notna() & df["Close"].notna()

    y_true = df.loc[valid, "Close"].astype(float).values
    y_pred = forecast_history.loc[valid, "yhat"].astype(float).values

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    return rmse, mae, mape, forecast_history


# Main Function #
def main():
    st.set_page_config(
        page_title="FB Prophet JXS Stock Prediction App",
        page_icon="📈",
        layout="centered",
    )

    st.markdown(
        """
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
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("JXS Prophet Stock Prediction App")

    st.write(
        """
        ### How the Model Works
        This app uses **Prophet** to forecast stock prices. Prophet is a time series forecasting model that decomposes data into trend, seasonality, and uncertainty.

        The app also calculates **moving averages (50-day, 100-day, and 200-day)** to help visualize shorter-term and longer-term price trends.
        """
    )

    st.warning(
        """
        ### Disclaimer
        **These stock value predictions are for testing and educational purposes only.** They should not be used for real-world financial decisions. Stock markets are volatile, and forecasts are uncertain.
        """
    )

    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA, XYZ):", "AAPL")
    prediction_days = st.slider("Select number of business days to predict (0-730):", 0, 730, 365)

    if st.button("Predict"):
        ticker = ticker.strip().upper()

        with st.spinner("Fetching data and making predictions..."):
            try:
                df, data_error = get_stock_data(ticker)

                if df.empty:
                    st.error(data_error or "No data found for the ticker. Please check the ticker symbol.")
                    st.info("Tip: Yahoo Finance symbols can differ by market. Examples: AAPL, MSFT, TSLA, SPY, SHOP.TO, TD.TO, BRK-B.")
                    return

                st.subheader(f"{ticker} Raw Stock Data")
                st.write(f"Data from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

                display_df = df.reset_index().rename(columns={"Date": "Dates"})
                st.dataframe(display_df)

                prophet_df = prepare_prophet_stock_data(df)

                if len(prophet_df) < 2:
                    st.error("Prophet needs at least 2 valid historical price points to run.")
                    return

                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.5,
                )
                model.fit(prophet_df)

                future = model.make_future_dataframe(periods=prediction_days, freq="B")
                forecast = model.predict(future)

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

                ax1.plot(df.index, df["Close"], label="Close Price", color="purple")
                ax1.plot(df.index, df["MA_50"], label="50-day MA", color="orange")
                ax1.plot(df.index, df["MA_100"], label="100-day MA", color="green")
                ax1.plot(df.index, df["MA_200"], label="200-day MA", color="red")
                ax1.set_title(f"{ticker} Historical Prices & Moving Averages", color="black")
                ax1.set_facecolor("white")
                ax1.legend()
                ax1.tick_params(colors="black")
                ax1.spines["bottom"].set_color("black")
                ax1.spines["left"].set_color("black")

                model.plot(forecast, ax=ax2)
                ax2.set_title(f"{ticker} {prediction_days}-Business-Day Price Prediction", color="black")
                ax2.set_xlabel("Date", color="black")
                ax2.set_ylabel("Price", color="black")
                ax2.set_facecolor("white")
                ax2.tick_params(colors="black")
                ax2.spines["bottom"].set_color("black")
                ax2.spines["left"].set_color("black")

                st.pyplot(fig)
                plt.close(fig)

                st.subheader("Forecast Components")
                components_fig = model.plot_components(forecast)
                st.pyplot(components_fig)
                plt.close(components_fig)

                st.subheader("Historical vs Predicted Prices with Error Metrics")
                rmse, mae, mape, forecast_history = calculate_metrics(df, forecast)

                fig2, ax3 = plt.subplots(figsize=(12, 6))
                ax3.plot(df.index, df["Close"], label="Historical Close Price", color="purple", linewidth=2)
                ax3.plot(forecast_history.index, forecast_history["yhat"], label="Model Predictions", color="orange", linestyle="--")
                ax3.fill_between(
                    forecast_history.index,
                    forecast_history["yhat_lower"].astype(float),
                    forecast_history["yhat_upper"].astype(float),
                    color="orange",
                    alpha=0.2,
                )

                metrics_text = f"""Error Metrics:
- RMSE: ${rmse:.2f}
- MAE: ${mae:.2f}
- MAPE: {mape:.2f}%"""

                ax3.text(
                    0.02,
                    0.95,
                    metrics_text,
                    transform=ax3.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

                ax3.set_title(f"{ticker} Historical vs Predicted Prices with Error Metrics", color="black")
                ax3.set_xlabel("Date", color="black")
                ax3.set_ylabel("Price", color="black")
                ax3.set_facecolor("white")
                ax3.legend(loc="lower right")
                ax3.tick_params(colors="black")
                ax3.spines["bottom"].set_color("black")
                ax3.spines["left"].set_color("black")

                st.pyplot(fig2)
                plt.close(fig2)

                st.subheader("Combined Historical and Predicted Values")
                combined_df = pd.concat(
                    [
                        df[["Close"]].rename(columns={"Close": "Historical Close"}),
                        forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]].rename(
                            columns={
                                "yhat": "Predicted Close",
                                "yhat_lower": "Predicted Lower Bound",
                                "yhat_upper": "Predicted Upper Bound",
                            }
                        ),
                    ],
                    axis=1,
                )
                st.dataframe(combined_df)

                st.subheader("Forecast Summary and Insights")

                last_close = float(df["Close"].iloc[-1])
                last_forecast = forecast.iloc[-1]
                predicted_close = float(last_forecast["yhat"])
                lower_bound = float(last_forecast["yhat_lower"])
                upper_bound = float(last_forecast["yhat_upper"])

                first_trend = float(forecast["trend"].iloc[0])
                last_trend = float(last_forecast["trend"])
                trend_change = ((last_trend - first_trend) / first_trend) * 100 if first_trend != 0 else 0

                yearly_effect = abs(float(forecast["yearly"].mean())) if "yearly" in forecast.columns else 0
                seasonality_label = "Significant" if yearly_effect > 0.5 else "Moderate"

                ma_status = "Golden Cross" if df["MA_50"].iloc[-1] > df["MA_200"].iloc[-1] else "Death Cross"
                momentum = "strong upward" if df["MA_50"].iloc[-1] > df["MA_200"].iloc[-1] else "downward"

                summary = f"""
                **Key Forecast Insights for {ticker}:**
                - **Final Historical Close**: ${last_close:.2f}
                - **Model Fit Metrics**: MAE ${mae:.2f}, RMSE ${rmse:.2f}, MAPE {mape:.2f}%
                - **{prediction_days}-Business-Day Price Prediction**: ${predicted_close:.2f}
                - **Prediction Range**: ${lower_bound:.2f} - ${upper_bound:.2f}
                - **Trend Direction**: {'Bullish' if trend_change > 0 else 'Bearish'} ({abs(trend_change):.1f}% {'increase' if trend_change > 0 else 'decrease'})
                - **Seasonality Impact**: {seasonality_label} yearly seasonality detected
                - **Moving Average Status**: {ma_status} pattern identified

                **Analysis Summary:**
                The forecast suggests a {trend_change:.1f}% {'growth' if trend_change > 0 else 'decline'} over the selected prediction window.
                Historical moving averages indicate {momentum} momentum.
                The model's average percentage error is {mape:.1f}% based on the fitted historical period.
                """

                st.markdown(summary)
                st.success("Prediction completed!")

            except Exception as e:
                st.exception(e)
                st.error("The app hit an unexpected error. The traceback above should show the exact line to debug.")


if __name__ == "__main__":
    main()
