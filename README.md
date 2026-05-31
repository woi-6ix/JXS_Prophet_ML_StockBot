# PROPHET Prediction StockBot 📈

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-purple)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)
![Model](https://img.shields.io/badge/Model-Prophet-black)
![Data](https://img.shields.io/badge/Data-Yahoo%20Finance-green)

A Streamlit-based stock forecasting dashboard that uses Yahoo Finance market data, Meta/Facebook Prophet time-series modelling, moving averages, and model fit metrics to generate educational stock price forecasts.

---

## 📌 Project Overview

**PROPHET Prediction StockBot** is an interactive financial forecasting web application built with Python and Streamlit. The app allows users to enter a stock ticker, retrieve recent historical market data, calculate moving averages, train a Prophet forecasting model, and visualize projected future price movement over a selected business-day forecast window.

This project was created as an educational machine learning and financial analytics dashboard. It combines time-series forecasting, technical analysis, and performance evaluation in one clean web interface.

---

## 🔍 What the App Does

The dashboard walks through a full stock forecasting workflow:

1. **User enters a stock ticker**

   * Example tickers: `AAPL`, `MSFT`, `TSLA`, `SPY`, `SHOP.TO`, `TD.TO`

2. **Yahoo Finance data is fetched**

   * The app retrieves daily historical stock data using `yfinance`.
   * Data is cached to reduce repeated requests and help avoid rate-limit issues.

3. **Technical indicators are calculated**

   * 50-day moving average
   * 100-day moving average
   * 200-day moving average

4. **Data is prepared for Prophet**

   * Closing prices are converted into Prophet’s required format:

     * `ds` = date column
     * `y` = target price column

5. **Prophet model is trained**

   * The model fits historical closing price data.
   * The user selects how many business days to forecast.

6. **Forecast results are displayed**

   * Historical price chart
   * Moving average chart
   * Prophet forecast chart
   * Forecast components
   * Historical vs predicted chart
   * Error metrics
   * Combined historical and forecast table
   * Final written forecast summary

---

## ✨ Key Features

* 📊 **Interactive Streamlit Interface**
  Simple web app layout with ticker input, prediction slider, and one-click forecasting.

* 📈 **Yahoo Finance Market Data**
  Fetches recent historical stock data using the `yfinance` library.

* ⚡ **Cached Data Loading**
  Uses Streamlit caching to reduce repeated data calls and improve app stability.

* 🤖 **Prophet Time-Series Forecasting**
  Applies Prophet to model stock price trends, seasonality, and uncertainty.

* 📉 **Technical Analysis Indicators**
  Calculates and visualizes:

  * 50-day moving average
  * 100-day moving average
  * 200-day moving average

* 📐 **Model Fit Metrics**
  Evaluates fitted predictions using:

  * RMSE — Root Mean Squared Error
  * MAE — Mean Absolute Error
  * MAPE — Mean Absolute Percentage Error

* 📋 **Forecast Summary Table**
  Combines historical closing prices with Prophet forecast values, including upper and lower prediction bounds.

* 🧠 **Interpretive Forecast Summary**
  Generates a written summary with:

  * Final historical close
  * Forecasted price
  * Prediction range
  * Bullish or bearish trend direction
  * Moving average signal
  * Seasonality label

* 🛡️ **Error Handling**
  Includes handling for invalid tickers, missing close price data, insufficient historical data, and Yahoo Finance rate-limit issues.

---

## 🧠 Model Methodology

The app uses **Prophet**, a time-series forecasting model designed to handle trend and seasonality patterns in sequential data.

For this project:

* The stock’s historical closing price is used as the target variable.
* The date column is converted into Prophet’s required `ds` format.
* The closing price column is converted into Prophet’s required `y` format.
* The model is trained on historical stock prices.
* A future dataframe is created based on the selected number of business days.
* Prophet generates:

  * `yhat` — predicted price
  * `yhat_lower` — lower forecast bound
  * `yhat_upper` — upper forecast bound

The forecast is then visualized and compared against historical price movement.

---

## 📊 Technical Indicators

The app calculates three moving averages:

| Indicator              | Description                  |
| ---------------------- | ---------------------------- |
| 50-Day Moving Average  | Shorter-term trend indicator |
| 100-Day Moving Average | Medium-term trend indicator  |
| 200-Day Moving Average | Longer-term trend indicator  |

The dashboard also identifies a basic moving-average signal:

| Signal       | Meaning                       |
| ------------ | ----------------------------- |
| Golden Cross | 50-day MA is above 200-day MA |
| Death Cross  | 50-day MA is below 200-day MA |

These indicators are used for educational trend interpretation only.

---

## 📁 Project Structure

```bash
PROPHET-Prediction-StockBot/
│
├── JXS_Prophet_ML_StockBot_Streamlit.py   # Main Streamlit application
├── requirements.txt                       # Python package dependencies
├── packages.txt                           # System dependencies for Streamlit Cloud
├── README.md                              # Project documentation
│
└── .streamlit/
    └── config.toml                        # Optional Streamlit configuration
```

---

## 🛠️ Tech Stack

| Category             | Tools                        |
| -------------------- | ---------------------------- |
| Programming Language | Python                       |
| Web Framework        | Streamlit                    |
| Data Source          | Yahoo Finance via `yfinance` |
| Forecasting Model    | Prophet                      |
| Data Handling        | pandas, numpy                |
| Visualization        | matplotlib                   |
| Model Evaluation     | scikit-learn                 |
| Deployment           | Streamlit Community Cloud    |

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/woi-6ix/JSS_Prophet_Future_PredictionBot.git
cd JSS_Prophet_Future_PredictionBot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run JXS_Prophet_ML_StockBot_Streamlit.py
```

---

## 📄 requirements.txt

Use a simple requirements file instead of a full `pip freeze` export:

```txt
streamlit
yfinance
prophet
pandas
numpy
matplotlib
scikit-learn
plotly
```

---

## 🧩 packages.txt

For Streamlit Cloud deployment, include:

```txt
libxml2-dev
libxslt1-dev
```

These system dependencies help prevent build issues with packages that may rely on XML-related libraries.

---

## ⚙️ Optional Streamlit Config

Create a `.streamlit/config.toml` file:

```toml
[server]
fileWatcherType = "none"
```

This can help reduce file-watcher issues on hosted Streamlit environments.

---

## 🚀 Deployment on Streamlit Cloud

To deploy the app:

1. Push all project files to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app.
4. Select the GitHub repository.
5. Set the main file path to:

```bash
JXS_Prophet_ML_StockBot_Streamlit.py
```

6. Use Python 3.11 if available.
7. Deploy the app.

Recommended repository files for deployment:

```bash
JXS_Prophet_ML_StockBot_Streamlit.py
requirements.txt
packages.txt
.streamlit/config.toml
README.md
```

---

## 🖥️ How to Use the App

1. Open the Streamlit app.
2. Enter a stock ticker.

   * Example: `AAPL`, `MSFT`, `TSLA`, `SPY`
3. Select the number of business days to forecast.
4. Click **Predict**.
5. Review:

   * Raw stock data
   * Historical price chart
   * Moving averages
   * Forecast chart
   * Forecast components
   * Error metrics
   * Combined prediction table
   * Forecast summary

---

## 📌 Example Tickers

| Market          | Example Tickers                        |
| --------------- | -------------------------------------- |
| U.S. Stocks     | `AAPL`, `MSFT`, `TSLA`, `NVDA`, `AMZN` |
| ETFs            | `SPY`, `QQQ`, `DIA`                    |
| Canadian Stocks | `SHOP.TO`, `TD.TO`, `RY.TO`            |

Yahoo Finance ticker symbols can vary by exchange. For Canadian stocks, `.TO` is commonly used.

---

## ⚠️ Troubleshooting

### Yahoo Finance Rate Limit

If the app shows a message like:

```bash
No Yahoo Finance data was returned
```

Yahoo Finance may be temporarily rate-limiting the request. Try:

* Waiting a few minutes
* Using a common ticker like `AAPL`
* Refreshing the app less often
* Keeping caching enabled
* Using a shorter historical data period

### Invalid Ticker

If no closing price data is found, confirm that the ticker exists on Yahoo Finance.

### Prophet Needs More Data

If the app says Prophet needs at least 2 valid historical price points, the selected ticker may not have enough usable data.

### Streamlit Cloud Dependency Error

If deployment fails during package installation, make sure the repo includes:

```bash
requirements.txt
packages.txt
```

and that Python 3.11 is selected where possible.

---

## 📊 Output Screens

The app generates several outputs:

* Raw historical stock data
* Historical closing price with moving averages
* Prophet forecast chart
* Forecast component charts
* Historical vs predicted comparison
* RMSE, MAE, and MAPE metrics
* Combined historical and predicted dataframe
* Forecast summary and interpretation

---

## 📚 Learning Objectives

This project demonstrates:

* Building a machine learning dashboard with Streamlit
* Fetching financial market data using Python
* Preparing time-series data for Prophet
* Training and evaluating a forecasting model
* Visualizing technical indicators
* Handling deployment and dependency errors
* Communicating model results through an interactive interface

---

## ⚠️ Financial Disclaimer

This application is for **educational and testing purposes only**.

The forecasts generated by this app should not be interpreted as financial advice, trading recommendations, or investment guidance. Stock markets are highly volatile, and machine learning forecasts are uncertain. Always conduct independent research and consult a qualified financial professional before making financial decisions.

---

## 👨‍💻 Author

**Woi-6ix**

Built as part of a machine learning, financial modelling, and Streamlit dashboard development project.

---

## 📜 License

This project is licensed under the MIT License.
