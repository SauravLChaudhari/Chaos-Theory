import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# Define high weightage stocks in Nifty 50
tickers = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'HDFC.NS', 'ICICIBANK.NS']

# Download historical stock price data
start_date = '2022-01-01'
end_date = '2023-01-01'
data = yf.download(tickers + ['^NSEI'], start=start_date, end=end_date)['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Define chaos theory function
def chaos_theory(data, threshold=0.01):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    diff = np.diff(scaled_data, axis=0)
    return np.sum(np.abs(diff) > threshold) / len(diff)

# Apply chaos theory function to Nifty 50 index
nifty_returns = returns['^NSEI']
chaos_index = chaos_theory(nifty_returns)

# Predict impact on high weightage stocks
model = LinearRegression()
stock_predictions = {}

for stock in tickers:
    X = nifty_returns.values.reshape(-1, 1)
    y = returns[stock].values
    model.fit(X, y)
    predicted_change = model.predict([[chaos_index]])
    stock_predictions[stock] = predicted_change[0]

# Display predictions
print(f"Chaos Index for Nifty 50: {chaos_index}")
print("Predicted Impact on High Weightage Stocks:")
for stock, prediction in stock_predictions.items():
    print(f"{stock}: {prediction*100:.2f}%")

# Save predictions to a CSV file
predictions_df = pd.DataFrame.from_dict(stock_predictions, orient='index', columns=['Predicted Change'])
predictions_df.to_csv('stock_predictions.csv', index=True)
