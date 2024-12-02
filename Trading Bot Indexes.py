import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
import datetime
import warnings
warnings.filterwarnings('ignore')

# Stock to analyze
stock = 'VOO'

# Time period
start_date = '2022-01-01'
end_date = datetime.datetime.now().strftime('%Y-%m-%d')

# Fetch data
data = yf.download(stock, start=start_date, end=end_date)

# Check data
if data.empty:
    print("No data was fetched. Please check your data source and parameters.")
    exit()

# Flatten MultiIndex columns if necessary
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Ensure 'Adj Close' is a Series
adj_close = data['Adj Close']
if isinstance(adj_close, pd.DataFrame):
    adj_close = adj_close.squeeze()

# Technical indicators
rsi_period = 14
data['RSI'] = RSIIndicator(close=adj_close, window=rsi_period).rsi()

def rsi_strategy(data, initial_capital=100000, position_size=1.0, transaction_fee=5,
                          oversold_level=35, overbought_level=65, stop_loss_pct=0.10, take_profit_pct=0.20):
    cash = initial_capital
    holdings = 0
    portfolio_value = []
    position = None
    entry_price = 0

    for date, row in data.iterrows():
        price = float(row['Adj Close'])
        rsi = float(row['RSI'])

        # Ensure RSI is not NaN
        if pd.notna(rsi):
            # Buy signal (RSI oversold)
            if position is None and rsi < oversold_level:
                # Invest all cash
                amount_to_invest = cash * position_size
                shares_to_buy = amount_to_invest // price
                if shares_to_buy > 0:
                    total_cost = shares_to_buy * price + transaction_fee
                    cash -= total_cost
                    holdings += shares_to_buy
                    position = 'Long'
                    entry_price = price

                    print(f"{date.date()}: Bought {shares_to_buy} shares at ${price:.2f} (RSI: {rsi:.2f})")

            elif position == 'Long':
                # Calculate current profit/loss
                profit_pct = (price - entry_price) / entry_price

                # Sell conditions
                sell_signal = False

                # Stop-loss condition
                if profit_pct <= -stop_loss_pct:
                    sell_signal = True
                    reason = 'Stop-Loss Triggered'

                # Take-profit condition
                elif profit_pct >= take_profit_pct:
                    sell_signal = True
                    reason = 'Take-Profit Reached'

                # RSI overbought condition
                elif rsi > overbought_level:
                    sell_signal = True
                    reason = 'RSI Overbought'

                if sell_signal:
                    total_revenue = holdings * price - transaction_fee
                    cash += total_revenue

                    print(f"{date.date()}: Sold {holdings} shares at ${price:.2f} - {reason} (RSI: {rsi:.2f})")
                    holdings = 0
                    position = None
                    entry_price = 0

        # Calculate total portfolio value
        total_value = cash + holdings * price
        portfolio_value.append({'Date': date, 'Portfolio Value': total_value})

    # Create a DataFrame for portfolio value over time
    portfolio_df = pd.DataFrame(portfolio_value)
    portfolio_df.set_index('Date', inplace=True)
    return portfolio_df

# RSI strategy
portfolio_adjusted_rsi = rsi_strategy(
    data,
    position_size=1.0,          # Invest 100% of available cash per trade
    oversold_level=35,          # RSI oversold level
    overbought_level=65,        # RSI overbought level
    stop_loss_pct=0.10,         # 10% stop-loss
    take_profit_pct=0.20,       # 20% take-profit
    transaction_fee=5
)

# Buy-and-hold strategy
def buy_and_hold(data, initial_capital=100000):
    starting_price = data['Adj Close'].iloc[0]
    shares = initial_capital / starting_price
    portfolio_values = data['Adj Close'] * shares
    portfolio_df = pd.DataFrame({'Date': data.index, 'Portfolio Value': portfolio_values})
    portfolio_df.set_index('Date', inplace=True)
    return portfolio_df

portfolio_bh = buy_and_hold(data)

# Calculate total returns
# Adjusted RSI Strategy Returns
initial_value_adjusted = portfolio_adjusted_rsi['Portfolio Value'].iloc[0]
final_value_adjusted = portfolio_adjusted_rsi['Portfolio Value'].iloc[-1]
total_return_adjusted = (final_value_adjusted - initial_value_adjusted) / initial_value_adjusted * 100

# Buy-and-Hold Returns
initial_value_bh = portfolio_bh['Portfolio Value'].iloc[0]
final_value_bh = portfolio_bh['Portfolio Value'].iloc[-1]
total_return_bh = (final_value_bh - initial_value_bh) / initial_value_bh * 100

# Print results
print(f"Adjusted RSI Strategy Final Portfolio Value: ${final_value_adjusted:,.2f}")
print(f"Adjusted RSI Strategy Total Return: {total_return_adjusted:.2f}%\n")

print(f"Buy-and-Hold Strategy Final Portfolio Value: ${final_value_bh:,.2f}")
print(f"Buy-and-Hold Strategy Total Return: {total_return_bh:.2f}%\n")

# Plot portfolios
plt.figure(figsize=(14, 7))
plt.plot(portfolio_adjusted_rsi.index, portfolio_adjusted_rsi['Portfolio Value'], label='Adjusted RSI Strategy')
plt.plot(portfolio_bh.index, portfolio_bh['Portfolio Value'], label='Buy and Hold Strategy')
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.show()
