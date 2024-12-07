import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import warnings
warnings.filterwarnings('ignore')

# Stock
stock = 'VOO'

# Time period
start_date = '2022-01-01'
end_date = '2024-12-01'

# GET DATA
data = yf.download(stock, start=start_date, end=end_date)
prices = data['Adj Close']
if isinstance(prices, pd.DataFrame):
    prices = prices.squeeze()
print(f"Data starts on: {prices.index.min().date()}")
print(f"Data ends on: {prices.index.max().date()}")
def buy_the_dip_spy(prices, drop_threshold=0.05, profit_threshold=0.05, trailing_stop_loss=0.05):
    cash = 100000  # Starting capital
    holdings = 0   # Number of shares
    portfolio_value = []
    positions = {'holding': False, 'buy_price': 0, 'highest_price': 0}
    all_time_high = 0
    transaction_fee = 10  # Transaction fee per trade in dollars

    for date in prices.index:
        price = prices.at[date]
        price = float(price)

        # Update all-time high
        if price > all_time_high:
            all_time_high = price

        # Calculate drop from all-time high
        drop = (all_time_high - price) / all_time_high if all_time_high != 0 else 0

        if not positions['holding'] and drop >= drop_threshold:
            # Buy signal
            shares_to_buy = cash // price
            if shares_to_buy > 0:
                total_cost = shares_to_buy * price + transaction_fee
                cash -= total_cost
                holdings += shares_to_buy
                positions['holding'] = True
                positions['buy_price'] = price
                positions['highest_price'] = price
                print(f"{date.date()}: Bought {shares_to_buy} shares at ${price:.2f}")

        elif positions['holding']:
            # Update highest price since purchase
            if price > positions['highest_price']:
                positions['highest_price'] = price

            # Calculate profit from purchase price
            profit = (price - positions['buy_price']) / positions['buy_price']

            # Trailing stop-loss condition
            trailing_stop_price = positions['highest_price'] * (1 - trailing_stop_loss)

            if price <= trailing_stop_price:
                # Sell signal (Trailing Stop-Loss)
                total_revenue = holdings * price - transaction_fee
                cash += total_revenue
                print(f"{date.date()}: Trailing stop-loss triggered. Sold {holdings} shares at ${price:.2f}")
                holdings = 0
                positions = {'holding': False, 'buy_price': 0, 'highest_price': 0}
            elif profit >= profit_threshold or price >= all_time_high:
                # Sell signal (Profit Target or Return to All-Time High)
                total_revenue = holdings * price - transaction_fee
                cash += total_revenue
                print(f"{date.date()}: Sold {holdings} shares at ${price:.2f}")
                holdings = 0
                positions = {'holding': False, 'buy_price': 0, 'highest_price': 0}

        # Calculate total portfolio value
        total_value = cash + holdings * price
        portfolio_value.append(total_value)

    # Create a DataFrame for portfolio value
    portfolio_df = pd.DataFrame({'Date': prices.index, 'Portfolio Value': portfolio_value})
    portfolio_df.set_index('Date', inplace=True)
    return portfolio_df

def buy_and_hold_spy(prices, initial_capital=100000):
    starting_price = prices.iloc[0]
    shares = initial_capital / starting_price
    portfolio_values = prices * shares
    portfolio_df = pd.DataFrame({'Date': prices.index, 'Portfolio Value': portfolio_values})
    portfolio_df.set_index('Date', inplace=True)
    return portfolio_df

# MAIN
portfolio_bt = buy_the_dip_spy(prices)
portfolio_bh = buy_and_hold_spy(prices)

# Calculate total returns
initial_value_bt = portfolio_bt['Portfolio Value'].iloc[0]
final_value_bt = portfolio_bt['Portfolio Value'].iloc[-1]
total_return_bt = (final_value_bt - initial_value_bt) / initial_value_bt * 100

initial_value_bh = portfolio_bh['Portfolio Value'].iloc[0]
final_value_bh = portfolio_bh['Portfolio Value'].iloc[-1]
total_return_bh = (final_value_bh - initial_value_bh) / initial_value_bh * 100

# Final monetary values and total returns
print(f"Buy-the-Dip Strategy Final Portfolio Value: ${final_value_bt:,.2f}")
print(f"Buy-the-Dip Strategy Total Return: {total_return_bt:.2f}%\n")

print(f"Buy-and-Hold Strategy Final Portfolio Value: ${final_value_bh:,.2f}")
print(f"Buy-and-Hold Strategy Total Return: {total_return_bh:.2f}%\n")

# Plot
plt.figure(figsize=(14, 7))
plt.plot(portfolio_bt.index, portfolio_bt['Portfolio Value'], label='Buy the Dip Strategy')
plt.plot(portfolio_bh.index, portfolio_bh['Portfolio Value'], label='Buy and Hold Strategy')
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.show()
