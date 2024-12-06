### Projects

# LSTM with Stock Forecasting
- Using deep learning I created a stock forecasting model that inputted single stock close values and outputted 30-100 day forecasts.

# Trading Bot S&P
- I was interested in creating an algo trading bot that would automatically make trades for me on the strategy 'buy at the dip'. This bot used simple python if-else statements but used various finance metrics like moving averages and rsi to make decisions on when to buy and sell.

Some Initialization Parameters:
  - Starting Capital : $100,000
  - Oversold/Underbought Ratio : 35:65
  - Stop-Loss Percentage : 0.10 (10%)
  - Profit Sell Signal : 0.20 (20%)
 
- Using these parameters, I created a bot using simple if-else statements in python to loop through historical data (backtesting). Once the bot validates that all the conditions have passed it will either conduct it's buy or sell signal.
- Here's the data if the test is run starting from 2022-01-01 to current day (2024-12-06).
- ![alt text](https://github.com/vnvashist/Projects/blob/master/S%26P%20Bot%202022.png?raw=True)
