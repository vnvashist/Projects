# Projects
[LSTM with Stock Forecasting](#https://github.com/vnvashist/Projects/blob/master/README.md#lstm-with-stock-forecasting)

# LSTM with Stock Forecasting
- Using deep learning I created a stock forecasting model that inputted single stock close values and outputted 30-100 day forecasts.

# Trading Bot S&P 
\
I was interested in creating an algo trading bot that would automatically make trades for me on the strategy 'buy at the dip'. This bot used simple python if-else statements but used various finance metrics like moving averages and rsi to make decisions on when to buy and sell.

<ins> Some Initialization Parameters: </ins> \
\
  Starting Capital : $100,000
  Oversold/Underbought Ratio : 35:65
  Stop-Loss Percentage : 0.10 (10%)
  Profit Sell Signal : 0.20 (20%)
 
  Using these parameters, I created a bot using simple if-else statements in python to loop through historical data (backtesting). Once the bot validates that all the conditions have passed it will either conduct it's buy or sell signal. \
  \
  Here's the data if the test is run starting from 2022-01-01 to current day (2024-12-06). \
  \
  ![alt text](https://github.com/vnvashist/Projects/blob/master/S%26P%20Bot%202022.png?raw=True) 
  \
  Awesome, the bot outperforms the market by a whopping 21%! 

<ins> Points of Improvement </ins> \
\
  Success.... right? Well unfortunately not really... This whole time I was backtesting on the single year and realized that I had not checked for any other years. Afterall, how useful is a bot that can only perform if you magically started on the year 2022.
  
  And of course... my fears were confirmed. \
  \
  ![alt text](https://github.com/vnvashist/Projects/blob/master/S%26P%20Bot%202021.png?raw=True)
  \
  The bot did not perform as well on every other year I checked. Attached above is checking from 2021-01-01 onward, but similar results were replicated for 2020, 2023, 2019, and onwards. Almost every other year I could have entered the market, the bot would underperform...
  \
  \
  So where do we go from here? Well next steps are definitely to refine the financial metrics involved within this bot. A little history lesson but 'buy at the dip' is a trading strategy that has historically performed well buy both individuals and large-scale algo firms, so I'm relativel convinced that the strategy is not impossible to work with. However, besides other financial metrics to consider, it might also be worth implementing other strategies given certain conditions... food for thought in the future...
  
