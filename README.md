# Projects
[LSTM with Stock Forecasting](#lstm-with-stock-forecasting)
\
[Trading Bot S&P](#trading-bot-sp)

# LSTM with Stock Forecasting
\
I had recently gotten into investing and naturally, like many of my peers, was interested to see how I could incorporate machine learning to optimize my investments. Of course, a brief google search will tell you machine learning models for stock prediction are generally not recommended due to the complexity and number of observations you would have to capture to accurately make any meaningful predictions, but... that didn't stop me.

<ins> Initialization and Justification </ins>
\
  I decided to predict AAPL (Apple) stock as it had a tremendous amount of historical data we could work with (ranging back from Dec.12 1980). 
  \
  Here are the layers in my neural net: \
  \
    LSTM \
    Dropout \
    LSTM \
    Dropout \
    Dense \
  \
  For those less familiar with LSTMs or Neural Nets, I will explain each of the above layers and my reasons for implementing them: \
  \
      LSTM (Long-Short-Term-Memory) - Good for time series, implements memory gates which control what to remember/forget (long term or short term memory aspect) \
  \
      Dropout - Prevents overfitting, disables some of the neurons so that the model does not overfit exactly to the historical data we are using \
  \
      Dense - Learning layer, connects to all available neurons from the previous layer, contains weights/biases. \
  \
  My justification for using these layers is primarily to consider the time series aspect of the data while preventing overfitting. I start off with an LSTM layer to 'set the tone' and then continue with dropout layers to prevent any overfitting that may occur (20% dropout). Finally, I add a dense layer at the end to sucessfully predict the next closing price that AAPL will have (based on previous time series data). \
  \
\
<ins> Visualization </ins> \
![alt text](https://github.com/vnvashist/Projects/blob/master/LSTM%20Stock%20Predictor/LSTM%20with%20Forecasting.png?raw=True)
  \
  Interestingly, the model performed fairly well. From the visualization we can see that the test predictions demonstrated similar trends to the actual data (predictions seem to lag by a small time period). \
  \
  Also interestingly, while the forecasting for the model suggests a downward curve, the model also predicts the value of the stock to increase for the next 100 days. \
  \
<ins> Points of Imporvement </ins>
\
\
  Fundamentally, I knew that this project was not going to be perfect. If I could accurately predict stocks using yfinance data and a laptop the U.S market would not survive a single day. However, despite that I do have a few points of improvement that I think, for future iterations, could be beneficial. 
  \
  First and foremost, understanding the time lag between the actual dataset and the model's predictions could be very helpful for improving accuracy. While the model captures trends well, it seems like there's a time component that is not being addressed fast enough (too many layers? maybe we need more nuance to each layer?). 
  \
  Secondly, more work needs to be done with the forecasting. I think that while I have started off in the right direction, the model's predictions and forecasts do not line up, which spells some kind of miscommunication in the code. \
  \
<ins> Final Thoughts </ins> \
\
  Overall, I conducted this project as I was enrolled in Deep Learning. A lot of the tweaks and commits I made were as I was learning better practices for not only neural nets but LSTMs and how best to optimize their hyperparameters. Of course, there's still a lot to learn in that regard and I am keen on returning to this project to implement some more data sources and/or tweaking my model ever so slightly. 
  \

# Trading Bot S&P 
\
I was interested in creating an algo trading bot that would automatically make trades for me on the strategy 'buy at the dip'. This bot used simple python if-else statements but used various finance metrics like moving averages and rsi to make decisions on when to buy and sell.

<ins> Some Initialization Parameters: </ins> \
\
  Starting Capital : $100,000
  \
  Oversold/Underbought Ratio : 35:65
  \
  Stop-Loss Percentage : 0.10 (10%)
  \
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
  
