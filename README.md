# Projects
[Reinforcement Learning in Age of Empires 2](#reinforcement-learning-in-age-of-empires-2)
\
[Pokemon Generator](#pokemon-generator)
\
[LSTM with Stock Forecasting](#lstm-with-stock-forecasting)
\
[Trading bot S&P](#trading-bot-sp)

# Reinforcement Learning in Age of Empires 2
\
Keeping in theme with video games, I recently completed a course in reinforcement learning and became interested to see the level of complexity I could factor into OpenAI's Gym package. The examples we covered in class were fairly binary problems that ranged from simple grid walks to stochastic differential equations that helped us park more efficiently.
\
Accordingly, I wanted to see if one of my favorite games of all time AOE 2 could be deciphered through using RL (reinforcement learning). If you aren't familiar with AOE2, I'll go over a brief description to lend a bit of context to this project.
\
\
<ins> Background </ins>
\
AOE2 is one of the greatest strategy games ever created where you, as a civilizations king/lord/commander, are responsible for deciding how your group of villagers/military units band together to form an empire to defeat at least one other player who is tasked with the same objective. Of course, any great conquest requires resources and there are four that are useful for you.
1. Food - Necessary to train both military units and villagers
2. Wood - Necessary to build houses which increase your population capacity
3. Gold - Necessary to train military units
4. Stone - Necessary for Castles (not implemented, i'll get into that in a bit)
5. Houses - Necessary to increase max_population size 


Given these resources, I tasked my agent to determine what the optimal set of moves are to maximize both resources and survivability. I used a reward system where positive actions like increasing the population or military unit count gave the agent points and negative actions whenver the agent would attempt to do something that there are not enough resources for. Here is a list of some of the other actions that can give rewards:

1. If the military unit count was greater than or equal to the (population_count/4) - Given a growing population we need a growing army to protect them (line 122-133)
2. If the population count is lower than the max_population size (line 99)
3. Building a house and also rewarding the agent extra if the house is built right before the population count hits the max_population count. (line 75-88)

Fundamentally the agent will iterate through actions that dictate either resource gathering, building houses, or creating villagers/soldiers. 

\
<ins> Modeling </ins>
\
Training an agent to perform these actions was fairly easy using OpenAI's gymnasium package. Using it, you can create a training environment that is customized to your needs. In this case I created my own environement (rather than using a pre-built one) and initialized various factors like discrete spaces (7 as there are 0-6 actions) or the observation space (what the agent 'sees' to help make it's decisions).
\
From there it was a matter of setting my starting class variables and the appropriate rewards for each 'step' that the agent makes. The foundational code here was:
\
if action == 0: (any number 0-6, gather wood in this case)
\
  reward += 1
\
The same format is applied to all 7 of the available actions with select modifications. For example, as mentioned before, rewarding the system more if it min/maxes the population-max_population ratio was important as we were only spending turns to gather resources when absolutely necessary. Likewise, it was important to incentivize a high military unit count as it would protect our citizens in the long run. Accordingly, due to how important these actions were, they had bonus reward and penalty systems to bring nuance to the agent.
\
Here's how the agent performed (after a lot of tinkering):
\
![alt text](https://github.com/vnvashist/Projects/blob/master/Reinforcement%20Learning%20in%20AOE2/RL%20with%20AOE2.png?raw=True)
\
<ins> Points of Improvement </ins>
\
So fundamentally the agent performed really well given the restrictions we placed on it. It maximized the population values, while keeping the soldier counts high, all the while getting a lot of resources under our belt to help with future expansions if our empire so wishes to. Another incredible observation is that, for the first 15 turns, the agent performed what is known as an 'ideal' start, a strategy that most pro-players perform to this day. It's where you create 3 villagers and put some on food gather. Then you build a house, and afterwards create more villagers and put them on wood gathering.
\
Here's the picture for the first couple of turns:
\
![alt text](https://github.com/vnvashist/Projects/blob/master/Reinforcement%20Learning%20in%20AOE2/RL%20AOE2%20start.png?raw=True)
\
Of course, if you were to compare the final resource values to most professional games at the 20 min mark, it would be very different from how AOE2 is played. Resources are generally not hoarded for a rainy day, but rather to maximize your army, buildings, or villager count. That is, there is no incentive to holding on to your resources. The objective is to spend your resources to defeat your enemy. In the future, I'd like to instill that ideology into the agent to ensure that, while they are keeping with good soldier-population practices, they could be maximizing their values better. Implementing a reward to hitting a certain number of soldiers at certain turn numbers could be a good way to do that. Overall though, I am pretty happy with how this turned out. The actions that the agent makes are generally what you would do when you are trying to build your economy in the game. Eventually, by adding military actions and reward/penalty systems, I think we could have a fully functioning agent to play against other players.
\
# Pokemon Generator 
\
Another topic that I became very interested in during my deep learning course was GANs (generative adversial networks). They work in an antagonistic relationship between two neural networks: Generator and Discriminator. I became interested in generating my own content, especially as generative AI has taken such a main focus on the world today. I thought that it would be interesting to see how a GAN would perform on generating fake pokemon given their names and their stats.
\
My focus for this project was 3-fold
1. Data engineer a pipeline to draw in pokemon data
2. Analyze the data to determine a relationship that I can leverage within the GAN
3. Implement the GAN to create names, stats (while being discouraged to avoid the relationship determined in Step 2)

<ins> Data Engineering/Analysis </ins> \
\
  The API call portion of this project was relatively simple. I limited my call to take in 1025 entries (as there are only 1025 pokemon) and filtered out the dataset to only the relevant fields that I was interested in (name, type, attack, speed, defense, etc).
  \
  From there I did a fair bit of data cleaning/transformation.
  1. I wanted to normalize the types into separate columns. Each pokemon has 1 main type and then two other 'subtypes'. Instead of having to decipher them all at once, I split them up to have their own columns
  2. I also had to normalize the abilities into separate columns. I did not end up using this but figured it could be useful for future applications (maybe generating abilities)
  3. I expanded the stats into separate columns. Again, the api retreived the pokemon's stats into one single column. I prefered to keep them separate for sanity.
  4. Lastly, I used a MinMaxScaler() to ensure that all of the data points fall within a similar scale (important for GANs and even LSTMs)

Finally, I plotted a few different values but found speed vs attack to be a relationship that made the most sense to enforce onto the GAN.
Here's the plot:
![alt text](https://github.com/vnvashist/Projects/blob/master/Pokemon%20Generator/Pokemon%20Speed%20vs%20Attack.png?raw=True)
\
  From here I was curious to see the relationship between each type and their attack-speed ratio. I calculated each of the slopes of the lines of best fit and averaged them resulting in a value of 0.3226 as the ratio between Speed and Attack for all Pokemon types. This would be useful soon. 

<ins> Modeling </ins> \
\
  For the sake of simplicity I'm not going to go through every line of code here but I am going to do my best to mention some of the highlights in the code. Overall, I decided that I wanted to generate names and stats, while keeping in mind that 0.3226 ratio I mentioned earlier. Like any GAN you have to build your generators and discriminators. Here I used the LeakyReLU activation function (normal ReLU causes dying neurons) and layered them with Dense layers.
\
  Here's where the cool part comes in. At this stage I had to train my GAN while incorporating the relationship between attack and speed. I had never done this before and thought about where I would introduce this. I started off with writing my GAN as I normally would. I looped through epochs and created real and fake labels, predictions, etc. As I was writing my loss function I then realized that this could be the point of influence that the relationship could have. A quick stackoverflow search also confirmed my suspicions. I decided to write the loss function while creating a variable called relationship_penalty. This variable would take in the generated attack data and use it to create a speed prediction (using the 0.3266 slope value). We would then subtract the speed prediction value from the GAN generated speed value to determine how different the value is. Using this variable we can multiply it to the lambda regularization parameter to easily tailor how much we want the relationshp penalty to affect the overal GAN stats. If this is confusing take a look at lines 76-83 and it might be a bit clearer. \
\
Ok so the final output? \
![alt text](https://github.com/vnvashist/Projects/blob/master/Pokemon%20Generator/Pokemon%20Generator.png?raw=True)
\
Ok so not amazing, but not bad either. The stats when analyzed demonstrate an average of 0.26 ratio of attack to speed values. This is pretty close to the measured average of 0.3226. I think with a larger generation batch we might see that number get closer to the average. The names are somewhat mangled together, most of them having parts that sound familiar. Overall the names do not sound cohesive enough to be considerd a success, but I will discuss that in my points of improvment.

<ins> Points of Improvement </ins> \
\
So, like always, there are always points of improvement. Chiefly, among all other reasons, is the lack of data. It's funny because when you think of pokemon you think of an endless number of creatures, but, in reality, there are only 1025. That means this GAN only had 1025 names to train itself on. Given that limitation I am moderately impressed with what it has outputted. In the future shifting this model to focus on (assuming we're still dealing with Pokemon) image data may be a useful consideration. Afterall there are an outstanding number of pictures of all 1025 Pokemon out on the internet.
# LSTM with Stock Forecasting
\
I had recently gotten into investing and naturally, like many of my peers, was interested to see how I could incorporate machine learning to optimize my investments. Of course, a brief google search will tell you machine learning models for stock prediction are generally not recommended due to the complexity and number of observations you would have to capture to accurately make any meaningful predictions, but... that didn't stop me.

<ins> Initialization and Justification </ins>
\
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
     * LSTM (Long-Short-Term-Memory) - Good for time series, implements memory gates which control what to remember/forget (long term or short term memory aspect) \
  \
     * Dropout - Prevents overfitting, disables some of the neurons so that the model does not overfit exactly to the historical data we are using \
  \
     * Dense - Learning layer, connects to all available neurons from the previous layer, contains weights/biases. \
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
<ins> Points of Improvement </ins>
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

# Trading bot S&P 
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
  So where do we go from here? Well next steps are definitely to refine the financial metrics involved within this bot. A little history lesson but 'buy at the dip' is a trading strategy that has historically performed well buy both individuals and large-scale algo firms, so I'm relatively convinced that the strategy is not impossible to work with. However, besides other financial metrics to consider, it might also be worth implementing other strategies given certain conditions... food for thought in the future...
  
