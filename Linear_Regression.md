---
title: Linear Regression Project

---

## **Predicting NBA Player Salary using Linear Regression and Web Scraping**

This is my second project as part of the Metis Data Science Bootcamp. Here, I web-scraped NBA player statistics from [Basketball-Reference](www.basketball-reference.com) to build a linear regression model to predict their salary.



**Introduction**

Because of my passion for basketball, I decided to focus my linear regression project on the NBA. Every free agency, we see bloated contracts handed out to underperforming players and value contracts handed out to budding stars that no one saw coming. These disparities between contract value and output can be the difference between championships and disappointment. The goal of this project is to predict NBA players' yearly salary using linear regression. With a data-driven approach to valuing a player, front offices can make more calculated decisions. They are better equipped to negotiate new contracts, trade for undervalued players, or trade away their own players they deem underperforming.



**Design**

To predict NBA salaries, the relationships between various box score statistics should be explored. The correlation between the stats and player salary, as well as each other for multicollinearity, should be considered. In addition, feature engineering will be implemented to account for interaction variables and/or dummy indicators that aren't immediately available.



After data cleaning and manipulation, I split the data into training and test sets, and tested three different models (OLS, Ridge Regression, and Lasso Regression). I modeled based on the training set and judged the efficacy of the models on the test data. I also judged the models on r squared, Mean Absolue Error, and intuitive fit.



**Web-Scraped Data**

I will be web scraping via BeautifulSoup from the following sources:

- Player statistics from the 2017-2018 to 2020-2021 seasons taken from [Basketball-Reference](www.basketball-reference.com)
- NBA salary data taken from [ESPN](http://www.espn.com/nba/salaries)



Luckily, basketball-reference has a coder-friendly interface that certainly helped simplify the web scraping process. Below is a snippet of the web scraping algorithm I used to convert player statistics on the website into a readable csv file. It iterates through each season and extracts relevant player statistics for all players who played in a given season.

![Screen Shot web-scrape](https://github.com/prathapr91/linear_regression_NBA/blob/main/images/Screen%20Shot%202022-01-11%20at%209.48.49%20PM.png?raw=true)



**Algorithm**

*Data Cleaning*

- Players that did not have a corresponding salary match were dropped from the dataset
- Blank rows are removed. These are typically attributed to players that are technically on a roster but haven't received any or significant playing time.

*Data Manipulation*

- The following variables were collected and considered: 
  - Player
  - Position
  - Age
  - Team
  - Games played/started
  - Minutes
  - Made and attempted Field Goals/Free Throws/3 Pointers
  - Shooting percentages
  - Offensive and defensive rebounds
  - Assists
  - Steals
  - Blocks
  - Turnovers
  - Fouls
  - Points
  - Year
  - Offensive/defensive rating
  - Salary.

- This tool is meant to predict an NBA player's salary. Due to current salary cap and contract regulations, this analysis won't be appropriate for all players. A player's first deal, a rookie contract, is set in stone and superstar athletes are limited by maximum contract provisions. To account for this, I will be flagging these players. For simplicity, I am assuming that players on rookie contracts are under the age of 23 and max contract players are those whose salary is 25% of the cap or greater. Also, I will be filtering out players who played fewer than 10 games in a given season for credibility purposes



After the initial cleaning, I worked iteratively to determine the best fitting model. VIF analysis was done to test multicollinearity and the r-squared was calculated as existing variables get dropped and feature-engineered variables get added.



*Feagure Engineering*

Variables Added:

- "Stocks". This is steals + blocks, and is commonly used as an aggregate term in NBA Analytics circles to simplify defensive stats

- 3 and D interaction term. Versatility is in high demand, so I am including an interaction term of 3 pointers made multiplied by steals and blocks.

- [True Shooting Percentage](https://www.breakthroughbasketball.com/stats/tsp_calc.html) (TS%). True Shooting Percentage is commonly used in NBA as a metric to judge overall shooting. It takes into account field goals, three pointers, and free throws into one comprehensive number. The formula is Points / (2 * (FGA + 0.44 * FTA))

- Past peak indicator. Typically, players start to decline once they hit the wrong side of 35. I would expect age to be positively correlated with salary prior to age 35, to account for player improvement and reputation/basketball IQ, but be negatively correlated as players decline in skill and take on reduced roles and/or ring chase. This indicator helps address the nonlinear relationship


- Fouls per Minute. Fouls and Turnovers are both positively correlated with Salary, despite both being detrimental to team performance. The reason for this is because the best players are on the court longer, increasing the likelihood of them committing fouls and turnovers. TOV% mitigates this phenomenon for turnovers, and fouls per minute will for personal fouls.
- Turnover Rate (TOV%). Turnovers after being controlled for time spent on court. The formula is Turnovers / (2 * (FGA + 0.44 * FTA + Turnovers))
- Salary Scale: A salary from 2018 should not be treated the same as a salary from 2021 due to player contracts increasing over time. To account for this, salaries will be scaled up via the salary cap for that given year. For example, if the salary cap in 2018 is $100m and in 2021 is $110m, all player contracts in 2018 will have a 1.1x adjustment factor applied to it.
- Square root transformation of Salary. Because independent variables are small nominally (typically under 100) while Salary is in the millions, we risk explosiveness and volatility in the model. This also reduces heteroskedasticity.



*Variable Selection*

- After accounting for redundancy and obvious multicollinearity, I narrowed the selection to the following variables: year, age, past peak, field goal attempts, 3 pointers made, free throw attempts, offensive and defensive rebounds, assists, stocks, fouls per minute, points, offensive and defensive rating, 3 and D, TS%, TOV%. I performed Variance Inflation Factor analysis to narrow it down further.
- Variance inflation factor is used to calculate multicollinearity. There is no hard threshold for variable removal, but ideally we would like to keep the VIF below 15.
- I originally added offensive and defensive ratings because it could add color by providing a metric that goes beyond the box score. However, due to high multicollinearity and low overall correlation with salary, these are dropped.
- TS% has multicollinearity issues and a surprisingly low correlation with salary. This will be removed.
- Field goal attempts has high multicollinearity and redundancy with points, so this will be removed.
- Turnovers are positively correlated with salary. While counterintuitive, this makes sense because better players get more playing time and run the offence more frequently, leading to more turnovers. In an attempt to control for this, I analyzed two commonly used metrics, assist-to-turnover ratio and turnover rate (or turnovers per 100 possessions estimate). Even after controlling for player skill and playing time, positive correlation still existed, suggesting that teams don’t typically punish turnovers. I kept TOV% out of the model for this reason.
- Although I experienced multicollinearity issues with points, I kept it in due to the obvious significance of this stat. Excluding this may lose credibility with potential stakeholders. This same logic applies to year and age.
- Despite controlling for salary cap changes, there is still a correlation between year and salary. Year has a slight negative relationship to salary, even after scaling for the increase in Salary Cap. The contracts are set in stone, with marginal raises, compared with the salary cap which has been increasing at a higher rate.
- The following variables will be used for regression analysis:
  - Year
  - Age
  - Past Peak
  - 3 Pointers made
  - Free Throw Attempts
  - Offensive Rebounds
  - Defensive Rebounds
  - Assists
  - Stocks
  - Fouls per Minute
  - Points
  - 3 and D
- As mentioned previously, the square root of salary scaled to 2021 salary cap will be the dependent variable
- In total, there are 1381 records of players from 2018-2021 that is being used for this analysis



The VIF table for modeled variables are below. This has been optimized to reduced multicollinearity; however, there are a few variables with a high VIF that I felt should stay in due to strong correlation with salary and importance in the game (such as points)



![vif](https://github.com/prathapr91/linear_regression_NBA/blob/main/images/Screen%20Shot%202022-01-12%20at%202.25.29%20PM.png?raw=true)



From here, three models were experimented with: Ordinary Least Squares, Lasso, and Ridge Regression. The latter two were included to help account for potential overfitting concerns and the penalty term alpha was optimized for intuitive fit and performance via cross validation.



**Results/Analysis**

Cross validation with the training data was conducted and the r squared, mean absolute error, and intuitive fit was assessed. After comparing the three models, below are the model performance results.

![performance](https://github.com/prathapr91/linear_regression_NBA/blob/main/images/Screen%20Shot%202022-01-12%20at%208.52.30%20AM.png?raw=true)



The MAE squared value is meant to express the MAE in real dollar terms. Because the square root of salary was taken in our model, squaring it back in can help explain the performance of this model in a more understandable way. Given the fact that NBA salaries are in the tens of millions of dollars per year, a margin of error less than $1 million is pretty robust!



As we can see here, all three models have very similar performance metrics, with a slight advantage for OLS in terms of MAE and a slight advantage for Lasso in terms of test r squared. Because the performances are not distinguishable enough to pick one over the other, let us take a look at the coefficients to see if these models pass the sanity test.



![coef](https://github.com/prathapr91/linear_regression_NBA/blob/main/images/Screen%20Shot%202022-01-12%20at%202.23.26%20PM.png?raw=true)



Although the OLS had slightly better results, the ridge regression was the best fit. OLS gave a negative coefficient for 3 pointers made, which is unrealistic given the benefits of this stat, and overestimated the fouls variable. On the other hand, Lasso removed the 3 pointers made, presumably due to it's close link to both 3 and D and points whereas Ridge tapered its impact. Because I believe 3 pointers are significant enough to keep and the difference in r^2 and MAE is very marginal, I went with Ridge. In addition, the difference between test and train r^2 suggests that there isn't significant overfitting and the MAE is relatively small, providing confidence in this model.

Below is a chart that outlines the weights given to each variable. Although age has the strongest link to salary, it is partially offset by the past peak indicator. Points is the second most valuable feature, but is not the only scoring feature that has a positive impact. Also, the high 3andD score seems to suggest that NBA teams value versatility highly.

![coefficients](https://github.com/prathapr91/linear_regression_NBA/blob/main/coefficients.png?raw=true)



As part of model validation, I visualized some diagnostics for the ridge regression model, as shown below:

![Diagnostics](https://github.com/prathapr91/linear_regression_NBA/blob/main/images/diagnostics.png?raw=true)

The residual plot seems to have a mean of 0, but there is a degree of heteroskedasticity even after the target variable was transformed. Also, the q-q plot resembling a straight line suggests a Gaussian distribution of data with minimal skew.



**Regression Model in Action**

As an example of this model performing accurately in practice, I reviewed the actual vs predicted for a number of players. Here, we see that Spencer Dinwiddie, according to our model, is valued quite similarly to what his contract would suggest.

![Dinwiddie](https://github.com/prathapr91/linear_regression_NBA/blob/main/images/Screen%20Shot%202022-01-12%20at%203.29.17%20PM.png?raw=true)



However, the real fun begins when we start to analyze anomalies. I would argue that this model is counterintuitive in that the real practical value comes from detecting players with large disparities. The reason for this is that once we spot anomalies, a team using this model can gauge which players are overrated and which ones are "diamonds in the rough". Below, I have three such examples in action:

![Anomalies](https://github.com/prathapr91/linear_regression_NBA/blob/main/images/Screen%20Shot%202022-01-12%20at%203.30.39%20PM.png?raw=true)



Pascal Siakam is a budding star for the Toronto Raptors. He won Most Improved Player in 2019 and played a key role in their championship run that year. While his actual salary is $2.4m in 2020, his gameplay suggests a salary that is 7 times that in my model!



On the other hand, Andrew Wiggins is widely considered a mediocre player. Despite this, the Minnesota Timberwolves rewarded him with a 5 year, $148m contract in 2017. As we can see here, there is a wide disparity between his annual salary and player output, something I’m sure hindered Timberwolves’ ability to add more talent and compete



This model does have it’s shortcomings, however, as this only takes into account box score statistics and not intangibles. For example, Draymond Green is someone who does not fill up the stat sheet, but impacts the game in other ways. His leadership, basketball IQ, and defense were very valuable during the Golden State Warriors championship runs, but those impacts are hard to quantify. Despite making the All NBA Defensive Team in 2019 and playing a key part in their finals run that year, my model views Draymond Green as an overvalued player.



To see my project in further detail, please visit my [GitHub Repo](https://github.com/prathapr91/linear_regression_NBA).



**Tools**

- Pandas for data manipulation
- Beautifulsoup for web scraping
- Numpy, Scikit-learn, and statsmodels for statistical analysis
- Matplotlib and Seaborn for visualization



