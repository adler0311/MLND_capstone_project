# Machine Learning Engineer Nanodegree
## Capstone Project
Joe Udacity  
December 31st, 2050

## I. Definition

### Project Overview

There has been a lot of tries to predict the stock market. As such, stock market prediction is a problem that
so many people wants to solve. One of my desire using machine learning to solve is also predicting stock market price.
The stock price is very very hard to predict but also known that it is not completely random walk. And it means
there has been somehow patterns to tackle. So I thought that the problem can be solved by using machine learning
algorithm not even 100% accurate but closer to the actual price. It is also known that there are so many company
trying to predict stock price with machine learning actually. 
In the capstone project, I created a web application that is predicting stock market price with input values 
such as tickle, which is symbol of specific company, and the future days. And training the past stock price of 
given company, the application shows the predicted value of the given future day, and the stock price graph. 

### Problem Statement

The problem to solve is very clear. The problem is to predict the future stock price and how close the predicted
price to the actual price. So I tried to predict the price of specific future day as close as possible based on
previous closing stock values. To solve this problem, I have to get the stock price data and I got the data
from 'quandl'. Because they provide very good API about the stock price. they also offer python API, so it's 
very convenient to get the stock price data. I don't need to save the stock data in csv files and load the file
to train the model or predict future price. I only do the API call and that's it.  

### Metrics

The model have to predict the specific value which means the model is regression model. So the metrics are 
related to how close the predicted value to the actual value. So I use the 'mse' or mean squared error for
metrics. Here's the definition of the MSE

![MSE image](./img/mse.png)
https://en.wikipedia.org/wiki/Mean_squared_error


## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration

The dataset is fetched by the quandl API. The size of the dataset varies from the company which to be predicted.
For example, when predict the company Amazon, I will use the recent 5 years data for the training, validating, and testing
The first 4 years dataset will be used training the predictor, and a year of the next dataset will be used
for validating the model, and the dataset from the first of this year to about March will be used to
test my model.
The dataset is obviously not included the saturday, and sunday, so the size of one year dataset is not like 
365 days.
And in this project, I only use one feature for training the predictor, Adj. Closing price. 
Here is the definition of difference between Closing and Adj. Close price.

An adjusted closing price is a stock's closing price on any given day of trading that
has been amended to include any distributions and corporate actions that occurred at any 
time before the next day's open
 https://www.investopedia.com/terms/a/adjusted_closing_price.asp

### Exploratory Visualization

![MSE image](./img/amzn_stock_price.png)

Here is the stock price of the AMZN. The stock price has been increased since 1997. For model to be fitted more accurately,
the data has to be normalized. But the data has no lower bound and upper bound. So nomalization for the whole dataset
is not a good idea. Because it is not appropriate for normalizing future price data with current value. 
Because the price keeps increasing and when normalizing whole data, the early times data would not affect much 
 to the model training. So, I decide for normalizing process to has been done for each window size or time step. 
 As we'll see below, the model I chosen to predict stock price is LSTM and window size or time step matters in the model.
 Through this normalization process, the predictor would learn the pattern no matter what the absolute prices are.  
 
 ### Algorithms and Techniques
 
 The LSTM is known for good performance in time series data prediction. So there has been many tries to predict 
 stock price with LSTM model. The LSTM is the improved model of RNN. The RNN also used for time series prediction.
 But the RNN model has gradient vanishing problem. But the LSTM model improved and solved this issue. So the LSTM
 model could train well with the data far from current time and has better performance for long term time series data.
 So I thought that this model is best for predicting stock price.

### Benchmark

I used Linear Regression model for benchmark because this model is very simple among regression models. This model 
simply predict values based on linear relation between feature and target value. Actually, I don't think that this
model has good performance because I don't think that this model couldn't train the time series patterns. But as we'll
see below, the Linear Regression model seems much more accurate than the LSTM model.


## III. Methodology

### Data Preprocessing
