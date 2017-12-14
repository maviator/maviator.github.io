---
layout: post
title:  "The Master tier Journey 1: November 2017"
date:   2017-11-28 16:03:47 +0100
categories: kaggle
tags: kaggle master-journey 
---

During this first month, I worked on 3 Kaggle competitions. The first two were tutorial competitions to practice classification and regression and the third one is a featured competition, that means it counts towards the goal. All of the notebooks and codes for these competitions is on my Github.

## [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

In this tutorial competition, the goal is to predict the price of houses in Ames, Iowa. The dataset has 79 variables (property surface, number of rooms, garage area…). The train data has 1460 rows and the test data has 1459. This is a regression competition as the goal is to predict a numerical variable (house price). The evaluation metric is the Root-Mean-Squared-Error (RMSE) between the log of the predicted price and the actual price.

After exploring the data, trying some models and implementing some of the ideas from previously submitted kernels, this was my final solution. I started by removing outliers that were affecting the predictive model, removing the features that had more than 50% of missing data. For the other features that had missing values, the missing data was either replaced by the mean value in case of a numerical value or by the mode in the case of categorical values. The next step was to reduce the target variable skewness by applying log transformation to the sale price. The last step before training the model is to use a one-hot encoding of categorical features.

The final model is the average prediction between Gradient Boosting Regression, LASSO regression and an Elastic Net regression. This solution was able to get me a top 13% score on the leaderboard. I can further improve the solution later, but it is satisfying for now.

Things that I learned from this competition:
- Model averaging and stacking
- Log transformation to reduce skewness
- The effect of outliers on regression models

## [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

For this second competition, the goal is to predict the survival of passengers aboard the Titanic based on 10 variables (age, sex, class, title…). This is a classification problem as the target variable has binary values. There are 891 rows for the train data and 418 rows for the test data. The evaluation metric is accuracy (the percentage of correct predictions).

In this competition I started with a baseline model that removed all features that required some engineering to have a baseline score to improve. At each iteration, I select one of the features that were removed, apply some feature engineering and test the model again. If the score improves, I keep the feature, otherwise I discard it. The final model was a voting system of 5 models. This was the equivalent of the model averaging but for binary target values. My submission got me to top 32% on the leaderboard which I think it is sufficient for the time being.

Things I learned from this competition:
- The importance of a simple baseline model
- The iterative process of feature engineering
- Grid search to find the best parameters for each model
- The voting classifier

## [Porto Seguro’s Safe Driver Prediction](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction)

This is my first featured competition. The goal here is to predict the probability that a driver will initiate an auto insurance claim in the next year. This is also a binary classification problem. The train data has 595212 rows and the test data has 892816 rows. The evaluation metric is the Normalized Gini coefficient.

This competition is still ongoing at the time of writing this post but I can already tell that it’s harder than the previous ones. It is already pushing the hardware on my PC (Intel i5 with 8 GB of RAM) to the limit. Creating multiple copies of the data without paying attention to memory usage limited my work. After looking for how other kagglers dealt with this problem, I found that converting features datatypes to a less memory consuming types saved me more than 50% in RAM. Since the data was unbalanced, undersampling saved me more space without losing significantly in prediction scores. In term of model training, it takes around 10 min to train with cross validation for the models. Will this be manageable with bigger datasets ? Probably not. But we will find a solution when faced with such case.

Things I learned so far:
- Data types conversion to save memory space
- Undersampling
- Target encoding of categorical features
- Out of fold model stacking

I will be updating on the progress in this competition in next month’s update but it’s been an awesome learning experience so far.

## Goal Progress:
- 0/2 Bronze medals
- 0/2 Silver medals
- 0/1 Gold medals
- No final rank yet for competitive competition

---

These posts are created entirely for my own personal progress tracking. If you are reading this, thank you and I hope you like it. If you have any thoughts, advice or topics that you want me to be write about, please reach out to me on my email.

Till next time!
