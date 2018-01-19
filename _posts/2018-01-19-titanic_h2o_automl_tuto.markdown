---
layout: post
title:  "Automatic model selection: H2O AutoML"
date:   2018-01-19 9:03:47 +0100
categories: other
tags: MLtopics tutorial
---
In this post, we will use [H2O AutoML](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) for auto model selection and tuning. This is an easy way to get a good tuned model with minimal effort on the model selection and parameter tuning side.

We will use the [Titanic dataset from Kaggle](https://www.kaggle.com/c/titanic) and apply some feature engineering on the data before using the H2O AutoML.

## Load Dataset

```python
# Handle table-like data and matrices
import numpy as np
import pandas as pd

# get titanic & test csv files as a DataFrame
train = pd.read_csv("../input/train.csv")
test    = pd.read_csv("../input/test.csv")
```

## Feature Engineering

First thing is to remove two features from the data. We remove the 'Cabin' and 'Ticket' features just because more complicated feature engineering is necessary and it is not the context of this post.

```python
train.pop('Cabin')
test.pop('Cabin')

train.pop('Ticket')
test.pop('Ticket')
```

We extract the passenger title from the name feature and group the titles in 4 categories.

```python
dataset_title = [i.split(',')[1].split('.')[0].strip() for i in train['Name']]
train['Title'] = dataset_title
train['Title'].head()

dataset_title = [i.split(',')[1].split('.')[0].strip() for i in test['Name']]
test['Title'] = dataset_title
test['Title'].head()

# Convert to categorical values Title
train["Title"] = train["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train["Title"] = train["Title"].map({"Master":'0', "Miss":'1', "Ms":'1', "Mme":'1', "Mlle":'1', "Mrs":'1', "Mr":'2', "Rare":'3'})


test["Title"] = test["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test["Title"] = test["Title"].map({"Master":'0', "Miss":'1', "Ms":'1', "Mme":'1', "Mlle":'1', "Mrs":'1', "Mr":'2', "Rare":'3'})

train.pop('Name')
test.pop('Name')
```

## Filling missing values
We fill missing values with the mean value for numerical features and the most frequent value for categorical features.

```python
train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())

train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mode()[0])

train['Fare'] = train['Fare'].fillna(train['Fare'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
```

## Mean target encoding
```python
means = train.groupby('Age').Survived.mean()

train['Age_mean_target'] = train['Age'].map(means)
test['Age_mean_target'] = test['Age'].map(means)


means = train.groupby('Pclass').Survived.mean()

train['PClass_mean_target'] = train['Pclass'].map(means)
test['PClass_mean_target'] = test['Pclass'].map(means)


means = train.groupby('Title').Survived.mean()

train['Title_mean_target'] = train['Title'].map(means)
test['Title_mean_target'] = test['Title'].map(means)


means = train.groupby('Embarked').Survived.mean()
train['Embarked_mean_target'] = train['Embarked'].map(means)
test['Embarked_mean_target'] = test['Embarked'].map(means)
```

## Log transformation for Fare
```python
train["Fare"] = train["Fare"].apply(log1p)
test["Fare"] = test["Fare"].apply(log1p)
```

## Convert numerical feature to categorical
```python
def num2cat(x):
    return str(x)

train['Pclass_cat'] = train['Pclass'].apply(num2cat)
test['Pclass_cat'] = test['Pclass'].apply(num2cat)
train.pop('Pclass')
test.pop('Pclass')
```
## Family size feature
We extract the family size for each passenger.

```python
train['Family'] = train['SibSp'] + train['Parch'] + 1
test['Family'] = test['SibSp'] + test['Parch'] + 1

train.pop('SibSp')
test.pop('SibSp')

train.pop('Parch')
test.pop('Parch')
```

## Getting Dummies from all other categorical features
Apply one hot encoding of categorical features
```python
for col in train.dtypes[train.dtypes == 'object'].index:
    for_dummy = train.pop(col)
    train = pd.concat([train, pd.get_dummies(for_dummy, prefix=col)], axis=1)

for col in test.dtypes[test.dtypes == 'object'].index:
    for_dummy = test.pop(col)
    test = pd.concat([test, pd.get_dummies(for_dummy, prefix=col)], axis=1)    
```

## Model selection and tuning
This is the core of this post. We will use H2O AutoML for model selection and tuning.

```python
import h2o
from h2o.automl import H2OAutoML

h2o.init()
```

|H2O cluster uptime:         |22 secs|
|H2O cluster version:        |3.16.0.3|
|H2O cluster version age:    |10 days|
|H2O cluster name:           |H2O_from_python_unknownUser_ogu663|
|H2O cluster total nodes:    |1|
|H2O cluster free memory:    |25.14 Gb|
|H2O cluster total cores:    |32|
|H2O cluster allowed cores:  |32|
|H2O cluster status:         |accepting new members, healthy|
|H2O connection url:         | http://127.0.0.1:54321 |
|H2O connection proxy:       | |
|H2O internal security:      |False|
|H2O API Extensions:         |XGBoost, Algos, AutoML, Core V3, Core V4|
|Python version:             |3.6.4 final|

We load the train and test data on H2O and select the training features and target feature.
```python
htrain = h2o.H2OFrame(train)
htest = h2o.H2OFrame(test)

x =htrain.columns
y ='Survived'
x.remove(y)

# This line is added in the case of classification
htrain[y] = htrain[y].asfactor()
#htest[y] = htest[y].asfactor()
```
For the AutoML function, we just specify how long we want to train for and we're set. For this example, we will train for 120 seconds.

```python
aml = H2OAutoML(max_runtime_secs = 120)
aml.train(x=x, y =y, training_frame=htrain)

lb = aml.leaderboard
print (lb)

print('Generate predictions...')

test_y = aml.leader.predict(htest)
test_y = test_y.as_data_frame()
```

|model_id                                              | auc      |logloss |
|                                                      |          |        |
|StackedEnsemble_AllModels_0_AutoML_20180119_093938    |0.878817  |0.399037|
|StackedEnsemble_BestOfFamily_0_AutoML_20180119_093938 |0.878523  |0.400079|
|GBM_grid_0_AutoML_20180119_093938_model_0             |0.877174  |0.419855|
|GBM_grid_0_AutoML_20180119_093938_model_3             |0.877033  |0.414869|
|DRF_0_AutoML_20180119_093938                          |0.874259  |0.535281|
|GBM_grid_0_AutoML_20180119_093938_model_2             |0.872272  |0.418382|
|GBM_grid_0_AutoML_20180119_093938_model_1             |0.871849  |0.419048|
|GLM_grid_0_AutoML_20180119_093938_model_0             |0.868148  |0.416836|
|DeepLearning_0_AutoML_20180119_093938                 |0.866239  |0.419353|
|XRT_0_AutoML_20180119_093938                          |0.866177  |0.786824|

[14 rows x 3 columns]

In 120 seconds, AutoML trained 14 models. Some of these models are Gradient Boosting, Extra trees, Random Forest and Deep learning models. Also, it performed stacking of these models to get better AUC score.

This very powerful and saves a lot of time when first deciding on the model choice and parameters and can put you on the right direction.
