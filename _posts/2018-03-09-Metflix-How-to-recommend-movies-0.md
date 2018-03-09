---
layout: post
title:  "Metflix: How to recommend movies - Part 0"
date:   2018-03-09 9:03:47 +0100
categories: other
tags: recommendation-engine
---

{% include image.html
            img="assets/metflix/rs_cover.jpg"
            title="coursera"
            %}

In this series of posts, I will try to build a recommendation engine to recommend similar movies to a selected title or recommend movies to a user that rates a couple of movies. This part 0 will be about how to get the data, load it and do some basic Exploratory Data Analysis (EDA). We will finish this post by creating a user-item interaction matrix that represents the ratings given by a particular user to a particular movie. In the next post, I will post how to implement:
- User Based Collaborative Filtering (UBCF)
- Item Based Collaborative filtering (IBCF)
- Alternating Least Squares (ALS) method for Matrix Factorization (MF)

As for future posts, my intentions are to take the simple model that we build all the way to deployment by running it as a web service. I don't know how to do that yet and that's why I'm creating this series of post to document the journey.

## The data

The data for this project is the [MovieLens](https://grouplens.org/datasets/movielens/) dataset. There two datasets that needs to be downloaded:
- `ml-latest-small` this data has 100.000 ratings and 1.300 tag applications applied to 9000 movies by 700 users. We will use this data for initial prototyping to go fast.
- `ml-latest` this data has 26.000.000 ratings and 750.000 tag applications to 45.000 movies by 270.000 users. This is much bigger data that we will use for final model deployment.

## Load MovieLens data
The links to download both datasets are presented here

```python
complete_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest.zip'
small_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
```


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.sparse as sparse

%matplotlib inline
```

Let's start with the ratings data


```python
ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
```


```python
ratings.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>31</td>
      <td>2.5</td>
      <td>1260759144</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1029</td>
      <td>3.0</td>
      <td>1260759179</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1061</td>
      <td>3.0</td>
      <td>1260759182</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1129</td>
      <td>2.0</td>
      <td>1260759185</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1172</td>
      <td>4.0</td>
      <td>1260759205</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings.userId.nunique(), ratings.movieId.nunique()
```




    (671, 9066)



We have 671 unique users and 9066 unique items. Let's see how many movies each user have rated.

But first let's disregard the time-stamp as it's not needed for this context. Time information can be helpful to us if we want to split the ratings data and make learning iterative by updating the ratings with time. This might be a good feature to add to a deployed model for example, by adding new incoming ratings daily or weekly or monthly and retraining the entire model.


```python
ratings = ratings[["userId", "movieId", "rating"]]
```


```python
data = ratings.groupby("userId", as_index=False).agg({"movieId": 'count'})
```


```python
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>76</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>51</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>204</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.movieId.hist()
```




{% include image.html
            img="assets/metflix/0.png"
            title="coursera"
            caption="Histogram of number of items rated by users"
            %}



```python
data.movieId.describe()
```




    count     671.000000
    mean      149.037258
    std       231.226948
    min        20.000000
    25%        37.000000
    50%        71.000000
    75%       161.000000
    max      2391.000000
    Name: movieId, dtype: float64



On average, users rated 149 movies. Interestingly, all users have rated at least 20 movies. This is useful when making the train and test data to evaluate our models.

Let's load the movies data


```python
movies = pd.read_csv("data/ml-latest-small/movies.csv")
```


```python
movies.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>



This data will be helpful to match movieId with the movie title

Now, let's make a user item interaction matrix. A user item interaction matrix is where each user is represented by a vector of length the number of unique items. From our ratings data, we convert each row into the specific user and item interaction and 0 everywhere else. This will result in a big matrix with a lot of zeros therefore we need to use sparse matrices to save memory. Keep in mind that the final data will have 45.000 movies and 270.000 users.


```python
users = list(np.sort(ratings.userId.unique())) # Get our unique customers
movies = list(ratings.movieId.unique()) # Get our unique products that were purchased
rating = list(ratings.rating) # All of our purchases

rows = ratings.userId.astype('category', categories = users).cat.codes
# Get the associated row indices
cols = ratings.movieId.astype('category', categories = movies).cat.codes
# Get the associated column indices
user_item = sparse.csr_matrix((rating, (rows, cols)), shape=(len(users), len(movies)))
```


```python
matrix_size = user_item.shape[0]*user_item.shape[1] # Number of possible interactions in the matrix
num_purchases = len(user_item.nonzero()[0]) # Number of items interacted with
sparsity = 100*(1 - (1.0*num_purchases/matrix_size))
sparsity
```




    98.35608583913366




```python
user_item
```




    <671x9066 sparse matrix of type '<type 'numpy.float64'>'
    	with 100004 stored elements in Compressed Sparse Row format>



The user item matrix has 671 unique users and 9066 unique movies which checks with what we found earlier in our data exploration. This matrix is 98.35% sparse which seems too low but it should be usable for the algorithms that we will use.

I will stop here for this post. Next post will be dedicated to the different algorithms that we will test. We will see UBCF, IBCF, ALS for Matrix Factorization and decide on an evaluation metric. All the code for this project is available on my [Github](https://github.com/maviator/metflix)
