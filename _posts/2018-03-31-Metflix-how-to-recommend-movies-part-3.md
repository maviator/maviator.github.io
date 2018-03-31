---
layout: post
title:  "Metflix: How to recommend movies - Part 3"
date:   2018-03-31 9:03:47 +0100
categories: other
tags: recommendation-engine
---

{% include image.html
            img="assets/metflix/3_cover.png"
            title="coursera"
            caption="Netflix Because you watched feature"
            %}

## Where are we at?

This is what we did so far
- In [part 0](https://maviator.github.io/2018/03/09/Metflix-How-to-recommend-movies-0/), we downloaded our data from [MovieLens](https://grouplens.org/datasets/movielens/), did some EDA and created our user item matrix. The matrix has 671 unique users, 9066 unique movies and is 98.35% sparse

- In [part 1](https://maviator.github.io/2018/03/16/Metflix-How-to-recommend-movies-1/), we described 3 of the most common recommendation methods: User Based Collaborative Filtering, Item Based Collaborative Filtering and Matrix Factorization

- In [part 2](https://maviator.github.io/2018/03/25/Metflix-how-to-recommend-movies-part-2/), we implemented Matrix Factorization through ALS and found similar movies

- In part 3, this part, we recommend movies to users based on what movies they've rated. We also make an attempt to clone Netflix's "because you watched X" feature and make a complete page recommendation with trending movies


## Recommending Movies to users

We pick up our code where we trained the ALS model from *implicit* library. Previous code to load and process the data can be found in the previous posts in this series or on my [Github]().

```python
model = implicit.als.AlternatingLeastSquares(factors=10,
                                             iterations=20,
                                             regularization=0.1,
                                             num_threads=4)
model.fit(user_item.T)
```

First let's write a function that returns the movies that a particular user had rated


```python
def get_rated_movies_ids(user_id, user_item, users, movies):
    """
    Input
    -----

    user_id: int
        User ID

    user_item: scipy.Sparse Matrix
        User item interaction matrix

    users: np.array
        Mapping array between user ID and index in the user item matrix

    movies: np.array
        Mapping array between movie ID and index in the user item matrix

    Output
    -----

    movieTableIDs: python list
        List of movie IDs that the user had rated

    """
    user_id = users.index(user_id)
    # Get matrix ids of rated movies by selected user
    ids = user_item[user_id].nonzero()[1]
    # Convert matrix ids to movies IDs
    movieTableIDs = [movies[item] for item in ids]

    return movieTableIDs
```


```python
movieTableIDs = get_rated_movies_ids(1, user_item, users, movies)
```


```python
rated_movies = pd.DataFrame(movieTableIDs, columns=['movieId'])
rated_movies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1029</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1061</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1129</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1172</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1263</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1287</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1293</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1339</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1343</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1371</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1405</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1953</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2105</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2150</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2193</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2294</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2455</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2968</td>
    </tr>
    <tr>
      <th>19</th>
      <td>3671</td>
    </tr>
  </tbody>
</table>
</div>




```python
def get_movies(movieTableIDs, movies_table):
    """
    Input
    -----

    movieTableIDs: python list
        List of movie IDs that the user had rated

    movies_table: pd.DataFrame
        DataFrame of movies info

    Output
    -----

    rated_movies: pd.DataFrame
        DataFrame of rated movies

    """

    rated_movies = pd.DataFrame(movieTableIDs, columns=['movieId'])

    rated_movies = pd.merge(rated_movies, movies_table, on='movieId', how='left')

    return rated_movies
```


```python
movieTableIDs = get_rated_movies_ids(1, user_item, users, movies)
df = get_movies(movieTableIDs, movies_table)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>31</td>
      <td>Dangerous Minds (1995)</td>
      <td>Drama</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1029</td>
      <td>Dumbo (1941)</td>
      <td>Animation|Children|Drama|Musical</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1061</td>
      <td>Sleepers (1996)</td>
      <td>Thriller</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1129</td>
      <td>Escape from New York (1981)</td>
      <td>Action|Adventure|Sci-Fi|Thriller</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1172</td>
      <td>Cinema Paradiso (Nuovo cinema Paradiso) (1989)</td>
      <td>Drama</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1263</td>
      <td>Deer Hunter, The (1978)</td>
      <td>Drama|War</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1287</td>
      <td>Ben-Hur (1959)</td>
      <td>Action|Adventure|Drama</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1293</td>
      <td>Gandhi (1982)</td>
      <td>Drama</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1339</td>
      <td>Dracula (Bram Stoker's Dracula) (1992)</td>
      <td>Fantasy|Horror|Romance|Thriller</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1343</td>
      <td>Cape Fear (1991)</td>
      <td>Thriller</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1371</td>
      <td>Star Trek: The Motion Picture (1979)</td>
      <td>Adventure|Sci-Fi</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1405</td>
      <td>Beavis and Butt-Head Do America (1996)</td>
      <td>Adventure|Animation|Comedy|Crime</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1953</td>
      <td>French Connection, The (1971)</td>
      <td>Action|Crime|Thriller</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2105</td>
      <td>Tron (1982)</td>
      <td>Action|Adventure|Sci-Fi</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2150</td>
      <td>Gods Must Be Crazy, The (1980)</td>
      <td>Adventure|Comedy</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2193</td>
      <td>Willow (1988)</td>
      <td>Action|Adventure|Fantasy</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2294</td>
      <td>Antz (1998)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2455</td>
      <td>Fly, The (1986)</td>
      <td>Drama|Horror|Sci-Fi|Thriller</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2968</td>
      <td>Time Bandits (1981)</td>
      <td>Adventure|Comedy|Fantasy|Sci-Fi</td>
    </tr>
    <tr>
      <th>19</th>
      <td>3671</td>
      <td>Blazing Saddles (1974)</td>
      <td>Comedy|Western</td>
    </tr>
  </tbody>
</table>
</div>


Now, let's recommend movieIDs for a particular user ID based on the movies that they rated.

```python
def recommend_movie_ids(user_id, model, user_item, users, movies, N=5):
    """
    Input
    -----

    user_id: int
        User ID

    model: ALS model
        Trained ALS model

    user_item: sp.Sparse Matrix
        User item interaction matrix so that we do not recommend already rated movies

    users: np.array
        Mapping array between User ID and user item index

    movies: np.array
        Mapping array between Movie ID and user item index

    N: int (default =5)
        Number of recommendations

    Output
    -----

    movies_ids: python list
        List of movie IDs
    """

    user_id = users.index(user_id)

    recommendations = model.recommend(user_id, user_item, N=N)

    recommendations = [item[0] for item in recommendations]

    movies_ids = [movies[ids] for ids in recommendations]

    return movies_ids
```


```python
movies_ids = recommend_movie_ids(1, model, user_item, users, movies, N=5)
movies_ids
```




    [1374, 1127, 1214, 1356, 1376]




```python
movies_rec = get_movies(movies_ids, movies_table)
movies_rec
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>1374</td>
      <td>Star Trek II: The Wrath of Khan (1982)</td>
      <td>Action|Adventure|Sci-Fi|Thriller</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1127</td>
      <td>Abyss, The (1989)</td>
      <td>Action|Adventure|Sci-Fi|Thriller</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1214</td>
      <td>Alien (1979)</td>
      <td>Horror|Sci-Fi</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1356</td>
      <td>Star Trek: First Contact (1996)</td>
      <td>Action|Adventure|Sci-Fi|Thriller</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1376</td>
      <td>Star Trek IV: The Voyage Home (1986)</td>
      <td>Adventure|Comedy|Sci-Fi</td>
    </tr>
  </tbody>
</table>
</div>



```python
from IPython.display import HTML
from IPython.display import display

def display_posters(df):

    images = '<p>'
    for ref in df.poster_path:
            if ref != '':
                link = 'http://image.tmdb.org/t/p/w185/' + ref
                images += "<img style='width: 120px; margin: 0px; \
                  float: left; border: 1px solid black;' src='%s' />" \
              % link
    images += '</p>'
    display(HTML(images))
```


```python
display_posters(movies_rec)
```


{% include image.html
            img="assets/metflix/3_1.png"
            title="coursera"
            %}



```python
movies_ids = recommend_movie_ids(100, model, user_item, users, movies, N=7)
movies_rec = get_movies(movies_ids, movies_table)
display_posters(movies_rec)
```

{% include image.html
            img="assets/metflix/3_2.png"
            title="coursera"
            %}

## Because You've watched

Let's implement Netflix latest features. It's about recommending movies based on what you've watched. This is similar to what we already did, but this time, it's more selective. Here's how we will do it: We will choose random 5 movies that a user had watched and for each movie recommend similar movies to it. Finally, we display all of them in a one page layout


```python
def similar_items(item_id, movies_table, movies, N=5):
    """
    Input
    -----

    item_id: int
        MovieID in the movies table

    movies_table: DataFrame
        DataFrame with movie ids, movie title and genre

    movies: np.array
        Mapping between movieID in the movies_table and id in the item user matrix

    N: int
        Number of similar movies to return

    Output
    -----
    df: DataFrame
        DataFrame with selected movie in first row and similar movies for N next rows
    """
    # Get movie user index from the mapping array
    user_item_id = movies.index(item_id)
    # Get similar movies from the ALS model
    similars = model.similar_items(user_item_id, N=N+1)    
    # ALS similar_items provides (id, score), we extract a list of ids
    l = [item[0] for item in similars[1:]]
    # Convert those ids to movieID from the mapping array
    ids = [movies[ids] for ids in l]
    # Make a dataFrame of the movieIds
    ids = pd.DataFrame(ids, columns=['movieId'])
    # Add movie title and genres by joining with the movies table
    recommendation = pd.merge(ids, movies_table, on='movieId', how='left')

    return recommendation
```


```python
def display_recommendations(df):

    images = ''
    for ref in df.poster_path:
            if ref != '':
                link = 'http://image.tmdb.org/t/p/w185/' + ref
                images += "<img style='width: 120px; margin: 0px; \
                  float: left; border: 1px solid black;' src='%s' />" \
              % link
    display(HTML(images))
```


```python
def similar_and_display(item_id, movies_table, movies, N=5):

    df = similar_items(item_id, movies_table, movies, N=N)

    df.dropna(inplace=True)

    display_recommendations(df)
```


```python
def because_you_watched(user, user_item, users, movies, k=5, N=5):
    """
    Input
    -----

    user: int
        User ID

    user_item: scipy sparse matrix
        User item interaction matrix

    users: np.array
        Mapping array between User ID and user item index

    movies: np.array
        Mapping array between Movie ID and user item index

    k: int
        Number of recommendations per movie

    N: int
        Number of movies already watched chosen

    """

    movieTableIDs = get_rated_movies_ids(user, user_item, users, movies)
    df = get_movies(movieTableIDs, movies_table)

    movieIDs = random.sample(df.movieId, N)

    for movieID in movieIDs:
        title = df[df.movieId == movieID].iloc[0].title
        print("Because you've watched ", title)
        similar_and_display(movieID, movies_table, movies, k)
```


```python
because_you_watched(500, user_item, users, movies, k=5, N=5)
```

("Because you watched ", 'Definitely, Maybe (2008)')



{% include image.html
            img="assets/metflix/3_3.png"
            title="coursera"
            %}


("Because you watched ", 'Pocahontas (1995)')

{% include image.html
            img="assets/metflix/3_4.png"
            title="coursera"
            %}


("Because you watched ", 'Simpsons Movie, The (2007)')

{% include image.html
            img="assets/metflix/3_5.png"
            title="coursera"
            %}

("Because you watched ", 'Catch Me If You Can (2002)')

{% include image.html
            img="assets/metflix/3_6.png"
            title="coursera"
            %}


("Because you watched ", 'Risky Business (1983)')

{% include image.html
            img="assets/metflix/3_7.png"
            title="coursera"
            %}

## Trending movies

Let's also implement trending movies. In our context, trending movies are movies that been rated the most by users


```python
def get_trending(user_item, movies, movies_table, N=5):
    """
    Input
    -----

    user_item: scipy sparse matrix
        User item interaction matrix to use to extract popular movies

    movies: np.array
        Mapping array between movieId and ID in the user_item matrix

    movies_table: pd.DataFrame
        DataFrame for movies information

    N: int
        Top N most popular movies to return

    """

    binary = user_item.copy()
    binary[binary !=0] = 1

    populars = np.array(binary.sum(axis=0)).reshape(-1)

    movieIDs = populars.argsort()[::-1][:N]

    movies_rec = get_movies(movieIDs, movies_table)

    movies_rec.dropna(inplace=True)

    print("Trending Now")

    display_posters(movies_rec)
```


```python
get_trending(user_item, movies, movies_table, N=6)
```

Trending Now


{% include image.html
            img="assets/metflix/3_8.png"
            title="coursera"
            %}

## Page recommendation

Let's put everything in a timeline method. The timeline method will get the user ID and display trending movies and recommendations based on similar movies that that user had watched.

```python
def my_timeline(user, user_item, users, movies, movies_table, k=5, N=5):

    get_trending(user_item, movies, movies_table, N=N)

    because_you_watched(user, user_item, users, movies, k=k, N=N)
```


```python
my_timeline(500, user_item, users, movies, movies_table, k=5, N=5)
```

Trending Now


{% include image.html
            img="assets/metflix/3_8.png"
            title="coursera"
            %}


("Because you watched ", 'Definitely, Maybe (2008)')



{% include image.html
            img="assets/metflix/3_3.png"
            title="coursera"
            %}


("Because you watched ", 'Pocahontas (1995)')

{% include image.html
            img="assets/metflix/3_4.png"
            title="coursera"
            %}


("Because you watched ", 'Simpsons Movie, The (2007)')

{% include image.html
            img="assets/metflix/3_5.png"
            title="coursera"
            %}

("Because you watched ", 'Catch Me If You Can (2002)')

{% include image.html
            img="assets/metflix/3_6.png"
            title="coursera"
            %}


("Because you watched ", 'Risky Business (1983)')

{% include image.html
            img="assets/metflix/3_7.png"
            title="coursera"
            %}

## Export trained models to be used in production

At this point, we want to get our model into production. We want to create a web service where a user will provide a userid to the service and the service will return all of the recommendations including the trending and the "because you've watched".

To do that, We first export the trained model and the used data for use in the web service.


```python
import scipy.sparse
scipy.sparse.save_npz('model/user_item.npz', user_item)
```


```python
np.save('model/movies.npy', movies)
np.save('model/users.npy', users)
movies_table.to_csv('model/movies_table.csv', index=False)
```


```python
from sklearn.externals import joblib
joblib.dump(model, 'model/model.pkl')
```




    ['model/model.pkl']

## Conclusion
In this post, we recommend movies to users based on their movie rating history. From there, we tried to clone the "because you watched" feature from Netflix and also display Trending movies as movies that were rated the most number of times. In the next post, we will try to put our work in a web service, where a user requests movie recommendations by providing its user ID.

Stay tuned!
