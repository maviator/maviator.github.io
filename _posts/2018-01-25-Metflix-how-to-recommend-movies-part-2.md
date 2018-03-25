---
layout: post
title:  "Metflix: How to recommend movies - Part 2"
date:   2018-03-25 9:03:47 +0100
categories: other
tags: recommendation-engine
---

{% include image.html
            img="assets/metflix/part2_cover.jpg"
            title="coursera"
            caption="https://unsplash.com/photos/ngMtsE5r9eI"
            %}

## Where are we at?

This is what we did so far
- In [part 0](https://maviator.github.io/2018/03/09/Metflix-How-to-recommend-movies-0/), we downloaded our data from [MovieLens](https://grouplens.org/datasets/movielens/), did some EDA and created our user item matrix. The matrix has 671 unique users, 9066 unique movies and is 98.35% sparse

- In [part 1](https://maviator.github.io/2018/03/16/Metflix-How-to-recommend-movies-1/), we described 3 of the most common recommendation methods: User Based Collaborative Filtering, Item Based Collaborative Filtering and Matrix Factorization

- In part 2, this part, we will implement Matrix Factorization through ALS and find similar movies

## Matrix Factorization

We want to factorize our user item interaction matrix into a User matrix and Item matrix. To do that, we will use the Alternating Least Squares (ALS) algorithm to factorize the matrix. We could write our own implementation of ALS like how it's been done in [this post](http://blog.ethanrosenthal.com/2016/01/09/explicit-matrix-factorization-sgd-als/) or [this post](https://jessesw.com/Rec-System/), or we can use the already available, fast implementation by [Ben Frederickson](http://www.benfrederickson.com/blog/). The ALS model here is from [*implicit*](https://github.com/benfred/implicit) and can easily be added to your Python packages with `pip` or with Anaconda package manager with `conda`.


```python
import implicit
```


```python
model = implicit.als.AlternatingLeastSquares(factors=10,
                                             iterations=20,
                                             regularization=0.1,
                                             num_threads=4)
model.fit(user_item.T)
```

Here, we called ALS with the following parameters:
- 10 factors. This indicates the number of latent factors to be used
- 20 iterations
- 0.1 regularization. This regularization term is the lambda in the loss function
- 4 threads. This code can be parallelized which makes it super fast. it takes about 5 sec to train.

One thing to note is that the input for the ALS model is a item user interaction matrix, so we just have to pass the transpose of our item user matrix to the model fit function

## Recommending similar movies

It's time to get some results. We want to find similar movies for a selected title. The implicit module offers a ready to use method that returns similar items by providing the movie index in the item user matrix. However, we need to translate that index to the movie ID in the movies table


```python
movies_table = pd.read_csv("data/ml-latest-small/movies.csv")
movies_table.head()
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

    recommendation: DataFrame
        DataFrame with selected movie in first row and similar movies for N next rows

    """
    # Get movie user index from the mapping array
    user_item_id = movies.index(item_id)
    # Get similar movies from the ALS model
    similars = model.similar_items(user_item_id, N=N+1)    
    # ALS similar_items provides (id, score), we extract a list of ids
    l = [item[0] for item in similars]
    # Convert those ids to movieID from the mapping array
    ids = [movies[ids] for ids in l]
    # Make a dataFrame of the movieIds
    ids = pd.DataFrame(ids, columns=['movieId'])
    # Add movie title and genres by joining with the movies table
    recommendation = pd.merge(ids, movies_table, on='movieId', how='left')

    return recommendation
```

Let's try it!

Let's see what similar movies do we get for a James Bond Movie: Golden Eye


```python
df = similar_items(10, movies_table, movies, 5)
df
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
      <td>10</td>
      <td>GoldenEye (1995)</td>
      <td>Action|Adventure|Thriller</td>
    </tr>
    <tr>
      <th>1</th>
      <td>208</td>
      <td>Waterworld (1995)</td>
      <td>Action|Adventure|Sci-Fi</td>
    </tr>
    <tr>
      <th>2</th>
      <td>316</td>
      <td>Stargate (1994)</td>
      <td>Action|Adventure|Sci-Fi</td>
    </tr>
    <tr>
      <th>3</th>
      <td>592</td>
      <td>Batman (1989)</td>
      <td>Action|Crime|Thriller</td>
    </tr>
    <tr>
      <th>4</th>
      <td>185</td>
      <td>Net, The (1995)</td>
      <td>Action|Crime|Thriller</td>
    </tr>
    <tr>
      <th>5</th>
      <td>153</td>
      <td>Batman Forever (1995)</td>
      <td>Action|Adventure|Comedy|Crime</td>
    </tr>
  </tbody>
</table>
</div>



Interesting recommendations. One thing to notice is that all recommended movies are also in the Action genre. Remember that there was no indication to the ALS algorithm about movies genres. Let's try another example


```python
df = similar_items(500, movies_table, movies, 5)
df
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
      <td>500</td>
      <td>Mrs. Doubtfire (1993)</td>
      <td>Comedy|Drama</td>
    </tr>
    <tr>
      <th>1</th>
      <td>586</td>
      <td>Home Alone (1990)</td>
      <td>Children|Comedy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>587</td>
      <td>Ghost (1990)</td>
      <td>Comedy|Drama|Fantasy|Romance|Thriller</td>
    </tr>
    <tr>
      <th>3</th>
      <td>597</td>
      <td>Pretty Woman (1990)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>539</td>
      <td>Sleepless in Seattle (1993)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>5</th>
      <td>344</td>
      <td>Ace Ventura: Pet Detective (1994)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>



Selected movie is a comedy movie and so are the recommendations. Another interesting thing to note is that recommended movies are in the same time frame (90s).


```python
df = similar_items(1, movies_table, movies, 5)
df
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
      <td>527</td>
      <td>Schindler's List (1993)</td>
      <td>Drama|War</td>
    </tr>
    <tr>
      <th>2</th>
      <td>356</td>
      <td>Forrest Gump (1994)</td>
      <td>Comedy|Drama|Romance|War</td>
    </tr>
    <tr>
      <th>3</th>
      <td>260</td>
      <td>Star Wars: Episode IV - A New Hope (1977)</td>
      <td>Action|Adventure|Sci-Fi</td>
    </tr>
    <tr>
      <th>4</th>
      <td>318</td>
      <td>Shawshank Redemption, The (1994)</td>
      <td>Crime|Drama</td>
    </tr>
    <tr>
      <th>5</th>
      <td>593</td>
      <td>Silence of the Lambs, The (1991)</td>
      <td>Crime|Horror|Thriller</td>
    </tr>
  </tbody>
</table>
</div>



This is a case where the recommendations are not relevant. Recommending Silence of the Lambs for a user that just watched Toy Story does not seem as a good idea.

## Make it fancy

So far, the recommendations are displayed in a DataFrame. Let's make it fancy by showing the movie posters instead of just titles. This might help us later when we deploy our model and separate the work into Front End and Back End. To do that we will download movies [metadata](https://www.kaggle.com/rounakbanik/the-movies-dataset/data) that I found on Kaggle. We will need the following data:
- movies_metadata.csv
- links.csv


```python
metadata = pd.read_csv('data/movies_metadata.csv')
metadata.head(2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>adult</th>
      <th>budget</th>
      <th>genres</th>
      <th>imdb_id</th>
      <th>...</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>30000000</td>
      <td>[{'id': 16, 'name': 'Animation'}, {'id': 35, '...</td>
      <td>tt0114709</td>
      <td>...</td>
      <td>Toy Story</td>
      <td>373554033.0</td>
      <td>81.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>65000000</td>
      <td>[{'id': 12, 'name': 'Adventure'}, {'id': 14, '...</td>
      <td>tt0113497</td>
      <td>...</td>
      <td>Jumanji</td>
      <td>262797249.0</td>
      <td>104.0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 24 columns</p>
</div>



From this metadata file we only need the `imdb_id` and `poster_path` columns.


```python
image_data = metadata[['imdb_id', 'poster_path']]
image_data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>imdb_id</th>
      <th>poster_path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0114709</td>
      <td>/rhIRbceoE9lR4veEXuwCC2wARtG.jpg</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0113497</td>
      <td>/vzmL6fP7aPKNKPRTFnZmiUfciyV.jpg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0113228</td>
      <td>/6ksm1sjKMFLbO7UY2i6G1ju9SML.jpg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0114885</td>
      <td>/16XOMpEaLWkrcPqSQqhTmeJuqQl.jpg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0113041</td>
      <td>/e64sOI48hQXyru7naBFyssKFxVd.jpg</td>
    </tr>
  </tbody>
</table>
</div>



We want to merge this column with the movies table. Therefore, we need the links file to map between imdb id and movieId


```python
links = pd.read_csv("data/links.csv")
links.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>imdbId</th>
      <th>tmdbId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>114709</td>
      <td>862.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>113497</td>
      <td>8844.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>113228</td>
      <td>15602.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>114885</td>
      <td>31357.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>113041</td>
      <td>11862.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
links = links[['movieId', 'imdbId']]
```

Merging the ids will be done in 2 steps:
- First merge the poster path with the mapping links
- Then merge with movies_table

But first we need to remove missing imdb ids and extract the integer ID


```python
image_data = image_data[~ image_data.imdb_id.isnull()]
```


```python
def app(x):
    try:
        return int(x[2:])
    except ValueError:
        print x
```


```python
image_data['imdbId'] = image_data.imdb_id.apply(app)

image_data = image_data[~ image_data.imdbId.isnull()]

image_data.imdbId = image_data.imdbId.astype(int)

image_data = image_data[['imdbId', 'poster_path']]

image_data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>imdbId</th>
      <th>poster_path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>114709</td>
      <td>/rhIRbceoE9lR4veEXuwCC2wARtG.jpg</td>
    </tr>
    <tr>
      <th>1</th>
      <td>113497</td>
      <td>/vzmL6fP7aPKNKPRTFnZmiUfciyV.jpg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>113228</td>
      <td>/6ksm1sjKMFLbO7UY2i6G1ju9SML.jpg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>114885</td>
      <td>/16XOMpEaLWkrcPqSQqhTmeJuqQl.jpg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>113041</td>
      <td>/e64sOI48hQXyru7naBFyssKFxVd.jpg</td>
    </tr>
  </tbody>
</table>
</div>




```python
posters = pd.merge(image_data, links, on='imdbId', how='left')

posters = posters[['movieId', 'poster_path']]

posters = posters[~ posters.movieId.isnull()]

posters.movieId = posters.movieId.astype(int)

posters.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>poster_path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>/rhIRbceoE9lR4veEXuwCC2wARtG.jpg</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>/vzmL6fP7aPKNKPRTFnZmiUfciyV.jpg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>/6ksm1sjKMFLbO7UY2i6G1ju9SML.jpg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>/16XOMpEaLWkrcPqSQqhTmeJuqQl.jpg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>/e64sOI48hQXyru7naBFyssKFxVd.jpg</td>
    </tr>
  </tbody>
</table>
</div>




```python
movies_table = pd.merge(movies_table, posters, on='movieId', how='left')
movies_table.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>poster_path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>/rhIRbceoE9lR4veEXuwCC2wARtG.jpg</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
      <td>/vzmL6fP7aPKNKPRTFnZmiUfciyV.jpg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
      <td>/6ksm1sjKMFLbO7UY2i6G1ju9SML.jpg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
      <td>/16XOMpEaLWkrcPqSQqhTmeJuqQl.jpg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
      <td>/e64sOI48hQXyru7naBFyssKFxVd.jpg</td>
    </tr>
  </tbody>
</table>
</div>



Now that we have the poster path, we need to download them from a website. One way to do it is to use the [TMDB API](https://www.themoviedb.org/documentation/api) to get movie posters. However, we will have to make an account on the website, apply to use the API and wait for approval to get a token ID. We don't have time for that, so we'll improvise.

All movie posters can be accessed through a base URL plus the movie poster path that we got, and using HTML module for Python we can display them directly in Jupyter Notebook.


```python
from IPython.display import HTML
from IPython.display import display

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
df = similar_items(500, movies_table, movies, 5)
display_recommendations(df)
```


<img style='width: 120px; margin: 0px;                   float: left; border: 1px solid black;' src='http://image.tmdb.org/t/p/w185//jx4NvllNGN7o5bpIhcCMh26Xwaj.jpg' /><img style='width: 120px; margin: 0px;                   float: left; border: 1px solid black;' src='http://image.tmdb.org/t/p/w185//8IWPBT1rkAaI8Kpk5V3WfQRklJ7.jpg' /><img style='width: 120px; margin: 0px;                   float: left; border: 1px solid black;' src='http://image.tmdb.org/t/p/w185//rtxy3cplRFPUvruZajpcoxOQ7bi.jpg' /><img style='width: 120px; margin: 0px;                   float: left; border: 1px solid black;' src='http://image.tmdb.org/t/p/w185//fgmdaCMxXClZm2ePteLzCPySB1n.jpg' /><img style='width: 120px; margin: 0px;                   float: left; border: 1px solid black;' src='http://image.tmdb.org/t/p/w185//afkYP15OeUOD0tFEmj6VvejuOcz.jpg' /><img style='width: 120px; margin: 0px;                   float: left; border: 1px solid black;' src='http://image.tmdb.org/t/p/w185//nZirljb8XYbKTWsRQTplDGhx39Q.jpg' />



Put all of it into one small method


```python
def similar_and_display(item_id, movies_table, movies, N=5):

    df = similar_items(item_id, movies_table, movies, N=N)

    display_recommendations(df)
```


```python
similar_and_display(10, movies_table, movies, 5)
```


<img style='width: 120px; margin: 0px;                   float: left; border: 1px solid black;' src='http://image.tmdb.org/t/p/w185//5c0ovjT41KnYIHYuF4AWsTe3sKh.jpg' /><img style='width: 120px; margin: 0px;                   float: left; border: 1px solid black;' src='http://image.tmdb.org/t/p/w185//yordVJcPLh3VNRL7bXzFIBEhXRr.jpg' /><img style='width: 120px; margin: 0px;                   float: left; border: 1px solid black;' src='http://image.tmdb.org/t/p/w185//39WsfbB5BshvdbPAYRFXdsjC481.jpg' /><img style='width: 120px; margin: 0px;                   float: left; border: 1px solid black;' src='http://image.tmdb.org/t/p/w185//kBf3g9crrADGMc2AMAMlLBgSm2h.jpg' /><img style='width: 120px; margin: 0px;                   float: left; border: 1px solid black;' src='http://image.tmdb.org/t/p/w185//gKDNaAFzT21cSVeKQop7d1uhoSp.jpg' /><img style='width: 120px; margin: 0px;                   float: left; border: 1px solid black;' src='http://image.tmdb.org/t/p/w185//eTMrHEhlFPHNxpqGwpGGTdAa0xV.jpg' />


## Conclusion

In this post we implemented ALS through the implicit module to find similar movies. Additionally we did some hacking to display the movie posters instead of just DataFrame. In the next post we will see how to make recommendations for users depending on what movies they've seen. We will also see how we can set up an evaluation scheme and optimize the ALS parameters for.

Stay tuned!
