import pandas as pd 
import utils as ut
import argparse
from typing import List

def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

def naive_recommender(ratings: pd.DataFrame, movies: pd.DataFrame, mu: int) -> List[int]:
    movie_avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
    movie_num_ratings = ratings.groupby('movieId')['rating'].count().reset_index()
    movie_num_ratings.columns = ['movieId', 'num_ratings']
    
    #normalization of ratings and num ratings
    movie_avg_ratings['rating_normalized'] = normalize(movie_avg_ratings['rating'])
    movie_num_ratings['num_ratings_normalized'] = normalize(movie_num_ratings['num_ratings'])
    
    #merging the 2 dataframes together and getting the harmonic mean, then sorting by this value
    movie_stats = pd.merge(movie_avg_ratings, movie_num_ratings, on='movieId')
    movie_stats['harmonic_mean'] = 2 / ((1 / movie_stats['rating_normalized']) + (1 / movie_stats['num_ratings_normalized']))
    movie_stats = movie_stats.sort_values(by='harmonic_mean', ascending=False)
    top_movies = movie_stats.head(mu)['movieId'].tolist()
    
    return top_movies, movie_stats.head(mu)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", default=5, type=int, help="Size of the recommendation set (sorted descendingly by relevance)")
    args = parser.parse_args()
    mu = args.mu 
    
    path_to_ml_latest_small = "../LAB_DATABASE/ml_latest_small/"
    dataset = ut.load_dataset_from_source(path_to_ml_latest_small)
    
    ratings, movies = dataset["ratings.csv"], dataset["movies.csv"]
    recommendations1, recom_df_1 = naive_recommender(ratings, movies,mu)
    
            
    print("---------------------------")
    print(recom_df_1.join(other=movies, on="movieId", how="left", lsuffix='_recom', rsuffix='_movie')["title"])

