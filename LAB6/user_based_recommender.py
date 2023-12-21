import pandas as pd
import numpy as np
import similarity as sim
import naive_recommender as nav
import utils as ut
from typing import List,Dict
import argparse
import time
import json
import matplotlib.pyplot as plt

def get_user_most_watched_genres(user_idx: int, ratings: pd.DataFrame, movies: pd.DataFrame, genre_matrix: pd.DataFrame) -> Dict[str, int]:
    genres_recommendation = {}
    
    movies_watched = (ratings.loc[user_idx])
    movies_watched = movies_watched[movies_watched != 0]

    for movie_id, rating in movies_watched.items():
        # Get the genres for the current movie
        movie_genres = genre_matrix.loc[movie_id]
        
        # Iterate through the genres and update the frequency dictionary
        for genre, value in movie_genres.items():
            if value == 1:
                genres_recommendation[genre] = genres_recommendation.get(genre, 0) + 1

    return dict(sorted(genres_recommendation.items(), key=lambda item: item[1], reverse=True))
    
    

def get_genre_freq_movie_info_and_print_it_out(recommendations: List[int], dataset: any, mu: int, print_info: bool) -> (List[str], Dict[str,int]):
    genres_recommendated = {} 
    movies_titles = {}
    
    if print_info:
        print("The recommended movies for this user are: \n")

    for recomendation in recommendations[:mu]:
        rec_movie = dataset["movies.csv"][dataset["movies.csv"]["movieId"]  == recomendation]
        
        #Print information
        if print_info:
            print(rec_movie)
            print (" Recomendation :Movie:{} (Genre: {})".format(rec_movie["title"].values[0], rec_movie["genres"].values[0]))
        
        #Store values
        movies_titles[recomendation] = rec_movie["title"].values[0]
        
        for genre in rec_movie["genres"].str.get_dummies().astype(str):
            
                genres_recommendated[genre] = genres_recommendated.get(genre, 0) + 1
        
    return movies_titles, dict(sorted(genres_recommendated.items(),key=lambda item: item[1], reverse=True))
        

def generate_m(movies_idx: pd.Series, users: pd.Series, ratings: pd.DataFrame) -> pd.DataFrame:
    filtered_ratings = ratings[ (ratings["movieId"].isin(movies_idx)) & (ratings["userId"].isin(users))]
    user_rating_matrix = filtered_ratings.pivot(index="userId", columns="movieId", values="rating")
    user_rating_matrix = user_rating_matrix.fillna(0)
    
    return user_rating_matrix
    
    
def user_based_recommender(target_user_idx: int, matrix: pd.DataFrame, phi: int) -> (List[int], List[int]):
    target_user_ratings = matrix.loc[target_user_idx]    
    other_users = matrix.drop(target_user_idx)
    target_user_avg_rating = (target_user_ratings.mask(target_user_ratings == 0)).mean()
        
    similarities = other_users.apply(lambda row: sim.compute_correlation_similarity(target_user_ratings[target_user_ratings != 0], row[row!=0]), axis=1).sort_values(ascending=False)
    neighbors = similarities.nlargest(phi)
    
    unrated_movies_by_target = target_user_ratings[target_user_ratings == 0].index
    
    recommendations = {}
    #for each movie we are going to apply the interest formula pred(a,s) = a_avg + sum(sim(a,b) - avg b ) / sum(sim(a,b))
    for movie in unrated_movies_by_target:
        sum1 = 0
        sum2 = 0
        for user, similarity in neighbors.items():
            user_ratings = other_users.loc[user]
            #If the user has seen the movie then its included on the computation
            if user_ratings[movie] != 0:
                sum1 += similarity * (user_ratings[movie] - user_ratings[user_ratings != 0].mean())
                sum2 += similarity 
                
        any_neighbor_seen_movie = sum2 != 0
        if any_neighbor_seen_movie: 
            #interest formula
            recommendations[movie] = target_user_avg_rating + sum1/sum2
                            
    sorted_recommendations = list(dict(sorted(recommendations.items(), key=lambda item: item[1], reverse=True)).keys())    
    return sorted_recommendations, neighbors  

def user_based_recommender_cosine(target_user_idx: int, matrix: pd.DataFrame, phi: int, item_means: pd.Series) -> (List[int], List[int]):
    target_user_ratings = matrix.loc[target_user_idx]    
    other_users = matrix.drop(target_user_idx)
    target_user_avg_rating = (target_user_ratings.mask(target_user_ratings == 0)).mean()
        
    similarities = other_users.apply(lambda row: sim.compute_adjusted_cosine_similarity(target_user_ratings[target_user_ratings != 0], row[row!=0], item_means), axis=1).sort_values(ascending=False)
    neighbors = similarities.nlargest(phi)
    
    unrated_movies_by_target = target_user_ratings[target_user_ratings == 0].index
    
    recommendations = {}
    #for each movie we are going to apply the interest formula pred(a,s) = a_avg + sum(sim(a,b) - avg b ) / sum(sim(a,b))
    for movie in unrated_movies_by_target:
        sum1 = 0
        sum2 = 0
        for user, similarity in neighbors.items():
            user_ratings = other_users.loc[user]
            #If the user has seen the movie then its included on the computation
            if user_ratings[movie] != 0:
                sum1 += similarity * (user_ratings[movie] - user_ratings[user_ratings != 0].mean())
                sum2 += similarity 
                
        any_neighbor_seen_movie = sum2 != 0
        if any_neighbor_seen_movie: 
            #interest formula
            recommendations[movie] = target_user_avg_rating + sum1/sum2
                            
    sorted_recommendations = list(dict(sorted(recommendations.items(), key=lambda item: item[1], reverse=True)).keys())    
    return sorted_recommendations, neighbors    



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--phi", default=5, type=int, help="Size of the neighborhood (How many similar users to the target are considered)")
    parser.add_argument("--mu", default=5, type=int, help="Size of the recommendation set (sorted descendingly by relevance)")
    parser.add_argument("--user", default=64, type=int, help="Target user for the recommendation")
    parser.add_argument("--theta", default=5, type=int, help="Number of most relevant genres to consider for each user")

    args = parser.parse_args()
    phi = args.phi
    mu = args.mu 
    target_user_idx = args.user
    theta = args.theta
        
    # Load the dataset, the genre matrix and the ratings data
    path_to_ml_latest_small = "../LAB_DATABASE/ml_latest_small/"
    dataset = ut.load_dataset_from_source(path_to_ml_latest_small)
    matrixmpa_genres = ut.matrix_genres(dataset["movies.csv"])

    # Ratings data
    val_movies = mu
    ratings_train, ratings_val = ut.split_users(dataset["ratings.csv"], val_movies)
    
    movies_idx = dataset["movies.csv"]["movieId"]
    users_idy = list(set(ratings_train["userId"].values))
    
    #GENERATE A MOVIE REVIEW MATRIX    
    m_train = generate_m(movies_idx, users_idy, ratings_train)
    m_validation = generate_m(movies_idx, users_idy, ratings_val)
    
    item_train_mean = m_train.replace(0, np.nan, inplace=False)
    item_train_mean = item_train_mean.mean()
    
    most_watched_genres = get_user_most_watched_genres(target_user_idx, ratings=m_train, movies=dataset["movies.csv"], genre_matrix=matrixmpa_genres)
                
    #GET A RECOMENDATION    
    recommendations_cos, neighbors_cos = user_based_recommender_cosine(target_user_idx, m_train,phi, item_train_mean)
    recommendations, neighbors = user_based_recommender(target_user_idx, m_train, phi)
    print(f"The neighbors of user {target_user_idx} are {neighbors} \n")
    print(f"The neighbors of user cosine {target_user_idx} are {neighbors_cos} \n")
     
    movies, genres = get_genre_freq_movie_info_and_print_it_out(
        recommendations=recommendations,
        dataset=dataset, mu=mu, print_info=True
    )
    
    print(f"Recommended genres")   
    print(genres)
    
    movies, genres = get_genre_freq_movie_info_and_print_it_out(
        recommendations=recommendations_cos,
        dataset=dataset, mu=mu, print_info=True
    )
    
    print(f"Recommended cosine genres")
    print(genres)
    
    print(f"\nActual most watched genres")
    print(most_watched_genres)
    print("\nValidation movies (Following ratings after recommendation)\n")
    
    # Validation  
    
    validation_movies = (ratings_val[ratings_val["userId"] == target_user_idx])
    validation_movies = validation_movies[validation_movies["rating"] != 0]
    print(validation_movies.join(dataset["movies.csv"], on="movieId", how="left", lsuffix="lol", rsuffix="juan")["title"])
