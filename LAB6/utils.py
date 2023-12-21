#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

def load_dataset_from_source(path_to_ml_latest_small: str) -> dict: 
    """
      Load a dataset from a directory containing MovieLens data files and extract movie release years.
    
      This function reads the MovieLens dataset files from the specified directory and extracts
      movie release years from the 'title' column in the 'movies.csv' file. It returns a dictionary
      containing the loaded dataframes and the added 'year' column in the 'movies.csv' dataframe.
    
      Args:
          path_to_ml_latest_small (str): The path to the directory containing MovieLens dataset files.
    
      Returns: 
          dict: A dictionary containing the following dataframes:
              - 'ratings.csv': User ratings data
              - 'tags.csv': User-generated tags data
              - 'movies.csv': Movie information data with an additional 'year' column
              - 'links.csv': Movie links data          
    """
    contents = os.listdir(path_to_ml_latest_small)
    expected_content = ['README.txt', 'ratings.csv', 'tags.csv', 
                        'movies.csv', 'links.csv']
    if set(contents) ==  set(expected_content): 
        dataset = {}
        for file in expected_content[1:]: 
            file_path = os.path.join(path_to_ml_latest_small, file)
            dataset[file] = pd.read_csv(file_path)
    else:
        raise ValueError(f"Expected files {expected_content} not found in {path_to_ml_latest_small}")
    
    dataset["movies.csv"]["year"] = dataset["movies.csv"]['title'].str.extract(r'(?:\((\d{4})\))?\s*$', expand=False)
        
    return dataset


def split_users(ratings: object , k: int = 5) -> tuple: 
    """
    Splits user ratings into training and validation sets.
    
    Args:
        ratings (DataFrame): The input ratings DataFrame containing "userId" column.
        k (int): The number of ratings to include in the validation set for each user.
    
    Returns:
        Tuple of DataFrames: Two DataFrames, the first containing training data, and the second containing validation data.
    """
    ratings_train, rating_validation = [], []
        
    for idx, ratings_users in ratings.groupby("userId"):
        ratings_users = ratings_users.sample(frac=1, random_state=42)
        rating_validation.append(ratings_users.iloc[:k]) # First top k rows
        ratings_train.append(ratings_users.iloc[k:]) # Remaining M-k rows
        
    return pd.concat(ratings_train), pd.concat(rating_validation)


def matrix_genres(movies: object) -> dict: 
    """
      Generate a binary movie-genre matrix from a DataFrame of movie information.
    
      This function takes a DataFrame containing movie information, extracts the genres associated with each movie,
      and generates a binary matrix representing the presence or absence of each genre for each movie.
    
      Args:
          movies (pd.DataFrame): A DataFrame containing movie information, including 'movieId' and 'genres' columns.
    
      Returns:
          pd.DataFrame: A binary movie-genre matrix where rows represent movies (indexed by 'movieId'),
                       and columns represent unique genres. Each cell in the matrix contains 1 if the movie belongs to
                       the genre, and 0 otherwise.
    """
    # Create a dictionary where keys are movieIds and values are sets of genres
    dict_movies = {movie["movieId"]: set(movie["genres"].split("|")) for _, movie in movies.iterrows()}
         
    unique_genres = set.union(*dict_movies.values())

    # Populate the DataFrame matrix with 1s for the genres each movie belongs to
    matrix = pd.DataFrame(np.zeros([len(dict_movies.keys()), len(unique_genres)]), 
                          columns=list(unique_genres), index=list(dict_movies.keys()))
    
    for idx_movie, genres in dict_movies.items():
        for genre in genres: matrix.loc[idx_movie][genre] = 1
     
    return matrix


if __name__ == "__main__":

    path_to_ml_latest_small = "../LAB_DATABASE/ml_latest_small/"
    dataset = load_dataset_from_source(path_to_ml_latest_small)

    movies = dataset["movies.csv"]
    test_matrix = matrix_genres(movies)
    assert test_matrix.sum().sum() == 22084.0

    ratings = dataset["ratings.csv"]
    ratings_train, ratings_val = split_users(ratings, 5)
    assert ratings_val.shape == (3050, 4)
