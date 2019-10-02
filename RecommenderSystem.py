# import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import  cosine_similarity
import itertools


# temporary features
features_all = ['genres', 'keywords', 'release_date', 'cast', 'director']
train_movie_info = pd.read_csv('movie_dataset.csv')


def get_movie_recs(base_movie):
    base_index = get_index(base_movie)

    for feature in features_all:
        train_movie_info[feature] = train_movie_info[feature].fillna('')
        train_movie_info['bag'] = train_movie_info.apply(feature_bag, axis=1)

    count_vectorizer = CountVectorizer()
    count_matrix = count_vectorizer.fit_transform(train_movie_info['bag'])
    similarity = cosine_similarity(count_matrix)
    sim_scores = list(enumerate(similarity[base_index]))
    sorted_sim = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]
    return sorted_sim


def feature_bag(example):
    return str(example['genres']) + str(example['keywords']) + str(example['release_date']) + str(example['cast'])+ str(example['director'])


def get_title(index):
    try:
        return train_movie_info[train_movie_info.index == index]['title'].values[0]
    except IndexError:
        return None


def get_index(title):
    try:
        return train_movie_info[train_movie_info.title == title]['index'].values[0]
    except IndexError:
        return None

def main():
    train_movie_info = pd.read_csv('movie_dataset.csv')
    # print(train_movie_info.head())
    # print(train_movie_info.tail())
    sim_list = get_movie_recs('Abc')

    for movie in itertools.islice(sim_list, 0, 10):
        print(get_title(movie[0]))



if __name__ == '__main__':
    main()




