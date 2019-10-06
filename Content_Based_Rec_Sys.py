import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import  cosine_similarity
import itertools


# temporary features
features_all = ['genres', 'keywords', 'release_date', 'cast', 'director']
train_movie_info = pd.read_csv('movie_dataset_rec.csv')


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
    return str(example['genres']) + str(example['keywords']) + str(example['genres']) + str(example['keywords']) + \
           str(example['release_date']) + str(example['cast']) + str(example['director'])


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


def parse_description(description):
    # keyword extraction to obtain essential words from movie/tv show descriptions
    pass


def main():
    sim_list = get_movie_recs('Batman Begins')

    if sim_list:
        for movie in itertools.islice(sim_list, 0, 10):
            print(get_title(movie[0]))
    else:
        print("Movie not found!")


if __name__ == '__main__':
    main()




