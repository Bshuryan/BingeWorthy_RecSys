import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
    if sorted_sim:
        recs_top = map(get_title, sorted_sim[:10])
        recs_top_full = list(map(get_additional_info, recs_top))
        # print(recs_top_full)
        # print('\n'.join(list(recs_top_full)))
    else:
        print("Movie not found!")
    return recs_top_full


def feature_bag(example):
    return str(example['genres']) * 6 + str(example['keywords']) * 5 + str(example['keywords']) + \
            str(example['cast']) * 3 + str(example['director']) * 3


# modified to accept a tuple where the first argument is the index
def get_title(index):
    # print(index)
    try:
        return train_movie_info[train_movie_info.index == index[0]]['title'].values[0]
    except IndexError:
        return None


def get_index(title):
    try:
        return train_movie_info[train_movie_info.title == title]['index'].values[0]
    except IndexError:
        return None


# function that will be mapped to movies array in order to get description, director, actors?, year, genre, runtime
# returns: [title, genres, overview, year, director, actors, runtime]
def get_additional_info(movie_title):
    # print(train_movie_info[train_movie_info.title == movie_title]['release_date'])
    movie_rundown = [movie_title, train_movie_info[train_movie_info.title == movie_title]['genres'].values[0],
                     train_movie_info[train_movie_info.title == movie_title]['overview'].values[0],
                     train_movie_info[train_movie_info.title == movie_title]['release_date'].values[0].split('-')[0],
                     train_movie_info[train_movie_info.title == movie_title]['director'].values[0],
                     train_movie_info[train_movie_info.title == movie_title]['cast'].values[0],
                     str(train_movie_info[train_movie_info.title == movie_title]['runtime'].values[0]).split('.')[0] + ' min']
    return movie_rundown



def parse_description(description):
    # keyword extraction to obtain essential words from movie/tv show descriptions
    pass


def main():
    movie_recs_final = get_movie_recs('Iron Man')
    print(movie_recs_final)





if __name__ == '__main__':
    main()
