import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import math


def main():
    imdb_titles = pd.read_csv('title.basics.tsv', sep='\t')
    print(imdb_titles.head())
    print(imdb_titles.tail())
    
if __name__ == '__main__':
    main()


