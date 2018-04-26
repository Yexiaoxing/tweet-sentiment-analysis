print("Loading datasets... It may take a longer time.")
import sentiment_mod as senti
import pickle
import pandas as pd
import numpy as np
from multiprocessing import cpu_count, Pool


print("The following result should be neg and 1.0")
print(senti.sentiment("He is an incapable person. His projects are totally senseless."))

cores = cpu_count()  # Number of CPU cores on your system
partitions = cores // 4 or 1  # Define as many partitions as you want


def parallelize(data, func):
    data_split = np.array_split(data, partitions)
    pool = Pool(cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


def par_func(cs):
    print("Processing batch of", len(cs.index))
    cs["sentiment"] = cs["text"].apply(senti.sentiment)
    return cs


def main(i):
    print("Reading tweet file", i)
    cs = pd.read_csv("full_tweets_" + str(i) + ".csv")
    print("Total tweets", len(cs.index))
    print("Detecting sentiment in parallel...")
    cs = parallelize(cs, par_func)
    cs.to_csv("full_tweets_with_sentiment_" + str(i) + ".csv")
    print("---------")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Calculate tweets sentiment')
    parser.add_argument("index")
    args = parser.parse_args()

    main(args.index)
