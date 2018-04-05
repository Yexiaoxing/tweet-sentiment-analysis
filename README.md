# Tweet Sentiment Analysis

This is a course project that analyzed the sentiment of tweets posted in 2016 U.S. Election Day.

We try to figure out whether using the social media can help predict the election result.

## Tweet Hydrator
Due to the Twitter's ToS, the data published only contains tweet IDs, so we need to hydrator it (aka, get the full tweet information).

Install requirements:

```
pip install -r requirements.txt
```

To hydrate, first you need a CSV file with only ID in each row. Then edit the `tweets_fetch.py` to fill information, and run it.

```
Usage: tweets_fetch.py -i input_file -o output_file -p proxy_address

Options:
  -h, --help           show this help message and exit
  -p str, --proxy=str  Proxy address
  -i FILE, --in=FILE   Input CSV file
  -o FILE, --out=FILE  Output CSV file
```

For example, I have a CSV file called "tweet_id_1.csv" and want to get an output of "full_tweets_1.csv", then run:

```
python tweets_fetch.py -i tweet_id_1.csv -o full_tweets_1.csv
```

It also supports proxy. Use the `-p` option.

## Sentiment Analysis
In this project, we utilized https://github.com/aalind0/NLP-Sentiment-Analysis-Twitter, which uses `nltk` and `Sklearn` to train and provides the best optimized sentiment analysis. To run the analysis, you need to do the following...

1. Clone the original repo
  1. ```
  $ git clone https://github.com/aalind0/Movie_Reviews-Sentiment_Analysis
  $ cd Movie_Reviews-Sentiment_Analysis
  ```
2. Install required packages and data
  1. Install `sklearn` with `pip install scikit-learn`
  2. Install `nltk` with `pip install nltk`
  3. Open a fresh python interpreter, run
    ```
    >>> import nltk
    >>> nltk.download('stopwords')
    >>> nltk.download('movie_reviews')
    >>> nltk.download('averaged_perceptron_tagger')
    >>> nltk.download('punkt')
    ```
3. Run the `Train_Classifiers.py` file to train models.
4. Copy all generated `.pickle` files to this project folder, as well as `sentiment_mod.py`.
5. Run `sentiment_cal.py` (it will use 1/4 of all your CPU cores to calculate) or `sentiment_cal_single_thread.py` (it will only utilize one core using one thread) to calculate the sentiment. You need to use this syntax: `python xxx.py <index>` and replace the `<index>` with the number of csv file. The filename is hardcoded so you may change it yourself.


## Reference
- Tweet IDs from https://github.com/chrisalbon/election_day_2016_twitter
