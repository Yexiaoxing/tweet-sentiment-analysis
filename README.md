# Tweet Sentiment Analysis

This is a course project that analyzed the sentiment of tweets posted in 2016 U.S. Election Day.

We try to figure out whether using the social media can help predict the election result.

## Tweet Hydrator

Due to the Twitter's ToS, the data published only contains tweet IDs, so we need to hydrator it (aka, get the full tweet information).

Install requirements:

```bash
pip install -r requirements.txt
```

To hydrate, first you need a CSV file with only ID in each row. Then edit the `tweets_fetch.py` to fill information, and run it.

```bash
Usage: tweets_fetch.py -i input_file -o output_file -p proxy_address

Options:
  -h, --help           show this help message and exit
  -p str, --proxy=str  Proxy address
  -i FILE, --in=FILE   Input CSV file
  -o FILE, --out=FILE  Output CSV file
```

For example, I have a CSV file called "tweet_id_1.csv" and want to get an output of "full_tweets_1.csv", then run:

```bash
python tweets_fetch.py -i tweet_id_1.csv -o full_tweets_1.csv
```

It also supports proxy. Use the `-p` option.

## Sentiment Analysis

In this project, we utilized [https://github.com/aalind0/NLP-Sentiment-Analysis-Twitter](https://github.com/aalind0/NLP-Sentiment-Analysis-Twitter), which uses `nltk` and `Sklearn` to train and provides the best optimized sentiment analysis. To run the analysis, you need to do the following...

1. Install required packages and data
    1. Install `sklearn` with `pip install scikit-learn`
    2. Install `nltk` with `pip install nltk`
    3. Open a fresh python interpreter, run
        ```python
        > import nltk
        > nltk.download('stopwords')
        > nltk.download('movie_reviews')
        > nltk.download('averaged_perceptron_tagger')
        > nltk.download('punkt')
        ```

2. Run the `train_classifiers.py` file to train models. Or you may use the pretrained models in this repo.
3. Run `sentiment_calculation_multithread.py` (it will use 1/4 of all your CPU cores to calculate) or `sentiment_calculation.py` (it will only utilize one core using one thread) to calculate the sentiment. You need to use this syntax: `python xxx.py <index>` and replace the `<index>` with the number of csv file. The filename is hardcoded so you may change it yourself.

### Resulting Accuracy

The accuracy varies because we randomly our training sets. But it should be stable at around $[65, 75]$. This is a demo run:

- Original Naive Bayes: 72.9607250755287
- Sklearn Multinomial Naive Bayes: 70.2416918429003
- Sklearn Bernoulli Naive Bayes: 72.35649546827794
- Sklearn Logistic Regression: 70.69486404833837
- Sklearn Linear SVC: 67.97583081570997
- Sklearn SGD classifier: 67.06948640483384

Voted Classifier: 71.75226586102718

## Reference

* Tweet IDs from [https://github.com/chrisalbon/election_day_2016_twitter](https://github.com/chrisalbon/election_day_2016_twitter)
