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

For example, I have a CSV file called "tweet_id_1.csv" and want to get an output of "full_1.csv", then run:

```
python tweets_fetch.py -i tweet_id_1.csv -o full_1.csv
```

It also supports proxy. Use the `-p` option.

## Reference
- Tweet IDs from https://github.com/chrisalbon/election_day_2016_twitter
