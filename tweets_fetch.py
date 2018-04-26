"""This module fetches full tweets (also called hydrate tweets) from given csv file."""
import pandas as pd
import tweepy
import csv
from tqdm import trange
from optparse import OptionParser
from typing import List

# Insert your Twitter API key here
consumer_key = ''
consumer_secret = ''

access_token = ''
access_secret = ''


def retrieve_tweets(tweet_ids: List[str],
                    output_file_name: str,
                    proxy: str = ""):
    """
    Retrieve tweets from list of tweet ids.

    @param tweet_ids: List of tweet id.
    @param output_file_name: The file to write.
    """
    print("Output:", output_file_name)

    # Authorization with Twitter
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(
        auth,
        wait_on_rate_limit=True,
        wait_on_rate_limit_notify=True,
        retry_count=3,
        retry_delay=5,
        retry_errors=set([401, 404, 500, 503]),
        proxy=proxy)

    # Create output file
    csvFile = open(output_file_name, 'w', encoding='utf-8')
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow([
        "text", "created_at", "geo", "lang", "place", "coordinates",
        "user.favourites_count", "user.statuses_count", "user.description",
        "user.location", "user.id", "user.created_at", "user.verified",
        "user.following", "user.url", "user.listed_count",
        "user.followers_count", "user.default_profile_image",
        "user.utc_offset", "user.friends_count", "user.default_profile",
        "user.name", "user.lang", "user.screen_name", "user.geo_enabled",
        "user.profile_background_color", "user.profile_image_url",
        "user.time_zone", "id", "favorite_count", "retweeted", "source",
        "favorited", "retweet_count"
    ])

    print("Total IDs", len(tweet_ids))
    # Append tweets to output file

    # Twitter allows a batch of 100 tweets
    for tweetid_batch in trange(len(tweet_ids) // 100):
        try:
            status_es = api.statuses_lookup(
                tweet_ids[tweetid_batch * 100:tweetid_batch * 100 + 100])
            for status in status_es:
                csvWriter.writerow([
                    status.text, status.created_at, status.geo, status.lang,
                    status.place, status.coordinates,
                    status.user.favourites_count, status.user.statuses_count,
                    status.user.description, status.user.location,
                    status.user.id, status.user.created_at,
                    status.user.verified, status.user.following,
                    status.user.url, status.user.listed_count,
                    status.user.followers_count,
                    status.user.default_profile_image, status.user.utc_offset,
                    status.user.friends_count, status.user.default_profile,
                    status.user.name, status.user.lang,
                    status.user.screen_name, status.user.geo_enabled,
                    status.user.profile_background_color,
                    status.user.profile_image_url, status.user.time_zone,
                    status.id, status.favorite_count, status.retweeted,
                    status.source, status.favorited, status.retweet_count
                ])
        except Exception as e:
            print(str(e))


def main(options, args):
    """Initialize the board, solver object and call the solve() function."""
    df = pd.read_csv(options.infile)
    retrieve_tweets(df.iloc[:, 0], options.out, proxy=options.proxy)


if __name__ == '__main__':
    parser = OptionParser(usage="Usage: %prog -i input_file" +
                          " -o output_file -p proxy_address")
    parser.add_option("-p", "--proxy", dest="proxy",
                      metavar="str", help="Proxy address", default="")
    parser.add_option("-i", "--in", dest="infile",
                      help="Input CSV file", metavar="FILE")
    parser.add_option("-o", "--out", dest="out",
                      help="Output CSV file", metavar="FILE")
    (options, args) = parser.parse_args()
    if not options.infile:
        parser.error('Input CSV filename not given')

    if not options.outfile:
        parser.error('Output CSV filename not given')
    main(*parser.parse_args())
