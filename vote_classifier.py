from nltk.classify import ClassifierI
from statistics import mode 

# Building our classifier class.
# Inheriting from NLTK's ClassifierI.
# Next,assigning the list of classifiers that are passed to our class to self._classifiers.
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def sentiment(self, features: list):
        votes = [c.classify(features) for c in self._classifiers]
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return mode(votes), conf

    #Creating our own classify method.
    #After iterating we return mode(votes), which just returns the most popular vote.
    def classify(self, features: list):
        votes = [c.classify(features) for c in self._classifiers]
        try:
            return mode(votes)
        except:
            return "pos"

    #Defining another parameter, confidence.
    #Since we have algorithms voting, we can tally the votes for and against the winning vote, and call this "confidence.
    def confidence(self, features: list):
        votes = [c.classify(features) for c in self._classifiers]
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf