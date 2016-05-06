import os
import pickle
import random
from math import floor

import nltk

import settings
from learning.classifiers.classifier import Classifier


class SNClassifier(Classifier):
    pickle_name = "sentiment_classifier.pickle"

    def __init__(self):
        super().__init__()

    def train(self, p_file="positive.txt", n_file="negative.txt"):
        p_file = os.path.join(settings.DATA_SENSETS, p_file)
        n_file = os.path.join(settings.DATA_SENSETS, n_file)

        if settings.DEBUG:
            print("Training sentiment classifier...")

        positive_content = open(p_file).read()
        positive_sentences = [sentence for sentence in positive_content.split('\n')]
        negative_content = open(n_file).read()
        negative_sentences = [sentence for sentence in negative_content.split('\n')]
        labeled_names = ([(sentence, 'positive') for sentence in positive_sentences] +
                         [(sentence, 'negative') for sentence in negative_sentences])
        random.shuffle(labeled_names)

        feature_sets = [(super(SNClassifier, self)._sentence_features(n), sentiment)
                        for (n, sentiment) in labeled_names]
        self.test_set = feature_sets[:floor(len(feature_sets) / 2)]
        self.training_set = feature_sets[floor(len(feature_sets) / 2):]
        self.classifier = nltk.NaiveBayesClassifier.train(self.training_set)

        if settings.DEBUG:
            print("Sentiment classifier trained successfully")
            self.print_accuracy()

    def to_pickle(self):
        pickle_name = os.path.join(settings.DATA_OUT, self.pickle_name)
        with open(pickle_name, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def from_pickle(self):
        pickle_name = os.path.join(settings.DATA_OUT, self.pickle_name)
        with open(pickle_name, 'rb') as file:
            return pickle.load(file)
