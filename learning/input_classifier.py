import os
import random

import nltk
from math import floor

import pickle

import settings
from learning.classifier import Classifier


class InputClassifier(Classifier):
    pickle_name = "input_classifier.pickle"

    def __init__(self):
        super().__init__()

    def train(self, q_out="questions_out.txt", s_out="sentences_out.txt"):
        q_out = os.path.join(settings.DATA_OUT, q_out)
        s_out = os.path.join(settings.DATA_OUT, s_out)

        if settings.DEBUG:
            print("Training input classifier...")

        questions_content = open(q_out).read()
        question_sentences = [sentence for sentence in questions_content.split('\n')]
        statements_content = open(s_out).read()
        statements_sentences = [sentence for sentence in statements_content.split('\n')]
        labeled_names = ([(sentence, 'question') for sentence in question_sentences] +
                         [(sentence, 'statement') for sentence in statements_sentences])
        random.shuffle(labeled_names)

        feature_sets = [(super(InputClassifier, self)._sentence_features(n), sentiment)
                        for (n, sentiment) in labeled_names]
        self.test_set = feature_sets[:floor(len(feature_sets) / 2)]
        self.training_set = feature_sets[floor(len(feature_sets) / 2):]
        self.classifier = nltk.NaiveBayesClassifier.train(self.training_set)

        if settings.DEBUG:
            print("Input classifier trained successfully")

    def to_pickle(self):
        pickle_name = os.path.join(settings.DATA_OUT, self.pickle_name)
        with open(pickle_name, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def from_pickle(self):
        pickle_name = os.path.join(settings.DATA_OUT, self.pickle_name)
        with open(pickle_name, 'rb') as file:
            return pickle.load(file)
