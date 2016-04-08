from collections import Counter

import nltk


class Classifier:

    def __init__(self):
        self.classifier = None
        self.test_set = []
        self.training_set = []

    def train(self, *args):
        raise NotImplementedError("All classes implementing Classifier should implement train()")

    @staticmethod
    def _sentence_features(sentence):
        """
        Gets a feature dictionary for a particular sentence to use with the NaiveBayesClassifier
        :param sentence: a sentence to 'featurize'
        :return: a feature dictionary for the sentence
        """
        features = {}

        word_tokens = nltk.word_tokenize(sentence)
        tagged_words = nltk.pos_tag(word_tokens)

        # This is a bag of words feature
        word_tokens = nltk.word_tokenize(sentence)
        for word in word_tokens:
            features[word] = True

        # This is a uni-gram count feature
        features["unigram_count"] = len(word_tokens)

        # This is a bi-gram feature
        for bigram in nltk.bigrams(word_tokens):
            features[bigram] = True

        # This is a tag feature
        for tag in tagged_words:
            features[tag[1]] = True

        # This is a tag bi-gram feature
        for tag_bigram in nltk.bigrams(tagged_words[1]):
            features[tag_bigram] = True

        # This is pos count feature
        features.update(Counter(tag for word, tag in tagged_words))

        return features

    def print_accuracy(self):
        print("Input classifier accuracy: {}%"
              .format(int(nltk.classify.accuracy(self.classifier, self.test_set) * 100)))

    def print_important_features(self, amount=15):
        print(self.classifier.show_most_informative_features(amount))

    def classify_text(self, query=""):
        return self.classifier.classify(self._sentence_features(query))

    def prob_classify_text(self, query=""):
        probabilities = self.classifier.prob_classify(self._sentence_features(query))
        for sample in probabilities.samples():
            return "{0}: {1}".format(sample, probabilities.prob(sample))