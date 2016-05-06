"""
CSCI 6660 Final Project

Author: John Persano
Date:   05/02/2016
"""

import nltk

from utils import Utils


class QueryHandler:
    """
    The QueryHandler class is used to query answers from the semantic network. The semantic
    network's information is queried in every direction to ensure the data is completely utilized.

    Answers are created with confidence levels. In general, the more the QueryHandler has to manipulate
    data to find an answer, the less confident it is that the answer is correct. It is possible, and
    actually quite probable, the QueryHandler will come up with the same answer multiple times. When this
    occurs, the QueryHandler will increase the confidence of the repeated answer.

    Answers are structured as follows:
        answer = (0.6298268967194809, 'alley cat', 'has a', 'blue coat')

    answer[0] - The first item in the tuple is the inverse confidence level. In other words, the QueryHandler is
                ~63% confident that it does NOT believe this is the correct answer in the above example.
    answer[1:] - The last three items represent the original data that was inserted into the network in the
                    correct order and can be parsed into an output
    """

    # Max sentiment difference an answer can have to be added to the answers list
    max_sn_difference = .75
    not_found_message = "Sorry, I don't know anything about that"

    def __init__(self, semantic_network, sn_classifier, debug=False):
        """
        Initialize the QueryHandler class.

        :param semantic_network: the semantic_network that contains the data to query
        :param sn_classifier: a trained sentiment classifier to assist queries
        :param debug: True is should display debug messages (this module can't see the settings class)
        """
        self.semantic_network_dictionary = semantic_network.node_dictionary
        self.sn_classifier = sn_classifier
        self.debug = debug

    def query(self, query):
        """
        Query the semantic network for relevant information. This function will dissect the query
        every way possible in an attempt to find a through relationship. This function has a relatively
        high time complexity so it may take a while to return.

        :param query: the query to query
        :return: the best answer string or an empty string
        """

        # Get rid of numbers and punctuation
        query = Utils.strip_sentence(query)
        query_tokens = nltk.word_tokenize(query)

        if self.debug:
            print("User input tokens: {}".format(query_tokens))
            user_input_sentiment = self.__calculate_sentiment(self.sn_classifier, query)
            print("User input sentiment: {}".format(user_input_sentiment))

        # Get the bigrams and unigrams of the query as lists
        bigrams = list(nltk.bigrams(query_tokens))
        unigrams = query_tokens

        # Bigrams are stored in the dictionary with hyphens (e.g. red dog becomes red-dog)
        bigram_keys = ["-".join(bigram) for bigram in bigrams]
        unigram_keys = unigrams  # For name consistency

        # This is the network where all of the data is stored
        if self.debug:
            print("Known dictionary keys: {}".format(self.semantic_network_dictionary.keys()))

        # Make a list of all the known bigrams
        known_bigrams = []
        for bigram_key in bigram_keys:
            node = self.semantic_network_dictionary.get(bigram_key, None)
            if node is not None:
                known_bigrams.append(node)

        # Make a list of all the known unigrams
        known_unigrams = []
        for unigram_key in unigram_keys:
            node = self.semantic_network_dictionary.get(unigram_key, None)
            if node is not None:
                known_unigrams.append(node)

        # A list of possible answer tuples [(sm_similarity, in_node, relation_node, out_node)]
        answer_contenders = []

        # Search known bigrams in the dictionary for through relationships
        for bigram_node in known_bigrams:
            if bigram_node.get_type() == "RelationNode":
                answer_contenders += self.__get_bigram_relationnode_contenders(
                    bigram_node, known_bigrams, known_unigrams, query)
            elif bigram_node.get_type() == "ObjectNode":
                answer_contenders += self.__get_bigram_objectnode_contenders(
                    bigram_node, known_bigrams, known_unigrams, query)
        if self.debug:
            print("Answer contenders after bigram searches: {}".format(answer_contenders))

        # Search known unigrams in the dictionary for through relationships, this may be less accurate
        for unigram_node in known_unigrams:
            if unigram_node.get_type() == "RelationNode":
                answer_contenders += self.__get_unigram_relationnode_contenders(
                    unigram_node, known_bigrams, known_unigrams, query)
            elif unigram_node.get_type() == "ObjectNode":
                answer_contenders += self.__get_unigram_objectnode_contenders(
                    unigram_node, known_bigrams, known_unigrams, query)
        if self.debug:
            print("Answer contenders after unigram searches: {}".format(answer_contenders))

        return self.__select_hypothesis(answer_contenders)

    @staticmethod
    def __calculate_sentiment(classifier, value):
        """
        Calculates the sentiment of a particular string. A -1.0 indicates a strong negative sentiment
        while a 1.0 indicates a strong positive sentiment.

        :param classifier: a trained SentimentClassifier
        :param value: the value to calculate the sentiment of
        :return: a float within the inclusive range of -1.0 to 1.0
        """
        classification = classifier.prob_classify_text(value)
        classification_probability = round(classification[1], 4)

        # Positive sentiment should have a positive value
        if classification[0] == 'positive':
            return classification_probability
        else:
            return classification_probability * -1

    @staticmethod
    def __get_bigram_relationnode_contenders(bigram_node, known_bigrams, known_unigrams, query):
        answer_contenders = []
        for out_object in bigram_node.out_objects:
            """
            This loop will find highly answers if the RL network knows "A dog has a red coat" and
            the query is "What has a red coat?"
            """
            for known_bigram in known_bigrams:
                if known_bigram.get_value() == out_object:
                    for in_relationship in known_bigram.in_relationships:
                        if in_relationship[1] == bigram_node:
                            answer_contenders.append((0, in_relationship[0], bigram_node.get_value(), out_object))
            """
            This loop will find highly answers if the RL network knows "A dog can have fur" and
            the query is "What can have fur?"
            """
            for known_unigram in known_unigrams:
                if known_unigram.get_value() == out_object:
                    for in_relationship in known_unigram.in_relationships:
                        if in_relationship[1] == bigram_node:
                            answer_contenders.append((0, in_relationship[0], bigram_node.get_value(), out_object))
        return answer_contenders

    def __get_bigram_objectnode_contenders(self, bigram_node, known_bigrams, known_unigrams, query):
        query_sentiment = self.__calculate_sentiment(self.sn_classifier, query)

        answer_contenders = []
        for in_relationship in bigram_node.in_relationships:
            in_relationship_sentiment = self.__calculate_sentiment(self.sn_classifier, in_relationship[1])
            """
            This loop will find highly likely answers if the RL network knows "A dog has a red coat" and
            the query is "What has a red coat?"
            """
            for known_bigram in known_bigrams:
                if known_bigram.get_value() == in_relationship[1]:
                    answer_contenders.append((0, in_relationship[0], in_relationship[1], bigram_node.get_value()))
            """
            This loop will find highly likely answers if the RL network knows "A dog has red fur" and
            the query is "What has red fur?"
            """
            for known_unigram in known_unigrams:
                if known_unigram.get_value() == in_relationship[1]:
                    answer_contenders.append((0, in_relationship[0], in_relationship[1], bigram_node.get_value()))

            """
            This loop will find possible answers if the RL network knows "A dog has a red coat" and
            the query is "What may have a red coat?"
            """
            sn_difference = abs(in_relationship_sentiment - query_sentiment)
            if sn_difference < self.max_sn_difference:
                answer_contenders.append(
                    (sn_difference, in_relationship[0], in_relationship[1], bigram_node.get_value()))

        for out_relationship in bigram_node.out_relationships:
            out_relationship_sentiment = self.__calculate_sentiment(self.sn_classifier, out_relationship[0])
            """
            This loop will find highly likely answers if the RL network knows "A dirty dog has a red coat" and
            the query is "A dirty dog has a what?"
            """
            for known_bigram in known_bigrams:
                if known_bigram.get_value() == out_relationship[0]:
                    answer_contenders.append((0, bigram_node.get_value(), out_relationship[0], out_relationship[1]))
            """
            This loop will find highly likely answers if the RL network knows "A dirty dog has red fur" and
            the query is "A dirty dog has what?"
            """
            for known_unigram in known_unigrams:
                if known_unigram.get_value() == out_relationship[0]:
                    answer_contenders.append((0, bigram_node.get_value(), out_relationship[0], out_relationship[1]))

            """
             This loop will find possible answers if the RL network knows "A dirty dog has red fur" and
             the query is "A dirty dog may have what?"
             """
            sn_difference = abs(out_relationship_sentiment - query_sentiment)
            if sn_difference < self.max_sn_difference:
                answer_contenders.append(
                    (sn_difference, bigram_node.get_value(), out_relationship[0], out_relationship[1]))

            """
             This loop will find possible answers if the RL network knows "A dirty dog has red fur", "Red is
             a color" and the query is "A dirty dog has what color fur?"
             """
            for known_unigram in known_unigrams:
                if known_unigram.get_value() == out_relationship[0]:
                    out_relationship_object = out_relationship[1]
                    out_relationship_object_tokens = nltk.word_tokenize(out_relationship_object)
                    for token in out_relationship_object_tokens:
                        node = self.semantic_network_dictionary.get(token, None)
                        if node.get_type() == "ObjectNode":
                            for inner_in_relationship in node.in_relationships:
                                for inner_known_unigram in known_unigrams:
                                    # If color exists in  sentence
                                    if inner_known_unigram.get_value() == inner_in_relationship[0]:
                                        inner_in_relationship_sentiment = self.__calculate_sentiment(
                                            self.sn_classifier, inner_in_relationship[1])
                                        inner_sn_difference = abs(
                                            out_relationship_sentiment - inner_in_relationship_sentiment)
                                        if inner_sn_difference < self.max_sn_difference:
                                            answer_contenders.append(
                                                (inner_sn_difference, bigram_node.get_value(), out_relationship[0],
                                                 out_relationship[1]))
                            for inner_out_relationship in node.out_relationships:
                                for inner_known_unigram in known_unigrams:
                                    # If color exists in  sentence
                                    if inner_known_unigram.get_value() == inner_out_relationship[1]:
                                        inner_out_relationship_sentiment = self.__calculate_sentiment(
                                            self.sn_classifier, inner_out_relationship[0])
                                        inner_sn_difference = abs(
                                            out_relationship_sentiment - inner_out_relationship_sentiment)
                                        if inner_sn_difference < self.max_sn_difference:
                                            answer_contenders.append(
                                                (inner_sn_difference, bigram_node.get_value(), out_relationship[0],
                                                    out_relationship[1]))
        return answer_contenders

    def __get_unigram_relationnode_contenders(self, unigram_node, bigram_keys, unigram_keys, query):
        answer_contenders = []
        answer_contenders += self.__get_bigram_relationnode_contenders(unigram_node, bigram_keys, unigram_keys, query)
        return answer_contenders

    def __get_unigram_objectnode_contenders(self, unigram_node, bigram_keys, unigram_keys, query):
        answer_contenders = []
        answer_contenders += self.__get_bigram_objectnode_contenders(unigram_node, bigram_keys, unigram_keys, query)

        for constituent_of in unigram_node.constituents_of:
            constituent_of_key = "-".join(nltk.word_tokenize(constituent_of))
            constituent_of_node = self.semantic_network_dictionary.get(constituent_of_key, None)
            answer_contenders += self.__get_bigram_objectnode_contenders(constituent_of_node, bigram_keys, unigram_keys, query)
        return answer_contenders

    @staticmethod
    def __select_hypothesis(answer_contenders):

        def in_hypotheses_list(hypotheses_list, value):
            for item in hypotheses_list:
                if item[2] == value:
                    return True
            return False

        if len(answer_contenders) < 1:
            return QueryHandler.not_found_message

        initial_hypotheses = []
        for answer in answer_contenders:
            answer_sentiment = answer[0]
            answer_text = "{} {} {}.".format(answer[1], answer[2], answer[3])
            hypothesis = [0, answer_sentiment, answer_text]
            initial_hypotheses.append(hypothesis)

        # Count all of the occurrences of a initial_hypotheses
        for hypothesis in initial_hypotheses:
            for inner_hypothesis in initial_hypotheses:
                if inner_hypothesis[2] == hypothesis[2]:
                    hypothesis[0] += 1

        initial_hypotheses.sort(key=lambda x: (int(x[0]), float(-x[1])), reverse=True)

        hypotheses = []
        for initial_hypothesis in initial_hypotheses:
            if not in_hypotheses_list(hypotheses, initial_hypothesis[2]):
                hypotheses.append(initial_hypothesis)

        if len(hypotheses) > 0:
            first_hypothesis = hypotheses[0]
            return first_hypothesis[2]
        else:
            return QueryHandler.not_found_message
