"""
CSCI 6660 Final Project

Author: John Persano
Date:   05/02/2016
"""

import nltk

from utils import Utils


class QueryHadndler:
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
                answer_contenders += self.__get_bigram_relationnode_contenders(bigram_node, bigram_keys, unigram_keys)
            elif bigram_node.get_type() == "ObjectNode":
                answer_contenders += self.__get_bigram_objectnode_contenders(bigram_node, bigram_keys, unigram_keys)
        if self.debug:
            print("Answer contenders after bigram searches: {}".format(answer_contenders))

        # Search known unigrams in the dictionary for through relationships, this may be less accurate
        for unigram_node in known_unigrams:
            if unigram_node.get_type() == "RelationNode":
                answer_contenders += self.__get_unigram_relationnode_contenders(unigram_node, bigram_keys, unigram_keys)
            elif unigram_node.get_type() == "ObjectNode":
                answer_contenders += self.__get_unigram_objectnode_contenders(unigram_node, bigram_keys, unigram_keys)
        if self.debug:
            print("Answer contenders after unigram searches: {}".format(answer_contenders))

        for unigram_node in known_unigrams:
            if unigram_node.get_type() == "ObjectNode":
                for in_relationship in unigram_node.in_relationships:
                    in_relationship_object = in_relationship[0]
                    in_relationship_object_tokens = nltk.word_tokenize(in_relationship_object)
                    in_relationship_object_key = "-".join(in_relationship_object_tokens)
                    node = self.semantic_network_dictionary.get(in_relationship_object_key, None)
                    answer_contenders += self.__get_unigram_objectnode_contenders(node, bigram_keys, unigram_keys)
                for out_relationship in unigram_node.out_relationships:
                    out_relationship_object = out_relationship[1]
                    out_relationship_object_tokens = nltk.word_tokenize(out_relationship_object)
                    out_relationship_object_key = "-".join(out_relationship_object_tokens)
                    node = self.semantic_network_dictionary.get(out_relationship_object_key, None)
                    answer_contenders += self.__get_unigram_objectnode_contenders(node, bigram_keys, unigram_keys)

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

    def __get_bigram_relationnode_contenders(self, bigram_node, bigram_keys, unigram_keys):
        answer_contenders = []
        # Check the in objects for any through relationships
        for in_object in bigram_node.in_objects:
            in_object_tokens = nltk.word_tokenize(in_object)
            in_object_key = "-".join(in_object_tokens)
            if in_object_key in bigram_keys or in_object_key in unigram_keys:
                node = self.semantic_network_dictionary.get(in_object_key, None)
                for out_relationship in node.out_relationships:
                    if out_relationship[0] == bigram_node.get_value():
                        answer_contenders.append((0, in_object, bigram_node.get_value(), out_relationship[1]))
            # Start cutting up any in object bigrams in the search for a relationship
            for in_object_token in in_object_tokens:
                if in_object_token in bigram_keys or in_object_token in unigram_keys:
                    node = self.semantic_network_dictionary.get(in_object_key, None)
                    for out_relationship in node.out_relationships:
                        if out_relationship[0] == bigram_node.get_value():
                            answer_contenders.append((0, in_object, bigram_node.get_value(), out_relationship[1]))

        # Check the out objects for any through relationships
        for out_object in bigram_node.out_objects:
            out_object_tokens = nltk.word_tokenize(out_object)
            out_object_key = "-".join(out_object_tokens)
            if out_object_key in bigram_keys or out_object_key in unigram_keys:
                node = self.semantic_network_dictionary.get(out_object_key, None)
                for in_relationship in node.in_relationships:
                    if in_relationship[1] == bigram_node.get_value():
                        answer_contenders.append((0, in_relationship[0], bigram_node.get_value(), out_object))
            # Start cutting up any out object bigrams in the search for a relationship
            for out_object_token in out_object_tokens:
                if out_object_token in bigram_keys or out_object_token in unigram_keys:
                    node = self.semantic_network_dictionary.get(out_object_key, None)
                    for in_relationship in node.in_relationships:
                        if in_relationship[1] == bigram_node.get_value():
                            answer_contenders.append((0, in_relationship[0], bigram_node.get_value(), out_object))
        return answer_contenders

    def __get_bigram_objectnode_contenders(self, bigram_node, bigram_keys, unigram_keys, sn_modifier=0.0):
        answer_contenders = []
        # Examine in relationships of the object node
        for in_relationship in bigram_node.in_relationships:
            print("In relationship = {}".format(in_relationship))

            # Calculate in relationship sentiment
            in_relation_sn_tuple = self.sn_classifier.prob_classify_text(in_relationship[1])
            in_relation_sentiment = in_relation_sn_tuple[1]
            if 'negative' in in_relation_sn_tuple[0]:
                in_relation_sentiment *= -1

            # Grab the relation, tokenize it, and get its key
            in_relationship_relation = in_relationship[1]
            in_relationship_relation_tokens = nltk.word_tokenize(in_relationship_relation)
            in_relationship_relation_key = "-".join(in_relationship_relation_tokens)

            # Check for a direct match of the in relationship to a sentence bigram
            if in_relationship_relation_key in bigram_keys or in_relationship_relation_key in unigram_keys:
                node = self.semantic_network_dictionary.get(in_relationship_relation_key, None)
                # Iterate through the nodes with an out relation that matches the
                for in_object in node.in_objects:
                    in_object_tokens = nltk.word_tokenize(in_object)
                    in_object_key = "-".join(in_object_tokens)
                    in_object_node = self.semantic_network_dictionary.get(in_object_key, None)
                    for out_relationship in in_object_node.out_relationships:
                        out_relationship_relation_tokens = nltk.word_tokenize(out_relationship[0])
                        out_relationship_relation_key = "-".join(out_relationship_relation_tokens)
                        if out_relationship_relation_key == in_relationship_relation_key:
                            answer_contenders.append((0, in_object, in_relationship_relation,
                                                      out_relationship[1]))

            # Start cutting up any in relationship bigrams in the search for a through relationship
            for in_relationship_token in in_relationship_relation_tokens:
                # Calculate in_relationship_token sentiment
                in_relationship_token_sn_tuple = self.sn_classifier.prob_classify_text(in_relationship_token)
                in_relationship_token_sentiment = in_relationship_token_sn_tuple[1]
                if 'negative' in in_relationship_token_sn_tuple[0]:
                    in_relationship_token_sentiment *= -1

                # The sentiment difference between the two will dictate if we can interchange the relationships
                sn_difference = abs(in_relation_sentiment - in_relationship_token_sentiment)
                sn_difference += sn_modifier
                # The chopped up relation was too different from the original
                if sn_difference > self.max_sn_difference:
                    continue
                # Check if the sentence contains any part of a known relationship
                if in_relationship_token in unigram_keys:
                    node = self.semantic_network_dictionary.get(in_relationship_relation_key, None)
                    for in_object in node.in_objects:
                        in_object_tokens = nltk.word_tokenize(in_object)
                        in_object_key = "-".join(in_object_tokens)
                        in_object_node = self.semantic_network_dictionary.get(in_object_key, None)
                        for out_relationship in in_object_node.out_relationships:
                            out_relationship_relation_tokens = nltk.word_tokenize(out_relationship[0])
                            out_relationship_relation_key = "-".join(out_relationship_relation_tokens)
                            if out_relationship_relation_key == in_relationship_relation_key:
                                answer_contenders.append((0, in_object, in_relationship_relation,
                                                          out_relationship[1]))

                # No part of the sentence contains any part of a known relation, but the sentiment is similar
                sn_difference += .20
                if sn_difference > self.max_sn_difference:
                    continue
                node = self.semantic_network_dictionary.get(in_relationship_relation_key, None)
                for in_object in node.in_objects:
                    in_object_tokens = nltk.word_tokenize(in_object)
                    in_object_key = "-".join(in_object_tokens)
                    in_object_node = self.semantic_network_dictionary.get(in_object_key, None)
                    for out_relationship in in_object_node.out_relationships:
                        out_relationship_relation_tokens = nltk.word_tokenize(out_relationship[0])
                        out_relationship_relation_key = "-".join(out_relationship_relation_tokens)
                        if out_relationship_relation_key == in_relationship_relation_key:
                            answer_contenders.append((0, in_object, in_relationship_relation,
                                                      out_relationship[1]))

        # Examine out relationships of the object node
        for out_relationship in bigram_node.out_relationships:
            # Calculate out relationship sentiment
            out_relation_sn_tuple = self.sn_classifier.prob_classify_text(out_relationship[0])
            out_relation_sentiment = out_relation_sn_tuple[1]
            if 'negative' in out_relation_sn_tuple[0]:
                out_relation_sentiment *= -1

            # Grab the relation, tokenize it, and get its key
            out_relationship_relation = out_relationship[0]
            out_relationship_relation_tokens = nltk.word_tokenize(out_relationship_relation)
            out_relationship_relation_key = "-".join(out_relationship_relation_tokens)

            # Check for a direct match of the out relationship to a sentence bigram
            if out_relationship_relation_key in bigram_keys or out_relationship_relation_key in unigram_keys:
                node = self.semantic_network_dictionary.get(out_relationship_relation_key, None)
                for out_object in node.out_objects:
                    out_object_tokens = nltk.word_tokenize(out_object)
                    out_object_key = "-".join(out_object_tokens)
                    out_object_node = self.semantic_network_dictionary.get(out_object_key, None)
                    for in_relationship in out_object_node.in_relationships:
                        in_relationship_relation_tokens = nltk.word_tokenize(in_relationship[1])
                        in_relationship_relation_key = "-".join(in_relationship_relation_tokens)
                        if in_relationship_relation_key == out_relationship_relation_key:
                            answer_contenders.append(
                                (0, in_relationship[0], out_relationship_relation, out_object))

            # Start cutting up any in relationship bigrams in the search for a through relationship
            for out_relationship_token in out_relationship_relation_tokens:
                # Calculate in_relationship_token sentiment
                out_relationship_token_sn_tuple = self.sn_classifier.prob_classify_text(out_relationship_token)
                out_relationship_token_sentiment = out_relationship_token_sn_tuple[1]
                if 'negative' in out_relationship_token_sn_tuple[0]:
                    out_relationship_token_sentiment *= -1

                # The sentiment difference between the two will dictate if we can interchange the relationships
                sn_difference = abs(out_relation_sentiment - out_relationship_token_sentiment)
                # The chopped up relation was too different from the original
                if sn_difference > self.max_sn_difference:
                    continue

                # Check if the sentence contains any part of a known relationship
                if out_relationship_token in unigram_keys:
                    node = self.semantic_network_dictionary.get(out_relationship_relation_key, None)
                    for out_object in node.out_objects:
                        out_object_tokens = nltk.word_tokenize(out_object)
                        out_object_key = "-".join(out_object_tokens)
                        out_object_node = self.semantic_network_dictionary.get(out_object_key, None)
                        for in_relationship in out_object_node.in_relationships:
                            in_relationship_relation_tokens = nltk.word_tokenize(in_relationship[1])
                            in_relationship_relation_key = "-".join(in_relationship_relation_tokens)
                            if out_relationship_relation_key == in_relationship_relation_key:
                                answer_contenders.append(
                                    (0, in_relationship[0], out_relationship_relation, out_object))
                # No part of the sentence contains any part of a known relation, but the sentiment is similar
                sn_difference += .20
                if sn_difference > self.max_sn_difference:
                    continue
                node = self.semantic_network_dictionary.get(out_relationship_relation_key, None)
                for out_object in node.out_objects:
                    out_object_tokens = nltk.word_tokenize(out_object)
                    out_object_key = "-".join(out_object_tokens)
                    out_object_node = self.semantic_network_dictionary.get(out_object_key, None)
                    for in_relationship in out_object_node.in_relationships:
                        in_relationship_relation_tokens = nltk.word_tokenize(in_relationship[1])
                        in_relationship_relation_key = "-".join(in_relationship_relation_tokens)
                        if out_relationship_relation_key == in_relationship_relation_key:
                            in_relationship_oblect_tokens = nltk.word_tokenize(in_relationship[1])
                            in_relationship_object_key = "-".join(in_relationship_oblect_tokens)
                            if in_relationship_object_key in bigram_keys or in_relationship_object_key in unigram_keys:
                                answer_contenders.append(
                                    (0, in_relationship[0], out_relationship_relation, out_object))
        return answer_contenders

    def __get_unigram_relationnode_contenders(self, unigram_node, bigram_keys, unigram_keys):
        return self.__get_bigram_relationnode_contenders(unigram_node, bigram_keys, unigram_keys)

    def __get_unigram_objectnode_contenders(self, unigram_node, bigram_keys, unigram_keys):
        answer_contenders = []
        answer_contenders += self.__get_bigram_objectnode_contenders(unigram_node, bigram_keys, unigram_keys,
                                                                     sn_modifier=0.20)

        for constituent_of in unigram_node.constituents_of:
            constituent_of_key = "-".join(nltk.word_tokenize(constituent_of))
            constituent_of_node = self.semantic_network_dictionary.get(constituent_of_key, None)

            # The node should never be None, but just in case
            if constituent_of_node is not None:
                answer_contenders += self.__get_bigram_objectnode_contenders(constituent_of_node, bigram_keys,
                                                                             unigram_keys, sn_modifier=0.30)
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
