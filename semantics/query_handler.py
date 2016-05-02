import string

import nltk

from utils import Utils


class QueryHandler:
    def __init__(self, semantic_network, sn_classifier, debug=False):
        self.semantic_network_dictionary = semantic_network.node_dictionary
        self.sn_classifier = sn_classifier
        self.debug = debug

    def query(self, query):

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
        for bigram_node in known_bigrams:
            if bigram_node.get_type() == "RelationNode":
                answer_contenders += self.__get_bigram_relationnode_contenders(bigram_node, bigram_keys, unigram_keys)

            if bigram_node.get_type() == "ObjectNode":
                answer_contenders += self.__get_bigram_objectnode_contenders(bigram_node, bigram_keys, unigram_keys)
        print(answer_contenders)

        first = answer_contenders[0]
        string = "A" + " " + first[1] + " " + first[2] + " " + first[3]

        return string


    @staticmethod
    def __calculate_sentiment(classifier, value):
        classification = classifier.prob_classify_text(value)
        classification_probability = round(classification[1], 4)
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

    def __get_bigram_objectnode_contenders(self, bigram_node, bigram_keys, unigram_keys):
        answer_contenders = []
        # Examine in relationships of the object node
        for in_relationship in bigram_node.in_relationships:

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
                            answer_contenders.append(
                                (0, in_object, in_relationship_relation, bigram_node.get_value()))

            # Start cutting up any in relationship bigrams in the search for a through relationship
            for in_relationship_token in in_relationship_relation_tokens:
                # Calculate in_relationship_token sentiment
                in_relationship_token_sn_tuple = self.sn_classifier.prob_classify_text(in_relationship_token)
                in_relationship_token_sentiment = in_relationship_token_sn_tuple[1]
                if 'negative' in in_relationship_token_sn_tuple[0]:
                    in_relationship_token_sentiment *= -1

                # The sentiment difference between the two will dictate if we can interchange the relationships
                sn_difference = abs(in_relation_sentiment - in_relationship_token_sentiment)
                # The chopped up relation was too different from the original
                if sn_difference > .5:
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
                                answer_contenders.append((sn_difference, in_object, in_relationship_relation,
                                                          bigram_node.get_value()))
                # No part of the sentence contains any part of a known relation, but the sentiment is similar
                sn_difference += .20
                if sn_difference > .5:
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
                            answer_contenders.append((sn_difference, in_object, in_relationship_relation,
                                                      bigram_node.get_value()))

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
                                (0, bigram_node.get_value(), out_relationship_relation, out_object))

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
                if sn_difference > .5:
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
                                    (sn_difference, bigram_node.get_value(), out_relationship_relation,
                                     out_object))
                # No part of the sentence contains any part of a known relation, but the sentiment is similar
                sn_difference += .20
                if sn_difference > .5:
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
                            answer_contenders.append(
                                (sn_difference, bigram_node.get_value(), out_relationship_relation,
                                 out_object))
        return answer_contenders

