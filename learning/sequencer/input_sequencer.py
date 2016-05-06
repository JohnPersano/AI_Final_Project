import difflib
import os
from queue import Queue
from random import randint
from xml.dom import minidom

import nltk
from math import floor

import pickle

import settings
from semantics.object_node import ObjectNode


class InputSequencer:
    pickle_name = "input_sequencer.pickle"

    def __init__(self):
        self.solutions_dict = {}
        self.pickle_path = os.path.join(settings.DATA_OUT, self.pickle_name)

    def load(self):
        if os.path.exists(self.pickle_path):
            with open(self.pickle_path, 'rb') as file:
                return pickle.load(file)
        return self

    def train(self, network_filename="network_set.xml"):

        # We loaded an existing set from a pickle
        if len(self.solutions_dict) > 0:
            if settings.DEBUG:
                print("Loaded from pickle!")
            return

        network_filename = os.path.join(settings.DATA_OUT, network_filename)

        # Files contain XML data
        xml_document = minidom.parse(network_filename)
        train_data_items = xml_document.getElementsByTagName('train-data')

        for train_data_item in train_data_items:
            input_sentence = train_data_item.getElementsByTagName('sentence')[0].firstChild.data
            network_string = train_data_item.getElementsByTagName('node')[0].firstChild.data
            self.train_sentence(input_sentence, network_string)

        # Save contents to a pickle
        with open(self.pickle_path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def train_sentence(self, sentence=None, correct_node_string=None):
        if sentence is None:
            print("No sentence was supplied to the InputParser.train_sentence function.")
        word_tokens = nltk.word_tokenize(sentence.lower())
        tagged_tokens = nltk.pos_tag(word_tokens)  # [('dog', 'NN'), ('is', 'VBZ'), ('mammal', 'JJ')]

        tag_list = []  # ['NN', 'VBZ', 'AMNN']

        for tagged_token in tagged_tokens:
            tag_list.append(tagged_token[1])

        count_tuple_list = []  # [['NN', 1], ['VBZ', 1], ['AMNN', 1]]
        for tag in tag_list:
            if not self.__tag_in_list(tag, count_tuple_list):
                tag_tuple = [tag, 0]
                for other_tag in tag_list:
                    if other_tag == tag_tuple[0]:
                        tag_tuple[1] += 1
                count_tuple_list.append(tag_tuple)

        initial_set = self.__generate_initial_set(count_tuple_list, tag_list)

        solution = self.__find_solution(initial_set, correct_node_string, tagged_tokens)

        if solution is None:
            print("Could not find solution for {}".format(tagged_tokens))
            return

        key = "-".join(tag_list)
        print("Added key: {}".format(key))
        print("For sentence: {}".format(sentence))

        self.solutions_dict[key] = solution

    def parse_to_node(self, sentence):
        """
        Returns a formatted node for a particular sentence
        :param sentence: the sentence to nodify
        :return: Node
        """
        if sentence is None:
            print("No sentence was supplied to the InputParser.train_sentence function.")

        word_tokens = nltk.word_tokenize(sentence.lower())
        pos_tuples = nltk.pos_tag(word_tokens)

        tag_list = []  # ['NN', 'VBZ', 'AMNN']
        for tagged_token in pos_tuples:
            tag_list.append(tagged_token[1])

        # The key will be a hyphenated tag list (NN-VBK-DT-NN)
        key = "-".join(tag_list)
        if settings.DEBUG:
            print("Searched for key: {}".format(key))

        if settings.DEBUG:
            print("\nDictionary keys-------------------\n")
            for dict_key in self.solutions_dict:
                print("Dictionary has key: {}".format(dict_key))

        solution_entry = self.solutions_dict.get(key, None)

        # Couldn't find an exact match, try to use the most similar
        if solution_entry is None:
            if settings.DEBUG:
                print("Parse to node: Could not find a direct match, searching for best match")
            best_ratio = 0
            best_key = None
            for dict_key in self.solutions_dict.keys():
                ratio = difflib.SequenceMatcher(None, dict_key, key).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_key = dict_key
            if settings.DEBUG:
                print("Parse to node: Best match key is {} with a {} similarity ratio".format(best_key, best_ratio))
            solution_entry = self.solutions_dict.get(best_key, None)

        return self.__generate_node(solution_entry[1], pos_tuples)

    @staticmethod
    def __tag_in_list(tag, count_tuple_list):
        for existing_tuple in count_tuple_list:
            if existing_tuple[0] == tag:
                return True
        return False

    @staticmethod
    def __generate_initial_set(count_tuple_list, tag_list):
        initial_set = []
        for i in range(1000):
            function_sequence = []
            for tag in tag_list:
                function_enumerator = randint(1, 3 + 1)  # Max functions plus 1
                parameter = []
                for count_tuple in count_tuple_list:
                    if count_tuple[0] == tag:
                        parameter_enumerator = randint(1, count_tuple[1])
                        parameter.append(parameter_enumerator)
                        parameter.append(tag)
                        break
                function_sequence.append((function_enumerator, parameter[0], parameter[1]))
            initial_set.append(function_sequence)
        return initial_set

    @staticmethod
    def __generate_node(function_sequence, tagged_tokens):
        node_builder = ObjectNode.Factory()
        for function_tuple in function_sequence:
            count = 0
            for tagged_token in tagged_tokens:
                if tagged_token[1] == function_tuple[2]:
                    count += 1
                    if count == function_tuple[1]:
                        data = tagged_token[0]
                        if function_tuple[0] == 1:
                            node_builder.add_value(data)
                        elif function_tuple[0] == 2:
                            node_builder.add_relation(data)
                        elif function_tuple[0] == 3:
                            node_builder.add_relation_object(data)
                        break
        return node_builder.build()

    def __find_solution(self, initial_set, correct_node_string, tagged_tokens, max_iterations=100):

        def mutate(child_sequence_set):
            if settings.DEBUG:
                print("Genetic algorithm: Child has mutated")
            index = randint(0, (len(child) - 1))

            child_function_sequence = child_sequence_set[index]
            temp_function_sequence = (randint(1, 4), child_function_sequence[1], child_function_sequence[2])

            child_sequence_set[index] = temp_function_sequence
            return child

        def get_corrected_ratio(f_ratio):
            return floor(f_ratio * pow(10, 10))

        ga_set_list = []
        temp_ga_set_list = []

        # Create initial GA set. tuple[0] = ratio and tuple[1] = function sequence
        for function_sequence in initial_set:
            ga_set_list.append([0, function_sequence])

        i = 0
        best_function_set = None
        best_ratio = 0
        while i <= max_iterations:
            if settings.DEBUG:
                print("Genetic algorithm: Iteration {}".format(i))
            i += 1
            for ga_set in ga_set_list:
                # ga_set[1] holds the function sequence
                potential_node_string = self.__generate_node(ga_set[1], tagged_tokens).to_string()

                ratio = difflib.SequenceMatcher(None, correct_node_string, potential_node_string).ratio()

                if ratio > best_ratio:
                    best_function_set = ga_set

                if correct_node_string == potential_node_string or ratio == 1.0:
                    if settings.DEBUG:
                        print("Genetic algorithm: Found exact match")
                        print(ratio)
                        print(ga_set)
                    return ga_set

                # Sort by integer so convert float ratio to integer
                temp_ga_set_list.append([get_corrected_ratio(ratio), ga_set[1]])

            ga_set_list.clear()
            ga_set_list += temp_ga_set_list
            temp_ga_set_list.clear()

            # Sort by ratio (descending)
            ga_set_list.sort(key=lambda x: int(x[0]), reverse=True)

            # Place all items in a queue, best parents will mate first
            ga_queue = Queue()
            for ga_set in ga_set_list:
                ga_queue.put(ga_set)

            while not ga_queue.empty():
                current_ga_set = ga_queue.get()
                next_ga_set = ga_queue.get()

                # Add child from XY
                parent_one = current_ga_set[1]
                parent_one_split = int(len(parent_one)/2)
                parent_one = parent_one[parent_one_split:]

                parent_two = next_ga_set[1]
                parent_two_split = int(len(parent_two)/2)
                parent_two = parent_two[:parent_two_split]

                child = parent_one + parent_two

                # Possibly mutate the child
                num = randint(250, 1000)
                chance = randint(11, 250)
                if num % chance == 0:
                    child = mutate(child)

                temp_ga_set_list.append([0, child])

                # Add child from YX
                parent_one = current_ga_set[1]
                parent_one_split = int(len(parent_one) / 2)
                parent_one = parent_one[:parent_one_split]

                parent_two = next_ga_set[1]
                parent_two_split = int(len(parent_two) / 2)
                parent_two = parent_two[parent_two_split:]

                child = parent_one + parent_two

                # Possibly mutate the child
                num = randint(250, 1000)
                chance = randint(11, 250)
                if num % chance == 0:
                    child = mutate(child)

                temp_ga_set_list.append([0, child])

            ga_set_list.clear()
            ga_set_list += temp_ga_set_list
            temp_ga_set_list.clear()

        return best_function_set
