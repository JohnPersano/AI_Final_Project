import difflib
import os
from queue import Queue
from random import randint
import xml.etree.ElementTree as ElementTree

import nltk

import settings
from semantics.node import Node
from semantics.semantic_network import SemanticNetwork


class InputParser:
    animal_node_noun = "AMNN"
    attribute_node_noun = "ATNN"

    def __init__(self):
        self.solutions_dict = {}

    def train(self, network_filename="network_set.xml"):
        network_filename = os.path.join(settings.DATA_NSETS, network_filename)
        network_file = open(network_filename, "r")

        network_data = network_file.read()
        network_file.close()

        # TODO
        # Files contain XML data
        # root_element = ElementTree.fromstring(network_data.strip())
        # for element in root_element.findall('target/qa/q'):
        #     element = element.text.strip()
        #     # There are random 'Other' elements in the set, we do not want these or list type questions
        #     if element != 'Other' and not element.endswith("."):
        #         # We do not want punctuation as a potential feature
        #         element = element.replace("?", "")
        #         self.q_out.write(element + "\n")

    def train_sentence(self, sentence=None, correct_network=None):
        if sentence is None:
            print("No sentence was supplied to the InputParser.train_sentence function.")
        if correct_network is None:
            raise ValueError("A correct output MUST be supplied to the InputParser.train_sentence function.")
        word_tokens = nltk.word_tokenize(sentence)
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

        solution = self.__find_solution(initial_set, correct_network, tagged_tokens)

        if solution is None:
            print("Could not find solution for {}".format(tagged_tokens))
            return

        key = "-".join(tag_list)
        print(key)
        self.solutions_dict[key] = solution

    def parse_to_node(self, sentence):
        """
        Returns a formatted node for a particular sentence
        :param sentence: the sentence to nodify
        :return: Node
        """
        if sentence is None:
            print("No sentence was supplied to the InputParser.train_sentence function.")

        word_tokens = nltk.word_tokenize(sentence)
        pos_tuples = nltk.pos_tag(word_tokens)

        tag_list = []  # ['NN', 'VBZ', 'AMNN']
        for tagged_token in pos_tuples:
            tag_list.append(tagged_token[1])

        # The key will be a hyphenated tag list (NN-VBK-DT-NN)
        key = "-".join(tag_list)

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
        node = Node()
        for function_tuple in function_sequence:
            count = 0
            for tagged_token in tagged_tokens:
                if tagged_token[1] == function_tuple[2]:
                    count += 1
                    if count == function_tuple[1]:
                        data = tagged_token[0]
                        if function_tuple[0] == 1:
                            node.set_name(data)
                        elif function_tuple[0] == 2:
                            node.add_attribute(data)
                        elif function_tuple[0] == 3:
                            node.add_inherited_by(data)
                        break
        return node

    def __find_solution(self, initial_set, correct_network, tagged_tokens, max_iterations=15):

        def mutate(child_sequence_set):
            index = randint(0, (len(child) - 1))

            child_function_sequence = child_sequence_set[index]
            temp_function_sequence = (randint(1, 4), child_function_sequence[1], child_function_sequence[2])

            child_sequence_set[index] = temp_function_sequence
            return child

        ga_set_list = []
        temp_ga_set_list = []

        # Create initial GA set. [0] = ratio and [1] = function sequence
        for function_sequence in initial_set:
            ga_set_list.append([0, function_sequence])

        i = 0
        best_function_set = None
        best_ratio = 0
        while i <= max_iterations:
            if settings.DEBUG:
                print("Genetic algorithm iteration {}".format(i))
            i += 1
            for ga_set in ga_set_list:
                # ga_set position 1 holds the function sequence
                potential_node = self.__generate_node(ga_set[1], tagged_tokens)
                new_network = SemanticNetwork()
                new_network.add_node(potential_node)

                ratio = difflib.SequenceMatcher(None, correct_network.to_string(), new_network.to_string()).ratio()

                if ratio > best_ratio:
                    best_function_set = ga_set

                if new_network.to_string() == correct_network.to_string() or ratio == 1.0:
                    if settings.DEBUG:
                        print("Genetic algorithm: Found exact match")
                        print(ratio)
                        print(ga_set)
                    return ga_set

                # Sort by integer so convert float ratio to integer
                temp_ga_set_list.append([ratio * 1000, ga_set[1]])

            ga_set_list.clear()
            ga_set_list += temp_ga_set_list
            temp_ga_set_list.clear()

            # Sort by ratio (descending)
            ga_set_list.sort(key=lambda x: int(x[0]), reverse=True)

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

                num = randint(0, 100)
                chance = randint(35, 45)

                if num % chance == 0:
                    if settings.DEBUG:
                        print("Genetic algorithm: Child has mutated.")
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

                num = randint(0, 100)
                chance = randint(35, 45)

                if num % chance == 0:
                    if settings.DEBUG:
                        print("Genetic algorithm: Child has mutated.")
                    child = mutate(child)
                temp_ga_set_list.append([0, child])

            ga_set_list.clear()
            ga_set_list += temp_ga_set_list
            temp_ga_set_list.clear()

        return best_function_set
