import os
import pickle

import settings
from semantics.node import Node


class SemanticNetwork:
    pickle_name = "semantic_network.pickle"

    # Instantiate the tree to include the root mammal node
    def __init__(self, ):
        self.node_dictionary = {}

    def add_node(self, node=None):
        if node is None:
            print("Cannot add a node of type None!")
            return

        # Do we already have this node in our tree?
        temp_node = self.node_dictionary.get(node.name, None)
        if temp_node is None:
            temp_node = node
        else:
            # Combine attributes and parents of new and old node
            for attribute in node.attributes:
                temp_node.add_attribute(attribute)
            for parent in node.inherited_by:
                temp_node.add_inherited_by(parent)
        node = temp_node

        for attribute_token in node.attribute_tokens:
            temp_attribute_token = self.node_dictionary.get(attribute_token, None)
            if temp_attribute_token is None:
                temp_attribute_token = Node(attribute_token)
            attribute_token = temp_attribute_token
            attribute_token.add_inherited_by(node.name)
            self.node_dictionary[attribute_token.name] = attribute_token

        self.node_dictionary[node.name] = node

    def __contains_item(self, node_attribute_tokens, token):
        for temp_attribute_token in node_attribute_tokens:
            if token == temp_attribute_token:
                return True
        return False

    # Print the dictionary values
    def print(self):
        for key in self.node_dictionary.keys():
            self.node_dictionary[key].print()

    def to_string(self):
        string = ""
        for key in self.node_dictionary.keys():
            string += self.node_dictionary[key].to_string()
        return string

    def to_pickle(self):
        pickle_name = os.path.join(settings.DATA_OUT, self.pickle_name)
        with open(pickle_name, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def from_pickle(self):
        pickle_name = os.path.join(settings.DATA_OUT, self.pickle_name)
        with open(pickle_name, 'rb') as file:
            return pickle.load(file)
