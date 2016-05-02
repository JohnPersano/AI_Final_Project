import os
import pickle

import nltk

import settings
from semantics.node import Node
from semantics.object_node import ObjectNode
from semantics.relation_node import RelationNode


class SemanticNetwork:
    pickle_name = "semantic_network.pickle"

    # Instantiate the tree to include the root mammal node
    def __init__(self, sn_classifier):
        self.node_dictionary = {}
        self.sn_classifier = sn_classifier

    def add_node(self, node=None):
        if node is None:
            print("Cannot add a node of type None!")
            return
        # Do we already have this node in our tree?
        temp_node = self.node_dictionary.get(node.get_key(), None)
        if temp_node is None:
            temp_node = node
        else:
            # Combine attributes and parents of new and old node
            temp_node += node
        node = temp_node

        # Ensure derivative children exist e.g. for "red fox", red and fox must exist
        if len(node.value_tokens) > 1:
            for value_token in node.value_tokens:
                temp_value_token = self.node_dictionary.get(value_token, None)
                if temp_value_token is None:
                    new_node = ObjectNode()
                    new_node.set_value(value_token)
                    temp_value_token = new_node
                value_token_object = temp_value_token
                value_token_object.add_constituent_of(node)
                self.node_dictionary[value_token] = value_token_object

        for in_relationship in node.in_relationships:
            object_key = RelationNode.create_key(in_relationship[0])
            temp_object = self.node_dictionary.get(object_key, None)
            if temp_object is None:
                new_object = ObjectNode()
                new_object.set_value(in_relationship[0])
                temp_object = new_object
            relation_object = temp_object

            relation_key = RelationNode.create_key(in_relationship[1])
            temp_relation = self.node_dictionary.get(relation_key, None)
            if relation_key is None:
                new_relation = RelationNode()
                new_relation.set_value(in_relationship[1], self.sn_classifier)
                temp_relation = new_relation
            relation = temp_relation
            relation.add_in_object(relation_object)
            relation.add_out_object(node)
            self.node_dictionary[relation_key] = relation

            relation_object.add_out_relationship((relation, node))
            self.node_dictionary[object_key] = relation_object
            if len(relation_object.value_tokens) > 1:
                for value_token in relation_object.value_tokens:
                    temp_value_token = self.node_dictionary.get(value_token, None)
                    if temp_value_token is None:
                        new_node = ObjectNode()
                        new_node.set_value(value_token)
                        temp_value_token = new_node
                    value_token_object = temp_value_token
                    value_token_object.add_constituent_of(relation_object)
                    self.node_dictionary[value_token] = value_token_object

        for out_relationship in node.out_relationships:
            object_key = RelationNode.create_key(out_relationship[1])
            temp_object = self.node_dictionary.get(object_key, None)
            if temp_object is None:
                new_object = ObjectNode()
                new_object.set_value(out_relationship[1])
                temp_object = new_object
            relation_object = temp_object

            print("In relat : {}".format(out_relationship[0]))

            relation_key = RelationNode.create_key(out_relationship[0])
            temp_relation = self.node_dictionary.get(relation_key, None)
            if temp_relation is None:
                new_relation = RelationNode()
                new_relation.set_value(out_relationship[0], self.sn_classifier)
                temp_relation = new_relation
            relation = temp_relation
            relation.add_in_object(node)
            relation.add_out_object(relation_object)
            self.node_dictionary[relation_key] = relation

            relation_object.add_in_relationship((node, relation))
            self.node_dictionary[object_key] = relation_object
            if len(relation_object.value_tokens) > 1:
                for value_token in relation_object.value_tokens:
                    temp_value_token = self.node_dictionary.get(value_token, None)
                    if temp_value_token is None:
                        new_node = ObjectNode()
                        new_node.set_value(value_token)
                        temp_value_token = new_node
                    value_token_object = temp_value_token
                    value_token_object.add_constituent_of(relation_object)
                    self.node_dictionary[value_token] = value_token_object

        self.node_dictionary[node.get_key()] = node

    @staticmethod
    def __contains_item(node_attribute_tokens, token):
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
