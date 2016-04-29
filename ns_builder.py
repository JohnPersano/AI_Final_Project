import os
from xml.dom.minidom import Document

import pickle

import settings
from semantics.object_node import ObjectNode


class NSBuilder:
    pickle_name = "ns_builder.pickle"

    def __init__(self, n_out="network_set.xml"):
        # Pipe output into the data output folder
        n_out = os.path.join(settings.DATA_OUT, n_out)

        self.pickle_path = os.path.join(settings.DATA_OUT, self.pickle_name)
        self.n_out = n_out
        self._build_list = []

    def load(self):
        if os.path.exists(self.pickle_path):
            with open(self.pickle_path, 'rb') as file:
                return pickle.load(file)
        return NSBuilder()

    def create_standard_set(self):
        # We loaded an existing set from a pickle
        if len(self._build_list) > 0:
            self.build()
            return

        # Add standard items to the network training set
        self.add("dog has a black coat", values=["dog"], relations=["has", "a"], relation_objects=["black", "coat"])
        self.add("black is a color", values=["black"], relations=["is", "a"], relation_objects=["color"])

        # Save contents to a pickle
        with open(self.pickle_path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

        self.build()

    def build(self):
        document = Document()
        root = document.createElement("root")
        document.appendChild(root)

        out_file = open(self.n_out, "w")

        for tuple_value in self._build_list:
            input_sentence = tuple_value[0]
            node_string = tuple_value[1]

            train_node = document.createElement("train-data")
            root.appendChild(train_node)

            input_element = document.createElement('sentence')
            input_element_content = document.createTextNode(input_sentence)
            input_element.appendChild(input_element_content)
            train_node.appendChild(input_element)

            node_element = document.createElement('node')
            node_element_content = document.createTextNode(node_string)
            node_element.appendChild(node_element_content)
            train_node.appendChild(node_element)

        document.writexml(out_file, indent="  ", addindent="  ", newl='\n')
        out_file.close()

    def add(self, sentence, values=None, relations=None, relation_objects=None):
        # Default values (cannot exist as parameter)
        if values is None:
            values = []
        if relations is None:
            relations = []
        if relation_objects is None:
            relation_objects = []

        # Begin creating new node with the node factory
        node_factory = ObjectNode.Factory()
        for value in values:
            node_factory.add_value(value)
        for relation in relations:
            node_factory.add_relation(relation)
        for relation_object in relation_objects:
            node_factory.add_relation_object(relation_object)

        # Build the node
        node = node_factory.build()
        node_string = node.to_string()

        self._build_list.append((sentence, node_string))

