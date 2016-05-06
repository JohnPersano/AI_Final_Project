"""
CSCI 6660 Final Project

Author: John Persano
Date:   05/01/2016
"""

import os
from xml.dom.minidom import Document

import pickle

import settings
from semantics.object_node import ObjectNode


class NSBuilder:
    """
    The NSBuilder class generates network sets to help train the InputSequencer. Data outside of the
    land mammalian domain was used to train the sequencer in order to avoid memory recall.

    The data used to train the InputSequencer can be found in network_set.xml in the data/out/ folder
    """
    pickle_name = "ns_builder.pickle"

    def __init__(self, sn_classifier, n_out="network_set.xml"):
        # Pipe output into the data output folder
        n_out = os.path.join(settings.DATA_OUT, n_out)

        self.sn_classifier = sn_classifier
        self.pickle_path = os.path.join(settings.DATA_OUT, self.pickle_name)
        self.n_out = n_out
        self._build_list = []

    def load(self):
        """
        Try to load the NSBuilder from a saved pickle.
        :return: a new NSBuilder if no pickle is found, the old one if it is found
        """
        if os.path.exists(self.pickle_path):
            with open(self.pickle_path, 'rb') as file:
                return pickle.load(file)
        return self

    def create_standard_set(self):
        """
        Create a standard working set of data for the InputParser to train on
        """
        # We loaded an existing set from a pickle, do not proceed further
        if len(self._build_list) > 0:
            self.__build()
            if settings.DEBUG:
                print("Loaded from pickle!")
            return

        if settings.DEBUG:
            print("Building network set...")

        # Add standard items to the network training set
        self.__add("a large fish has shiny scales", values=["large", "fish"], relations=["has"],
                   relation_objects=["shiny", "scales"])
        self.__add("a bass is a vertebrate", values=["bass"], relations=["is", "a"],
                   relation_objects=["vertebrate"])
        self.__add("a sun fish has shiny scales", values=["sun", "fish"], relations=["has"],
                   relation_objects=["shiny", "scales"])
        self.__add("a whale has blue skin", values=["whale"], relations=["has"],
                   relation_objects=["blue", "skin"])
        self.__add("water is a liquid", values=["water"], relations=["is", "a"],
                   relation_objects=["liquid"])
        self.__add("a strong whale eats plankton", values=["strong", "whale"], relations=["eats"],
                   relation_objects=["plankton"])
        self.__add("plankton are animals", values=["plankton"], relations=["are"],
                   relation_objects=["animals"])
        self.__add("whales are gentle", values=["whales"], relations=["are"],
                   relation_objects=["gentle"])
        self.__add("white sharks are mean", values=["white", "sharks"], relations=["are"],
                   relation_objects=["mean"])
        self.__add("sharks have sharp teeth", values=["sharks"], relations=["have"],
                   relation_objects=["sharp", "teeth"])

        # Save contents to a pickle
        with open(self.pickle_path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        self.__build()

    def __build(self):
        """
        Build the network_set.xml file. This is an internal method and should not be called directly.
        """
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

        if settings.DEBUG:
            print("Network Set built!")

    def __add(self, sentence, values=None, relations=None, relation_objects=None):
        """
        Add a node to the network set.
        """
        # Default values (cannot exist as parameter)
        if values is None:
            values = []
        if relations is None:
            relations = []
        if relation_objects is None:
            relation_objects = []

        # Begin creating new node with the node factory
        node_factory = ObjectNode.Factory()
        node_factory.set_sn_classifier(self.sn_classifier)
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
