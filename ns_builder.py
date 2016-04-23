import os
from xml.dom.minidom import Document
from semantics.node import Node

import settings
from semantics.semantic_network import SemanticNetwork


class NSBuilder:

    def __init__(self, n_out="network_set.xml"):
        # Pipe output into the data output folder
        n_out = os.path.join(settings.DATA_OUT, n_out)
        self.n_out = open(n_out, "w")
        self._build_list = []

    def build(self):
        document = Document()
        root = document.createElement("root")
        document.appendChild(root)

        for tuple_value in self._build_list:
            input_sentence = tuple_value[0]
            network_string = tuple_value[1]

            train_node = document.createElement("train-data")
            root.appendChild(train_node)

            input_element = document.createElement('sequencer')
            input_element_content = document.createTextNode(input_sentence)
            input_element.appendChild(input_element_content)
            train_node.appendChild(input_element)

            network_element = document.createElement('network')
            network_element_content = document.createTextNode(network_string)
            network_element.appendChild(network_element_content)
            train_node.appendChild(network_element)

        document.writexml(self.n_out, indent="  ", addindent="  ", newl='\n')
        self.n_out.close()

    def add(self, sentence, name="", attribute=None, parent=None):
        node = Node()
        node.set_name(name)
        if attribute is not None:
            node.add_attribute(attribute)
        if parent is not None:
            node.add_inherited_by(parent)

        semantic_network = SemanticNetwork()
        semantic_network.add_node(node)

        self._build_list.append((sentence, semantic_network.to_string()))

