# Class that handles all node operations
import nltk
import re

from genus.animal_node import AnimalNode
from genus.attribute_node import AttributeNode
from genus.base_node import BaseNode


class Genus:
    # Root node name will never change
    root_node_name = "mammal"

    # Instantiate the tree to include the root mammal node
    def __init__(self, ):
        self.node_dictionary = {self.root_node_name: AnimalNode(self.root_node_name)}
        self.legal_attribute_pos = re.compile("NN|VB|(VB(D|G|N|P)|Z)|RB|(RB(R|S))|JJ|(JJ(R|S))")

    def append_animal(self, new_node):

        # Ensure node is the correct type
        if new_node.get_type() is not BaseNode.Animal:
            print("Node is not AnimalType")
            return

        existing_node = self.node_dictionary.get(new_node.get_name(), None)

        # We already have this node in our tree, use append existing function
        if existing_node is not None:
            # noinspection PyTypeChecker
            self._append_existing(new_node, existing_node)
            return

        # New node came in with a non-empty parent parameter
        if new_node.genus.parent != "":
            new_node_parent = self.node_dictionary.get(new_node.genus.parent, None)

            # New node came in with existing parent
            if new_node_parent is not None:
                new_node_parent.genus.append(child=new_node.get_name())
                self.node_dictionary[new_node_parent.get_name()] = new_node_parent

            # New node came in with unknown parent
            else:
                root_parent = self.node_dictionary[self.root_node_name]

                new_parent_node = AnimalNode(new_node.genus.parent)
                new_parent_node.genus.append(child=new_node.get_name(), parent=root_parent.get_name())

                root_parent.genus.append(child=new_parent_node.get_name())
                self.node_dictionary[root_parent.get_name()] = root_parent
                self.node_dictionary[new_parent_node.get_name()] = new_parent_node

        new_node_parent = self.node_dictionary.get(new_node.genus.parent, None)

        if new_node_parent is None:

            # New node does not already exist, and does not have a parent
            root_parent = self.node_dictionary[self.root_node_name]
            new_node.genus.parent = root_parent.get_name()
            root_parent.genus.append(child=new_node.get_name())
            self.node_dictionary[root_parent.get_name()] = root_parent

        # Add the new node to the dictionary
        self.node_dictionary[new_node.get_name()] = new_node

        # New node came in with children
        if len(new_node.genus.children) > 0:

            for child_node_name in new_node.genus.children:
                existing_child_node = self.node_dictionary.get(child_node_name, None)

                # This child exists already
                if existing_child_node is not None:
                    existing_child_node_parent = self.node_dictionary.get(existing_child_node.genus.parent, None)
                    existing_child_node_parent.genus.remove_child(existing_child_node)
                    existing_child_node.genus.append(parent=new_node.get_name())
                    self.node_dictionary[existing_child_node.get_name()] = existing_child_node

                else:
                    child_node = AnimalNode(child_node_name)
                    child_node.genus.append(parent=new_node.get_name())
                    self.append_animal(child_node)

    # Modifies an existing node instead of creating a new node
    def _append_existing(self, new_node, existing_node):

        # All existing nodes will have a parent, there's no way to add a node without a parent being set
        existing_parent = self.node_dictionary.get(existing_node.genus.parent, None)

        # New node came in with a non-empty parent parameter
        if new_node.genus.parent != "":

            new_node_parent = self.node_dictionary.get(existing_node.genus.parent, None)

            # We know the parent already and should switch parents
            if new_node_parent is not None:
                existing_parent.genus.remove_child(existing_node)
                self.node_dictionary[existing_parent.get_name()] = existing_parent

            # Recursively add new parent
            new_node_new_parent = AnimalNode(new_node.genus.parent)
            new_node_new_parent.genus.append(child=new_node.get_name())
            self.append_animal(new_node_new_parent)

        # Existing node append without specifying new parent should keep old parent
        if new_node.genus.parent == "":
            new_node.genus.parent = existing_node.genus.parent

        # Add node to dictionary
        self.node_dictionary[existing_node.get_name()] = new_node

        # The new node has children, merge old children with new children
        if len(new_node.genus.children) > 0:

            # Make sure the new children exist, add them if not
            for child_name in new_node.genus.children:
                child = self.node_dictionary.get(child_name, None)

                if child is None:
                    new_child = AnimalNode(child_name)
                    new_child.genus.append(parent=new_node.get_name())
                    self.append_animal(new_child)
                else:
                    existing_child_parent = self.node_dictionary.get(child.genus.parent, None)
                    existing_child_parent.genus.remove_child(child)
                    child.genus.append(parent=new_node.get_name())
                    self.node_dictionary[child.get_name()] = child

            # Add old children to new children
            new_node.genus.children += existing_node.genus.children
            self.node_dictionary[existing_node.get_name()] = new_node

    #TODO
    def append_attribute(self, animal_node, attribute_chunk):

        # Ensure node is the correct type
        if animal_node.get_type() is not BaseNode.Animal:
            print("Node is not AnimalType")
            return

        existing_node = self.node_dictionary.get(animal_node.get_name(), None)

        if existing_node is None:
            # noinspection PyTypeChecker
            self.append_animal(animal_node)
            self.append_attribute(animal_node, attribute_chunk)
            return

        attribute = ""
        for word_tuple in attribute_chunk.leaves():
            # Is this a valid POS?
            if self.legal_attribute_pos.match(word_tuple[1]):
                if attribute != "":
                    attribute += " "
                attribute += word_tuple[0]

        for word_tuple in attribute_chunk.leaves():
            # Is this a valid POS?
            if self.legal_attribute_pos.match(word_tuple[1]):
                if word_tuple[1] == "NN":
                    attribute_node = AttributeNode(AttributeNode.Noun, word_tuple[0])
                    attribute_node.append_connection(animal_node.get_name())

                    self.node_dictionary[attribute_node.get_name()] = attribute_node

                    self._back_propagate_attribute(existing_node, attribute_node, attribute)

    #TODO
    def _back_propagate_attribute(self, animal_node, attribute_node, attribute):

        animal_node.append(attribute)
        self.node_dictionary[animal_node.get_name()] = animal_node

        parent = self.node_dictionary.get(animal_node.genus.parent, None)

        print("Animal {} parent {}".format(animal_node.get_name(), parent.get_name()))

        while parent is not None:
            attribute_node.append_connection(parent.get_name())
            self.node_dictionary[attribute_node.get_name()] = attribute_node

            parent.append(attribute)
            self.node_dictionary[parent.get_name()] = parent
            print("11Animal {} parent {}".format(animal_node.get_name(), parent.get_name()))

            parent = self.node_dictionary.get(parent.genus.parent, None)

    # Print the dictionary values
    def print(self):

        for key in self.node_dictionary.keys():
            self.node_dictionary[key].print()
