from tree.node import Node


# Class that handles all node operations
class Tree:
    # Root node name will never change
    root_node_name = "mammal"

    # Instantiate the tree to include the root mammal node
    def __init__(self, node_dictionary=None):

        # Mutable default argument
        if node_dictionary is None:
            node_dictionary = {self.root_node_name: Node(self.root_node_name)}

        self.node_dictionary = node_dictionary

    """
        Adds a new node to the tree, this handles all necessary operations for the node tree
    """
    def append(self, new_node):

        existing_node = self.node_dictionary.get(new_node.name, None)

        # We already have this node in our tree, use append existing function
        if existing_node is not None:
            # noinspection PyTypeChecker
            self.__append_existing(new_node, existing_node)
            return

        # New node came in with a non-empty parent parameter
        if new_node.parent != "":
            new_node_parent = self.node_dictionary.get(new_node.parent, None)

            # New node came in with existing parent
            if new_node_parent is not None:
                new_node_parent.append(child=new_node.name)
                self.node_dictionary[new_node_parent.name] = new_node_parent

            # New node came in with unknown parent
            else:
                root_parent = self.node_dictionary[self.root_node_name]

                new_parent_node = Node(new_node.parent)
                new_parent_node.append(child=new_node.name, parent=root_parent.name)

                root_parent.append(child=new_parent_node.name)
                self.node_dictionary[root_parent.name] = root_parent
                self.node_dictionary[new_parent_node.name] = new_parent_node

        if len(new_node.children) > 0:

            for child_node_name in new_node.children:
                existing_child_node = self.node_dictionary.get(child_node_name, None)

                if existing_child_node is not None:
                    existing_child_node_parent = self.node_dictionary.get(existing_child_node.parent, None)
                    existing_child_node_parent.remove_child(existing_child_node)
                    existing_child_node.append(parent=new_node.name)
                    self.node_dictionary[existing_child_node.name] = existing_child_node

                else:
                    self.append(Node(child_node_name))

        new_node_parent = self.node_dictionary.get(new_node.parent, None)

        if new_node_parent is None:

            # New node does not already exist, and does not have a parent
            root_parent = self.node_dictionary[self.root_node_name]
            new_node.parent = root_parent.name
            root_parent.append(child=new_node.name)
            self.node_dictionary[root_parent.name] = root_parent

        # Add the new node to the dictionary
        self.node_dictionary[new_node.name] = new_node

    # Modifies an existing node instead of creating a new node
    def __append_existing(self, new_node, existing_node):

        # All existing nodes will have a parent, there's no way to add a node without a parent being set
        existing_parent = self.node_dictionary.get(existing_node.parent, None)

        # New node came in with a non-empty parent parameter
        if new_node.parent != "":

            new_node_parent = self.node_dictionary.get(existing_node.parent, None)

            # We know the parent already and should switch parents
            if new_node_parent is not None:
                existing_parent.remove_child(existing_node)
                self.node_dictionary[existing_parent.name] = existing_parent

            # Recursively add new parent
            new_node_new_parent = Node(new_node.parent)
            new_node_new_parent.append(child=new_node.name)
            self.append(new_node_new_parent)

        # The new node has children, merge old children with new children
        if len(new_node.children) > 0:

            # Make sure the new children exist, add them if not
            for child_name in new_node.children:
                child = self.node_dictionary.get(child_name, None)

                if child is None:
                    self.append(Node(child_name))
                else:
                    existing_child_parent = self.node_dictionary.get(child.parent, None)
                    existing_child_parent.remove_child(child)
                    child.append(parent=new_node.name)
                    self.node_dictionary[child.name] = child

            # Add old children to new children
            new_node.children += existing_node.children

        # Existing node append without specifying new parent should keep old parent
        if new_node.parent == "":
            new_node.parent = existing_node.parent

        # Add node to dictionary
        self.node_dictionary[existing_node.name] = new_node

    # Print the dictionary values
    def print(self):

        for key in self.node_dictionary.keys():
            self.node_dictionary[key].print()
