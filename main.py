import nltk

from learning.sequencer.input_sequencer import InputSequencer
from ns_builder import NSBuilder
from semantics.semantic_network import SemanticNetwork
from utils.general_utils import GeneralUtils

if __name__ == "__main__":

    # Erase all files in the out directory
    GeneralUtils().clean_start()

    print("Begin")
    print(nltk.word_tokenize("dog has a black coat"))
    print(nltk.pos_tag(nltk.word_tokenize("dog has a black coat")))
    print("-----------------------------------------------------\n\n")

    # Will load from a pickle if one exists
    ns_builder = NSBuilder().load()
    ns_builder.create_standard_set()

    input_sequencer = InputSequencer()
    input_sequencer.train()

    test_network = SemanticNetwork()
    test_network.add_node(input_sequencer.parse_to_node("cat has blue coat"))
    test_network.print()