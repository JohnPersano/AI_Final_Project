import os
import random
import string

import nltk
from nltk.corpus import inaugural
import xml.etree.ElementTree as ElementTree
import settings


class QSBuilder:
    """
    Class that handles question/statement file building for the Bayes classifier
    """
    qsets_files = ["QA2004_testset.xml", "QA2005_testset.xml", "QA2006_testset.xml", "QA2007_testset.xml"]
    punct_regex = nltk.re.compile('[%s]' % nltk.re.escape(string.punctuation))

    def __init__(self, qsets_files=None, q_out="questions_out.txt", s_out="sentences_out.txt"):
        # Load parameter set instead of default set
        if qsets_files is not None:
            self.qsets_files = qsets_files

        # Pipe output into the data output folder
        q_out = os.path.join(settings.DATA_OUT, q_out)
        s_out = os.path.join(settings.DATA_OUT, s_out)

        # Open files with append permission
        self.q_out = open(q_out, "w")
        self.s_out = open(s_out, "w")

    def generate_qs_files(self):
        """
        Generate a fresh set of question/statement files
        :return: None
        """
        if settings.DEBUG:
            print("Generating question/statement files...")
        self.__append_corpus_data()
        self.__append_aux_questions()
        self.q_out.close()
        self.s_out.close()
        if settings.DEBUG:
            print("Question/statement files generated")

    def __append_aux_questions(self):
        """
        Appends auxiliary questions since there are more statements than questions in typical corpora
        :return: None
        """
        qsets_directory = settings.DATA_QSETS

        for file_name in self.qsets_files:
            file = open(os.path.join(qsets_directory, file_name))
            file_data = file.read()
            file.close()

            # Files contain XML data
            root_element = ElementTree.fromstring(file_data.strip())
            for element in root_element.findall('target/qa/q'):
                element = element.text.strip()
                # There are random 'Other' elements in the set, we do not want these or list type questions
                if element != 'Other' and not element.endswith("."):
                    # We do not want punctuation as a potential feature
                    element = element.replace("?", "")
                    self.q_out.write(element + "\n")

    def __append_corpus_data(self):
        """
        Appends data to the questions and statements files from the Gutenberg corpus
        :return: None
        """
        sentences = []

        # Use the Presidential inaugural addresses corpus
        for fileid in inaugural.fileids():
            raw_text = inaugural.raw(fileid)
            sentence_tokens = nltk.sent_tokenize(raw_text)
            sentences += sentence_tokens

        random.shuffle(sentences)
        random.shuffle(sentences)
        random.shuffle(sentences)

        # Write sentences to the sentences and questions files
        for sentence in sentences:
            if sentence and 10 < len(sentence) < 75:
                if sentence.endswith('?'):
                    self.q_out.write(self.__strip_sentence(sentence) + '\n')
                else:
                    self.s_out.write(self.__strip_sentence(sentence) + '\n')

    def __strip_sentence(self, sentence):
        """
        Strips a sentence of all punctuation and numbers. Some corpera have numbers
        assigned to verses.
        :param sentence: the sentence to strip
        :return: the stripped sentence
        """
        sentence = ''.join([i for i in sentence if not i.isdigit()])
        sentence = sentence.lower()
        return self.punct_regex.sub('', sentence)
