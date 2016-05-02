import string
import nltk


class Utils:
    punct_regex = nltk.re.compile('[%s]' % nltk.re.escape(string.punctuation))

    @staticmethod
    def strip_sentence(input_query):
        """
        Strips a sentence of all punctuation and numbers. Some corpera have numbers
        assigned to verses.
        :param input_query: the sentence to strip
        :return: the stripped sentence
        """
        input_query = ''.join([i for i in input_query if not i.isdigit()])
        input_query = input_query.lower()
        return Utils.punct_regex.sub('', input_query)




