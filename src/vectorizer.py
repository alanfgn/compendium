import bs4
import nltk
import shutil
import os
import json
import pickle
import unicodedata
from nltk import FreqDist
from nltk.text import TextCollection

from readers import PickledCorpusReader


class Vectorizer(object):

    def __init__(self, corpus, target, cat_file_path, stopwords=nltk.corpus.stopwords.words("portuguese"), **kwargs):
        self.target = target
        self.corpus = corpus
        self.cat_file_path = cat_file_path
        self.stopwords = stopwords

    def clean_data(self, fileids):
        for word in self.corpus.words(fileids):
            if word.lower() in self.stopwords:
                continue
            if all(unicodedata.category(char).startswith('P') for char in word):
                continue
            yield word

    def vect_frequency(self):
        def frequency(fileid):
            fre = FreqDist(self.clean_data(fileid))
            return dict(fre.items())

        for fileid in self.corpus.fileids():
            self.save("frequency", frequency(fileid), fileid)

    def vect_tf_idf(self):
        corpus = {fileid: list(self.clean_data(fileid))
                  for fileid in self.corpus.fileids()}

        texts = TextCollection(list(corpus.values()))

        for fileid, doc in corpus.items():
            document = {
                term: texts.tf_idf(term, doc)
                for term in doc
            }

            self.save("tf-idf", document, fileid)

    def vect_tf(self):
        corpus = {fileid: list(self.clean_data(fileid))
                  for fileid in self.corpus.fileids()}

        texts = TextCollection(list(corpus.values()))

        for fileid, doc in corpus.items():
            document = {
                term: texts.tf(term, doc)
                for term in doc
            }

            self.save("tf", document, fileid)

    def vect_one_hot(self):
        def one_hot(fileid):
            return {
                token: True
                for token in self.clean_data(fileid)
            }

        for fileid in self.corpus.fileids():
            self.save("one_hot", one_hot(fileid), fileid)

    def save(self, method, vector, fileid):

        basename = os.path.join(self.target, method)
        target = os.path.normpath(os.path.join(basename, fileid))

        parent = os.path.dirname(target)

        if not os.path.exists(parent):
            os.makedirs(parent)

        with open(target, 'wb') as f:
            pickle.dump(vector, f, pickle.HIGHEST_PROTOCOL)

        line = fileid + " " +" ".join(self.corpus.categories(fileid)) + "\n"
        cat_file_path = os.path.join(parent, "cats.txt")

        with open(cat_file_path, "a") as f:
            f.write(line)
