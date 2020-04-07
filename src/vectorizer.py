import bs4
import nltk
import shutil
import os
import json
import pickle
import unicodedata
from nltk import FreqDist
from nltk.text import TextCollection
from gensim.models import Word2Vec
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

from readers import PickledCorpusReader


class Vectorizer(object):

    def __init__(self, corpus, target, cat_file_path, stopwords=nltk.corpus.stopwords.words("portuguese"), **kwargs):
        self.target = target
        self.corpus = corpus
        self.cat_file_path = cat_file_path
        self.stopwords = stopwords

    def clean(self, words):
        for word in words:
            if word.lower() in self.stopwords:
                continue
            if all(unicodedata.category(char).startswith('P') for char in word):
                continue
            yield word

    def clean_data(self, fileids):
        return self.clean(self.corpus.words(fileids))

    def vect_frequency(self):
        print("\nStart Vectorizing frequency\n")

        def frequency(fileid):
            fre = FreqDist(self.clean_data(fileid))
            return dict(fre.items())

        for fileid in self.corpus.fileids():
            print("%s ..." % fileid)

            self.save("frequency", frequency(fileid), fileid)

        print("\nFinish Vectorizing frequency\n")



    def vect_tf_idf(self):
        print("\nStart tf-idf frequency\n")

        corpus = {fileid: list(self.clean_data(fileid))
                  for fileid in self.corpus.fileids()}

        texts = TextCollection(list(corpus.values()))

        for fileid, doc in corpus.items():
            print("%s ..." % fileid)

            document = {
                term: texts.tf_idf(term, doc)
                for term in doc
            }

            self.save("tf-idf", document, fileid)
        
        print("\nFinish tf-idf frequency\n")


    def vect_tf(self):
        print("\nStart tf frequency\n")

        corpus = {fileid: list(self.clean_data(fileid))
                  for fileid in self.corpus.fileids()}

        texts = TextCollection(list(corpus.values()))

        for fileid, doc in corpus.items():
            print("%s ..." % fileid)

            document = {
                term: texts.tf(term, doc)
                for term in doc
            }

            self.save("tf", document, fileid)

        print("\nFinish tf frequency\n")


    def vect_word2vec(self):
        print("\nStart word2vec frequency\n")

        def word2vec(fileid):
            document = [list(self.clean([token[0] for token in sent]))
                        for sent in self.corpus.sentences(fileid)]

            return Word2Vec(document, min_count=1, size= 50, workers=3, window =3, sg = 1)

        for fileid in self.corpus.fileids():
            print("%s ..." % fileid)

            self.save("word2vec", word2vec(fileid), fileid)

        print("\nFinish word2vec frequency\n")



    def vect_one_hot(self):
        print("\nStart one hot frequency\n")

        def one_hot(fileid):
            return {
                token: True
                for token in self.clean_data(fileid)
            }

        for fileid in self.corpus.fileids():
            print("%s ..." % fileid)

            self.save("one_hot", one_hot(fileid), fileid)

        print("\nFinish one hot frequency\n")


    def save(self, method, vector, fileid):

        basename = os.path.join(self.target, method)
        target = os.path.normpath(os.path.join(basename, fileid))

        parent = os.path.dirname(target)

        if not os.path.exists(parent):
            os.makedirs(parent)

        with open(target, 'wb') as f:
            pickle.dump(vector, f, pickle.HIGHEST_PROTOCOL)

        line = fileid + " " + " ".join(self.corpus.categories(fileid)) + "\n"
        cat_file_path = os.path.join(parent, "cats.txt")

        with open(cat_file_path, "a") as f:
            f.write(line)
