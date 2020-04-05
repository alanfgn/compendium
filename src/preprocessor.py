import os
import pickle

from parser import html_cleaner
from lexical import paragraph_segmentation, sentence_segmentation, word_segmentation, part_of_speech_tagger
from utils import save_pickle


class Preprocessor(object):

    def __init__(
            self,
            target=None,
            parser= html_cleaner,
            functions=[
                paragraph_segmentation,
                sentence_segmentation,
                part_of_speech_tagger],
            cat_file_name="cats.txt",
            **kwargs):

        self.target = target
        self.parser = parser
        self.functions = functions
        self.cat_file_name = cat_file_name

        if not os.path.exists(target):
            os.makedirs(target)

    def get_name(self, fileid):
        name, _ = os.path.splitext(os.path.basename(fileid))
        return name

    def save_cat_file(self, names):
        file = ""
        for name, labels in names:
            file += name + " " + " ".join(labels) + "\n"

        open(os.path.join(self.target, self.cat_file_name), "w").write(file)

    def get_tokens(self, document):
        def extract(doc, funcs):
            if len(funcs) == 1:
                return funcs[0](doc)
            else:
                docs = funcs[0](doc)
                return [extract(x, funcs[1:]) for x in docs]

        return extract(document, self.functions)

    def preprocess(self, corpus, save_fun=save_pickle, extension="pickle"):
        cat_file = []

        for fileid in corpus.fileids():
            name = self.get_name(fileid) + "." + extension

            text = self.parser(next(corpus.raws(fileid))['text'])
            document = self.get_tokens(text)

            labels = next(corpus.labels(fileid))

            cat_file.append((name, labels))
            save_fun(os.path.join(self.target, name), document)

            del document

        self.save_cat_file(cat_file)
