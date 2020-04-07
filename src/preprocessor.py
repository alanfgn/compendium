import os
import pickle

from parser import html_cleaner
from lexical import paragraph_segmentation, sentence_segmentation, word_segmentation, part_of_speech_tagger
from utils import save_pickle


class Preprocessor(object):

    def __init__(
            self,
            corpus,
            target,
            parser=html_cleaner,
            functions=[
                paragraph_segmentation,
                sentence_segmentation,
                part_of_speech_tagger],
            cat_file_name="cats.txt",
            **kwargs):

        self.corpus = corpus
        self.target = target
        self.parser = parser
        self.functions = functions
        self.cat_file_name = cat_file_name

    def get_name(self, fileid):
        name, _ = os.path.splitext(os.path.basename(fileid))
        return name

    def save_cat_file(self, target, names):
        file = ""
        for name, labels in names:
            file += name + " " + " ".join(labels) + "\n"

        open(os.path.join(target, self.cat_file_name), "w").write(file)

    def get_tokens(self, document):
        def extract(doc, funcs):
            if len(funcs) == 1:
                return funcs[0](doc)
            else:
                docs = funcs[0](doc)
                return [extract(x, funcs[1:]) for x in docs]

        return extract(document, self.functions)

    def preprocess(self, target=None, save_fun=save_pickle, extension="pickle", only_parse=False):
        cat_file = []

        target = target if target is not None else self.target

        if not os.path.exists(target):
            os.makedirs(target)

        for fileid in self.corpus.fileids():
            name = self.get_name(fileid) + "." + extension

            text = self.parser(next(self.corpus.raws(fileid))['text'])
            document = self.get_tokens(text) if not only_parse else text

            labels = next(self.corpus.labels(fileid))

            cat_file.append((name, labels))
            save_fun(os.path.join(target, name), document)

            del document

        self.save_cat_file(target, cat_file)
