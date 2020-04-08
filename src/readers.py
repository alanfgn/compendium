import codecs
import json
import pickle

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader

PICKLE_PATTERN = r'[a-z0-9-#_\.]+\.pickle'
CAT_FILE_NAME = 'cats.txt'

class RawCorpusReader(CorpusReader):

    def __init__(self, root, fileids, encoding='utf8', **kwargs):
        CorpusReader.__init__(self, root, fileids, encoding)

    def raws(self, fileids=None):
        for path, encoding in self.abspaths(fileids, include_encoding=True):
            with codecs.open(path, 'r', encoding=encoding) as f:
                yield json.load(f)

    def labels(self, fileids=None):
        for json in self.raws(fileids):
            yield json['labels']


class PickledCorpusReader(CategorizedCorpusReader, CorpusReader):

    def __init__(self, root, fileids=PICKLE_PATTERN, cat_file="cats.txt", **kwargs):
        
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_file'] = CAT_FILE_NAME
        
        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids)

    def resolve(self, fileids, categories):
        if categories is not None:
            return self.fileids(categories)
        
        return fileids

    def documents(self, fileids=None, categories=None):
        fileids = self.resolve(fileids, categories)

        for path, enc, fileid in self.abspaths(fileids, True, True):
            with open(path, 'rb') as f:
                yield pickle.load(f)

    def paragraphs(self, fileids=None, categories=None):
        for document in self.documents(fileids, categories):
            for paragraph in document:
                yield paragraph

    def sentences(self, fileids=None, categories=None):
        for paragraph in self.paragraphs(fileids, categories):
            for sentence in paragraph:
                yield sentence

    def tokens(self, fileids=None, categories=None):
        for sentence in self.sentences(fileids, categories):
            for token in sentence:
                yield token

    def words(self, fileids=None, categories=None):
        for token in self.tokens(fileids, categories):
            yield token[0]


class PickledVectorizeCorpusReader(CategorizedCorpusReader, CorpusReader):

    def __init__(self, root, fileids=PICKLE_PATTERN, **kwargs):

        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_file'] = CAT_FILE_NAME

        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids)

    def resolve(self, fileids, categories):
        if categories is not None:
            return self.fileids(categories)
        
        return fileids

    def vectors(self, fileids=None, categories=None):
        fileids = self.resolve(fileids, categories)

        for path, enc, fileid in self.abspaths(fileids, True, True):
            with open(path, 'rb') as f:
                yield pickle.load(f)
