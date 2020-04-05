import os
from gathering import SimpleRequestGathering
from preprocessor import Preprocessor
from vectorizer import Vectorizer
from readers import RawCorpusReader, PickledCorpusReader
from utils import save_json_file


class Nlp(object):

    def __init__(self, urls, path="./data", **kwargs):

        if not os.path.exists(path):
            os.makedirs(path)

        self.urls = urls
        self.path = path

        self.raw_path = os.path.join(self.path, 'raw')
        self.corpus_path = os.path.join(self.path, 'corpus')
        self.pickle_path = os.path.join(self.path, 'pickle')
        self.tokens_path = os.path.join(self.pickle_path, 'tokens')
        self.vectorization_path = os.path.join(
            self.pickle_path, 'vectorization')

        self.docs_path = os.path.join(self.path, 'docs')

        self.raws = None
        self.corpus = None

        self.vectorizer = None

    def gathering(self):
        for json in SimpleRequestGathering(self.urls).collect():
            save_json_file(self.raw_path, json['fileId'], json)

        self.read_raw()

    def read_raw(self):
        self.raws = RawCorpusReader(self.raw_path, r'[\w0-9-#_\.]+\.json')

    def preprocess(self):
        Preprocessor(self.tokens_path).preprocess(self.raws)

    def read_corpus(self):
        self.corpus = PickledCorpusReader(
            self.tokens_path, r'[a-z0-9-#_\.]+\.pickle', cat_file="cats.txt")

    def vectorize(self):
        if self.vectorizer is None:
            self.vectorizer = Vectorizer(self.corpus, self.vectorization_path,
                            cat_file_path=os.path.join(self.tokens_path, "cats.txt"))
        
        return self.vectorizer


def main():
    urls = [
        (["lula-livre", "folha"], 'https://www1.folha.uol.com.br/poder/2019/11/ex-presidente-lula-e-solto-apos-580-dias-preso-na-policia-federal-em-curitiba.shtml'),
        (["lula-livre", "veja"],
         'https://veja.abril.com.br/politica/lula-deixa-cadeia-apos-580-dias-veja-como-foi/'),
        (["lula-livre", "g1"], 'https://g1.globo.com/pr/parana/noticia/2019/11/08/lula-deixa-a-prisao-em-curitiba-apos-decisao-do-stf.ghtml'),
        (["lula-preso", "folha"],
         'https://www1.folha.uol.com.br/poder/2018/04/lula-e-preso.shtml'),
        (["lula-preso", "veja"],
         'https://veja.abril.com.br/politica/lula-e-preso-ex-presidente-se-entrega-a-policia-federal/'),
        (["lula-preso", "g1"], 'https://g1.globo.com/sp/sao-paulo/noticia/lula-se-entrega-a-pf-para-cumprir-pena-por-corrupcao-e-lavagem-de-dinheiro.ghtml'),
        (["lula-preso", "elpais"],
         'https://brasil.elpais.com/brasil/2018/04/05/politica/1522917041_563602.html'),
        (["lula-preso", "the-intercept"],
         'https://theintercept.com/2018/04/07/a-prisao-de-lula-e-politica/'),
        (["lula-preso", "uol"],
         'https://noticias.uol.com.br/politica/ultimas-noticias/2018/04/07/lula-prisao.htm'),
        (["lula-preso", "r7"],
         'https://noticias.r7.com/brasil/a-prisao-de-lula'),
        (["lula-preso", "istoe"],
         'https://istoe.com.br/rojoes-sao-disparados-por-todo-o-brasil-apos-prisao-de-lula/'),
        (["lula-preso", "metropoles"],
         'https://www.metropoles.com/brasil/lula-preso-os-bastidores-da-historica-prisao-do-ex-presidente'),
        (["lula-preso", "estadao"],
         'https://politica.estadao.com.br/blogs/fausto-macedo/lula-deixa-sindicato-para-a-prisao-da-lava-jato/'),
        (["lula-preso", "agenciabrasil"],
         'https://agenciabrasil.ebc.com.br/internacional/noticia/2018-04/imprensa-internacional-destaca-prisao-de-lula')
    ]

    nlp = Nlp(urls)

    nlp.gathering()
    nlp.read_raw()
    nlp.preprocess()
    nlp.read_corpus()

    nlp.vectorize().vect_frequency()
    nlp.vectorize().vect_tf_idf()
    nlp.vectorize().vect_tf()
    nlp.vectorize().vect_one_hot()



if __name__ == "__main__":
    main()
