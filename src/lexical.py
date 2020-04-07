import nltk
from nltk.tokenize import word_tokenize
from nltk import DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger
from nltk.corpus import floresta
from nltk.corpus import mac_morpho
from nltk.tag import tnt
import numpy as np


def simplify_tag(t):
    if "+" in t:
        return t[t.index("+") + 1:]
    else:
        return t


tsents = mac_morpho.tagged_sents()
tsents = [[(w.lower(), simplify_tag(t)) for (w, t) in sent]
          for sent in tsents if sent]

np.random.shuffle(tsents)

train_tsents = tsents[100:]
test_tsents = tsents[:100]

UNIGRAM_TAGGER = UnigramTagger(train_tsents, backoff=DefaultTagger("n"))
BIGRAM_TAGGER = BigramTagger(train_tsents, backoff=UNIGRAM_TAGGER)
TRIGRAM_TAGGER = TrigramTagger(train_tsents, backoff=BIGRAM_TAGGER)

TNT_TAGGER = tnt.TnT(unk=DefaultTagger('N'), Trained=True)
TNT_TAGGER.train(train_tsents)


def paragraph_segmentation(text):
    return text.split("\n\n")


def sentence_segmentation(text):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
    return sent_tokenizer.tokenize(text)


def word_segmentation(text):
    return word_tokenize(text, language="portuguese")


def part_of_speech_tagger(text, tagger=TNT_TAGGER):
    return tagger.tag(word_segmentation(text))


def main():
    print("Unigram Tagger:", UNIGRAM_TAGGER.evaluate(test_tsents))
    print("Bigram Tagger:", BIGRAM_TAGGER.evaluate(test_tsents))
    print("Trigram Tagger:", TRIGRAM_TAGGER.evaluate(test_tsents))
    print("Tnt Tagger:", TNT_TAGGER.evaluate(test_tsents))

    # Unigram Tagger:           0.8218864468864469
    # Bigram Tagger:            0.8507326007326007
    # Trigram Tagger:           0.8489010989010989
    # Tnt Tagger:               0.913003663003663


if __name__ == "__main__":
    main()
