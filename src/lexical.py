import nltk
from nltk.tokenize import word_tokenize
from nltk import DefaultTagger, UnigramTagger, BigramTagger
from nltk.corpus import floresta

def simplify_tag(t):
    if "+" in t:
        return t[t.index("+") + 1:]
    else:
        return t

tsents = floresta.tagged_sents()
tsents = [[(w.lower(),simplify_tag(t)) for (w,t) in sent] for sent in tsents if sent]

train_tsents = tsents[100:]
test_tsents = tsents[:100]

uni_tagger = UnigramTagger(train_tsents, backoff= DefaultTagger("n"))
bi_tagger = BigramTagger(train_tsents, backoff=uni_tagger)

def paragraph_segmentation(text):
    return text.split("\n\n")

def sentence_segmentation(text):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
    return sent_tokenizer.tokenize(text)

def word_segmentation(text):
    return word_tokenize(text, language="portuguese")

def part_of_speech_tagger(text):
    return bi_tagger.tag(word_segmentation(text))
