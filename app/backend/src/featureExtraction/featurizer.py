from corpy.udpipe import Model
import os
from app.backend.helper import IMPROVED_HURTLEX
from app.backend.src.featureExtraction.models import udpipe_models, UD_VERSION
import wget
import csv

HL_VERSION = "1.2"


class HurtLexFeaturizer:
    def __init__(self, language, improved_hurtlex, level="conservative"):
        self.language = language
        self.improved_hurtlex = improved_hurtlex
        self.model = self.load_model()
        self.lexicon = self.read_lexicon(level)

    # download pre-trained model from  UDPipe website and create the model with it to extract features from sentences
    def load_model(self):
        extension = "-ud-{0}.udpipe".format(UD_VERSION)
        udpipe_model = udpipe_models[self.language] + extension
        model_file = os.path.join(os.path.dirname(__file__), '../..', 'UDPipeModels', udpipe_model)
        if not os.path.exists(model_file):
            url = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/{0}".format(udpipe_model)
            print(url)
            wget.download(url, out=os.path.join(os.path.dirname(__file__), '../..', 'UDPipeModels'))
        return Model(model_file)

    # get vector of lemmas of the given sentence using pre-trained model from UDPipe
    def lemmatize(self, text):
        sentences = list(self.model.process(text))
        lemmas = [t.lemma for t in sentences[0].words[1:]]
        return lemmas

    # get vector of type of lemmas of the given sentence, e.g: verb, noun, punct, adj...
    def pos(self, text):
        sentences = list(self.model.process(text))
        pos = [t.upostag for t in sentences[0].words[1:]]
        return pos

    # create a lexicon of words together with the categories for every word used in lexica file
    # generated categories vector: self.categories= ['an', 'asf', 'asm', 'cds', 'ddf', 'ddp', 'dmc', 'is', 'om',
    # 'or', 'pa', 'pr', 'ps', 'qas', 'rci', 're', 'svp']
    # e.g: 'ass-rape': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    def read_lexicon(self, level):
        lexicon = dict()
        self.categories = []
        lexicon_filename = "hurtlex_{0}.tsv".format(self.language)
        lexicon_path = os.path.join(os.path.dirname(__file__), '../..', 'originalHurtlexLexica',
                                    self.language, HL_VERSION, lexicon_filename)

        with open(lexicon_path) as originLexica:
            # read categories
            readerOrigin = csv.DictReader(originLexica, delimiter="\t")
            for row in readerOrigin:
                self.categories.append(row["category"])
            self.categories = sorted(list(set(self.categories)))
            originLexica.seek(0)
            readerOrigin = csv.DictReader(originLexica, delimiter="\t")
            for row in readerOrigin:
                if row["level"] != "conservative" and row["level"] != level:
                    continue
                if not row["lemma"] in lexicon:
                    lexicon[row["lemma"]] = [0 for self.category in self.categories]
                lexicon[row["lemma"]][self.categories.index(row["category"])] += 1

        if self.improved_hurtlex == IMPROVED_HURTLEX:
            # use the improved hurtlex file
            lexicon_filename_improved = "hurtlex_{0}_improved.tsv".format(self.language)
            lexicon_path_improved = os.path.join(os.path.dirname(__file__), '../..', 'improvedHurtlexLexica',
                                                 self.language, lexicon_filename_improved)
            if os.path.exists(lexicon_path_improved):
                with open(lexicon_path_improved) as improvedLexica:
                    readerImproved = csv.DictReader(improvedLexica, delimiter="\t")
                    for row in readerImproved:
                        if row["level"] != "conservative" and row["level"] != level:
                            continue
                        if not row["lemma"] in lexicon:
                            lexicon[row["lemma"]] = [0 for self.category in self.categories]
                        lexicon[row["lemma"]][self.categories.index(row["category"])] += 1
        return lexicon

    # generate categories vector for the given text
    def process(self, text):
        # prefill feature_vector with zeros
        feature_vector = [0 for category in self.categories]
        for lemma in self.lemmatize(text):
            # if given lemma exist in lexicon then return correspondent category vector otherwise vector with zeros
            lemma_vector = self.lexicon.get(lemma, [0 for category in self.categories])
            # iterate simultaneously to count the number of times each categories appears in the text
            feature_vector = [i + j for i, j in zip(lemma_vector, feature_vector)]
        return feature_vector
