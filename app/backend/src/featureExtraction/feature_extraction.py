import csv
import os
from app.backend.src.featureExtraction.featurizer import HurtLexFeaturizer as hurtlexFeaturizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import numpy as np
from app.backend.helper import *


class featureExtraction:
    def __init__(self, language):
        self.language = language

    def extractUsingHurtlex(self, isImproved):
        lexicon_src_filename = 'train_data_' + self.language + '.tsv'
        lexicon_dest_filename = 'hurtlex_features_' + self.language + '.tsv'
        src_file_path = os.path.join(os.path.dirname(__file__), '../..', 'dataSet', self.language, lexicon_src_filename)
        dest_file_path = os.path.join(os.path.dirname(__file__), '../..', 'dataSet', self.language,
                                      lexicon_dest_filename)
        if isImproved:
            hurtlex = hurtlexFeaturizer(self.language, IMPROVED_HURTLEX)
        else:
            hurtlex = hurtlexFeaturizer(self.language, HURTLEX)
        with open(dest_file_path, 'w') as dest_file:
            with open(src_file_path, 'r') as src_file:
                tsvreader = csv.reader(src_file, delimiter="\t")
                id = 1
                for line in tsvreader:
                    hurtlexTab = hurtlex.process(line[1])
                    all_zeros = np.any(hurtlexTab)
                    if all_zeros:
                        dest_file.write(str(id) + '\t')
                        for i in hurtlexTab:
                            dest_file.write(str(i) + '\t')
                        dest_file.write(str(line[2]) + '\n')
                        id += 1

    def extractUsingTfidfVectorizer(self):
        corpus = []
        if self.language == FRENCH:
            stop_words = set(stopwords.words(get_language_name(self.language)))
        elif self.language == ENGLISH:
            stop_words = get_language_name(self.language)
        lexicon_src_filename = 'train_data_' + self.language + '.tsv'
        lexicon_dest_filename = 'tfidf_features_' + self.language + '.tsv'
        src_file_path = os.path.join(os.path.dirname(__file__), '../..', 'dataSet', self.language, lexicon_src_filename)
        dest_file_path = os.path.join(os.path.dirname(__file__), '../..', 'dataSet', self.language,
                                      lexicon_dest_filename)
        with open(src_file_path, 'r') as src_file:
            tsvreader = csv.reader(src_file, delimiter="\t")
            for line in tsvreader:
                corpus.append(str(line[1]))

        vectorizer = TfidfVectorizer(
            min_df=5,
            max_df=0.95,
            max_features=8000,
            stop_words=stop_words
        )
        X = vectorizer.fit_transform(corpus)
        pca = PCA(n_components=100)
        pca = pca.fit_transform(X.todense())

        with open(dest_file_path, 'w') as dest_file:
            with open(src_file_path, 'r') as src_file:
                tsvreader = csv.reader(src_file, delimiter="\t")
                for i, j in zip(tsvreader, pca):
                    dest_file.write(str(i[0]) + '\t')
                    for k in j:
                        dest_file.write(str(k) + '\t')
                    dest_file.write(str(i[2]) + '\n')

    def extractUsingHurtlexAndTfidfVectorizer(self):
        hurtlex_src_filename = 'hurtlex_features_' + self.language + '.tsv'
        tfidf_src_filename = 'tfidf_features_' + self.language + '.tsv'
        lexicon_dest_filename = 'hurtlexAndTfidf_features_' + self.language + '.tsv'
        src_file_hurtlex_path = os.path.join(os.path.dirname(__file__), '../..', 'dataSet', self.language,
                                             hurtlex_src_filename)
        src_file_tfidf_path = os.path.join(os.path.dirname(__file__), '../..', 'dataSet', self.language,
                                           tfidf_src_filename)
        dest_file_path = os.path.join(os.path.dirname(__file__), '../..', 'dataSet', self.language,
                                      lexicon_dest_filename)
        hurtlexVector = []
        tfidfVector = []
        if os.path.exists(src_file_hurtlex_path) and os.path.exists(src_file_tfidf_path):
            with open(dest_file_path, 'w') as dest_file:
                with open(src_file_hurtlex_path, 'r') as src_file_hurtlex:
                    tsvHurtlexReader = csv.reader(src_file_hurtlex, delimiter="\t")

                    for line in tsvHurtlexReader:
                        hurtlexVector.append(line[0:-1])

                with open(src_file_tfidf_path, 'r') as src_file_tfidf:
                    tsvTfidfReader = csv.reader(src_file_tfidf, delimiter="\t")
                    for line in tsvTfidfReader:
                        tfidfVector.append(line[1:])
                for i in range(len(hurtlexVector)):
                    to_add = ''
                    for x in hurtlexVector[i]:
                        to_add += str(x) + '\t'
                    for x in tfidfVector[i]:
                        to_add += str(x) + '\t'
                    to_add = to_add[:-1]
                    dest_file.write(to_add + '\n')
        else:
            self.extractUsingHurtlex(False)
            self.extractUsingTfidfVectorizer()
            with open(dest_file_path, 'w') as dest_file:
                with open(src_file_hurtlex_path, 'r') as src_file_hurtlex:
                    tsvHurtlexReader = csv.reader(src_file_hurtlex, delimiter="\t")

                    for line in tsvHurtlexReader:
                        hurtlexVector.append(line[0:-1])

                with open(src_file_tfidf_path, 'r') as src_file_tfidf:
                    tsvTfidfReader = csv.reader(src_file_tfidf, delimiter="\t")
                    for line in tsvTfidfReader:
                        tfidfVector.append(line[1:])
                for i in range(len(hurtlexVector)):
                    to_add = ''
                    for x in hurtlexVector[i]:
                        to_add += str(x) + '\t'
                    for x in tfidfVector[i]:
                        to_add += str(x) + '\t'
                    to_add = to_add[:-1]
                    dest_file.write(to_add + '\n')


if __name__ == "__main__":
    featureExtraction = featureExtraction(FRENCH)
    featureExtraction.extractUsingHurtlexAndTfidfVectorizer()
