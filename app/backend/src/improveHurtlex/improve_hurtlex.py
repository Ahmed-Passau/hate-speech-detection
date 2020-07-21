import csv
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from app.backend.helper import *
from app.backend.src.featureExtraction.featurizer import HL_VERSION
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
import numpy as np


class improveHurtlex:
    def __init__(self, language, classifierName):
        self.classifierName = classifierName
        self.language = language
        self.encodePos = encodeLable(POS)
        self.encodeCategory = encodeLable(CATEGORY)
        self.encodeStereotype = encodeLable(STEREOTYPE)
        self.encodeLevel = encodeLable(LEVEL)

    def extractFeatureFromOriginalHurtlexUsingTfidfVectorizer(self):
        lexicon_filename = "hurtlex_{0}.tsv".format(self.language)
        lexicon_feature_filename = "hurtlex_{0}_feature.tsv".format(self.language)
        src_file_path = os.path.join(os.path.dirname(__file__), '../..', 'originalHurtlexLexica', self.language,
                                     HL_VERSION, lexicon_filename)
        dest_dir_path = os.path.join(os.path.dirname(__file__), '../..', 'improvedHurtlexLexica', self.language)
        if not os.path.exists(dest_dir_path):
            os.makedirs(dest_dir_path)
        dest_file_path = dest_dir_path + '/' + lexicon_feature_filename

        if not os.path.exists(dest_file_path):
            corpus = []
            if self.language == FRENCH:
                stop_words = set(stopwords.words(get_language_name(self.language)))
            elif self.language == ENGLISH:
                stop_words = get_language_name(self.language)

            with open(src_file_path, 'r') as src_file:
                tsvreader = csv.reader(src_file, delimiter="\t")
                next(src_file)
                for line in tsvreader:
                    corpus.append(str(line[4]))

            vectorizer = TfidfVectorizer(
                min_df=5,
                max_df=0.95,
                max_features=8000,
                stop_words=stop_words
            )
            X = vectorizer.fit_transform(corpus)
            pca = PCA(n_components=17)
            pca = pca.fit_transform(X.todense())

            with open(dest_file_path, 'w') as dest_file:
                with open(src_file_path, 'r') as src_file:
                    tsvreader = csv.reader(src_file, delimiter="\t")
                    id = 1
                    next(src_file)
                    for i, j in zip(tsvreader, pca):
                        pos = list(self.encodePos.classes_).index(str(i[1]))
                        category = list(self.encodeCategory.classes_).index(str(i[2]))
                        stereotype = list(self.encodeStereotype.classes_).index(str(i[3]))
                        level = list(self.encodeLevel.classes_).index(str(i[5]))
                        dest_file.write(
                            str(id) + '\t' + str(pos) + '\t' + str(category) + '\t' + str(stereotype) + '\t')
                        for k in j:
                            dest_file.write(str(k) + '\t')
                        dest_file.write(str(level) + '\n')
                        id += 1

    def extractFeatureFromNewHate(self):
        corpus = []
        if self.language == FRENCH:
            stop_words = set(stopwords.words(get_language_name(self.language)))
        elif self.language == ENGLISH:
            stop_words = get_language_name(self.language)
        lexicon_src_filename = 'new_hurtlex_' + self.language + '.tsv'
        lexicon_dest_filename = 'new_hurtlex_' + self.language + '_feature.tsv'
        src_file_path = os.path.join(os.path.dirname(__file__), '../..', 'improvedHurtlexLexica', self.language,
                                     lexicon_src_filename)
        dest_file_path = os.path.join(os.path.dirname(__file__), '../..', 'improvedHurtlexLexica', self.language,
                                      lexicon_dest_filename)
        with open(src_file_path, 'r') as src_file:
            if self.language == FRENCH:
                tsvreader = csv.reader(src_file, delimiter="\t")
            else:
                tsvreader = csv.reader(src_file, delimiter=",")
            next(src_file)
            for line in tsvreader:
                corpus.append(str(line[0]))
        vectorizer = TfidfVectorizer(
            stop_words=stop_words
        )
        X = vectorizer.fit_transform(corpus)
        pca = PCA(n_components=17)
        pca = pca.fit_transform(X.todense())

        with open(dest_file_path, 'w') as dest_file:
            for i, j in zip(corpus, pca):
                to_add = str(i) + '\t'
                for vector in j:
                    to_add += str(vector) + '\t'
                to_add = to_add[:-1]
                dest_file.write(to_add + '\n')

    def improveHurtlex(self):
        lexicon_src_filename = 'new_hurtlex_' + self.language + '_feature.tsv'
        lexicon_dest_filename = 'hurtlex_' + self.language + '_improved.tsv'
        src_file_path = os.path.join(os.path.dirname(__file__), '../..', 'improvedHurtlexLexica', self.language,
                                     lexicon_src_filename)
        dest_file_path = os.path.join(os.path.dirname(__file__), '../..', 'improvedHurtlexLexica', self.language,
                                      lexicon_dest_filename)
        with open(dest_file_path, 'w') as dest_file:
            with open(src_file_path, 'r') as src_file:
                tsvreader = csv.reader(src_file, delimiter="\t")
                next(src_file)
                id = 1
                dest_file.write('id\tpos\tcategory\tstereotype\tlemma\tlevel\n')
                for line in tsvreader:
                    hateWordVector = []
                    for i in range(1, len(line)):
                        hateWordVector.append(float(line[i]))
                    predictionPos = self.predictNewHateLabel(POS, hateWordVector)
                    predictionCategory = self.predictNewHateLabel(CATEGORY, hateWordVector)
                    predictionStereotype = self.predictNewHateLabel(STEREOTYPE, hateWordVector)

                    dest_file.write(self.language + 'IMP' + str(id) + '\t'
                                    + str(self.encodePos.inverse_transform([predictionPos])[0]) + '\t'
                                    + str(self.encodeCategory.inverse_transform([predictionCategory])[0]) + '\t'
                                    + str(self.encodeStereotype.inverse_transform([predictionStereotype])[0]) + '\t'
                                    + line[0] + '\t'
                                    + 'conservative \n'
                                    )
                    id += 1

                    # print(line[0] + '\t'
                    #       + 'pos: '+ str(self.encodePos.inverse_transform([predictionPos])[0]) + '\t'
                    #       + 'category: ' + str(self.encodeCategory.inverse_transform([predictionCategory])[0]) + '\t'
                    #       + 'stereotype: ' + str(self.encodeStereotype.inverse_transform([predictionStereotype])[
                    #       0]) + '\n')

    def predictNewHateLabel(self, labelName, hateWord):
        model_filename = self.classifierName + '_improveHurtlex_' + labelName + '_' + self.language + '.pickle.dat'
        model_path = os.path.join(os.path.dirname(__file__), '../../..', 'GeneratedModels', self.language,
                                  self.classifierName, model_filename)
        # predict
        clf = pickle.load(open(model_path, "rb"))
        prediction = clf.predict(np.array([hateWord]))[0]
        return int(prediction)
