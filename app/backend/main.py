import os
from numpy import loadtxt
from app.backend.src.classifiers.classifier import classifier as cl
from app.backend.src.preprocessing.pre_processing import preProcessing as prep
from app.backend.src.featureExtraction.feature_extraction import featureExtraction as fex
from app.backend.helper import *


class MainClass:
    def __init__(self, language, methodName, classifierName, labelName=None):
        self.preProcessing = prep(language)
        self.featureExtraction = fex(language)
        self.language = language
        self.methodName = methodName
        self.classifierName = classifierName
        self.labelName = labelName
        self.y = []
        self.x = []

    def setLanguage(self, language):
        self.language = language

    def preProcessDataFunc(self):
        if self.methodName == IMPROVE_HURTLEX:
            print('pre_process data in ' + get_language_name(self.language) + ' for CONAN dataset')
            if self.language == FRENCH:
                self.preProcessing.preProcessCONANDataSet()
        else:
            print('pre_process data in ' + get_language_name(self.language))
            self.preProcessing.normalizeTweetsUsingNltk()

    def featureExtractionFunc(self):
        print('Extract features in ' + get_language_name(self.language) + ' using:  ' + self.methodName)
        if self.methodName == TFTIDF:
            self.featureExtraction.extractUsingTfidfVectorizer()
        elif self.methodName == HURTLEX_AND_TFTIDF:
            self.featureExtraction.extractUsingHurtlexAndTfidfVectorizer()
        elif self.methodName == IMPROVED_HURTLEX:
            self.featureExtraction.extractUsingHurtlex(True)
        else:
            self.featureExtraction.extractUsingHurtlex(False)

    def loadDataFunc(self):
        print('load ' + self.methodName + ' data in ' + get_language_name(self.language))
        if self.methodName == TFTIDF:
            src_filename = 'tfidf_features_' + self.language + '.tsv'
            src_file_path = os.path.join(os.path.dirname(__file__), 'dataSet', self.language, src_filename)
            dataset = loadtxt(src_file_path, delimiter="\t")
            self.x = dataset[:, 1:101]
            self.y = dataset[:, 101:102]
        elif self.methodName == HURTLEX_AND_TFTIDF:
            src_filename = 'hurtlexAndTfidf_features_' + self.language + '.tsv'
            src_file_path = os.path.join(os.path.dirname(__file__), 'dataSet', self.language, src_filename)
            dataset = loadtxt(src_file_path, delimiter="\t")
            self.x = dataset[:, 1:118]
            self.y = dataset[:, 118:119]
        elif self.methodName == IMPROVE_HURTLEX:
            src_filename = "hurtlex_{0}_feature.tsv".format(self.language)
            src_file_path = os.path.join(os.path.dirname(__file__), 'improvedHurtlexLexica', self.language,
                                         src_filename)
            dataset = loadtxt(src_file_path, delimiter="\t")
            self.x = dataset[:, 4:21]
            if self.labelName == POS:
                self.y = dataset[:, 1:2]
            elif self.labelName == CATEGORY:
                self.y = dataset[:, 2:3]
            elif self.labelName == STEREOTYPE:
                self.y = dataset[:, 3:4]
            else:
                print('wrong params')
                exit(1)
        else:
            src_filename = 'hurtlex_features_' + self.language + '.tsv'
            src_file_path = os.path.join(os.path.dirname(__file__), 'dataSet', self.language, src_filename)
            dataset = loadtxt(src_file_path, delimiter="\t")
            self.x = dataset[:, 1:18]
            self.y = dataset[:, 18:19]

    def classifyFunc(self):
        self.loadDataFunc()
        print('classify ' + self.methodName + ' data using:  ' + self.classifierName + ' classifier')
        if self.labelName is not None:
            print('label in use: ' + self.labelName)
        classifier = cl(self.language, self.x, self.y)
        classifier.classifierModel(self.methodName, self.classifierName, self.labelName)
        return classifier.getModelExtraOutput()


if __name__ == "__main__":
    pass
    ########################################## Predict french tweet hatespeech #########################################
    # process = MainClass(FRENCH, HURTLEX, XGBOOST)
    # process.preProcessDataFunc()
    # process.featureExtractionFunc()
    # process.classifyFunc()
    ######################################### Predict english tweet hatespeech #########################################
    # process = MainClass(ENGLISH, HURTLEX, SVM)
    # process.preProcessDataFunc()
    # process.featureExtractionFunc()
    # process.classifyFunc()
    ############################################## Improve hurtlex french ##############################################
    # # pre-process conan dataset
    # process = MainClass(FRENCH, IMPROVE_HURTLEX, XGBOOST)
    # process.preProcessDataFunc()
    # # feature extraction
    # improveHurtlex = imph(FRENCH, XGBOOST)
    # improveHurtlex.extractFeatureFromOriginalHurtlexUsingTfidfVectorizer()
    # improveHurtlex.extractFeatureFromNewHate()
    # # generate model that predict the improved hurtlex
    # process = MainClass(FRENCH, IMPROVE_HURTLEX, XGBOOST, POS)
    # process.classifyFunc()
    # process = MainClass(FRENCH, IMPROVE_HURTLEX, XGBOOST, CATEGORY)
    # process.classifyFunc()
    # process = MainClass(FRENCH, IMPROVE_HURTLEX, XGBOOST, STEREOTYPE)
    # process.classifyFunc()
    # #  predict the improved hurtlex: the data is saved under "improvedHurtlexLexica/FR/hurtlex_FR_improved.tsv"
    # improveHurtlex.improveHurtlex()
    ############################################## Improve hurtlex english #############################################
    # # feature extraction
    # improveHurtlex = imph(ENGLISH, XGBOOST)
    # improveHurtlex.extractFeatureFromOriginalHurtlexUsingTfidfVectorizer()
    # improveHurtlex.extractFeatureFromNewHate()
    # # generate model that predict the improved hurtlex
    # process = MainClass(ENGLISH, IMPROVE_HURTLEX, XGBOOST, POS)
    # process.classifyFunc()
    # process = MainClass(ENGLISH, IMPROVE_HURTLEX, XGBOOST, CATEGORY)
    # process.classifyFunc()
    # process = MainClass(ENGLISH, IMPROVE_HURTLEX, XGBOOST, STEREOTYPE)
    # process.classifyFunc()
    # #  predict the improved hurtlex: the data is saved under "improvedHurtlexLexica/EN/hurtlex_EN_improved.tsv"
    # improveHurtlex.improveHurtlex()
