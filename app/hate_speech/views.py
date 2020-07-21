import pickle
from django.conf import settings
import os
from django.contrib import messages
from django.shortcuts import render, redirect
import numpy as np
from .forms import InputFieldForm, InputFieldImproveForm, ModelForm, InputLanguageForm
from app.backend.main import MainClass as MainObject
from app.backend.src.featureExtraction.featurizer import HurtLexFeaturizer as HurtLexFeaturizerObject
from app.backend.src.improveHurtlex.improve_hurtlex import improveHurtlex as improveHurtlexObject
from app.backend.src.featureExtraction.feature_extraction import featureExtraction as featureExtractionObject
from app.backend.helper import *


def processTweet(methodName, classifierName, tweet, language):
    dest_file_path = classifierName + '_' + methodName + '_' + language + '.pickle.dat'
    model_path = os.path.join(settings.MODELS_DIR, language, classifierName, dest_file_path)
    if not os.path.exists(model_path):
        return 'Model does not exist'
    clf = pickle.load(open(model_path, "rb"))
    is_not_zeros = True
    if methodName == HURTLEX or methodName == IMPROVED_HURTLEX:
        hurtlexFeaturizer = HurtLexFeaturizerObject(language, methodName)
        vector = hurtlexFeaturizer.process(tweet)
        is_not_zeros = np.any(vector)
        prediction = clf.predict(np.array([vector]))
    else:
        # generate feature using tfidf for the entered tweet and run prediction

        # append new tweet to data train file
        lexicon_dest_filename = 'train_data_' + language + '.tsv'
        lexicon_dest_filename = os.path.join(settings.DATASET_DIR, language, lexicon_dest_filename)
        with open(lexicon_dest_filename, 'r+') as dest_file:
            for last_line in dest_file:
                pass
            words = last_line.split('\t')
            id = int(words[0]) + 1
            dest_file.write(str(id) + '\t' + tweet + '\t' + '1')

        # run feature extraction using tfidf on the new train data file and get the the feature vector of the entered
        # tweet
        featureExtraction = featureExtractionObject(language)
        featureExtraction.extractUsingTfidfVectorizer()
        feature_src_filename = 'tfidf_features_' + language + '.tsv'
        feature_src_filename = os.path.join(settings.DATASET_DIR, language, feature_src_filename)
        vector = []
        if methodName == HURTLEX_AND_TFTIDF:
            hurtlexFeaturizer = HurtLexFeaturizerObject(language, methodName)
            vector = hurtlexFeaturizer.process(tweet)
        with open(feature_src_filename, 'r') as src_file:
            for last_line in src_file:
                pass
            words = last_line.split('\t')
            words = words[1:101]
            for word in words:
                vector.append(float(word))
        # predict the tweet
        prediction = clf.predict(np.array([vector]))
        # add generated prediction to train data file
        with open(lexicon_dest_filename, 'rb+') as dest_file_data:
            dest_file_data.seek(-1, os.SEEK_END)
            dest_file_data.truncate()
        with open(lexicon_dest_filename, 'a') as dest_file_data:
            dest_file_data.write(str(int(prediction[0])) + '\n')

    # return prediction output
    if language == ENGLISH:
        if is_not_zeros:
            if prediction[0] == 1.0:
                formOutput = 'Hate Speech'
            else:
                formOutput = 'Non Hate Speech'
        else:
            formOutput = 'The corresponding Tweet doesn\'t exist in Hurtlex lexica: Please help improving hurtlex by ' \
                         'selecting the target words'
    else:
        if is_not_zeros:
            if prediction[0] == 1.0:
                formOutput = 'Discours de haine'
            else:
                formOutput = 'Discours non haineux'
        else:
            formOutput = 'Les mots entrés sur Tweet n\'existent pas dans Hurtlex lexica: utilisez plutôt TFIDF'
    return formOutput


def addHateWords(words, language):
    lexicon_dest_filename = 'new_hurtlex_' + language + '.tsv'
    dest_file_path = os.path.join(settings.IMPROVE_DIR, language, lexicon_dest_filename)
    try:
        if not os.path.exists(dest_file_path):
            with open(dest_file_path, 'w') as dest_file:
                words = words.split(",")
                for word in words:
                    if word != "":
                        dest_file.write(word + '\n')
        else:
            with open(dest_file_path, 'a') as dest_file:
                words = words.split(",")
                for word in words:
                    if word != "":
                        dest_file.write(word + '\n')
        return True
    except:
        return False


def buildModelClassifier(methodName, classifierName, language):
    process = MainObject(language, methodName, classifierName)
    process.preProcessDataFunc()
    process.featureExtractionFunc()
    return process.classifyFunc()


def improveHurtlex(request, language):
    # pre-process conan dataset
    if language == FRENCH:
        process = MainObject(language, IMPROVE_HURTLEX, XGBOOST)
        process.preProcessDataFunc()
    # feature extraction
    improveHurtlex = improveHurtlexObject(language, XGBOOST)
    improveHurtlex.extractFeatureFromOriginalHurtlexUsingTfidfVectorizer()
    improveHurtlex.extractFeatureFromNewHate()
    # generate model that predict the improved hurtlex
    process = MainObject(language, IMPROVE_HURTLEX, XGBOOST, POS)
    process.classifyFunc()
    process = MainObject(language, IMPROVE_HURTLEX, XGBOOST, CATEGORY)
    process.classifyFunc()
    process = MainObject(language, IMPROVE_HURTLEX, XGBOOST, STEREOTYPE)
    process.classifyFunc()
    #  predict the improved hurtlex: the data is saved under "improvedHurtlexLexica/FR/hurtlex_FR_improved.tsv"
    improveHurtlex.improveHurtlex()
    messages.success(request, 'Improve Hurtlex completed')
    return redirect('main-page', language=language)


def getMainPage(request, language):
    formInput = InputFieldForm({'language': language, 'method_name': HURTLEX, 'classifier_name': XGBOOST})
    formOutput = ''
    formImprove = InputFieldImproveForm()
    formLanguage = InputLanguageForm({'language': language})
    formModel = ModelForm({'language': language, 'method_name': HURTLEX, 'classifier_name': XGBOOST})
    logs = []
    if request.method == 'POST':
        if 'prediction' in request.POST:
            formInput = InputFieldForm(request.POST)
            formOutput = processTweet(request.POST['method_name'], request.POST['classifier_name'],
                                      request.POST['tweet'], request.POST['language'])
        if 'improve' in request.POST:
            onSaveWords = addHateWords(request.POST['improve'], request.POST['language'])
            if onSaveWords:
                messages.success(request, 'successfully add words: ' + request.POST['improve'])
                messages.success(request, 'Under File: ' + 'new_hurtlex_' + request.POST['language'] + '.tsv')
            else:
                messages.warning(request, 'Form submission error')
        if 'model' in request.POST:
            formModel = ModelForm(request.POST)
            logs = buildModelClassifier(request.POST['method_name'], request.POST['classifier_name'],
                                        request.POST['language'])
    if formOutput == 'Not Exist':
        messages.warning(request, request.POST['method_name'] + ' method is not implemented yet')
    elif formOutput == 'Model does not exist':
        messages.info(request, 'Model is not created yet')
        messages.info(request, 'Go to Build Model section to generate it')

    context = {'formInput': formInput, 'formOutput': formOutput, 'formImprove': formImprove,
               'formLanguage': formLanguage, 'formModel': formModel, 'logs': logs, 'language': language}
    return render(request, 'hate_speech/hateSpeechPage.html', context)
