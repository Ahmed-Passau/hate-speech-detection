from django import forms
from app.backend.helper import *

LANGUAGE = [
    (FRENCH, 'French'),
    (ENGLISH, 'English'),
]
METHOD_NAMES = [
    (HURTLEX, 'Hurtlex'),
    (IMPROVED_HURTLEX, 'Improved Hurtlex Lexica'),
    (TFTIDF, 'TfidfVectorizer'),
    (HURTLEX_AND_TFTIDF, 'Hurtlex and TfidfVectorizer'),
]

CLASSIFIER_NAMES = [
    (XGBOOST, 'XGBoost classifier'),
    (SVM, 'SVM classifier'),
]


class InputFieldForm(forms.Form):
    language = forms.CharField(label='Select language: ', widget=forms.Select(choices=LANGUAGE))
    method_name = forms.CharField(label='Select feature extraction method: ', widget=forms.Select(choices=METHOD_NAMES))
    classifier_name = forms.CharField(label='Select classifer: ', widget=forms.Select(choices=CLASSIFIER_NAMES))
    tweet = forms.CharField(label='Tweet', max_length=100)


class InputFieldImproveForm(forms.Form):
    improve = forms.CharField(label='Add hateful words to improve hurtlex lexica:', max_length=100, required=False,
                              initial=',')


class InputLanguageForm(forms.Form):
    language = forms.CharField(label='Select language: ', widget=forms.Select(choices=LANGUAGE))


class ModelForm(forms.Form):
    language = forms.CharField(label='Select language: ', widget=forms.Select(choices=LANGUAGE))
    method_name = forms.CharField(label='Select feature extraction method: ', widget=forms.Select(choices=METHOD_NAMES))
    classifier_name = forms.CharField(label='Select classifer: ', widget=forms.Select(choices=CLASSIFIER_NAMES))
