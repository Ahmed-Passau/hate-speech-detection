import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn import preprocessing

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

TFTIDF = 'tfidf'
HURTLEX = 'hurtlex'
IMPROVED_HURTLEX = 'improvedhurtlexlexica'
IMPROVE_HURTLEX = 'improveHurtlex'
HURTLEX_AND_TFTIDF = 'hurtlexAndTfidf'
XGBOOST = 'XGBoost'
SVM = 'SVM'
SVM_OPTIMIZER = 'SVMOptimiser'
FRENCH = 'FR'
ENGLISH = 'EN'
POS = 'pos'
CATEGORY = 'category'
STEREOTYPE = 'stereotype'
LEVEL = 'level'


def get_language_name(language):
    if language == ENGLISH:
        return 'english'
    else:
        return 'french'


def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


def normalizer(tweet, language):
    stop_words = set(stopwords.words(get_language_name(language)))
    wordnet_lemmatizer = WordNetLemmatizer()
    only_letters = clean_tweet(tweet)
    tokens = nltk.word_tokenize(only_letters)
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    if len(filtered_result) >= 2:
        lemmas = ' '.join(wordnet_lemmatizer.lemmatize(t) for t in filtered_result)
    else:
        lemmas = ''
    return lemmas


def encodeLable(labelName):
    encode = preprocessing.LabelEncoder()
    if labelName == POS:
        encode.fit(['a', 'av', 'n', 'v'])
    elif labelName == CATEGORY:
        encode.fit(
            ['an', 'asf', 'asm', 'cds', 'ddf', 'ddp', 'dmc', 'is', 'om', 'or', 'pa', 'pr', 'ps', 'qas', 'rci', 're',
             'svp'])
    elif labelName == STEREOTYPE:
        encode.fit(['yes', 'no'])
    elif labelName == LEVEL:
        encode.fit(['inclusive', 'conservative'])
    else:
        print('wrong params to encode')
        exit(1)
    return encode
