from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pickle
from app.backend.helper import *
import os


class classifier:
    def __init__(self, language, x, y):
        self.language = language
        self.x = x
        self.y = y
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.accuracy = ''
        self.model_path = ''
        self.model_info_message = ''

    def getModelExtraOutput(self):
        extraOutput = [self.model_info_message, '', self.accuracy, '', self.model_path]
        return extraOutput

    def splitDataToTrainAndTest(self):
        seed = 7
        test_size = 0.2
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=test_size,
                                                                                random_state=seed)

    def classifierModel(self, methodName, classifierName, labelName=None):
        self.splitDataToTrainAndTest()
        if classifierName == SVM:
            model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        elif classifierName == SVM_OPTIMIZER:
            return self.classifierOptimizerSVM(methodName, classifierName)
        else:
            model = XGBClassifier()

        model.fit(self.x_train, self.y_train)
        # plot feature importance
        # plot_importance(model)
        # pyplot.show()
        # make predictions for test data
        y_pred = model.predict(self.x_test)
        predictions = [round(value) for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(self.y_test, predictions)
        self.accuracy = 'Accuracy: %.2f%%' % (accuracy * 100.0)
        print("Accuracy: %.2f%%" % (accuracy * 100.0) + '\n')
        # save model to file
        dest_dir_path = os.path.join(os.path.dirname(__file__), '../../..', 'GeneratedModels', self.language,
                                     classifierName)
        if not os.path.exists(dest_dir_path):
            os.makedirs(dest_dir_path)
        if labelName == None:
            dest_file_path = dest_dir_path + '/' + classifierName + '_' + methodName + '_' + self.language + \
                             '.pickle.dat'
            self.model_path = 'Generated model:  ' + classifierName + '_' + methodName + '_' + self.language + \
                              '.pickle.dat'
        else:
            dest_file_path = dest_dir_path + '/' + classifierName + '_' + methodName + '_' + labelName + '_' + \
                             self.language + '.pickle.dat'
            self.model_path = 'Generated model:  ' + classifierName + '_' + methodName + '_' + labelName + '_' + \
                              self.language + '.pickle.dat'
        self.model_info_message = str(
            'Extract features using ' + methodName + ' method, from ' + get_language_name(self.language) \
            + ' dataset and pass them to ' + classifierName + ' classifier to generate trained model ')
        pickle.dump(model, open(dest_file_path, "wb"))

    def classifierOptimizerSVM(self, methodName, classifierName):
        kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
        # Create the pipeline SVM
        pipeline_svm = make_pipeline(StandardScaler(),
                                     SVC(probability=True, kernel="linear", class_weight="balanced", max_iter=-1))

        # Grid search : find the best parameters to have the highest f1 score
        grid_svm = GridSearchCV(pipeline_svm,
                                param_grid={'svc__C': [0.01, 0.1, 10, 100, 1000]},
                                cv=kfolds,
                                scoring="f1",
                                verbose=1,
                                n_jobs=-1)
        grid_svm.fit(self.x_train, self.y_train)
        # make predictions for test data
        y_pred = grid_svm.predict(self.x_test)
        predictions = [round(value) for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(self.y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0) + '\n')

        # save model to file
        dest_dir_path = os.path.join(os.path.dirname(__file__), '../../..', 'GeneratedModels', self.language,
                                     classifierName)
        if not os.path.exists(dest_dir_path):
            os.makedirs(dest_dir_path)
        dest_file_path = dest_dir_path + '/' + classifierName + '_' + methodName + '_' + self.language + '.pickle.dat'
        if not os.path.exists(dest_file_path):
            pickle.dump(grid_svm, open(dest_file_path, "wb"))
