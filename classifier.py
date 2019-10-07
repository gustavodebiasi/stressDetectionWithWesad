import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest
from shooter import Shooter

class Classifier(object):
    subjects = []
    window = 0
    window_overlap = False
    selector = ''

    def get_data(self, base_path, signal, subject, data_type):
        features = []
        labels = []
        if (data_type == 'training'):
            subjects = self.subjects[:]
            subjects.remove(subject)
        else:
            subjects = [subject]
        
        for i in subjects:
            subject = 'S' + str(i)
            path = base_path + subject + '/data/chest_' + signal + '/'
            os.chdir(path)

            labels2 = np.asarray(np.loadtxt('labels_' + str(self.window) + '_' + str(self.window_overlap) + '.txt'))
            labels.extend(labels2)
            features2 = np.asarray(np.loadtxt('features_' + str(self.window) + '_' + str(self.window_overlap) + '_' + self.selector + '.txt'))
            features.extend(features2)

        features = np.asarray(features)
        i = 0
        for i in range(len(labels)):
            if (int(labels[i]) == 3):
                labels[i] = 1
        labels = np.asarray(labels)

        return labels, features

    def random_forest_classifier(self, training_data, training_labels, testing_data):
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, oob_score=True)
        rf = rf.fit(training_data, training_labels)
        rf.class_weight()
        predictions = rf.predict(testing_data)

        return predictions

    def svm_classifier(self, training_data, training_labels, testing_data):
        clf = svm.SVC(gamma='scale')
        clf = clf.fit(training_data, training_labels)
        predictions = clf.predict(testing_data)

        return predictions

    def knn_classifier(self, training_data, training_labels, testing_data):
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='kd_tree')
        nbrs = nbrs.fit(training_data, training_labels)
        distances, indices = nbrs.kneighbors(testing_data)
        predictions = []
        for indice in indices:
            i = 0
            count_stress = 0
            for i in range(len(indice)):
                if (training_labels[indice[i]] == 2.0):
                    count_stress += 1
            if (count_stress >= 3):
                predictions.append(2.)
            else:
                predictions.append(1.)

        return predictions

    def calc_sensibility(self, confusion):
        return (confusion[0][0] / (confusion[0][0] + confusion[1][0]))

    def calc_specificity(self, confusion):
        return (confusion[1][1] / (confusion[1][1] + confusion[0][1]))

    def print_results(self, predictions, testing_labels):
        # print('Predictions = ', predictions)
        print('acuracy= ', accuracy_score(testing_labels, predictions))
        matrix = confusion_matrix(testing_labels, predictions)
        print('matrix = ', matrix)
        print('sensibility= ', self.calc_sensibility(matrix))
        print('specificity= ', self.calc_specificity(matrix))
        print('--------------------------------')

    def execute(self, base_path, signal, subjects, window, window_overlap, selector):
        self.subjects = subjects
        self.window = window
        self.window_overlap = window_overlap
        self.selector = selector
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, oob_score=True)
        clf = svm.SVC(gamma='auto')

        predictsRF = []
        predictsCLF = []
        predictsNBRS = []
        testings = []
        
        for sub in subjects:
            print('sujeito = ', sub)
            training_labels, training_features = self.get_data(base_path, signal, sub, 'training')
            testing_labels, testing_features = self.get_data(base_path, signal, sub, 'testing')

            testings.extend(testing_labels)

            # print('labels true = ', testing_labels)
            # dt = dt.fit(training_features, training_labels)
            # predictions = dt.predict(testing_features)
            # print('--------------------------------')
            # print('DECISION TREE ', sub)
            # self.print_results(predictions, testing_labels)

            # predictions = random_forest_classifier(training_features, training_labels, testing_features)
            # print(training_features.shape)
            # print(training_features)
            # print(training_labels)

            # testing_features.reshape(-1, 1)
            # training_features.reshape(-1, 1)
            
            rf = rf.fit(training_features, training_labels)
            predictions = rf.predict(testing_features)
            predictsRF.extend(predictions)
            # print('--------------------------------')
            # print('RANDOM FOREST ', sub)
            # self.print_results(predictions, testing_labels)

            # predictions = svm_classifier(training_features, training_labels, testing_features)
            clf = clf.fit(training_features, training_labels)
            predictions = clf.predict(testing_features)
            predictsCLF.extend(predictions)
            # print('--------------------------------')
            # print('SVM ', sub)
            # self.print_results(predictions, testing_labels)
            # KNN -
            # print('--------------------------------')
            # print('KNN ', sub)
            predictions = self.knn_classifier(training_features, training_labels, testing_features)
            predictsNBRS.extend(predictions)
            # self.print_results(predictions, testing_labels)

        shoot = Shooter()
        new_results = shoot.choose(predictsRF, predictsNBRS, predictsCLF)

        print('RF')
        self.print_results(predictsRF, testings)
        print('SVM')
        self.print_results(predictsCLF, testings)
        print('KNN')
        self.print_results(predictsNBRS, testings)
        print('Shooter')
        self.print_results(new_results, testings)

from classifier import Classifier

if __name__ == '__main__':
    subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    # all_subjects = [2, 3]
    signal = 'ecg'
    base_path = '/Volumes/My Passport/TCC/WESAD/'
    window = 20
    window_overlap = True
    selector = 'pca'
    classification = Classifier()
    classification.execute(base_path, signal, subjects, window, window_overlap, selector)
    