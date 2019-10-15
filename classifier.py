import os
import numpy as np
import math  
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
from sklearn import preprocessing
import matplotlib.pyplot as plt

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

            features2 = np.asarray(np.loadtxt('features_' + str(self.window) + '_' + str(self.window_overlap) + '_' + self.selector + '.txt'))
            features.extend(features2)
            labels2 = np.asarray(np.loadtxt('labels_' + str(self.window) + '_' + str(self.window_overlap) + '.txt'))
            labels.extend(labels2)

        i = 0
        # all_delete = []
        for i in range(len(labels)):
            if (int(labels[i]) == 3):
                # all_delete.append(i)
                labels[i] = 1

        # all_delete.sort(reverse = True)

        # for j in all_delete:
        #     # print(j)
        #     # print(labels)
        #     del labels[j]
        #     del features[j]
        labels = np.asarray(labels)
        features = np.asarray(features)

        return labels, features

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

    def save_results(self, base_path, subject, signal, predicts_rf2, predicts_clf2, predicts_nbrs2, new_results, testing_labels, times):
        all_results = []
        matrix_rf = confusion_matrix(testing_labels, predicts_rf2)
        sensibility_rf = self.calc_sensibility(matrix_rf)
        specificity_rf = self.calc_specificity(matrix_rf)
        matrix_clf = confusion_matrix(testing_labels, predicts_clf2)
        sensibility_clf = self.calc_sensibility(matrix_clf)
        specificity_clf = self.calc_specificity(matrix_clf)
        matrix_nbrs = confusion_matrix(testing_labels, predicts_nbrs2)
        sensibility_nbrs = self.calc_sensibility(matrix_nbrs)
        specificity_nbrs = self.calc_specificity(matrix_nbrs)
        matrix_shoot = confusion_matrix(testing_labels, new_results)
        sensibility_shoot = self.calc_sensibility(matrix_shoot)
        specificity_shoot = self.calc_specificity(matrix_shoot)
        all_results.append([
            sensibility_rf, specificity_rf, sensibility_clf, specificity_clf, sensibility_nbrs, specificity_nbrs, sensibility_shoot, specificity_shoot
        ])
        
        np.savetxt(base_path + 'S' + str(subject) + '/data/chest_' + signal + '/results' + str(times) + '.txt', all_results, fmt="%s")
        matrixs = [
            self.just_numbers(matrix_rf),
            self.just_numbers(matrix_clf),
            self.just_numbers(matrix_nbrs),
            self.just_numbers(matrix_shoot)
        ]
        np.savetxt(base_path + 'S' + str(subject) + '/data/chest_' + signal + '/matrix' + str(times) + '.txt', matrixs, fmt="%s")

    def just_numbers(self, matrix):
        new_matrix = ''
        for m in matrix:
            for n in m:
                new_matrix += ' ' + str(n)
        
        return new_matrix

    def execute(self, base_path, signal, subjects, window, window_overlap, selector, times):
        self.subjects = subjects
        self.window = window
        self.window_overlap = window_overlap
        self.selector = selector

        predicts_rf = []
        predicts_clf = []
        predicts_nbrs = []
        testings = []

        for sub in subjects:
            print('sujeito = ', sub)
            training_labels, training_features = self.get_data(base_path, signal, sub, 'training')
            testing_labels, testing_features = self.get_data(base_path, signal, sub, 'testing')

            testings.extend(testing_labels)

            rf = RandomForestClassifier(n_estimators=100, max_depth=5, oob_score=True)
            rf = rf.fit(training_features, training_labels)
            predictions = rf.predict(testing_features)
            predicts_rf.extend(predictions)
            predicts_rf2 = predictions
            print('RF')
            # self.print_results(predictions, testing_labels)

            clf = svm.SVC(gamma='scale', C=4)
            clf = clf.fit(training_features, training_labels)
            predictions = clf.predict(testing_features)
            predicts_clf.extend(predictions)
            predicts_clf2 = predictions
            print('SVM')
            # self.print_results(predictions, testing_labels)
        
            predictions = self.knn_classifier(training_features, training_labels, testing_features)
            predicts_nbrs.extend(predictions)
            predicts_nbrs2 = predictions
            print('KNN')
            # self.print_results(predictions, testing_labels)
            
            shoot2 = Shooter()
            new_results = shoot2.choose(predicts_rf2, predicts_nbrs2, predicts_clf2)
            print('Shooter')
            # self.print_results(new_results, testing_labels)
            self.save_results(base_path, sub, signal, predicts_rf2, predicts_clf2, predicts_nbrs2, new_results, testing_labels, times)

            print('----------------------------------------------------------------')

        shoot = Shooter()
        new_results = shoot.choose(predicts_rf, predicts_nbrs, predicts_clf)

        print('RF')
        # self.print_results(predicts_rf, testings)
        print('SVM')
        # self.print_results(predicts_clf, testings)
        print('KNN')
        # self.print_results(predicts_nbrs, testings)
        print('Shooter')
        # self.print_results(new_results, testings)

    def calc_med(self, base_path, signal, subjects, times):
        for sub in subjects:
            print('sujeito = ', sub)
            all_data = []
            i = 0
            for i in range(times):
                all_data.append(np.asarray(np.loadtxt(base_path + 'S' + str(sub) + '/data/chest_' + signal + '/results' + str(i) + '.txt')))

            meds = []
            k = 0
            for k in range(6):
                j = 0
                med = 0
                for j in range(len(all_data)):
                    if (not math.isnan(all_data[j][k])):
                        med += all_data[j][k]

                meds.extend([(med) / len(all_data)])

            np.savetxt(base_path + 'S' + str(sub) + '/data/chest_' + signal + '/all_results_' + str(times) + '.txt', meds, fmt="%f")

from classifier import Classifier

if __name__ == '__main__':
    subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    # all_subjects = [2, 3]
    signal = 'ecg'
    base_path = '/Volumes/My Passport/TCC/WESAD/'
    window = 20
    window_overlap = True
    selector = 'pca'
    times = 1
    classification = Classifier()
    i = 0
    for i in range(times):
        classification.execute(base_path, signal, subjects, window, window_overlap, selector, i)

    classification.calc_med(base_path, signal, subjects, times)
