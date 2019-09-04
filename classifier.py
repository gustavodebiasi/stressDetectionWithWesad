import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

def get_training_data(oque = 0):
    # features = []
    features = []
    # subjects = [3]
    labels = []
    if (oque == 0):
        subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]
    if (oque == 1):
        subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17]
    if (oque == 2):
        subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17]
    if (oque == 3):
        subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17]
    if (oque == 4):
        subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17]
    if (oque == 5):
        subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17]
    if (oque == 6):
        subjects = [2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17]
    if (oque == 7):
        subjects = [2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 17]
    if (oque == 8):
        subjects = [2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17]
    if (oque == 9):
        subjects = [2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    if (oque == 10):
        subjects = [2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    if (oque == 11):
        subjects = [2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    if (oque == 12):
        subjects = [2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    if (oque == 13):
        subjects = [2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    if (oque == 14):
        subjects = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    for i in subjects:
        subject = 'S' + str(i)
        path = '/Volumes/My Passport/TCC/WESAD/' + subject + '/data/raw/chest_ecg/'
        os.chdir(path)

        labels2 = np.asarray(np.loadtxt('labels_false.txt'))
        # for label in labels2:
        labels.extend(labels2)
        # features = [
        #     np.loadtxt('kurtosis_false.txt'),
        #     np.loadtxt('max_false.txt'),
        #     np.loadtxt('mean_false.txt'),
        #     np.loadtxt('median_false.txt'),
        #     np.loadtxt('min_false.txt'),
        #     np.loadtxt('std_false.txt'),
        #     np.loadtxt('variance_false.txt')
        # ]
        kurtosis = np.asarray(np.loadtxt('kurtosis_false.txt'))
        max_var = np.asarray(np.loadtxt('max_false.txt'))
        mean_var = np.asarray(np.loadtxt('mean_false.txt'))
        median_var = np.asarray(np.loadtxt('median_false.txt'))
        min_var = np.asarray(np.loadtxt('min_false.txt'))
        std_var = np.asarray(np.loadtxt('std_false.txt'))
        variance_var = np.asarray(np.loadtxt('variance_false.txt'))
        k=0
        for k in range(len(kurtosis)):
            features.append([
                kurtosis[k],
                max_var[k],
                mean_var[k],
                median_var[k],
                min_var[k],
                std_var[k],
                variance_var[k]
            ])


        # features = np.asarray([
        #     np.loadtxt('kurtosis_false.txt'),
            # np.loadtxt('max_false.txt'),
            # np.loadtxt('mean_false.txt'),
            # np.loadtxt('median_false.txt'),
            # np.loadtxt('min_false.txt'),
            # np.loadtxt('std_false.txt'),
            # np.loadtxt('variance_false.txt')
        # ])
        # features[0].append(np.loadtxt('max_false.txt'))
        # features[0].append(np.loadtxt('mean_false.txt'))
        # features[0].append(np.loadtxt('median_false.txt'))
        # features[0].append(np.loadtxt('min_false.txt'))
        # features[0].append(np.loadtxt('std_false.txt'))
        # features[0].append(np.loadtxt('variance_false.txt'))

    features = np.asarray(features)
    labels = np.asarray(labels)

    return labels, features

def get_testing_data(oque = 0):
    # features = []
    # subjects = [2]
    labels = []
    # subs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    if (oque == 0):
        subjects = [17]
    if (oque == 1):
        subjects = [16]
    if (oque == 2):
        subjects = [15]
    if (oque == 3):
        subjects = [14]
    if (oque == 4):
        subjects = [13]
    if (oque == 5):
        subjects = [11]
    if (oque == 6):
        subjects = [10]
    if (oque == 7):
        subjects = [9]
    if (oque == 8):
        subjects = [8]
    if (oque == 9):
        subjects = [7]
    if (oque == 10):
        subjects = [6]
    if (oque == 11):
        subjects = [5]
    if (oque == 12):
        subjects = [4]
    if (oque == 13):
        subjects = [3]
    if (oque == 14):
        subjects = [2]
    for i in subjects:
        subject = 'S' + str(i)
        path = '/Volumes/My Passport/TCC/WESAD/' + subject + '/data/raw/chest_ecg/'
        os.chdir(path)

        labels2 = np.asarray(np.loadtxt('labels_false.txt'))
        # for label in labels2:
        labels.extend(labels2)
        # features = [
        #     np.loadtxt('kurtosis_false.txt'),
        #     np.loadtxt('max_false.txt'),
        #     np.loadtxt('mean_false.txt'),
        #     np.loadtxt('median_false.txt'),
        #     np.loadtxt('min_false.txt'),
        #     np.loadtxt('std_false.txt'),
        #     np.loadtxt('variance_false.txt')
        # ]
        # features = np.asarray(np.loadtxt('kurtosis_false.txt'))
        kurtosis = np.asarray(np.loadtxt('kurtosis_false.txt'))
        max_var = np.asarray(np.loadtxt('max_false.txt'))
        mean_var = np.asarray(np.loadtxt('mean_false.txt'))
        median_var = np.asarray(np.loadtxt('median_false.txt'))
        min_var = np.asarray(np.loadtxt('min_false.txt'))
        std_var = np.asarray(np.loadtxt('std_false.txt'))
        variance_var = np.asarray(np.loadtxt('variance_false.txt'))
        features = []
        k=0
        for k in range(len(kurtosis)):
            features.append([
                kurtosis[k],
                max_var[k],
                mean_var[k],
                median_var[k],
                min_var[k],
                std_var[k],
                variance_var[k]
            ])

        # features = np.asarray([
        #     np.loadtxt('kurtosis_false.txt'),
            # np.loadtxt('max_false.txt'),
            # np.loadtxt('mean_false.txt'),
            # np.loadtxt('median_false.txt'),
            # np.loadtxt('min_false.txt'),
            # np.loadtxt('std_false.txt'),
            # np.loadtxt('variance_false.txt')
        # ])
        # features[0].append(np.loadtxt('max_false.txt'))
        # features[0].append(np.loadtxt('mean_false.txt'))
        # features[0].append(np.loadtxt('median_false.txt'))
        # features[0].append(np.loadtxt('min_false.txt'))
        # features[0].append(np.loadtxt('std_false.txt'))
        # features[0].append(np.loadtxt('variance_false.txt'))

    features = np.asarray(features)
    labels = np.asarray(labels)

    return labels, features

def decision_tree_classifier(training_data, training_labels, testing_data):
    dt = DecisionTreeClassifier(random_state=0, max_depth=2)
    dt = dt.fit(training_data, training_labels)
    predictions = dt.predict(testing_data)

    return predictions

def random_forest_classifier(training_data, training_labels, testing_data):
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, oob_score=True)
    rf = rf.fit(training_data, training_labels)
    predictions = rf.predict(testing_data)

    return predictions

def svm_classifier(training_data, training_labels, testing_data):
    clf = svm.SVC(gamma='scale')
    clf = clf.fit(training_data, training_labels)
    predictions = clf.predict(testing_data)

    return predictions

def execute():
    # training_labels, training_features = get_training_data()
    # testing_labels, testing_features = get_testing_data()
    
    # predictions = decision_tree_classifier(training_features, training_labels, testing_features)
    dt = DecisionTreeClassifier(random_state=0, max_depth=2)
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, oob_score=True)
    clf = svm.SVC(gamma='scale')
    i = 0
    for i in range(15):
        training_labels, training_features = get_training_data(i)
        testing_labels, testing_features = get_testing_data(i)
        dt = dt.fit(training_features, training_labels)
        predictions = dt.predict(testing_features)
        print('DECISION TREE ', i)
        print('Predictions:')
        print(predictions)
        print('Correct:')
        print(testing_labels)
        print('ACURACY')
        print('Accuracy is ', accuracy_score(testing_labels, predictions) * 100)
        print('--------------------------------')

        # predictions = random_forest_classifier(training_features, training_labels, testing_features)
        
        rf = rf.fit(training_features, training_labels)
        predictions = rf.predict(testing_features)
        print('RANDOM FOREST ', i)
        print('Predictions:')
        print(predictions)
        print('Correct:')
        print(testing_labels)
        print('ACURACY')
        print('Accuracy is ', accuracy_score(testing_labels,predictions) * 100)
        print('--------------------------------')

        # predictions = svm_classifier(training_features, training_labels, testing_features)
        clf = clf.fit(training_features, training_labels)
        predictions = clf.predict(testing_features)
        print('SVM ', i)
        print('Predictions:')
        print(predictions)
        print('Correct:')
        print(testing_labels)
        print('ACURACY')
        print('Accuracy is ', accuracy_score(testing_labels,predictions) * 100)
        print('--------------------------------')

if __name__ == '__main__':
    execute()
    