import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

def get_training_data():
    # features = []
    features = []
    # subjects = [3]
    labels = []
    subjects = [3, 4, 5, 6, 7, 8, 9, 10]
    for i in subjects:
        subject = 'S' + str(i)
        path = '/Volumes/My Passport/TCC/WESAD/' + subject + '/data/raw/chest_eda/'
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

def get_testing_data():
    # features = []
    subjects = [2]
    labels = []
    # subs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    for i in subjects:
        subject = 'S' + str(i)
        path = '/Volumes/My Passport/TCC/WESAD/' + subject + '/data/raw/chest_eda/'
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
    training_labels, training_features = get_training_data()
    testing_labels, testing_features = get_testing_data()
    
    predictions = decision_tree_classifier(training_features, training_labels, testing_features)
    print('DECISION TREE')
    print('Predictions:')
    print(predictions)
    print('Correct:')
    print(testing_labels)
    print('--------------------------------')

    predictions = random_forest_classifier(training_features, training_labels, testing_features)
    print('RANDOM FOREST')
    print('Predictions:')
    print(predictions)
    print('Correct:')
    print(testing_labels)
    print('--------------------------------')

    predictions = svm_classifier(training_features, training_labels, testing_features)
    print('SVM')
    print('Predictions:')
    print(predictions)
    print('Correct:')
    print(testing_labels)
    print('--------------------------------')

if __name__ == '__main__':
    execute()
    