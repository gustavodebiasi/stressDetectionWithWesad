import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

def get_training_data():
    # features = []
    subjects = [3]
    # subs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    for i in subjects:
        subject = 'S' + str(i)
        path = '/Volumes/My Passport/TCC/WESAD/' + subject + '/data/raw/chest_eda/'
        os.chdir(path)

        labels = np.loadtxt('labels_false.txt')
        # features = [
        #     np.loadtxt('kurtosis_false.txt'),
        #     np.loadtxt('max_false.txt'),
        #     np.loadtxt('mean_false.txt'),
        #     np.loadtxt('median_false.txt'),
        #     np.loadtxt('min_false.txt'),
        #     np.loadtxt('std_false.txt'),
        #     np.loadtxt('variance_false.txt')
        # ]
        features = np.asarray(np.loadtxt('kurtosis_false.txt'))
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

    return labels, features

def get_testing_data():
    # features = []
    subjects = [2]
    # subs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    for i in subjects:
        subject = 'S' + str(i)
        path = '/Volumes/My Passport/TCC/WESAD/' + subject + '/data/raw/chest_eda/'
        os.chdir(path)

        labels = np.loadtxt('labels_false.txt')
        # features = [
        #     np.loadtxt('kurtosis_false.txt'),
        #     np.loadtxt('max_false.txt'),
        #     np.loadtxt('mean_false.txt'),
        #     np.loadtxt('median_false.txt'),
        #     np.loadtxt('min_false.txt'),
        #     np.loadtxt('std_false.txt'),
        #     np.loadtxt('variance_false.txt')
        # ]
        features = np.asarray(np.loadtxt('kurtosis_false.txt'))
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

    return labels, features


def execute():
    labels, features = get_training_data()
    # print(labels)
    # print(features)
    data = features
    print(data.shape)
    print(labels.shape)
    data = data.reshape(-1, 1)
    # data = np.vstack((data, features))
    # print(data.shape)
    # exit
    # X = features[:, :16]  # 16 features
    # y = data[:, 16]
    # print(X.shape)
    # print(X)
    # print(y.shape)
    # print(y)
    # train_features, test_features, train_labels, test_labels = train_test_split(X, y,
    #                                                                             test_size=0.25)
    # print('Training Features Shape:', train_features.shape)
    # print('Training Labels Shape:', train_labels.shape)
    # print('Testing Features Shape:', test_features.shape)
    # print('Testing Labels Shape:', test_labels.shape)

    clf = RandomForestClassifier(n_estimators=100, max_depth=5, oob_score=True)
    clf.fit(data, labels)
    labels, features = get_testing_data()
    data = features
    print(data.shape)
    print(labels.shape)
    data = data.reshape(-1, 1)
    predictions = clf.predict(data)
    print(predictions)
    print(labels)
    # print(clf.feature_importances_)
    # print(clf.oob_decision_function_)
    # print(clf.oob_score_)
    # predictions = clf.predict(test_features)
    # errors = abs(predictions - test_labels)
    # print("M A E: ", np.mean(errors))
    # print(np.count_nonzero(errors), len(test_labels))
    # print("Accuracy:", np.count_nonzero(errors)/len(test_labels))


if __name__ == '__main__':
    execute()
    