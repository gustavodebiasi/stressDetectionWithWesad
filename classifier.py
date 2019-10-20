import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from shooter import Shooter

class Classifier(object):
    subjects = []
    window = 0
    window_overlap = False
    selector = ''
    with_all_signals = False

    def get_data(self, base_path, signal, subject, data_type):
        features = []
        labels = []

        selector_text = ''
        if (self.selector != ''):
            selector_text += '_' + self.selector

        if (data_type == 'training'):
            subjects = self.subjects[:]
            subjects.remove(subject)
        else:
            subjects = [subject]
        
        for i in subjects:
            subject = 'S' + str(i)
            if (self.with_all_signals):
                path = base_path + subject + '/data/'
            else:
                path = base_path + subject + '/data/chest_' + signal + '/'

            features2 = np.asarray(np.loadtxt(path + 'features_' + str(self.window) + '_' + str(self.window_overlap) + selector_text + '.txt'))
            features.extend(features2)
            labels2 = np.asarray(np.loadtxt(path + 'labels_' + str(self.window) + '_' + str(self.window_overlap) + '.txt'))
            labels.extend(labels2)

        i = 0
        for i in range(len(labels)):
            if (int(labels[i]) == 3):
                labels[i] = 1

        labels = np.asarray(labels)
        features = np.asarray(features)

        return labels, features

    def knn_classifier(self, training_data, training_labels, testing_data):
        # n_neighbors = 5
        # fifty_neighbors = int(n_neighbors/2)
        # print(fifty_neighbors)
        nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree')
        nbrs = nbrs.fit(training_data, training_labels)
        distances, indices = nbrs.kneighbors(testing_data)
        predictions = []
        for indice in indices:
            i = 0
            count_stress = 0
            for i in range(len(indice)):
                if (training_labels[indice[i]] == 2.0):
                    count_stress += 1
            if (count_stress >= 6):
                predictions.append(2.)
            else:
                predictions.append(1.)

        return np.asarray(predictions)

    def execute(self, base_path, signal, subjects, window, window_overlap, selector, use_classifiers, with_all_signals, times):
        self.subjects = subjects
        self.window = window
        self.window_overlap = window_overlap
        self.selector = selector
        self.with_all_signals = with_all_signals

        predicts_rf = []
        predicts_clf = []
        predicts_nbrs = []
        predicts_shooter = []
        testings = []

        for sub in subjects:
            # print('sujeito = ', sub)
            training_labels, training_features = self.get_data(base_path, signal, sub, 'training')
            testing_labels, testing_features = self.get_data(base_path, signal, sub, 'testing')

            if (np.isscalar(training_features[0])):
                training_features = training_features.reshape(-1,1)
                testing_features = testing_features.reshape(-1,1)

            testings.insert(sub, testing_labels)

            if 'forest' in use_classifiers:
                rf = RandomForestClassifier(n_estimators=100, max_depth=5, oob_score=True)
                rf = rf.fit(training_features, training_labels)
                predictions = rf.predict(testing_features)
                predicts_rf.insert(sub, predictions)
                predicts_rf_subject = predictions

            if 'svm' in use_classifiers:
                clf = svm.SVC(gamma='scale', C=4)
                clf = clf.fit(training_features, training_labels)
                predictions = clf.predict(testing_features)
                predicts_clf.insert(sub, predictions)
                predicts_clf_subject = predictions
        
            if 'knn' in use_classifiers:
                predictions = self.knn_classifier(training_features, training_labels, testing_features)
                predicts_nbrs.insert(sub, predictions)
                predicts_nbrs_subject = predictions
            
            if 'shooter' in use_classifiers:
                shoot2 = Shooter()
                predictions = shoot2.choose(predicts_rf_subject, predicts_nbrs_subject, predicts_clf_subject)
                predicts_shooter.insert(sub, predictions)

        return predicts_rf, predicts_clf, predicts_nbrs, predicts_shooter, testings

from classifier import Classifier
from evaluator import Evaluator

if __name__ == '__main__':
    subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    signal = 'ecg'
    base_path = '/Volumes/My Passport/TCC/WESAD2/'
    window = 20
    window_overlap = True
    selector = 'pca'
    use_classifiers = ['svm', 'forest', 'knn', 'shooter']
    with_all_signals = False
    times = 2
    classification = Classifier()
    evaluate = Evaluator()
    i = 0
    predicts_rf = []
    predicts_clf = []
    predicts_nbrs = []
    predicts_shooter = []
    testings = []
    for i in range(times):
        print('times = ', i)
        predicts_rf.insert(i, [])
        predicts_clf.insert(i, [])
        predicts_nbrs.insert(i, [])
        predicts_shooter.insert(i, [])
        testings.insert(i, [])
        predicts_rf[i], predicts_clf[i], predicts_nbrs[i], predicts_shooter[i], testings[i] = classification.execute(base_path, signal, subjects, window, window_overlap, selector, use_classifiers, with_all_signals, i)

    print('\n\n\n --------------------------------')

    evaluate.report(subjects, times, testings, predicts_rf, predicts_clf, predicts_nbrs, predicts_shooter, '/Volumes/My Passport/TCC/Resultados/teste.txt')
