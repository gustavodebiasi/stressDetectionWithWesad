import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class Selector(object):
    def get_labels_and_features(self, base_path, signal, subjects, window, window_overlap, with_all_signals):
        features = []
        labels = []

        for i in subjects:
            subject = 'S' + str(i)
            path = base_path + subject + '/data/chest_' 

            labels2 = np.asarray(np.loadtxt(path + signal + '/' + 'labels_' + str(window) + '_' + str(window_overlap) + '.txt'))
            labels.extend(labels2)

            #### FEATURES ####
            if (with_all_signals):
                features2 = []
                features_ecg = np.asarray(np.loadtxt(path + 'ecg/' + 'features_' + str(window) + '_' + str(window_overlap) + '.txt'))
                features_eda = np.asarray(np.loadtxt(path + 'eda/' + 'features_' + str(window) + '_' + str(window_overlap) + '.txt'))
                features_resp = np.asarray(np.loadtxt(path + 'resp/' + 'features_' + str(window) + '_' + str(window_overlap) + '.txt'))
                # features_emg = np.asarray(np.loadtxt(path + 'emg/' + 'features_' + str(window) + '_' + str(window_overlap) + '.txt'))

                all_features = []
                size = len(features_ecg)
                if (len(features_eda) < size):
                    size = len(features_eda)
                if (len(features_resp) < size):
                    size = len(features_resp)
                for j in range(size):
                    all_features.insert(j, [])
                    all_features[j].extend(features_ecg[j])
                    all_features[j].extend(features_eda[j])
                    all_features[j].extend(features_resp[j])
                    # all_features[j].extend(features_emg[j])
                features2.extend(all_features)
            else:
                features2 = np.asarray(np.loadtxt(path + signal + '/' + 'features_' + str(window) + '_' + str(window_overlap) + '.txt'))

            features.extend(features2)
        features = np.asarray(features)

        return labels, features

    def execute(self, base_path, signal, subjects, window, window_overlap, selection_type, with_all_signals):
        variances = []
        first = 0
        first_variance = -1
        for subject in subjects:
            all_subjects_except_test = subjects[:]
            all_subjects_except_test.remove(subject)
            train_labels, train_features = self.get_labels_and_features(base_path, signal, all_subjects_except_test, window, window_overlap, with_all_signals)

            test_labels, test_features = self.get_labels_and_features(base_path, signal, [subject], window, window_overlap, with_all_signals)

            if (with_all_signals):
                self.save_files(test_features, test_labels, selection_type, signal, with_all_signals, base_path, subject, window, window_overlap)
                continue

            if (selection_type == 'pca'):
                reduction = PCA()

            if (selection_type == 'lda'):
                reduction = LinearDiscriminantAnalysis(n_components=None)

            sc = StandardScaler()
            train_features = sc.fit_transform(train_features)

            if (selection_type == 'lda' or selection_type == 'pca'):
                reduction.fit(train_features, train_labels)

            test_features = sc.transform(test_features)

            if (selection_type == 'lda' or selection_type == 'pca'):
                principal_components = reduction.transform(test_features)

                principal_df = pd.DataFrame(data = principal_components)
                exp = reduction.explained_variance_ratio_
                variances.append(exp)

                sum_exp = 0
                objects_sum = 0
                if (first == 0):
                    for e in exp:
                        if (sum_exp < 0.95 or objects_sum <= 1):
                            sum_exp += e
                            objects_sum += 1
                    
                    first_variance = objects_sum
                    first = 1
                else:
                    objects_sum = first_variance

                new_features = []
                p = 0
                for p in range(len(principal_df)):
                    array_new_features = []
                    k = 0
                    for k in range(objects_sum):
                        array_new_features.extend([principal_df[k][p]])

                    new_features.append(array_new_features)
            else:
                new_features = test_features

            self.save_files(new_features, test_labels, selection_type, signal, with_all_signals, base_path, subject, window, window_overlap)

        if (with_all_signals):
            return []

        if (selection_type == 'lda' or selection_type == 'pca'):
            return self.calc_variances_std(variances)
        
        return ['Sem variações']

    def save_files(self, features, labels, selection_type, signal, with_all_signals, base_path, subject, window, window_overlap):
        if (with_all_signals):
            select_text = ''
            if (selection_type != ''):
                select_text = '_' + selection_type

            np.savetxt(base_path + 'S' + str(subject) + '/data/features_' + str(window) + '_' + str(window_overlap) + select_text + '.txt', features, fmt="%f")
            np.savetxt(base_path + 'S' + str(subject) + '/data/labels_' + str(window) + '_' + str(window_overlap) + '.txt', np.asarray(labels), fmt="%f")
        else:
            np.savetxt(base_path + 'S' + str(subject) + '/data/chest_' + signal + '/features_' + str(window) + '_' + str(window_overlap) + '_' + selection_type + '.txt', features, fmt="%f")
            np.savetxt(base_path + 'S' + str(subject) + '/data/chest_' + signal + '/labels_' + str(window) + '_' + str(window_overlap) + '.txt', np.asarray(labels), fmt="%f")

    def calc_variances_std(self, variances):
        new_variances = []

        i = 0
        for i in range(len(variances[0])):
            new_variances.insert(i, [])
            for v in variances:
                new_variances[i].extend([v[i]])

        variance = []
        i = 0
        for i in range(len(new_variances)):
            variance.insert(i, [])
            variance[i].extend([
                np.mean(new_variances[i]),
                np.std(new_variances[i])
            ])

        return variance

from selector import Selector

if __name__ == '__main__':
    subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    base_path = '/Volumes/My Passport/TCC/WESAD2/'
    signal = 'ecg'
    selection_type = 'lda'
    window = 20
    window_overlap = True
    with_all_signals = False
    select = Selector()
    variance = select.execute(base_path, signal, subjects, window, window_overlap, selection_type, with_all_signals)
    print(variance)