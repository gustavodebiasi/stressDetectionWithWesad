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
                features_emg = np.asarray(np.loadtxt(path + 'emg/' + 'features_' + str(window) + '_' + str(window_overlap) + '.txt'))

                all_features = []
                for j in range(len(features_ecg)):
                    all_features.insert(j, [])
                    all_features[j].extend(features_ecg[j])
                    all_features[j].extend(features_eda[j])
                    all_features[j].extend(features_resp[j])
                    all_features[j].extend(features_emg[j])
                features2.extend(all_features)
            else:
                features2 = np.asarray(np.loadtxt(path + signal + '/' + 'features_' + str(window) + '_' + str(window_overlap) + '.txt'))

            features.extend(features2)
        features = np.asarray(features)

        return labels, features

    def execute(self, base_path, signal, subjects, window, window_overlap, selection_type, with_all_signals):
        variances = []
        for subject in subjects:
            all_subjects_except_test = subjects[:]
            all_subjects_except_test.remove(subject)
            train_labels, train_features = self.get_labels_and_features(base_path, signal, all_subjects_except_test, window, window_overlap, with_all_signals)

            if (selection_type == 'pca'):
                sc = StandardScaler()
                train_features = sc.fit_transform(train_features)

                pca_select = PCA()
                pca_select.fit(train_features)

                test_labels, test_features = self.get_labels_and_features(base_path, signal, [subject], window, window_overlap, with_all_signals)

                test_features = sc.transform(test_features)

                principal_components = pca_select.transform(test_features)

                principal_df = pd.DataFrame(data = principal_components)
                exp = pca_select.explained_variance_ratio_
                variances.append(exp)

                sum_95 = 0
                objets_sum_95 = 0
                for e in exp:
                    if (sum_95 < 0.90 or objets_sum_95 <= 1):
                        sum_95 += e
                        objets_sum_95 += 1
                
                objets_sum_95 = objets_sum_95 if objets_sum_95 >= 2 else 2 

                new_features_pca = []
                p = 0
                for p in range(len(principal_df)):
                    array_new_features = []
                    k = 0
                    for k in range(objets_sum_95):
                        array_new_features.extend([principal_df[k][p]])

                    new_features_pca.append(array_new_features)

                self.save_files(new_features_pca, test_labels, selection_type, signal, with_all_signals, base_path, subject, window, window_overlap)

            if (selection_type == 'lda'):
                lda = LinearDiscriminantAnalysis(n_components=None)
                lda.fit(train_features, train_labels)

                test_labels, test_features = self.get_labels_and_features(base_path, signal, [subject], window, window_overlap)

                new_features_lda = lda.transform(test_features)

                # self.save_files(new_features_lda, selection_type, signal, with_all_signals, base_path, subject, window, window_overlap)
                # 

        return self.calc_variances_std(variances)

    def save_files(self, features, labels, selection_type, signal, with_all_signals, base_path, subject, window, window_overlap):
        if (with_all_signals):
            np.savetxt(base_path + 'S' + str(subject) + '/data/features_' + str(window) + '_' + str(window_overlap) + '_' + selection_type + '.txt', features, fmt="%f")
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
    base_path = '/Volumes/My Passport/TCC/WESAD/'
    signal = 'ecg'
    selection_type = 'pca'
    window = 20
    window_overlap = True
    with_all_signals = True
    select = Selector()
    variance = select.execute(base_path, signal, subjects, window, window_overlap, selection_type, with_all_signals)
    print(variance)