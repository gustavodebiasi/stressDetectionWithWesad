import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

class Selector(object):
    def get_labels_and_features(self, base_path, signal, subjects, window, window_overlap):
        features = []
        labels = []
        quantities = {}

        for i in subjects:
            subject = 'S' + str(i)
            path = base_path + subject + '/data/chest_' + signal + '/'
            os.chdir(path)

            labels2 = np.asarray(np.loadtxt('labels_' + str(window) + '_' + str(window_overlap) + '.txt'))
            labels.extend(labels2)
            features2 = np.asarray(np.loadtxt('features_' + str(window) + '_' + str(window_overlap) + '.txt'))
            quantities.update({
                subject: len(features2)
            })
            features.extend(features2)

        return labels, features

    def execute(self, base_path, signal, subjects, window, window_overlap, selection_type):
        for subject in subjects:
            all_subjects_except_test = subjects[:]
            all_subjects_except_test.remove(subject)
            train_labels, train_features = self.get_labels_and_features(base_path, signal, all_subjects_except_test, window, window_overlap)

            if (selection_type == 'pca'):
                pca_select = PCA()
                pca_select.fit(train_features)

                test_labels, test_features = self.get_labels_and_features(base_path, signal, [subject], window, window_overlap)

                principal_components = pca_select.transform(test_features)

                principal_df = pd.DataFrame(data = principal_components)
                exp = pca_select.explained_variance_ratio_

                print('component percentage')
                sum_95 = 0
                objets_sum_95 = 0
                for e in exp:
                    if (sum_95 < 0.95 or objets_sum_95 <= 1):
                        sum_95 += e
                        objets_sum_95 += 1

                    print('%.9f' % (e * 100))
                
                objets_sum_95 = objets_sum_95 if objets_sum_95 >= 2 else 2 

                new_features_pca = []
                p = 0
                for p in range(len(principal_df)):
                    array_new_features = []
                    k = 0
                    for k in range(objets_sum_95):
                        array_new_features.extend([principal_df[k][p]])

                    new_features_pca.append(array_new_features)

                np.savetxt(base_path + 'S' + str(subject) + '/data/chest_' + signal + '/features_' + str(window) + '_' + str(window_overlap) + '_pca.txt', new_features_pca, fmt="%f")

            if (selection_type == 'lda'):
                print('selection LDA')
        print('terminou')        


from selector import Selector

if __name__ == '__main__':
    # subjects = [13, 14, 15, 16, 17]
    subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    base_path = '/Volumes/My Passport/TCC/WESAD/'
    signal = 'ecg'
    selection_type = 'pca'
    window = 20
    window_overlap = True
    select = Selector()
    select.execute(base_path, signal, subjects, window, window_overlap, selection_type)