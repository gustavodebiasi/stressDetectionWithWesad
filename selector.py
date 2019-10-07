import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

class Selector(object):
    def execute(self, base_path, signal, subjects, window, window_overlap):
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

        pca_select = PCA()
        principal_components = pca_select.fit_transform(features)
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

        quantity_sj = 0
        j = 0
        for i in subjects:
            subject = 'S' + str(i)
            path = base_path + subject + '/data/chest_' + signal + '/'
            os.chdir(path)

            new_features_pca = []
            final_quantity = quantity_sj + quantities.get(subject)
            j = quantity_sj
            print("inicio J = ", j)
            while (j < final_quantity):
                array_new_features = []
                for k in range(objets_sum_95):
                    array_new_features.extend(principal_df[k][j])

                new_features_pca.append(array_new_features)
                j += 1
            quantity_sj+=quantities.get(subject)

            np.savetxt(base_path + subject + '/data/chest_' + signal + '/features_' + str(window) + '_' + str(window_overlap) + '_pca.txt', new_features_pca, fmt="%f")

        print('terminou')


from selector import Selector

if __name__ == '__main__':
    subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    base_path = '/Volumes/My Passport/TCC/WESAD/'
    signal = 'emg'
    window = 20
    window_overlap = True
    select = Selector()
    select.execute(base_path, signal, subjects, window, window_overlap)