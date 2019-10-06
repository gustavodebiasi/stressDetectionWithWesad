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

        # selection = SelectKBest(k=15)
        pca_select = PCA()
        # x_new = selection.fit_transform(features, labels)
        principalComponents = pca_select.fit_transform(features)
        # indexes = selection.get_support()
        principalDf = pd.DataFrame(data = principalComponents)
        exp = pca_select.explained_variance_ratio_
        # for e in exp:
            # print('%.9f' % (e * 100))
        # print(exp)
        # '''
        # print(quantities)
        # print(quantities.get(subject))
        # print(principalDf)
        # print(principalDf[0][0])
        # print(principalDf[0][1])
        # print(len(principalDf))
        quantity_sj = 0
        j = 0
        for i in subjects:
            subject = 'S' + str(i)
            path = base_path + subject + '/data/chest_' + signal + '/'
            os.chdir(path)

        #     features3 = np.asarray(np.loadtxt('features_' + str(window) + '_' + str(window_overlap) + '.txt'))
        #     new_features = []
        #     for feature in features3:
        #         feature_array = feature.tolist()
        #         k = 0
        #         new_array = []
        #         for feat in feature_array:
        #             if (indexes[k]):
        #                 new_array.extend([feat])
        #             i += k
        #         new_features.append(new_array)
        #     np.savetxt(base_path + subject + '/data/chest_' + signal + '/features_' + str(window) + '_' + str(window_overlap) + '_selected.txt', new_features, fmt="%f")

            new_features_pca = []
            final_quantity = quantity_sj + quantities.get(subject)
            j = quantity_sj
            print("inicio J = ", j)
            while (j < final_quantity):
                new_features_pca.append([
                    principalDf[0][j],
                    principalDf[1][j],
                    principalDf[2][j],
                    principalDf[3][j],
                    # principalDf[4][j],
                    # principalDf[5][j],
                    # principalDf[6][j],
                    # principalDf[7][j],
                ])
                j += 1
            quantity_sj+=quantities.get(subject)
            print("final J = ", quantity_sj)

            np.savetxt(base_path + subject + '/data/chest_' + signal + '/features_' + str(window) + '_' + str(window_overlap) + '_pca.txt', new_features_pca, fmt="%f")
        #     break
        print('terminou')
        # '''




from selector import Selector

if __name__ == '__main__':
    subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    base_path = '/Volumes/My Passport/TCC/WESAD/'
    signal = 'eda'
    window = 20
    window_overlap = True
    select = Selector()
    select.execute(base_path, signal, subjects, window, window_overlap)