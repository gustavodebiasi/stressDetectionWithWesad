import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

def execute():
    subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    features = []
    labels = []

    for i in subjects:
        subject = 'S' + str(i)
        path = '/Volumes/My Passport/TCC/WESAD/' + subject + '/data/chest_ecg/'
        os.chdir(path)

        labels2 = np.asarray(np.loadtxt('labels_20_True.txt'))
        labels.extend(labels2)
        features2 = np.asarray(np.loadtxt('features_20_True.txt'))
        features.extend(features2)


    selection = SelectKBest(k=15)
    x_new = selection.fit_transform(features, labels)
    indexes = selection.get_support()
    
    for i in subjects:
        subject = 'S' + str(i)
        path = '/Volumes/My Passport/TCC/WESAD/' + subject + '/data/chest_ecg/'
        os.chdir(path)

        features3 = np.asarray(np.loadtxt('features_20_True.txt'))
        new_features = []
        for feature in features3:
            feature_array = feature.tolist()
            i = 0
            new_array = []
            for feat in feature_array:
                if (indexes[i]):
                    new_array.extend([feat])
                i += 1
            new_features.append(new_array)
        np.savetxt('/Volumes/My Passport/TCC/WESAD/' + subject + '/data/chest_ecg/features_20_True_selected.txt', new_features, fmt="%f")

    # pca = PCA(n_components=2)
    # principalComponents = pca.fit_transform(x)
    # principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

if __name__ == '__main__':
    execute()