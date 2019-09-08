import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def execute():
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

if __name__ == '__main__':
    execute()