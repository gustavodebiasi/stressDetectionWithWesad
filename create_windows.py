import os
import pickle
import numpy as np

def execute(data_set_path):
    os.chdir(data_set_path)
    
    subs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    os.chdir(subject)
    print('executando')



if __name__ == '__main__':
    path = "/Volumes/My Passport/TCC/WESAD"
    # path_input = input('Digite o caminho (padrão = "/Volumes/My Passport/TCC/WESAD"')
    # if path_input:
    #     path = path_input
    # print("path = {}".format(path))
    execute(path)