import os
import pickle
import numpy as np
from Enums.Types import Types
from Helpers.read_data_one_subject import read_data_one_subject

directory = ''
window = 20

def filter_types(labels, experience_types):
    indexes = []
    for i in experience_types:
        indexes.extend(np.asarray([idx for idx, val in enumerate(labels) if val == i]))

    return indexes

def process_another_rate(labels, experience_types, new_rate):
    each = int(np.ceil(700 / new_rate))

    j = 0
    new_labels = []
    each700 = 0
    counting = 0
    for j in range(len(labels)):
        each700 += 1
        if ((j % each) == 0):
            new_labels.append(labels[j])
            counting += 1
        if ((each700) == 700):
            if (counting < new_rate):
                if ((j % each) == 0):
                    new_labels.append(labels[j+1])
                else:
                    new_labels.append(labels[j])
            counting = 0
            each700 = 0

    indexes_new_rate = filter_types(new_labels, experience_types)

    np.savetxt(directory + '/indexes_' + str(new_rate) + '.txt', indexes_new_rate, fmt='%d')    
    np.savetxt(directory + '/labels_' + str(new_rate) + '.txt', new_labels, fmt='%d')

    return indexes_new_rate


def process_labels(labels, experience_types):
    np.savetxt(directory + '/chest_labels_all.txt', labels, fmt='%d')

    indexes = filter_types(labels, experience_types)
    
    new_labels = labels[indexes]
    np.savetxt(directory + '/indexes_700.txt', indexes, fmt='%d')
    np.savetxt(directory + '/chest_labels_filtered.txt', new_labels, fmt='%d')

    index_64 = process_another_rate(labels, experience_types, 64)
    index_32 = process_another_rate(labels, experience_types, 32)
    index_4 = process_another_rate(labels, experience_types, 4)

    all_indexes = {
        '700': indexes,
        '64': index_64,
        '32': index_32,
        '4': index_4
    }

    return all_indexes

def create_directory(data_path, subject):
    global directory
    directory = data_path + '/' + subject + '/data/raw/'
    if not (os.path.isdir(directory)):
        os.makedirs(directory)

def execute(data_set_path, experience_types, subjects):
    obj_data = {}
    labels = {}
    # all_data = {}
    for i in subjects:
        subject = 'S' + str(i)
        create_directory(data_set_path, subject)

        print("Reading data", subject)
        obj_data[subject] = read_data_one_subject(data_set_path, subject)
        
        # Read the labels from the pkl file and save as txt
        labels[subject] = obj_data[subject].get_labels()
        all_indexes = process_labels(labels[subject], experience_types)

        # Read the wrist signs from the pkl file 
        wrist_data_dict = obj_data[subject].get_wrist_data()
        wrist_dict_length = {key: len(value) for key, value in wrist_data_dict.items()}
        
        np.savetxt(directory + '/wrist_bvp_all.txt', wrist_data_dict['BVP'], fmt='%f')
        np.savetxt(directory + '/wrist_eda_all.txt', wrist_data_dict['EDA'], fmt='%f')
        np.savetxt(directory + '/wrist_acc_all.txt', wrist_data_dict['ACC'], fmt='%f')
        np.savetxt(directory + '/wrist_temp_all.txt', wrist_data_dict['TEMP'], fmt='%f')

        bvp_filtered = wrist_data_dict['BVP'][all_indexes['64']]
        np.savetxt(directory + '/wrist_bvp_filtered.txt', bvp_filtered, fmt='%f')

        eda_filtered = wrist_data_dict['EDA'][all_indexes['4']]
        np.savetxt(directory + '/wrist_eda_filtered.txt', eda_filtered, fmt='%f')

        acc_filtered = wrist_data_dict['ACC'][all_indexes['32']]
        np.savetxt(directory + '/wrist_acc_filtered.txt', acc_filtered, fmt='%f')

        temp_filtered = wrist_data_dict['TEMP'][all_indexes['4']]
        np.savetxt(directory + '/wrist_temp_filtered.txt', temp_filtered, fmt='%f')


        chest_data_dict = obj_data[subject].get_chest_data()
        chest_dict_length = {key: len(value) for key, value in chest_data_dict.items()}
        print(chest_dict_length)

        np.savetxt(directory + '/chest_ecg_all.txt', np.squeeze(chest_data_dict['ECG']), fmt='%f')
        np.savetxt(directory + '/chest_eda_all.txt', np.squeeze(chest_data_dict['EDA']), fmt='%f')
        np.savetxt(directory + '/chest_emg_all.txt', np.squeeze(chest_data_dict['EMG']), fmt='%f')
        np.savetxt(directory + '/chest_temp_all.txt', np.squeeze(chest_data_dict['Temp']), fmt='%f')
        np.savetxt(directory + '/chest_acc_all.txt', np.squeeze(chest_data_dict['ACC']), fmt='%f')
        np.savetxt(directory + '/chest_resp_all.txt', np.squeeze(chest_data_dict['Resp']), fmt='%f')

        ecg_filtered = chest_data_dict['ECG'][all_indexes['700']]
        np.savetxt(directory + '/chest_ecg_filtered.txt', ecg_filtered, fmt='%f')
        eda_filtered = chest_data_dict['EDA'][all_indexes['700']]
        np.savetxt(directory + '/chest_eda_filtered.txt', eda_filtered, fmt='%f')
        emg_filtered = chest_data_dict['EMG'][all_indexes['700']]
        np.savetxt(directory + '/chest_emg_filtered.txt', emg_filtered, fmt='%f')
        temp_filtered = chest_data_dict['Temp'][all_indexes['700']]
        np.savetxt(directory + '/chest_temp_filtered.txt', temp_filtered, fmt='%f')
        acc_filtered = chest_data_dict['ACC'][all_indexes['700']]
        np.savetxt(directory + '/chest_acc_filtered.txt', acc_filtered, fmt='%f')
        resp_filtered = chest_data_dict['Resp'][all_indexes['700']]
        np.savetxt(directory + '/chest_resp_filtered.txt', resp_filtered, fmt='%f')
        

        print('finish')

        # 'ACC' : 3, 'ECG' 1: , 'EDA' : 1, 'EMG': 1, 'RESP': 1, 'Temp': 1  ===> Total dimensions : 8
        # No. of Labels ==> 8 ; 0 = not defined / transient, 1 = baseline, 2 = stress, 3 = amusement,
        # 4 = meditation, 5/6/7 = should be ignored in this dataset
        
if __name__ == '__main__':
    path = "/Volumes/My Passport/TCC/WESAD"
    types = [
        Types.BASELINE.value,
        Types.STRESS.value,
        Types.AMUSEMENT.value,
    ]
    
    # subs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    subs = [17]
    
    execute(path, types, subs)

    # backup:
    # path_input = input('Digite o caminho (padrão = "/Volumes/My Passport/TCC/WESAD"')
    # if path_input:
    #     path = path_input
    # print("path = {}".format(path))

