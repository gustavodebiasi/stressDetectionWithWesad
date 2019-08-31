import os
import pickle
import numpy as np
from Types import Types

directory = ''
window = 20

class read_data_one_subject:
    def __init__(self, path, subject):
        self.keys = ['label', 'subject', 'signal']
        self.signal_keys = ['wrist', 'chest']
        self.chest_sensor_keys = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
        self.wrist_sensor_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
        os.chdir(path)
        os.chdir(subject)
        with open(subject + '.pkl', 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        self.data = data

    def get_labels(self):
        return self.data[self.keys[0]]

    def get_wrist_data(self):
        signal = self.data[self.keys[2]]
        wrist_data = signal[self.signal_keys[0]]
        return wrist_data

    def get_chest_data(self):
        signal = self.data[self.keys[2]]
        chest_data = signal[self.signal_keys[1]]
        return chest_data

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

        # Do for each subject
        '''
        baseline = np.asarray([idx for idx, val in enumerate(labels[subject]) if val == 1])
        print("Baseline:", chest_data_dict['ECG'][baseline].shape)
        print(baseline.shape)

        stress = np.asarray([idx for idx, val in enumerate(labels[subject]) if val == 2])
        print(stress.shape)

        amusement = np.asarray([idx for idx, val in enumerate(labels[subject]) if val == 3])
        print(amusement.shape)
        
        baseline_data = extract_one(chest_data_dict, baseline, l_condition=1)
        stress_data = extract_one(chest_data_dict, stress, l_condition=2)
        amusement_data = extract_one(chest_data_dict, amusement, l_condition=3)

        full_data = np.vstack((baseline_data, stress_data, amusement_data))
        print("One subject data", full_data.shape)
        all_data[subject] = full_data

    i = 0
    for k, v in all_data.items():
        if i == 0:
            data = all_data[k]
            i += 1
        print(all_data[k].shape)
        data = np.vstack((data, all_data[k]))

    print(data.shape)
    return data
    '''

if __name__ == '__main__':
    path = "/Volumes/My Passport/TCC/WESAD"
    types = [
        Types.BASELINE.value,
        Types.STRESS.value,
        Types.AMUSEMENT.value,
    ]
    
    # subs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    subs = [2]
    
    execute(path, types, subs)

    # backup:
    # path_input = input('Digite o caminho (padrão = "/Volumes/My Passport/TCC/WESAD"')
    # if path_input:
    #     path = path_input
    # print("path = {}".format(path))

