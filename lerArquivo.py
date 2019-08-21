import os
import pickle
import numpy as np

def load_data(path, subject):
    """Given path and subject, load the data of the subject"""
    os.chdir(path)
    os.chdir(subject)
    with open(subject + '.pkl', 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    return data

def read_data_one_subject(self, path, subject):
    self.keys = ['label', 'subject', 'signal']
    self.signal_keys = ['wrist', 'chest']
    self.chest_sensor_keys = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
    self.wrist_sensor_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
    self.data = load_data(path, subject)
    # os.chdir(path)
    # os.chdir(subject)
    # with open(subject + '.pkl', 'rb') as file:
    #     data = pickle.load(file, encoding='latin1')
    # self.data = data

def get_labels(self):
    return self.data[self.keys[0]]

def get_wrist_data(self):
    """"""
    #label = self.data[self.keys[0]]
    #assert subject == self.data[self.keys[1]]
    signal = self.data[self.keys[2]]
    wrist_data = signal[self.signal_keys[0]]
    #wrist_ACC = wrist_data[self.wrist_sensor_keys[0]]
    #wrist_ECG = wrist_data[self.wrist_sensor_keys[1]]
    return wrist_data

def get_chest_data(self):
    """"""
    signal = self.data[self.keys[2]]
    chest_data = signal[self.signal_keys[1]]
    return chest_data

def execute(data_set_path):
    obj_data = {}
    labels = {}
    # all_data = {}
    subs = [3]
    for i in subs:
        subject = 'S' + str(i)
        print("Reading data", subject)
        obj_data[subject] = read_data_one_subject(data_set_path, subject)
        
        labels[subject] = obj_data[subject].get_labels()
        print(labels[subject])

        wrist_data_dict = obj_data[subject].get_wrist_data()
        wrist_dict_length = {key: len(value) for key, value in wrist_data_dict.items()}
        print(wrist_dict_length)

        chest_data_dict = obj_data[subject].get_chest_data()
        chest_dict_length = {key: len(value) for key, value in chest_data_dict.items()}
        print(chest_dict_length)
        chest_data = np.concatenate((chest_data_dict['ACC'], chest_data_dict['ECG'], chest_data_dict['EDA'],
                                     chest_data_dict['EMG'], chest_data_dict['Resp'], chest_data_dict['Temp']), axis=1)
        # Get labels


        # 'ACC' : 3, 'ECG' 1: , 'EDA' : 1, 'EMG': 1, 'RESP': 1, 'Temp': 1  ===> Total dimensions : 8
        # No. of Labels ==> 8 ; 0 = not defined / transient, 1 = baseline, 2 = stress, 3 = amusement,
        # 4 = meditation, 5/6/7 = should be ignored in this dataset

        # Do for each subject
        '''
        baseline = np.asarray([idx for idx, val in enumerate(labels[subject]) if val == 1])
        # print("Baseline:", chest_data_dict['ECG'][baseline].shape)
        # print(baseline.shape)

        stress = np.asarray([idx for idx, val in enumerate(labels[subject]) if val == 2])
        # print(stress.shape)

        amusement = np.asarray([idx for idx, val in enumerate(labels[subject]) if val == 3])
        # print(amusement.shape)

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
    # path_input = input('Digite o caminho (padrão = "/Volumes/My Passport/TCC/WESAD"')
    # if path_input:
    #     path = path_input
    # print("path = {}".format(path))
    execute(path)

