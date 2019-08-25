import os
import pickle
import numpy as np

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

def execute(data_set_path):
    obj_data = {}
    labels = {}
    # all_data = {}
    # subs = [3]
    subs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    for i in subs:
        subject = 'S' + str(i)
        directory = data_set_path + '/' + subject + '/data/raw/'
        if not (os.path.isdir(directory)):
            os.makedirs(directory)

        print("Reading data", subject)
        obj_data[subject] = read_data_one_subject(data_set_path, subject)
        
        # Read the labels from the pkl file and save as txt
        labels[subject] = obj_data[subject].get_labels()
        np.savetxt(directory + '/chest_labels.txt', labels[subject], fmt='%d')
        j = 0
        labels32hz = []
        labels4hz = []
        labels64hz = []
        cada700x64 = 0
        cada700x32 = 0
        contagem64 = 0
        contagem32 = 0
        for j in range(len(labels[subject])):
            cada700x64 += 1
            cada700x32 += 1
            if ((j % 22) == 0):
                labels32hz.append(labels[subject][j])
                contagem32 += 1
            if ((cada700x32) == 700):
                if (contagem32 < 32):
                    if ((j % 22) == 0):
                        labels32hz.append(labels[subject][j+1])
                    else:
                        labels32hz.append(labels[subject][j])
                contagem32 = 0
                cada700x32 = 0
            if ((j % 175) == 0):
                labels4hz.append(labels[subject][j])
            if ((j % 11) == 0):
                labels64hz.append(labels[subject][j])
                contagem64 += 1
            if ((cada700x64) == 700):
                if (contagem64 < 64):
                    if ((j % 11) == 0):
                        labels64hz.append(labels[subject][j+1])
                    else:
                        labels64hz.append(labels[subject][j])
                contagem64 = 0
                cada700x64 = 0
            

        np.savetxt(directory + '/labels32hz.txt', labels32hz, fmt='%d')
        np.savetxt(directory + '/labels4hz.txt', labels4hz, fmt='%d')
        np.savetxt(directory + '/labels64hz.txt', labels64hz, fmt='%d')


        # Read the wrist signs from the pkl file 
        wrist_data_dict = obj_data[subject].get_wrist_data()
        # wrist_dict_length = {key: len(value) for key, value in wrist_data_dict.items()}
        
        np.savetxt(directory + '/wrist_bvp.txt', wrist_data_dict['BVP'], fmt='%f')
        np.savetxt(directory + '/wrist_eda.txt', wrist_data_dict['EDA'], fmt='%f')
        np.savetxt(directory + '/wrist_acc.txt', wrist_data_dict['ACC'], fmt='%f')
        np.savetxt(directory + '/wrist_temp.txt', wrist_data_dict['TEMP'], fmt='%f')

        chest_data_dict = obj_data[subject].get_chest_data()
        # chest_dict_length = {key: len(value) for key, value in chest_data_dict.items()}
        # print(chest_dict_length)

        np.savetxt(directory + '/chest_ecg.txt', np.squeeze(chest_data_dict['ECG']), fmt='%f')
        np.savetxt(directory + '/chest_eda.txt', np.squeeze(chest_data_dict['EDA']), fmt='%f')
        np.savetxt(directory + '/chest_emg.txt', np.squeeze(chest_data_dict['EMG']), fmt='%f')
        np.savetxt(directory + '/chest_temp.txt', np.squeeze(chest_data_dict['Temp']), fmt='%f')
        np.savetxt(directory + '/chest_acc.txt', np.squeeze(chest_data_dict['ACC']), fmt='%f')
        np.savetxt(directory + '/chest_resp.txt', np.squeeze(chest_data_dict['Resp']), fmt='%f')

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
    # path_input = input('Digite o caminho (padrão = "/Volumes/My Passport/TCC/WESAD"')
    # if path_input:
    #     path = path_input
    # print("path = {}".format(path))
    execute(path)

