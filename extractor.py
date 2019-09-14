import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import neurokit as nk
import pandas as pd
from scipy.stats import kurtosis

class Extractor(object):
    registers700 = 0
    registers64 = 0
    registers32 = 0
    registers4 = 0
    path = ''

    def calc_registers(self, window):
        self.registers700 = 700 * window
        self.registers64 = 64 * window
        self.registers32 = 32 * window
        self.registers4 = 4 * window 

    def extract_default_features(self, data):
        var_mean = np.mean(data)
        var_min = np.amin(data)
        var_max = np.amax(data)
        var_std = np.std(data)
        var_median = np.median(data)
        var_variance = np.var(data)
        var_kurtosis = kurtosis(data)

        all_features = {
            'mean': var_mean,
            'min': var_min,
            'max': var_max,
            'std': var_std,
            'median': var_median,
            'variance': var_variance,
            'kurtosis': var_kurtosis,
        }

        return all_features

    def read_subject_basic_info(self):
        return []

    def extract_emg(self, data):
        if (len(data) <= 15):
            return []
        return nk.emg_process(data, 700, envelope_freqs=[10, 300])

    def extract_ecg(self, data_ecg, data_resp):
        return []

    def extract_resp(self, data_resp):
        return []

    def extract_eda(self, data):
        return []

    def extract_temp(self, data):
        return []

    def remove_files(self, device, which):
        try: 
            os.remove(device + '_' + which + '/labels_false.txt')
        except Exception as e:
            a = 'a'

        folder = path + device + '_' + which + '/'
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

    def process(self, labels, device, which, registers):
        data = np.loadtxt(device + '_' + which + '_filtered.txt')
        if not (os.path.isdir(device + '_' + which)):
            os.makedirs(device + '_' + which)

        remove_files(device, which)
        data_window = []
        i = 0
        label_anterior = labels[i]
        for i in range(len(data)):
            if (i == len(data)):
                data_window.append(data[i])

            if ((i == len(data) and len(data_window) > 0) or (i != 0 and (len(data_window) % registers) == 0) or (label_anterior != labels[i] and len(data_window) > 0)):
                all_features = extract_default_features(data_window)
                data_window = []
                
                for key,val in all_features.items():
                    with open(device + '_' + which + '/' + key + '_false.txt', 'a') as myfile:
                        myfile.write(str(float(val)) + '\n')
                    
                with open(device + '_' + which + '/labels_false.txt', 'a') as myfile:
                    myfile.write(str(int(labels[i])) + '\n')
                
            data_window.append(data[i])
            label_anterior = labels[i]


    def execute(self, base_path, window, window_overlap, subjects):
        calc_registers(window)

        for i in subjects:
            subject = 'S' + str(i)
            self.path = base_path + subject + '/data/'
            os.chdir(self.path)
        
            labels = np.loadtxt('chest_labels_filtered.txt')

            # emg_data = np.loadtxt('chest_emg_filtered.txt')
            # resp_data = np.loadtxt('chest_resp_filtered.txt')
            # ecg_data = np.loadtxt('chest_ecg_filtered.txt')
            # eda_data = np.loadtxt('chest_eda_filtered.txt')
            # temp_data = np.loadtxt('chest_temp_filtered.txt')

            # dados = extract_chest(ecg_data, eda_data, emg_data, resp_data, temp_data)

            # processamento = extrair_emg(dados)
            # print(processamento)
            # process(labels, 'chest', 'eda', registers700)
            # process(labels, 'chest', 'resp', registers700)
            # process(labels, 'chest', 'ecg', registers700)
            process(labels, 'chest', 'emg', self.registers700)
            # process(labels, 'chest', 'temp', registers700)
            # process(labels, 'wrist', 'bvp', registers64)
            # process(labels, 'wrist', 'eda', registers4)
            # process(labels, 'wrist', 'temp', registers4)

from extractor import Extractor

if __name__ == '__main__':
    window = 20
    window_overlap = False
    subjects = [4]
    # subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    extract = Extractor()
    extract.execute(window, window_overlap, subjects)
    
    