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
    window_overlap = False
    window = 0

    def calc_registers(self):
        self.registers700 = 700 * self.window
        self.registers64 = 64 * self.window
        self.registers32 = 32 * self.window
        self.registers4 = 4 * self.window 

    def extract_default_features(self, data):
        var_mean = np.mean(data)
        var_min = np.amin(data)
        var_max = np.amax(data)
        var_std = np.std(data)
        var_median = np.median(data)
        var_variance = np.var(data)
        var_kurtosis = kurtosis(data)

        return [
            var_mean,
            var_min,
            var_max,
            var_std,
            var_median,
            var_variance,
            var_kurtosis,
        ]

    def read_subject_basic_info(self):
        return []

    def extract_emg(self, data):
        if (len(data) <= 15):
            return []
        return nk.emg_process(data, 700, envelope_freqs=[10, 300])

    def extract_ecg(self, data_ecg, data_resp):
        return nk.ecg_process(data_ecg, data_resp, 700)

    def extract_resp(self, data_resp):
        return []

    def extract_eda(self, data):
        return []

    def extract_temp(self, data):
        return []

    def read_file(self, device, which):
        data = np.loadtxt(device + '_' + which + '_filtered.txt')
        if not (os.path.isdir(device + '_' + which)):
            os.makedirs(device + '_' + which)
        return data

    def extract_features(self, data_window, which, data_window_resp):
        print('aaaaaa')
        print(which)
        if (which == 'eda'):
            return self.extract_default_features(data_window)
        elif (which == 'resp'):
            return self.extract_default_features(data_window)
        elif (which == 'ecg'):
            print('teste')
            data = self.extract_ecg(data_window, data_window_resp)
            print('k')
        elif (which == 'emg'):
            return self.extract_default_features(data_window)
        elif (which == 'temp'):
            return self.extract_default_features(data_window)
        elif (which == 'bvp'):
            return self.extract_default_features(data_window)


    def process(self, labels, device, which, registers):
        data = self.read_file(device, which)
        if (which == 'ecg'):
            data_resp = self.read_file(device, 'resp')

        data_window = []
        data_window_resp = []
        i = 0
        label_anterior = labels[i]
        all_features = []
        window_labels = []
        data_size = len(data) - 1
        while (i <= data_size):
            if (i == data_size):
                data_window.append(data[i])
                if (which == 'ecg'):
                    data_window_resp.append(data_resp[i])

            if ((i == data_size and len(data_window) > 0) or (i > 0 and (len(data_window) % registers) == 0) or (label_anterior != labels[i] and len(data_window) > 0)):
                all_features.append(self.extract_features(data_window, which, data_window_resp))
                window_labels.append(labels[i])
                data_window = []
                data_window_resp = []
                
                if (i != data_size and self.window_overlap and len(data_window) % registers == 0 and label_anterior == labels[i]):
                    i = i - int(registers / 2)
                
            # if (i == data_size):
            #     all_feat = np.asarray(all_features)
            #     final_path_feat = self.path + '/' + device + '_' + which + '/features_' + str(self.window) + '_' + str(self.window_overlap) + '.txt'
            #     final_path_label = self.path + '/' + device + '_' + which + '/labels_' + str(self.window) + '_' + str(self.window_overlap) + '.txt'
            #     np.savetxt(final_path_feat, all_feat, fmt="%f")
            #     np.savetxt(final_path_label, window_labels, fmt="%d")
                                
            data_window.append(data[i])
            if (which == 'ecg'):
                data_window_resp.append(data_resp[i])
            label_anterior = labels[i]
            i += 1

    def execute(self, base_path, window, window_overlap, subjects):
        self.window_overlap = window_overlap
        self.window = window
        self.calc_registers()

        for i in subjects:
            subject = 'S' + str(i)
            self.path = base_path + subject + '/data/'
            os.chdir(self.path)
        
            labels = np.loadtxt('chest_labels_filtered.txt')

            # self.process(labels, 'chest', 'eda', self.registers700)
            # self.process(labels, 'chest', 'resp', self.registers700)
            self.process(labels, 'chest', 'ecg', self.registers700)
            # self.process(labels, 'chest', 'emg', self.registers700)
            # self.process(labels, 'chest', 'temp', self.registers700)
            # self.process(labels, 'wrist', 'bvp', self.registers64)
            # self.process(labels, 'wrist', 'eda', self.registers4)
            # self.process(labels, 'wrist', 'temp', self.registers4)

from extractor import Extractor

if __name__ == '__main__':
    window = 20
    window_overlap = True
    path = '/Volumes/My Passport/TCC/WESAD/'
    subjects = [2]
    # subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    extract = Extractor()
    extract.execute(path, window, window_overlap, subjects)
    
    