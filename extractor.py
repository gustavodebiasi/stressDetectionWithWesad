#%%
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import neurokit as nk
import pandas as pd
import math  
from pandas import DataFrame
from scipy.stats import kurtosis
from scipy.stats import skew
from statsmodels import robust

class Extractor(object):
    registers700 = 0
    registers64 = 0
    registers32 = 0
    registers4 = 0
    path = ''
    window_overlap = False
    window = 0
    overlap = 0

    def calc_registers(self):
        self.registers700 = 700 * self.window
        self.registers64 = 64 * self.window
        self.registers32 = 32 * self.window
        self.registers4 = 4 * self.window 
        self.overlap = (self.window - 20) * 700

    def extract_default_features(self, data):
        var_mean = np.mean(data)
        var_min = np.amin(data)
        var_max = np.amax(data)
        var_std = np.std(data)
        var_median = np.median(data)
        var_variance = np.var(data)
        var_kurtosis = kurtosis(data)
        var_skew = self.select_data_from_string(skew(data))
        var_range = self.select_data_from_string(var_max - var_min)

        return [
            np.float32(var_mean),
            np.float32(var_min),
            np.float32(var_max),
            np.float32(var_std),
            np.float32(var_median),
            np.float32(var_variance),
            np.float32(var_kurtosis),
            np.float32(var_skew),
            np.float32(var_range)
        ]

    def extract_root_mean_square(self, data):
        data_sum = 0
        for i in range(len(data)):
            data_sum += data[i] ** 2
            
        data_no_root = data_sum / len(data) 
        return math.sqrt(data_no_root)
    
    def extract_mean_absolute_deviation(self, data):
        return robust.mad(data)

    def extract_mean_absolute_value(self, data):
        data_sum = 0
        for i in range(len(data)):
            data_sum += abs(data[i])

        return data_sum / len(data)

    def extract_slope_sign_change(self, data):
        data_sum = 0
        i = 1
        data_size = len(data) - 1
        while (i < data_size):
            if ((data[i] < data[i+1] and data[i] < data[i-1]) or (data[i] > data[i+1] and data[i] > data[i-1])):
                data_sum += 1
                
            i += 1

        return data_sum

    def extract_zero_crossing(self, data):
        data_sum = 0
        i = 0
        data_size = len(data) - 1
        while (i < data_size):
            if ((data[i] > 0 and data[i+1] < 0) or (data[i] < 0 and data[i+1] > 0)):
                data_sum += 1
                
            i += 1

        return data_sum

    def extract_emg(self, data_emg):
        default_features = self.extract_default_features(data_emg)
        rms = self.extract_root_mean_square(data_emg)
        default_features.extend([
            rms,
            np.log(rms),
            self.extract_mean_absolute_value(data_emg),
            self.extract_slope_sign_change(data_emg),
            self.extract_zero_crossing(data_emg),
        ])
        return default_features

    def select_data_from_string(self, string):
        if (np.isnan(string)):
            return 0.
        return string

    def select_data_from_array(self, array, stat):
        try:
            result = np.float32(array[stat])
            return self.select_data_from_string(result)
        except:
            return 0.

    def extract_ecg(self, data_ecg):
        # try:
        data = nk.ecg_process(ecg=data_ecg, rsp=None, sampling_rate=700)
        # except:
            # return []

        default_features = self.extract_default_features(data_ecg)
        default_features.extend([
            self.select_data_from_array(data['ECG']['HRV'],'CVSD'),
            self.select_data_from_array(data['ECG']['HRV'],'HF'),
            self.select_data_from_array(data['ECG']['HRV'],'HF/P'),
            self.select_data_from_array(data['ECG']['HRV'],'HFn'),
            self.select_data_from_array(data['ECG']['HRV'],'LF'),
            self.select_data_from_array(data['ECG']['HRV'],'LF/HF'),
            self.select_data_from_array(data['ECG']['HRV'],'LF/P'),
            self.select_data_from_array(data['ECG']['HRV'],'LFn'),
            self.select_data_from_array(data['ECG']['HRV'],'RMSSD'),
            self.select_data_from_array(data['ECG']['HRV'],'Total_Power'),
            self.select_data_from_array(data['ECG']['HRV'],'Triang'),
            self.select_data_from_array(data['ECG']['HRV'],'VHF'),
            self.select_data_from_array(data['ECG']['HRV'],'cvNN'),
            self.select_data_from_array(data['ECG']['HRV'],'madNN'),
            self.select_data_from_array(data['ECG']['HRV'],'mcvNN'),
            self.select_data_from_array(data['ECG']['HRV'],'meanNN'),
            self.select_data_from_array(data['ECG']['HRV'],'medianNN'),
            self.select_data_from_array(data['ECG']['HRV'],'pNN20'),
            self.select_data_from_array(data['ECG']['HRV'],'pNN50'),
            self.select_data_from_array(data['ECG']['HRV'],'sdNN'),
        ])
        
        return default_features

    def extract_resp(self, data_resp):
        # try:
        data = nk.rsp_process(data_resp, 700)
        # except:
            # return False 

        default_features = self.extract_default_features(data_resp)
        default_features.extend([
            self.select_data_from_array(data['RSP']['Respiratory_Variability'],'RSPV_RMSSD'),
            self.select_data_from_array(data['RSP']['Respiratory_Variability'],'RSPV_RMSSD_Log'),
            self.select_data_from_array(data['RSP']['Respiratory_Variability'],'RSPV_SD'),
        ])
        return default_features

    def extract_eda(self, data_eda, label):
        # try:
        data = nk.eda_process(data_eda, 700)
        # except:
            # return []

        default_features = self.extract_default_features(data_eda)
        default_features.extend([
            np.mean(data['EDA']['SCR_Peaks_Amplitudes']),
            (len(data['EDA']['SCR_Peaks_Indexes']) / len(data_eda))
        ])

        return default_features

    def read_file(self, device, which):
        data = np.loadtxt(device + '_' + which + '_filtered.txt')
        if not (os.path.isdir(device + '_' + which)):
            os.makedirs(device + '_' + which)
        return data

    def extract_features(self, data_window, which, label):
        if (which == 'eda'):
            return self.extract_eda(data_window, label)
        elif (which == 'resp'):
            return self.extract_resp(data_window)
        elif (which == 'ecg'):
            return self.extract_ecg(data_window)
        elif (which == 'emg'):
            return self.extract_emg(data_window)

    def process(self, labels, device, which, registers):
        data = self.read_file(device, which)
        data_window = []
        i = 0
        label_anterior = labels[i]
        all_features = []
        window_labels = []
        data_size = len(data) - 1
        while (i <= data_size):
            if (i == data_size):
                data_window.append(data[i])

            

            # if ((i == data_size and len(data_window) > 0) or (i > 0 and (len(data_window) % registers) == 0) or (label_anterior != labels[i] and len(data_window) > 0)):
            
            if (i > 0 and (len(data_window) % registers) == 0):
                print(len(data_window))
                result = self.extract_features(data_window, which, labels[i])
                # if (result):
                all_features.append(result)
                window_labels.append(labels[i])

                data_window = []
                
                if (i != data_size and self.window_overlap and len(data_window) % registers == 0 and label_anterior == labels[i]):
                    i = i - self.overlap
            else:
                if (label_anterior != labels[i]):
                    data_window = []
                
            if (i == data_size):
                all_feat = np.asarray(all_features)
                final_path_feat = self.path + '/' + device + '_' + which + '/features_' + str(self.window) + '_' + str(self.window_overlap) + '.txt'
                final_path_label = self.path + '/' + device + '_' + which + '/labels_' + str(self.window) + '_' + str(self.window_overlap) + '.txt'
                np.savetxt(final_path_feat, all_feat, fmt="%f")
                np.savetxt(final_path_label, window_labels, fmt="%d")
                                
            data_window.append(data[i])
            label_anterior = labels[i]
            i += 1

    def execute(self, base_path, window, window_overlap, subjects):
        self.window_overlap = window_overlap
        self.window = window
        self.calc_registers()

        for i in subjects:
            subject = 'S' + str(i)
            print('iniciando sujeito = ', subject)
            self.path = base_path + subject + '/data/'
            os.chdir(self.path)
        
            labels700 = np.loadtxt('chest_labels_filtered.txt')

            self.process(labels700, 'chest', 'ecg', self.registers700)
            self.process(labels700, 'chest', 'emg', self.registers700)
            self.process(labels700, 'chest', 'eda', self.registers700)
            self.process(labels700, 'chest', 'resp', self.registers700)

from extractor import Extractor

if __name__ == '__main__':
    window = 30
    window_overlap = True
    path = '/Volumes/My Passport/TCC/WESAD/'
    subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    extract = Extractor()
    extract.execute(path, window, window_overlap, subjects)
    
    

#%%
