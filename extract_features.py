import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import neurokit as nk
import pandas as pd
from scipy.stats import kurtosis

window = 0
registers700 = 0
registers64 = 0
registers32 = 0
registers4 = 0

def extract_default_features(data):
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

def process_eda():
    eda = np.loadtxt('chest_eda_filtered.txt')
    if not (os.path.isdir('chest_eda')):
        os.makedirs('chest_eda')

    eda_window = []
    print(len(eda))
    i = 0
    files = 0
    for i in range(len(eda)):
        eda_window.append(eda[i])
        if ((i == len(eda) and eda_window) or (i != 0 and (i % registers700) == 0)):
            files += 1
            eda_features = nk.eda_process(eda=eda_window, sampling_rate=700)
            np.savetxt('chest_eda/' + str(files) + '_Onsets.txt', np.squeeze(eda_features['EDA']['SCR_Onsets']), fmt='%d')
            np.savetxt('chest_eda/' + str(files) + '_peaks_indexes.txt', np.squeeze(eda_features['EDA']['SCR_Peaks_Indexes']), fmt='%d')
            np.savetxt('chest_eda/' + str(files) + '_recovery.txt', np.squeeze(eda_features['EDA']['SCR_Recovery_Indexes']), fmt='%d')
            np.savetxt('chest_eda/' + str(files) + '_amplitudes.txt', np.squeeze(eda_features['EDA']['SCR_Peaks_Amplitudes']), fmt='%f')
            eda_window = []
            np.savetxt('chest_eda/' + str(files) + '_min.txt', [np.amin(eda_window)], fmt='%f')
            np.savetxt('chest_eda/' + str(files) + '_max.txt', [np.amax(eda_window)], fmt='%f')
            np.savetxt('chest_eda/' + str(files) + '_std.txt', [np.std(eda_window)], fmt='%f')
            np.savetxt('chest_eda/' + str(files) + '_mean.txt', [np.mean(eda_window)], fmt='%f')

    # eda2 = np.loadtxt('wrist_eda.txt')
    # if not (os.path.isdir('wrist_eda')):
    #     os.makedirs('wrist_eda')

    # eda2_window = []
    # print(len(eda2))
    # print(min(eda2))
    # print(max(eda2))
    # i = 0
    # files = 0
    # for i in range(len(eda2)):
    #     eda2_window.append((eda2[i]+5))
    #     if ((i == len(eda2) and eda2_window) or (i != 0 and (i % 79) == 0)):
    #         files += 1
    #         print(eda2_window)
    #         print(len(eda2_window))
    #         eda_features = nk.eda_process(eda=eda2_window, sampling_rate=4)
    #         np.savetxt('wrist_eda/' + str(files) + '_Onsets.txt', np.squeeze(eda_features['EDA']['SCR_Onsets']), fmt='%d')
    #         np.savetxt('wrist_eda/' + str(files) + '_peaks_indexes.txt', np.squeeze(eda_features['EDA']['SCR_Peaks_Indexes']), fmt='%d')
    #         np.savetxt('wrist_eda/' + str(files) + '_recovery.txt', np.squeeze(eda_features['EDA']['SCR_Recovery_Indexes']), fmt='%d')
    #         np.savetxt('wrist_eda/' + str(files) + '_amplitudes.txt', np.squeeze(eda_features['EDA']['SCR_Peaks_Amplitudes']), fmt='%f')
    #         eda2_window = []


def process_resp():
    rsp = np.loadtxt('chest_resp.txt')
    if not (os.path.isdir('chest_resp')):
        os.makedirs('chest_resp')

    rsp_window = []
    print(len(rsp))
    i = 0
    files = 0
    for i in range(len(rsp)):
        rsp_window.append(rsp[i])
        if ((i == len(rsp) and rsp_window) or (i != 0 and (i % registers700) == 0)):
            files += 1
            rsp_features = nk.rsp_process(rsp=rsp_window, sampling_rate=700)
            np.savetxt('chest_resp/' + str(files) + '_Onsets.txt', np.squeeze(rsp_features['RSP']['Cycles_Onsets']), fmt='%d')
            np.savetxt('chest_resp/' + str(files) + '_Expiration_Onsets.txt', np.squeeze(rsp_features['RSP']['Expiration_Onsets']), fmt='%d')
            np.savetxt('chest_resp/' + str(files) + '_Cycles_Length.txt', rsp_features['RSP']['Cycles_Length'], fmt='%f')
            respiratory_variability = []
            respiratory_variability.append(rsp_features['RSP']['Respiratory_Variability']['RSPV_SD'])
            respiratory_variability.append(rsp_features['RSP']['Respiratory_Variability']['RSPV_RMSSD'])
            respiratory_variability.append(rsp_features['RSP']['Respiratory_Variability']['RSPV_RMSSD_Log'])
            np.savetxt('chest_resp/' + str(files) + '_Respiratory_Variability.txt', respiratory_variability, fmt='%f')
            np.savetxt('chest_resp/' + str(files) + '_min.txt', [np.amin(rsp_window)], fmt='%f')
            np.savetxt('chest_resp/' + str(files) + '_max.txt', [np.amax(rsp_window)], fmt='%f')
            np.savetxt('chest_resp/' + str(files) + '_std.txt', [np.std(rsp_window)], fmt='%f')
            np.savetxt('chest_resp/' + str(files) + '_mean.txt', [np.mean(rsp_window)], fmt='%f')
            
            rsp_window = []
    
    print('resp')

def process_ecg():
    print('ecg')
    ecg = np.loadtxt('chest_ecg.txt')
    rsp = np.loadtxt('chest_resp.txt')
    if not (os.path.isdir('chest_ecg')):
        os.makedirs('chest_ecg')

    ecg_window = []
    rsp_window = []
    print(len(ecg))
    i = 0
    files = 0
    for i in range(len(ecg)):
        ecg_window.append(ecg[i])
        rsp_window.append(rsp[i])
        if ((i == len(ecg) and ecg_window) or (i != 0 and (i % registers700) == 0)):
            files += 1
            ecg_features = nk.ecg_process(ecg_window, rsp=rsp_window, sampling_rate=700)
            print(ecg_features)
            break

def process_emg(registers700):
    emg = np.loadtxt('chest_emg_filtered.txt')
    if not (os.path.isdir('chest_emg')):
        os.makedirs('chest_emg')

    emg_window = []
    print(len(emg))
    i = 0
    files = 0
    # for i in range(len(emg)):
    for i in range(registers700):
        emg_window.append(emg[i])
            
        # if ((i == len(emg) and emg_window) or (i != 0 and (i % registers700) == 0)):
        #     emg_features = nk.emg_process(emg=emg_window, sampling_rate=700)
        #     np.savetxt('chest_emg/' + str(files) + 'WINDOW.txt', emg_window, fmt='%f')
        #     files += 1
        #     # np.savetxt('chest_emg/' + str(files) + '_Onsets.txt', np.squeeze(emg_features['EDA']['SCR_Onsets']), fmt='%f')
        #     np.savetxt('chest_emg/' + str(files) + '_peaks_indexes.txt', np.squeeze(eda_features['EDA']['SCR_Peaks_Indexes']), fmt='%f')
        #     np.savetxt('chest_emg/' + str(files) + '_recovery.txt', np.squeeze(eda_features['EDA']['SCR_Recovery_Indexes']), fmt='%f')
        #     np.savetxt('chest_emg/' + str(files) + '_amplitudes.txt', np.squeeze(eda_features['EDA']['SCR_Peaks_Amplitudes']), fmt='%f')
        #     eda_window = []
    
    emg_features = nk.emg_process(emg=emg_window, sampling_rate=700)
    print(emg_features)

def execute(data_set_path):
    os.chdir(path)

    # subs = [3]
    # subs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    # for i in subs:
        # subject = 'S' + str(i)
        # directory = data_set_path + '/' + subject + '/data/raw/'
        # os.chdir(data_set_path)

    # process_eda()
    # process_resp()
    # process_ecg()
    process_emg(registers700)
        


if __name__ == '__main__':
    # path = "/Users/gustavodebiasi/Documents/DadosTCC/data/raw/"
    path = "/Volumes/My Passport/TCC/WESAD/S2/data/raw/"
    window = 20
    registers700 = 700 * window
    registers64 = 64 * window
    registers32 = 32 * window
    registers4 = 4 * window 
    execute(path)
    
    