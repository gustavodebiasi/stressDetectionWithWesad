import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import neurokit as nk
import pandas as pd

def process_eda():
    eda = np.loadtxt('chest_eda.txt')
    if not (os.path.isdir('chest_eda')):
        os.makedirs('chest_eda')

    eda_window = []
    print(len(eda))
    i = 0
    files = 0
    for i in range(len(eda)):
        eda_window.append(eda[i])
        if ((i == len(eda) and eda_window) or (i != 0 and (i % 14000) == 0)):
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
        if ((i == len(rsp) and rsp_window) or (i != 0 and (i % 14000) == 0)):
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
        if ((i == len(ecg) and ecg_window) or (i != 0 and (i % 14000) == 0)):
            files += 1
            ecg_features = nk.ecg_process(ecg_window, rsp=rsp_window, sampling_rate=700)
            print(ecg_features)
            break

def process_emg():
    emg = np.loadtxt('chest_emg.txt')
    if not (os.path.isdir('chest_emg')):
        os.makedirs('chest_emg')

    emg_window = []
    print(len(emg))
    i = 0
    files = 0
    for i in range(14000):
        emg_window.append(emg[i])
        # if ((i == len(emg) and emg_window) or (i != 0 and (i % 14000) == 0)):
        #     files += 1
        #     np.savetxt('chest_emg/' + str(files) + '_Onsets.txt', np.squeeze(eda_features['EDA']['SCR_Onsets']), fmt='%f')
        #     np.savetxt('chest_emg/' + str(files) + '_peaks_indexes.txt', np.squeeze(eda_features['EDA']['SCR_Peaks_Indexes']), fmt='%f')
        #     np.savetxt('chest_emg/' + str(files) + '_recovery.txt', np.squeeze(eda_features['EDA']['SCR_Recovery_Indexes']), fmt='%f')
        #     np.savetxt('chest_emg/' + str(files) + '_amplitudes.txt', np.squeeze(eda_features['EDA']['SCR_Peaks_Amplitudes']), fmt='%f')
        #     eda_window = []
    
    print(min(emg_window))
    print(max(emg_window))
    np.savetxt('chest_emg/' + str(files) + 'WINDOW.txt', emg_window, fmt='%f')
    emg_features = nk.emg_process(emg=emg_window, sampling_rate=700)
    print(emg_window)
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
    process_emg()
        


if __name__ == '__main__':
    path = "/Users/gustavodebiasi/Documents/DadosTCC/data/raw/"
    execute(path)
    '''
    os.chdir(path)
    # dados_ecg = np.loadtxt('chest_ecg.txt')
    # dados_resp = np.loadtxt('chest_resp.txt')
    dados_eda = np.loadtxt('chest_eda.txt')
    novos_dados_eda = []
    # novos_dados_ecg = []
    # novos_dados_resp = []
    i = 0
    for i in range(14000):
        # novos_dados_ecg.append(dados_ecg[i])
        # novos_dados_resp.append(dados_resp[i])
        novos_dados_eda.append(dados_eda[i])
    
    ecg_features = nk.ecg_process(ecg=novos_dados_ecg, rsp=novos_dados_resp, sampling_rate=700)
    eda_features = nk.eda_process(eda=novos_dados_eda, sampling_rate=700)
    print(eda_features['EDA'])
    print(eda_features['EDA']['SCR_Onsets'])
    print(eda_features['EDA']['SCR_Peaks_Indexes'])
    print(eda_features['EDA']['SCR_Recovery_Indexes'])
    print(eda_features['EDA']['SCR_Peaks_Amplitudes'])
    print('-----------------------------------')

    novos_dados_eda2 = []
    for i in range(14000, 28001):
        novos_dados_eda2.append(dados_eda[i])
    
    eda_features = nk.eda_process(eda=novos_dados_eda2, sampling_rate=700)
    print(eda_features['EDA'])
    print(eda_features['EDA']['SCR_Onsets'])
    print(eda_features['EDA']['SCR_Peaks_Indexes'])
    print(eda_features['EDA']['SCR_Recovery_Indexes'])
    print(eda_features['EDA']['SCR_Peaks_Amplitudes'])
    '''
    