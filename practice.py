
# import base64
# import traceback
# import os
# import json
# import time
# from flask import Flask, request, jsonify
# from flask_restful import http_status_message
# import re
# import cv2
# import dlib
# import keras as ks
# import tensorflow as tf

# import time
# import shutil
# import json
# from gevent.pywsgi import WSGIServer
# from flask import Flask, render_template, request
# from flask_socketio import SocketIO, emit
# import os
# import base64
# import numpy as np
# import subprocess
# import threading

# import os
# import cv2
# import math
# from tqdm import tqdm
# import dlib
# import re
# from PIL import Image
# import time
# import torch
# import numpy as np
# import pandas as pd
# import heartpy as hp
# from torch import nn
# from imutils.video import FileVideoStream
# import heartpy as hp
# from scipy import signal
# from scipy import linalg
# from scipy import signal
# import random
# from scipy import sparse
# from scipy import stats
# from scipy import integrate
# from scipy import io as sci
# from scipy import stats as st
# from imutils import face_utils
# import matplotlib.pyplot as plt
# import torch.nn.functional as F
# from skimage.util import img_as_float
# from moviepy.editor import VideoFileClip
# from heartpy.datautils import rolling_mean
# from heartpy.preprocessing import enhance_peaks
# from scipy.signal import butter, filtfilt
# from sklearn.metrics import mean_squared_error
# from sklearn.decomposition import FastICA, PCA
# import PyEMD

# import os
# import cv2
# import math
# import dlib
# import numpy as np
# import pandas as pd
# import heartpy as hp
# import neurokit2 as nk

# from scipy import signal, sparse
# from imutils import face_utils
# import matplotlib.pyplot as plt
# from heartpy.datautils import rolling_mean
# import keras as ks
# from kapre import STFT, Magnitude, MagnitudeToDecibel
# import time
# import tensorflow as tf
# import lightgbm



# import sys
# import json
# import os



# ################################################

# dependencies = {
#         'ReLU': ks.layers.ReLU,
#         'STFT': STFT,
#         'Magnitude': Magnitude,
#         'MagnitudeToDecibel': MagnitudeToDecibel
# }

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('dlib_shape_predictor_68_face_landmarks.dat')

# model_bp = ks.models.load_model('./resnet_ppg_nonmixed.h5', custom_objects=dependencies)

# model_spo2 = tf.keras.models.load_model('./bidmc_hr_FCN_Residual_huber_loss.h5')

# clf_asth = lightgbm.Booster(model_file='lgbr_cust_slicegtppg_91.txt')



# def calc_hr_hp(wave,sampling_rate):
#   wd1,m1=hp.process(wave, sampling_rate, report_time = False)
#   hp_hr=m1['bpm']
#   return hp_hr

# def calc_hp_metrics(wave,sampling_rate):
#   wd1,m1=hp.process(wave, sampling_rate, report_time = False)
#   return m1['ibi'], m1['sdnn'], m1['rmssd'], m1['pnn20'], m1['pnn50'], m1['sdnn']/m1['rmssd'], int(math.floor(m1['breathingrate'] * 60))

# def calc_hr_rr(wave,sampling_rate):
#   rol_mean = rolling_mean(wave, windowsize = 1.05, sample_rate = sampling_rate)
#   wd = hp.peakdetection.fit_peaks(wave, rol_mean, sampling_rate)
#   peak_times = np.array(wd['peaklist']) / sampling_rate
#   inter_peak_intervals = np.diff(peak_times)
#   heart_rate_bpm = 1 / np.median(inter_peak_intervals) * 60
#   return heart_rate_bpm


# def pred_adv(wave):
#   try:
#     global model_bp, model_spo2
#     bp_signal = wave
#     t1=bp_signal[0:875].reshape(1,875,1)
#     sys,di=model_bp.predict(t1)
#     sp_signal = wave
#     t1=sp_signal[0:156].reshape(1,156,1)
#     pred=model_spo2.predict(t1)
#     pred = min([[99.998]],pred)
#     return sys,di,pred
#   except:
#     return np.array([[115]]), np.array([[85]]), np.array([[98]])
    



# def calculate_mode_50ms_bins(ibi):
#     # Bin the data in 50ms bins
#     bins = np.arange(min(ibi), max(ibi) + 50, 50)
#     histogram, bins = np.histogram(ibi, bins=bins)
#     max_count_index = np.argmax(histogram)
#     mode_50ms = (bins[max_count_index] + bins[max_count_index + 1]) / 2
#     return mode_50ms, histogram[max_count_index] / len(ibi) * 100

# def calculate_Stress_Index(wave,sampling_rate):
#     wd, m = hp.process(wave, sample_rate = sampling_rate)
#     peak_indices = wd['peaklist']
#     ibi = np.diff(peak_indices) / sampling_rate #* 1000
#     mode_50ms, amplitude = calculate_mode_50ms_bins(ibi)
#     sdnn = m['sdnn']
#     SI = amplitude / (2 * mode_50ms) * 3.92 * sdnn
#     return SI



# def calculate_cardio_fit(age,heart_rate):
#   MHR = 208-0.7 * age
#   RHR = heart_rate
#   #normal vo2 range >28 for men-women age 20-35
#   Vo2 = 15.3*(MHR/RHR)
#   return Vo2


# def calc_all_params(Age, Gender, Weight, Height, sys, di, heart_rate):
#   Height = Height/100

#   HR = heart_rate
#   HR_MAX = 208-0.7 * Age
#   HR_Reserve = HR_MAX - HR
#   THR = (HR_Reserve * 0.2) + HR

#   Stroke_Volume = (sys-di)*2
#   Cardiac_OP = (Stroke_Volume * HR)/1000

#   Mean_Arterial_Pressure = di + (1/3*(sys-di))

#   heart_utilized = (HR/HR_MAX)*100

#   if Gender == 0:
#     Blood_Volume = ((0.3669 * Height**3) + (0.03219 * Weight) + 0.6041)  * 1000
#   else:
#     Blood_Volume = ((0.3561 * Height**3) + (0.03308 * Weight) + 0.1833)  * 1000

#   if Gender == 0:
#     TBW = 2.447 - 0.09156 * Age + 0.1074 * (Height*100) + 0.3362 * Weight
#   else:
#     TBW = -2.097 + (0.1069*(Height*100)) + (0.2466 * Weight)

#   Body_water = (TBW/Weight)*100

#   BMI = Weight / Height**2

#   Body_Fat = -44.988 + (0.503 * Age) + (10.689 * Gender) + (3.172 * BMI) - (0.026 * BMI**2) + (0.181 * BMI * Gender) - (0.02 * BMI * Age) - (0.005 * BMI**2 * Gender) + (0.00021 * BMI**2 * Age)


#   try:
#     print(str(Cardiac_OP[0, 0]))
#   except:
#     pass
  
#   try:
#     print(str(Mean_Arterial_Pressure[0, 0]))
#   except:
#     pass
  
#   if isinstance(Cardiac_OP, np.ndarray) and isinstance(Mean_Arterial_Pressure, np.ndarray):
#     return [str(HR_MAX), str(HR_Reserve), str(THR), str(Cardiac_OP[0,0]), str(Mean_Arterial_Pressure[0,0]), str(heart_utilized), str(Blood_Volume), str(TBW), str(Body_water), str(BMI), str(Body_Fat)]
  
#   else:
#     return [str(HR_MAX), str(HR_Reserve), str(THR), str(Cardiac_OP), str(Mean_Arterial_Pressure), str(heart_utilized), str(Blood_Volume), str(TBW), str(Body_water), str(BMI), str(Body_Fat)]

  
  

#     # print('Maximum Heart Rate:'+str(HR_MAX)+' bpm')
#     # print('Heart Rate Reserve:'+str(HR_Reserve)+' bpm')
#     # print('Target Heart Rate:'+str(THR)+' bpm')

#     # print('Cardiac Output is: '+str(Cardiac_OP)+' L/min')
#     # print('Mean Arterial Pressure is: '+str(Mean_Arterial_Pressure)+' mmHg')
#     # print('Heart Utilized is: '+str(heart_utilized)+' %')
#     # print('Blood Volume is: '+str(Blood_Volume)+' mL')
#     # print('Total Body Water is: '+str(TBW)+' %')
#     # print('Body Water Percentage is: '+str(Body_water)+' %')
#     # print('Body-Mass Index is: '+str(BMI)+' kg/m2')
#     # print('Body Fat is: '+str(Body_Fat)+' %')



# def calculate_mode_50ms_bins(ibi):
#     # Bin the data in 50ms bins
#     bins = np.arange(min(ibi), max(ibi) + 50, 50)
#     histogram, bins = np.histogram(ibi, bins=bins)
#     max_count_index = np.argmax(histogram)
#     mode_50ms = (bins[max_count_index] + bins[max_count_index + 1]) / 2
#     return mode_50ms, histogram[max_count_index] / len(ibi) * 100

# def calculate_Stress_Index(wave,sampling_rate):
#     wd, m = hp.process(wave, sample_rate = sampling_rate)
#     peak_indices = wd['peaklist']
#     ibi = np.diff(peak_indices) / sampling_rate #* 1000
#     mode_50ms, amplitude = calculate_mode_50ms_bins(ibi)
#     sdnn = m['sdnn']
#     SI = amplitude / (2 * mode_50ms) * 3.92 * sdnn
#     return SI

# #***********    ASTHAMA CODE    *********************************************************************#




# def find_dominant_frequencies(IMFs, sampling_rate):
#     n_imfs, n_samples = IMFs.shape
#     frequencies = np.fft.rfftfreq(n_samples, d=1/sampling_rate)
#     dominant_frequencies = []
#     for imf in IMFs:
#         windowed_imf = imf * np.hanning(n_samples)
#         fft_values = np.fft.rfft(windowed_imf)
#         magnitude = np.abs(fft_values)
#         positive_frequencies = frequencies
#         positive_magnitude = magnitude
#         dominant_frequency_index = np.argmax(positive_magnitude[1:]) + 1
#         dominant_frequency = positive_frequencies[dominant_frequency_index]
#         dominant_frequencies.append(dominant_frequency)
#     return dominant_frequencies

# def filter_respiratory_IMFs(dominant_frequencies, lower_bound=0.1, upper_bound=0.7):
#       respiratory_IMFs_indices = [i for i, freq in enumerate(dominant_frequencies) if lower_bound <= freq <= upper_bound]
#       if len(respiratory_IMFs_indices)<1:
#         respiratory_IMFs_indices = [i for i, freq in enumerate(dominant_frequencies) if 0.015 <= freq <= 0.9]
#       return respiratory_IMFs_indices

# def get_respirate_CEEMDAN_PCA(wave):
#   ceemdan = PyEMD.CEEMDAN(parallel=True, processes=6)
#   ceemdan.noise_seed(seed=2009)
#   imfs = ceemdan(wave)
#   num_imfs, length = imfs.shape

#   IMFs = imfs
#   sampling_rate = 30  # Hz
#   n_imfs, n_samples = IMFs.shape

#   dominant_frequencies = find_dominant_frequencies(IMFs, sampling_rate)
#   respiratory_IMFs_indices = filter_respiratory_IMFs(dominant_frequencies)

#   if len(respiratory_IMFs_indices)>1:
#     temp_pca = np.array(respiratory_IMFs_indices)
#     temp_pca_max = (temp_pca.max()+1)
#     temp_pca_min = temp_pca.min()
#     #print(temp_pca_min,temp_pca_max)
#     selected_MRIs = imfs[temp_pca_min:temp_pca_max, :]
#     #print(selected_MRIs)
#     pca = PCA(n_components=1)
#     PC1 = pca.fit_transform(selected_MRIs.T).flatten()
#     respiratory_wave = PC1
#   elif len(respiratory_IMFs_indices)==1:
#     respiratory_wave = imfs[respiratory_IMFs_indices[0],:]
#   else:
#     respiratory_IMFs_indices = np.argmax(dominant_frequencies)
#     respiratory_wave = imfs[respiratory_IMFs_indices,:]
#   return respiratory_wave
# #-------------------------------------------------------------------------------

# def calc_ins_exp(normalized_signal,start_inspiration,end_inspiration,start_expiration,end_expiration):
#   expi2 = stats.skew(normalized_signal[start_expiration:end_expiration], axis=0, bias=True, nan_policy='propagate', keepdims=False)
#   inspi2 = stats.skew(normalized_signal[start_inspiration:end_inspiration], axis=0, bias=True, nan_policy='propagate', keepdims=False)
#   skew_ratio = round((expi2/inspi2),6)


#   time = np.array(list(range(len(normalized_signal))))
#   amplitude = normalized_signal
#   inspiration_area = integrate.trapz(amplitude[start_inspiration:end_inspiration], time[start_inspiration:end_inspiration])
#   expiration_area = integrate.trapz(amplitude[start_expiration:end_expiration], time[start_expiration:end_expiration])
#   area_ratio = round((expiration_area / inspiration_area),6)

#   inspiration_time = (time[end_inspiration]-time[start_inspiration])/30
#   expiration_time = (time[end_expiration]-time[start_expiration])/30
#   time_ratio = round((expiration_time/inspiration_time),6)

#   return skew_ratio,area_ratio,time_ratio

# def respi_featext_II(b):
#   signal2 = b
#   peak_signal, info = nk.rsp_peaks(signal2,sampling_rate=30,method="khodadad2018", amplitude_min=0.15)
#   min_value = np.min(signal2)
#   max_value = np.max(signal2)
#   normalized_signal = (signal2 - min_value) / (max_value - min_value)
#   normalized_signal = np.array(normalized_signal)

#   q=np.array(peak_signal['RSP_Peaks'])
#   q2=np.where(q==1)
#   qi=np.array(peak_signal['RSP_Troughs'])
#   q2i=np.where(qi==1)
#   q2,q2i

#   inspiration = np.full(len(normalized_signal), np.nan)
#   inspiration[q2] = 1.0
#   inspiration[q2i] = 0.0

#   troughs0=np.where(inspiration==0)
#   peaks0=np.where(inspiration==1)

#   if len(troughs0[0])<len(peaks0[0]):
#     diff=len(peaks0[0]-troughs0[0])
#     peaks0[0]=peaks0[0][:len(troughs0[0])]
#   elif len(troughs0[0])>len(peaks0[0]):
#     diff=len(troughs0[0]-peaks0[0])
#     if diff==1:
#       pass
#     elif diff>1:
#       troughs0[0]=troughs0[0][:len(peaks0[0])+1]
#   else:
#     pass

#   skew_total, area_total, time_total = [], [], []

#   for i in range(len(peaks0[0])):
#     inspi_start = troughs0[0][i]
#     inspi_end, expi_start = peaks0[0][i], peaks0[0][i]
#     if len(troughs0[0])>len(peaks0[0]) or i<len(peaks0[0])-1:
#       expi_end = troughs0[0][i+1]
#       buffer_oe =  expi_end-expi_start
#     elif i==len(peaks0[0])-1:
#       expi_end = expi_start+buffer_oe
#       if expi_end>len(normalized_signal):
#         expi_end=len(normalized_signal)-2

#     skew_ratio, area_ratio, time_ratio = calc_ins_exp(normalized_signal,inspi_start,inspi_end,expi_start,expi_end)
#     skew_total.append(skew_ratio)
#     area_total.append(area_ratio)
#     time_total.append(time_ratio)

#   skew_total = np.array(skew_total).mean()
#   area_total = np.array(area_total).mean()
#   time_total = np.array(time_total).mean()

#   if math.isnan(skew_total) or math.isnan(area_total) or math.isnan(time_total) :
#     return 999,999,999,999
#   skew_total = round(skew_total,5)
#   area_total = round(area_total,5)
#   time_total = round(time_total,5)

#   sig_kurtosis = stats.kurtosis(normalized_signal, axis=0, fisher=False, bias=True, nan_policy='propagate', keepdims=False)
#   sig_kurtosis = round(sig_kurtosis,5)

#   return sig_kurtosis,skew_total,area_total,time_total


# def get_asthama_riskscore(ppg_wave):
#   global clf_asth
#   respi_wave = get_respirate_CEEMDAN_PCA(ppg_wave)
#   tt_kurtosis, tt_skewr, tt_area, tt_time = respi_featext_II(respi_wave)
#   if tt_kurtosis==999 or tt_skewr==999:
#     y_pred_proba = 10
#   else:
#     ip_sample = []
#     ip_sample.append(tt_kurtosis)
#     ip_sample.append(tt_skewr)
#     ip_sample.append(tt_area)
#     ip_sample.append(tt_time)
#     for i in range(len(respi_wave)):
#       ip_sample.append(respi_wave[i])
#     if len(ip_sample)>904:
#       ip_sample=ip_sample[0:904]
#     elif len(ip_sample)<904:
#       padding_size = 904-len(ip_sample)
#       ip_sample=np.pad(ip_sample, (0, padding_size), mode='constant', constant_values=0)
#     ip_sample=np.array(ip_sample).reshape(1,len(ip_sample))
#     score_temp = clf_asth.predict(ip_sample, raw_score=False, pred_leaf=False, pred_contrib=False)
#     y_pred_proba = (round(score_temp[0]*100,4))
#   return y_pred_proba


# #********************************************************************************#



# def detrend(input_signal, lambda_value):
#     signal_length = input_signal.shape[0]
#     # observation matrix
#     H = np.identity(signal_length)
#     ones = np.ones(signal_length)
#     minus_twos = -2 * np.ones(signal_length)
#     diags_data = np.array([ones, minus_twos, ones])
#     diags_index = np.array([0, 1, 2])
#     D = sparse.spdiags(diags_data, diags_index,
#                 (signal_length - 2), signal_length).toarray()
#     filtered_signal = np.dot(
#         (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
#     return filtered_signal


# def _process_video(frames):
#     RGB = []
#     for frame in frames:
#         summation = np.sum(np.sum(frame, axis=0), axis=0)
#         RGB.append(summation / (frame.shape[0] * frame.shape[1]))
#     return np.asarray(RGB)

# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'secret!'
# socketio = SocketIO(app, cors_allowed_origins="*", pingTimeout=60000, pingInterval=20000)



# def POS_WANG(frames, fs):
#     WinSec = 32 / fs
#     RGB = _process_video(frames)
#     N = RGB.shape[0]
#     H = np.zeros((1, N))
#     l = math.ceil(WinSec * fs)

#     for n in range(N):
#         m = n - l
#         if m >= 0:
#             Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
#             Cn = np.mat(Cn).H
#             S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
#             h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
#             mean_h = np.mean(h)
#             for temp in range(h.shape[1]):
#                 h[0, temp] = (h[0, temp] - mean_h)
#             H[0, m:n] = H[0, m:n] + (h[0])
#     BVP = H
#     BVP = detrend(np.mat(BVP).H, 100)
#     BVP = np.asarray(np.transpose(BVP))[0]
#     b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
#     BVP = signal.filtfilt(b, a, BVP.astype(np.double))
#     return BVP
  
  
# def POS_WANG2(frames, fs):
#     WinSec = 32 / fs
#     RGB = _process_video(frames)
#     N = RGB.shape[0]
#     H = np.zeros((1, N))
#     l = math.ceil(WinSec * fs)

#     for n in range(N):
#         m = n - l
#         if m >= 0:
#             Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
#             Cn = np.mat(Cn).H
#             S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
#             h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
#             mean_h = np.mean(h)
#             for temp in range(h.shape[1]):
#                 h[0, temp] = (h[0, temp] - mean_h) / np.std(h)
#             H[0, m:n] = H[0, m:n] + (h[0])
#     BVP = H
#     BVP = detrend(np.mat(BVP).H, 100)
#     BVP = np.asarray(np.transpose(BVP))[0]
#     b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
#     BVP = signal.filtfilt(b, a, BVP.astype(np.double))
#     return BVP


# def get_ROI(frames):
#   left_cheek_frames = []
#   right_cheek_frames = []
#   # filename = input_file
#   # st = time.time()
#   # v_cap = FileVideoStream(filename).start()
#   # v_len = int(v_cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))
#   # v_fps = int(v_cap.stream.get(cv2.CAP_PROP_FPS))
#   # frames = []

#   v_len = len(frames)
#   v_fps = 30
#   frame_counter=0
#   batch_size = 4
#   frame_flag_start = 0
#   frame_flag_end = 0
#   collection_frames = []
#   collection_frames_fin = []

#   stride_fin = int(v_fps)*2

#   num_iterations = 0
#   num_iterations_face = 0
#   flag_done=0
#   LAST_DET=False
#   NO_DETECT_COUNT=1
#   dup_res_inx=0
#   prec_inx=0
#   face_nan=0

#   def detect_faces(fr):
      
#     faces = detector(fr)
#     if len(faces)==0:
#       return 0
#     else:
#       shape = predictor(fr, faces[0])
#       shape = face_utils.shape_to_np(shape)
#       RC_r = (shape[29][1],shape[33][1]),(shape[4][0],shape[12][0])
#       LC_r = (shape[29][1],shape[33][1]),(shape[4][0],shape[48][0])
#       return RC_r, LC_r

#   def concatenate_slices(frame, coords1, coords2):
#       slice1 = frame[coords1[0]:coords1[1], coords1[2]:coords1[3]]
#       return slice1

#   for j in range(0,v_len):
#       frame = frames[j]
#       # frame = v_cap.read()
#       # frame  = cv2.resize(frame,(230,300)) #230,300
#       # cv2_imshow(frame)
#       # break
#       frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)#cv2.ROTATE_180) #cv2.ROTATE_90_COUNTERCLOCKWISE)
#       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#       num_iterations+=1

#       if j % 15 == 0:
#         NO_DETECT_COUNT=1
#       else:
#         NO_DETECT_COUNT=0

#       if j % stride_fin == 0 or frame_flag_start==0 or NO_DETECT_COUNT==1:

#         result_indexes = detect_faces(gray)

#         if result_indexes!=0:
#           if j % stride_fin == 0:
#             num_iterations_face+=1
#             frame_flag_start=99
#             prec_inx = result_indexes
#           else:
#             result_indexes=prec_inx


#         else:
#           result_indexes=dup_res_inx
#           collection_frames.append(frame)
#           face_nan+=1
#           continue

#       if frame_flag_start==99 or frame_flag_start == 999:
#         collection_frames.append(frame)
#         temp1 = j+1
#         if temp1 % stride_fin == 0 or LAST_DET == True:
#           frame_flag_end = 99
#           if (v_len-j) <= stride_fin and LAST_DET == False :
#             LAST_DET = True
#             frame_flag_start = 0

#           R_row_start = result_indexes[0][0][0]
#           R_row_end = result_indexes[0][0][1]
#           R_col_start = result_indexes[0][1][0]
#           R_col_end = result_indexes[0][1][1]

#           L_row_start = result_indexes[1][0][0]
#           L_row_end = result_indexes[1][0][1]
#           L_col_start = result_indexes[1][1][0]
#           L_col_end = result_indexes[1][1][1]
#           for k in range(len(collection_frames)):
#             collection_frames_fin.append(concatenate_slices(collection_frames[k],(R_row_start,R_row_end,R_col_start,R_col_end),(L_row_start,L_row_end,L_col_start,L_col_end)))
#           collection_frames = []
#           dup_res_inx = result_indexes
#           # break
#       if face_nan > ((v_len//15)*0.6):
#         print('Adjust Face: Face not founddd')
#         break
#   print(len(collection_frames_fin))
#   return v_fps,collection_frames_fin


# ################################################

# def read_png_frames(folder_path):
#     # Get all PNG files and sort them by the numeric part of the filename
#     png_files = sorted([file for file in os.listdir(folder_path) if file.endswith('.png')],
#                        key=lambda x: int(re.search(r'(\d+)', x).group(1)))

#     frames = []

#     for png_file in png_files:
#         try:
#             file_path = os.path.join(folder_path, png_file)
#             print(file_path)
#             image = cv2.imread(file_path)
#             # print(type(image))
#             # frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
#             frames.append(image)
#         except:
#             pass

#     return frames





# ############################################################################


# def process_text_file(sessionDirPath):
    
#     if ( sessionDirPath in os.listdir('./')):
        
#         frames = read_png_frames(f'./{sessionDirPath}')
        
#         _, frames = get_ROI(frames)

#         sampling_rate = 30
#         wave = POS_WANG(frames, sampling_rate)

#         heart_rate_bpm = calc_hr_rr(wave, sampling_rate=sampling_rate)
#         ibi, sdnn, rmssd, pnn20, pnn50, hrv, rr = calc_hp_metrics(wave, sampling_rate)
#         sysbp, diabp, spo2 = pred_adv(wave)

#         age = 21
#         gender = 0
#         weight = 71
#         height = 172

#         vo2max = calculate_cardio_fit(age,heart_rate_bpm)
#         [mhr, hrr, thr, co, map, 
#         hu, bv, tbw, bwp, bmi, bf] = calc_all_params(age, gender, weight, height, sysbp, diabp, heart_rate_bpm)
#         si = calculate_Stress_Index(wave, sampling_rate)
#         wave2 = POS_WANG2(frames, sampling_rate)
#         asth_rs = get_asthama_riskscore(wave2)
        
        
#         # print({
#         #     'hr': (math.floor(heart_rate_bpm)),  
#         #     "ibi": (round(float(ibi), 1)), 
#         #     "sdnn": (round(float(sdnn), 1)), 
#         #     "rmssd": (round(float(rmssd), 1)), 
#         #     "pnn20": (round(float(pnn20) * 100, 1)), 
#         #     "pnn50": (round(float(pnn50) * 100, 1)), 
#         #     "hrv": (hrv),
#         #     "rr": (round(float(rr), 2)), 
#         #     "sysbp": (math.floor(sysbp[0, 0])), 
#         #     "diabp": (math.floor(diabp[0,0])), 
#         #     # "spo2": (math.floor(spo2[0 ,0])),
#         #     "vo2max": (round(vo2max, 1)), 
#         #     "si": (round(si, 1)), 
#         #     "mhr": (math.floor(float(mhr))), 
#         #     "hrr": (math.floor(float(hrr))), 
#         #     "thr": (math.floor(float(thr))), 
#         #     "co": (round(float(co), 1)),
#         #     "map": (round(float(map), 1)), 
#         #     "hu": (round(float(hu), 1)), 
#         #     "bv": (bv), 
#         #     "tbw": (float(tbw)), 
#         #     "bwp": (round(float(bwp), 1)), 
#         #     "bmi": (round(float(bmi), 1)), 
#         #     "bf": (round(float(bf), 1)), 
#         #     "asth_risk": round(float(asth_rs),1)
        
#         # })
        
    
    
#     fileName = f'./{sessionDirPath}_results.txt'
#     filePath = os.path.join(sessionDirPath, fileName)

#     # Process text data and generate JSON
    
#     if isinstance(co, np.ndarray) and isinstance(map, np.ndarray):
      
#       if isinstance(spo2, np.ndarray):
        
#         jsonData = {
#                 'hr': (math.floor(heart_rate_bpm)),  
#                 "ibi": (round(float(ibi), 1)), 
#                 "sdnn": (round(float(sdnn), 1)), 
#                 "rmssd": (round(float(rmssd), 1)), 
#                 "pnn20": (round(float(pnn20) * 100, 1)), 
#                 "pnn50": (round(float(pnn50) * 100, 1)), 
#                 "hrv": (hrv),
#                 "rr": (round(float(rr), 2)), 
#                 "sysbp": (math.floor(sysbp[0, 0])), 
#                 "diabp": (math.floor(diabp[0,0])), 
#                 "spo2": (math.floor(spo2[0 ,0])),
#                 "vo2max": (round(vo2max, 1)), 
#                 "si": (round(si, 1)), 
#                 "mhr": (math.floor(float(mhr))), 
#                 "hrr": (math.floor(float(hrr))), 
#                 "thr": (math.floor(float(thr))), 
#                 "co": (round(float(co[0,0]), 1)),
#                 "map": (round(float(map[0,0]), 1)), 
#                 "hu": (round(float(hu), 1)), 
#                 "bv": round(float(bv)), 
#                 "tbw": (float(tbw)), 
#                 "bwp": (round(float(bwp), 1)), 
#                 "bmi": (round(float(bmi), 1)), 
#                 "bf": (round(float(bf), 1)), 
#                 "asth_risk": round(float(asth_rs),1)
            
#             }
        
#         print(f'Writing json file')

#         jsonFileName = fileName.replace('.txt', '.json')
#         jsonFilePath = os.path.join(sessionDirPath, jsonFileName)

#         with open(jsonFilePath, 'w') as file:
#             json.dump(jsonData, file)
            
#       else:
#         jsonData = {
#                 'hr': (math.floor(heart_rate_bpm)),  
#                 "ibi": (round(float(ibi), 1)), 
#                 "sdnn": (round(float(sdnn), 1)), 
#                 "rmssd": (round(float(rmssd), 1)), 
#                 "pnn20": (round(float(pnn20) * 100, 1)), 
#                 "pnn50": (round(float(pnn50) * 100, 1)), 
#                 "hrv": (hrv),
#                 "rr": (round(float(rr), 2)), 
#                 "sysbp": (math.floor(sysbp[0, 0])), 
#                 "diabp": (math.floor(diabp[0,0])), 
#                 "spo2": (math.floor(spo2)),
#                 "vo2max": (round(vo2max, 1)), 
#                 "si": (round(si, 1)), 
#                 "mhr": (math.floor(float(mhr))), 
#                 "hrr": (math.floor(float(hrr))), 
#                 "thr": (math.floor(float(thr))), 
#                 "co": (round(float(co[0,0]), 1)),
#                 "map": (round(float(map[0,0]), 1)), 
#                 "hu": (round(float(hu), 1)), 
#                 "bv": round(float(bv)), 
#                 "tbw": (float(tbw)), 
#                 "bwp": (round(float(bwp), 1)), 
#                 "bmi": (round(float(bmi), 1)), 
#                 "bf": (round(float(bf), 1)), 
#                 "asth_risk": round(float(asth_rs),1)
            
#             }
        
#         print(f'Writing json file')

#         jsonFileName = fileName.replace('.txt', '.json')
#         jsonFilePath = os.path.join(sessionDirPath, jsonFileName)

#         with open(jsonFilePath, 'w') as file:
#             json.dump(jsonData, file)
#     else:
      
#       if isinstance(spo2, np.ndarray):
        
#         jsonData = {
#                 'hr': (math.floor(heart_rate_bpm)),  
#                 "ibi": (round(float(ibi), 1)), 
#                 "sdnn": (round(float(sdnn), 1)), 
#                 "rmssd": (round(float(rmssd), 1)), 
#                 "pnn20": (round(float(pnn20) * 100, 1)), 
#                 "pnn50": (round(float(pnn50) * 100, 1)), 
#                 "hrv": (hrv),
#                 "rr": (round(float(rr), 2)), 
#                 "sysbp": (math.floor(sysbp[0, 0])), 
#                 "diabp": (math.floor(diabp[0,0])), 
#                 "spo2": (math.floor(spo2[0 ,0])),
#                 "vo2max": (round(vo2max, 1)), 
#                 "si": (round(si, 1)), 
#                 "mhr": (math.floor(float(mhr))), 
#                 "hrr": (math.floor(float(hrr))), 
#                 "thr": (math.floor(float(thr))), 
#                 "co": (round(float(co), 1)),
#                 "map": (round(float(map), 1)), 
#                 "hu": (round(float(hu), 1)), 
#                 "bv": round(float(bv)), 
#                 "tbw": (float(tbw)), 
#                 "bwp": (round(float(bwp), 1)), 
#                 "bmi": (round(float(bmi), 1)), 
#                 "bf": (round(float(bf), 1)), 
#                 "asth_risk": round(float(asth_rs),1)
            
#             }
        
#         print(f'Writing json file')

#         jsonFileName = fileName.replace('.txt', '.json')
#         jsonFilePath = os.path.join(sessionDirPath, jsonFileName)

#         with open(jsonFilePath, 'w') as file:
#             json.dump(jsonData, file)
            
#       else:
#         jsonData = {
#                 'hr': (math.floor(heart_rate_bpm)),  
#                 "ibi": (round(float(ibi), 1)), 
#                 "sdnn": (round(float(sdnn), 1)), 
#                 "rmssd": (round(float(rmssd), 1)), 
#                 "pnn20": (round(float(pnn20) * 100, 1)), 
#                 "pnn50": (round(float(pnn50) * 100, 1)), 
#                 "hrv": (hrv),
#                 "rr": (round(float(rr), 2)), 
#                 "sysbp": (math.floor(sysbp[0, 0])), 
#                 "diabp": (math.floor(diabp[0,0])), 
#                 "spo2": (math.floor(spo2)),
#                 "vo2max": (round(vo2max, 1)), 
#                 "si": (round(si, 1)), 
#                 "mhr": (math.floor(float(mhr))), 
#                 "hrr": (math.floor(float(hrr))), 
#                 "thr": (math.floor(float(thr))), 
#                 "co": (round(float(co), 1)),
#                 "map": (round(float(map), 1)), 
#                 "hu": (round(float(hu), 1)), 
#                 "bv": round(float(bv)), 
#                 "tbw": (float(tbw)), 
#                 "bwp": (round(float(bwp), 1)), 
#                 "bmi": (round(float(bmi), 1)), 
#                 "bf": (round(float(bf), 1)), 
#                 "asth_risk": round(float(asth_rs),1)
            
#             }
        
#         print(f'Writing json file')

#         jsonFileName = fileName.replace('.txt', '.json')
#         jsonFilePath = os.path.join(sessionDirPath, jsonFileName)

#         with open(jsonFilePath, 'w') as file:
#             json.dump(jsonData, file)
# if __name__ == "__main__":
#     sessionDirPath = sys.argv[1]

#     process_text_file(sessionDirPath)



########## END OF JS-PYTHON BACKEND #########


import base64
import traceback
import os
import json
import time
from flask import Flask, request, jsonify
from flask_restful import http_status_message
import re
import cv2
import dlib
import keras as ks
import tensorflow as tf

import time
import shutil
import json
from gevent.pywsgi import WSGIServer
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import os
import base64
import numpy as np
import subprocess
import threading

import os
import cv2
import math
from tqdm import tqdm
import dlib
import re
from PIL import Image
import time
import torch
import numpy as np
import pandas as pd
import heartpy as hp
from torch import nn
from imutils.video import FileVideoStream
import heartpy as hp
from scipy import signal
from scipy import linalg
from scipy import signal
import random
from scipy import sparse
from scipy import stats
from scipy import integrate
from scipy import io as sci
from scipy import stats as st
from imutils import face_utils
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage.util import img_as_float
from heartpy.datautils import rolling_mean
from heartpy.preprocessing import enhance_peaks
from scipy.signal import butter, filtfilt
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import FastICA, PCA
import PyEMD

import os
import cv2
import math
import dlib
import numpy as np
import pandas as pd
import heartpy as hp
import neurokit2 as nk

from scipy import signal, sparse
from imutils import face_utils
import matplotlib.pyplot as plt
from heartpy.datautils import rolling_mean
import keras as ks
from kapre import STFT, Magnitude, MagnitudeToDecibel
import time
import tensorflow as tf
import lightgbm


app = Flask(__name__)

################################################

dependencies = {
        'ReLU': ks.layers.ReLU,
        'STFT': STFT,
        'Magnitude': Magnitude,
        'MagnitudeToDecibel': MagnitudeToDecibel
}

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('dlib_shape_predictor_68_face_landmarks.dat')

model_bp = ks.models.load_model('./resnet_ppg_nonmixed.h5', custom_objects=dependencies)

model_spo2 = tf.keras.models.load_model('./bidmc_hr_FCN_Residual_huber_loss.h5')

clf_asth = lightgbm.Booster(model_file='lgbr_cust_slicegtppg_91.txt')



def calc_hr_hp(wave,sampling_rate=30):
  wd1,m1=hp.process(wave, sampling_rate, report_time = False)
  hp_hr=m1['bpm']
  return hp_hr

def calc_hp_metrics(wave,sampling_rate=30):
  wd1,m1=hp.process(wave, sampling_rate, report_time = False)
  return m1['ibi'], m1['sdnn'], m1['rmssd'], m1['pnn20'], m1['pnn50'], m1['sdnn']/m1['rmssd'], int(math.floor(m1['breathingrate'] * 60))

def calc_hr_rr(wave,sampling_rate=30):
  rol_mean = rolling_mean(wave, windowsize = 1.05, sample_rate = sampling_rate)
  wd = hp.peakdetection.fit_peaks(wave, rol_mean, sampling_rate)
  peak_times = np.array(wd['peaklist']) / sampling_rate
  inter_peak_intervals = np.diff(peak_times)
  heart_rate_bpm = 1 / np.median(inter_peak_intervals) * 60
  return heart_rate_bpm


def pred_adv(wave):
    global model_bp, model_spo2
    bp_signal = wave
    t1=bp_signal[0:875].reshape(1,875,1)
    sys,di=model_bp.predict(t1)
    sp_signal = wave
    t1=sp_signal[0:156].reshape(1,156,1)
    pred=model_spo2.predict(t1)
    pred = min([[99.998]],pred)
    return sys,di,pred



def calculate_mode_50ms_bins(ibi):
    # Bin the data in 50ms bins
    bins = np.arange(min(ibi), max(ibi) + 50, 50)
    histogram, bins = np.histogram(ibi, bins=bins)
    max_count_index = np.argmax(histogram)
    mode_50ms = (bins[max_count_index] + bins[max_count_index + 1]) / 2
    return mode_50ms, histogram[max_count_index] / len(ibi) * 100

def calculate_Stress_Index(wave,sampling_rate):
    wd, m = hp.process(wave, sample_rate = sampling_rate)
    peak_indices = wd['peaklist']
    ibi = np.diff(peak_indices) / sampling_rate #* 1000
    mode_50ms, amplitude = calculate_mode_50ms_bins(ibi)
    sdnn = m['sdnn']
    SI = amplitude / (2 * mode_50ms) * 3.92 * sdnn
    return SI



def calculate_cardio_fit(age,heart_rate):
  MHR = 208-0.7 * age
  RHR = heart_rate
  #normal vo2 range >28 for men-women age 20-35
  Vo2 = 15.3*(MHR/RHR)
  return Vo2


def calc_all_params(Age, Gender, Weight, Height, sys, di, heart_rate):
  Height = Height/100
  try:
    HR = heart_rate
    HR_MAX = 208-0.7 * Age
    HR_Reserve = HR_MAX - HR
    THR = (HR_Reserve * 0.2) + HR

    Stroke_Volume = (sys-di)*2
    Cardiac_OP = (Stroke_Volume * HR)/1000

    Mean_Arterial_Pressure = di + (1/3*(sys-di))

    heart_utilized = (HR/HR_MAX)*100

    if Gender == 0:
      Blood_Volume = ((0.3669 * Height**3) + (0.03219 * Weight) + 0.6041)  * 1000
    else:
      Blood_Volume = ((0.3561 * Height**3) + (0.03308 * Weight) + 0.1833)  * 1000

    if Gender == 0:
      TBW = 2.447 - 0.09156 * Age + 0.1074 * (Height*100) + 0.3362 * Weight
    else:
      TBW = -2.097 + (0.1069*(Height*100)) + (0.2466 * Weight)

    Body_water = (TBW/Weight)*100

    BMI = Weight / Height**2

    Body_Fat = -44.988 + (0.503 * Age) + (10.689 * Gender) + (3.172 * BMI) - (0.026 * BMI*2) + (0.181 * BMI * Gender) - (0.02 * BMI * Age) - (0.005 * BMI * Gender) + (0.00021 * BMI*2 * Age)



    return [str(HR_MAX), str(HR_Reserve), str(THR), str(Cardiac_OP[0, 0]), str(Mean_Arterial_Pressure[0, 0])
            , str(heart_utilized), str(Blood_Volume), str(TBW), str(Body_water), str(BMI), str(Body_Fat)]

    # print('Maximum Heart Rate:'+str(HR_MAX)+' bpm')
    # print('Heart Rate Reserve:'+str(HR_Reserve)+' bpm')
    # print('Target Heart Rate:'+str(THR)+' bpm')

    # print('Cardiac Output is: '+str(Cardiac_OP)+' L/min')
    # print('Mean Arterial Pressure is: '+str(Mean_Arterial_Pressure)+' mmHg')
    # print('Heart Utilized is: '+str(heart_utilized)+' %')
    # print('Blood Volume is: '+str(Blood_Volume)+' mL')
    # print('Total Body Water is: '+str(TBW)+' %')
    # print('Body Water Percentage is: '+str(Body_water)+' %')
    # print('Body-Mass Index is: '+str(BMI)+' kg/m2')
    # print('Body Fat is: '+str(Body_Fat)+' %')

  except:
    print('Check your input prameters and try again!')


def calculate_mode_50ms_bins(ibi):
    # Bin the data in 50ms bins
    bins = np.arange(min(ibi), max(ibi) + 50, 50)
    histogram, bins = np.histogram(ibi, bins=bins)
    max_count_index = np.argmax(histogram)
    mode_50ms = (bins[max_count_index] + bins[max_count_index + 1]) / 2
    return mode_50ms, histogram[max_count_index] / len(ibi) * 100

def calculate_Stress_Index(wave,sampling_rate):
    wd, m = hp.process(wave, sample_rate = sampling_rate)
    peak_indices = wd['peaklist']
    ibi = np.diff(peak_indices) / sampling_rate #* 1000
    mode_50ms, amplitude = calculate_mode_50ms_bins(ibi)
    sdnn = m['sdnn']
    SI = amplitude / (2 * mode_50ms) * 3.92 * sdnn
    return SI

#****    ASTHAMA CODE    ************************#




def find_dominant_frequencies(IMFs, sampling_rate):
    n_imfs, n_samples = IMFs.shape
    frequencies = np.fft.rfftfreq(n_samples, d=1/sampling_rate)
    dominant_frequencies = []
    for imf in IMFs:
        windowed_imf = imf * np.hanning(n_samples)
        fft_values = np.fft.rfft(windowed_imf)
        magnitude = np.abs(fft_values)
        positive_frequencies = frequencies
        positive_magnitude = magnitude
        dominant_frequency_index = np.argmax(positive_magnitude[1:]) + 1
        dominant_frequency = positive_frequencies[dominant_frequency_index]
        dominant_frequencies.append(dominant_frequency)
    return dominant_frequencies

def filter_respiratory_IMFs(dominant_frequencies, lower_bound=0.1, upper_bound=0.7):
      respiratory_IMFs_indices = [i for i, freq in enumerate(dominant_frequencies) if lower_bound <= freq <= upper_bound]
      if len(respiratory_IMFs_indices)<1:
        respiratory_IMFs_indices = [i for i, freq in enumerate(dominant_frequencies) if 0.015 <= freq <= 0.9]
      return respiratory_IMFs_indices

def get_respirate_CEEMDAN_PCA(wave):
  ceemdan = PyEMD.CEEMDAN(parallel=True, processes=6)
  ceemdan.noise_seed(seed=2009)
  imfs = ceemdan(wave)
  num_imfs, length = imfs.shape

  IMFs = imfs
  sampling_rate = 30  # Hz
  n_imfs, n_samples = IMFs.shape

  dominant_frequencies = find_dominant_frequencies(IMFs, sampling_rate)
  respiratory_IMFs_indices = filter_respiratory_IMFs(dominant_frequencies)

  if len(respiratory_IMFs_indices)>1:
    temp_pca = np.array(respiratory_IMFs_indices)
    temp_pca_max = (temp_pca.max()+1)
    temp_pca_min = temp_pca.min()
    #print(temp_pca_min,temp_pca_max)
    selected_MRIs = imfs[temp_pca_min:temp_pca_max, :]
    #print(selected_MRIs)
    pca = PCA(n_components=1)
    PC1 = pca.fit_transform(selected_MRIs.T).flatten()
    respiratory_wave = PC1
  elif len(respiratory_IMFs_indices)==1:
    respiratory_wave = imfs[respiratory_IMFs_indices[0],:]
  else:
    respiratory_IMFs_indices = np.argmax(dominant_frequencies)
    respiratory_wave = imfs[respiratory_IMFs_indices,:]
  return respiratory_wave
#-------------------------------------------------------------------------------

def calc_ins_exp(normalized_signal,start_inspiration,end_inspiration,start_expiration,end_expiration):
  expi2 = stats.skew(normalized_signal[start_expiration:end_expiration], axis=0, bias=True, nan_policy='propagate', keepdims=False)
  inspi2 = stats.skew(normalized_signal[start_inspiration:end_inspiration], axis=0, bias=True, nan_policy='propagate', keepdims=False)
  skew_ratio = round((expi2/inspi2),6)


  time = np.array(list(range(len(normalized_signal))))
  amplitude = normalized_signal
  inspiration_area = integrate.trapz(amplitude[start_inspiration:end_inspiration], time[start_inspiration:end_inspiration])
  expiration_area = integrate.trapz(amplitude[start_expiration:end_expiration], time[start_expiration:end_expiration])
  area_ratio = round((expiration_area / inspiration_area),6)

  inspiration_time = (time[end_inspiration]-time[start_inspiration])/30
  expiration_time = (time[end_expiration]-time[start_expiration])/30
  time_ratio = round((expiration_time/inspiration_time),6)

  return skew_ratio,area_ratio,time_ratio

def respi_featext_II(b):
  signal2 = b
  peak_signal, info = nk.rsp_peaks(signal2,sampling_rate=30,method="khodadad2018", amplitude_min=0.15)
  min_value = np.min(signal2)
  max_value = np.max(signal2)
  normalized_signal = (signal2 - min_value) / (max_value - min_value)
  normalized_signal = np.array(normalized_signal)

  q=np.array(peak_signal['RSP_Peaks'])
  q2=np.where(q==1)
  qi=np.array(peak_signal['RSP_Troughs'])
  q2i=np.where(qi==1)
  q2,q2i

  inspiration = np.full(len(normalized_signal), np.nan)
  inspiration[q2] = 1.0
  inspiration[q2i] = 0.0

  troughs0=np.where(inspiration==0)
  peaks0=np.where(inspiration==1)

  if len(troughs0[0])<len(peaks0[0]):
    diff=len(peaks0[0]-troughs0[0])
    peaks0[0]=peaks0[0][:len(troughs0[0])]
  elif len(troughs0[0])>len(peaks0[0]):
    diff=len(troughs0[0]-peaks0[0])
    if diff==1:
      pass
    elif diff>1:
      troughs0[0]=troughs0[0][:len(peaks0[0])+1]
  else:
    pass

  skew_total, area_total, time_total = [], [], []

  for i in range(len(peaks0[0])):
    inspi_start = troughs0[0][i]
    inspi_end, expi_start = peaks0[0][i], peaks0[0][i]
    if len(troughs0[0])>len(peaks0[0]) or i<len(peaks0[0])-1:
      expi_end = troughs0[0][i+1]
      buffer_oe =  expi_end-expi_start
    elif i==len(peaks0[0])-1:
      expi_end = expi_start+buffer_oe
      if expi_end>len(normalized_signal):
        expi_end=len(normalized_signal)-2

    skew_ratio, area_ratio, time_ratio = calc_ins_exp(normalized_signal,inspi_start,inspi_end,expi_start,expi_end)
    skew_total.append(skew_ratio)
    area_total.append(area_ratio)
    time_total.append(time_ratio)

  skew_total = np.array(skew_total).mean()
  area_total = np.array(area_total).mean()
  time_total = np.array(time_total).mean()

  if math.isnan(skew_total) or math.isnan(area_total) or math.isnan(time_total) :
    return 999,999,999,999
  skew_total = round(skew_total,5)
  area_total = round(area_total,5)
  time_total = round(time_total,5)

  sig_kurtosis = stats.kurtosis(normalized_signal, axis=0, fisher=False, bias=True, nan_policy='propagate', keepdims=False)
  sig_kurtosis = round(sig_kurtosis,5)

  return sig_kurtosis,skew_total,area_total,time_total


def get_asthama_riskscore(ppg_wave):
  global clf_asth
  respi_wave = get_respirate_CEEMDAN_PCA(ppg_wave)
  tt_kurtosis, tt_skewr, tt_area, tt_time = respi_featext_II(respi_wave)
  if tt_kurtosis==999 or tt_skewr==999:
    y_pred_proba = 10
  else:
    ip_sample = []
    ip_sample.append(tt_kurtosis)
    ip_sample.append(tt_skewr)
    ip_sample.append(tt_area)
    ip_sample.append(tt_time)
    for i in range(len(respi_wave)):
      ip_sample.append(respi_wave[i])
    if len(ip_sample)>904:
      ip_sample=ip_sample[0:904]
    elif len(ip_sample)<904:
      padding_size = 904-len(ip_sample)
      ip_sample=np.pad(ip_sample, (0, padding_size), mode='constant', constant_values=0)
    ip_sample=np.array(ip_sample).reshape(1,len(ip_sample))
    score_temp = clf_asth.predict(ip_sample, raw_score=False, pred_leaf=False, pred_contrib=False)
    y_pred_proba = (round(score_temp[0]*100,4))
  return y_pred_proba


#****************************#



def detrend(input_signal, lambda_value):
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return filtered_signal


def _process_video(frames):
    RGB = []
    for frame in frames:
        summation = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(summation / (frame.shape[0] * frame.shape[1]))
    return np.asarray(RGB)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", pingTimeout=60000, pingInterval=20000)

def POS_WANG(frames, fs):
    WinSec = 32 / fs
    RGB = _process_video(frames)
    N = RGB.shape[0]
    H = np.zeros((1, N))
    l = math.ceil(WinSec * fs)

    for n in range(N):
        m = n - l
        if m >= 0:
            Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
            Cn = np.mat(Cn).H
            S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
            h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
            mean_h = np.mean(h)
            for temp in range(h.shape[1]):
                h[0, temp] = (h[0, temp] - mean_h)
            H[0, m:n] = H[0, m:n] + (h[0])
    BVP = H
    BVP = detrend(np.mat(BVP).H, 100)
    BVP = np.asarray(np.transpose(BVP))[0]
    b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
    BVP = signal.filtfilt(b, a, BVP.astype(np.double))
    return BVP
  
  
def POS_WANG2(frames, fs):
    WinSec = 32 / fs
    RGB = _process_video(frames)
    N = RGB.shape[0]
    H = np.zeros((1, N))
    l = math.ceil(WinSec * fs)

    for n in range(N):
        m = n - l
        if m >= 0:
            Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
            Cn = np.mat(Cn).H
            S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
            h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
            mean_h = np.mean(h)
            for temp in range(h.shape[1]):
                h[0, temp] = (h[0, temp] - mean_h) / np.std(h)
            H[0, m:n] = H[0, m:n] + (h[0])
    BVP = H
    BVP = detrend(np.mat(BVP).H, 100)
    BVP = np.asarray(np.transpose(BVP))[0]
    b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
    BVP = signal.filtfilt(b, a, BVP.astype(np.double))
    return BVP


def get_ROI(frames):
  left_cheek_frames = []
  right_cheek_frames = []
  # filename = input_file
  # st = time.time()
  # v_cap = FileVideoStream(filename).start()
  # v_len = int(v_cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))
  # v_fps = int(v_cap.stream.get(cv2.CAP_PROP_FPS))
  # frames = []

  v_len = len(frames)
  v_fps = 30
  frame_counter=0
  batch_size = 4
  frame_flag_start = 0
  frame_flag_end = 0
  collection_frames = []
  collection_frames_fin = []

  stride_fin = int(v_fps)*2

  num_iterations = 0
  num_iterations_face = 0
  flag_done=0
  LAST_DET=False
  NO_DETECT_COUNT=1
  dup_res_inx=0
  prec_inx=0
  face_nan=0

  def detect_faces(fr):
      
    faces = detector(fr)
    if len(faces)==0:
      return 0
    else:
      shape = predictor(fr, faces[0])
      shape = face_utils.shape_to_np(shape)
      RC_r = (shape[29][1],shape[33][1]),(shape[4][0],shape[12][0])
      LC_r = (shape[29][1],shape[33][1]),(shape[4][0],shape[48][0])
      return RC_r, LC_r

  def concatenate_slices(frame, coords1, coords2):
      slice1 = frame[coords1[0]:coords1[1], coords1[2]:coords1[3]]
      return slice1

  for j in range(0,v_len):
      frame = frames[j]
      # frame = v_cap.read()
      # frame  = cv2.resize(frame,(230,300)) #230,300
      # cv2_imshow(frame)
      # break
      frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)#cv2.ROTATE_180) #cv2.ROTATE_90_COUNTERCLOCKWISE)
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      num_iterations+=1

      if j % 15 == 0:
        NO_DETECT_COUNT=1
      else:
        NO_DETECT_COUNT=0

      if j % stride_fin == 0 or frame_flag_start==0 or NO_DETECT_COUNT==1:

        result_indexes = detect_faces(gray)

        if result_indexes!=0:
          if j % stride_fin == 0:
            num_iterations_face+=1
            frame_flag_start=99
            prec_inx = result_indexes
          else:
            result_indexes=prec_inx


        else:
          result_indexes=dup_res_inx
          collection_frames.append(frame)
          face_nan+=1
          continue

      if frame_flag_start==99 or frame_flag_start == 999:
        collection_frames.append(frame)
        temp1 = j+1
        if temp1 % stride_fin == 0 or LAST_DET == True:
          frame_flag_end = 99
          if (v_len-j) <= stride_fin and LAST_DET == False :
            LAST_DET = True
            frame_flag_start = 0

          R_row_start = result_indexes[0][0][0]
          R_row_end = result_indexes[0][0][1]
          R_col_start = result_indexes[0][1][0]
          R_col_end = result_indexes[0][1][1]

          L_row_start = result_indexes[1][0][0]
          L_row_end = result_indexes[1][0][1]
          L_col_start = result_indexes[1][1][0]
          L_col_end = result_indexes[1][1][1]
          for k in range(len(collection_frames)):
            collection_frames_fin.append(concatenate_slices(collection_frames[k],(R_row_start,R_row_end,R_col_start,R_col_end),(L_row_start,L_row_end,L_col_start,L_col_end)))
          collection_frames = []
          dup_res_inx = result_indexes
          # break
      if face_nan > ((v_len//15)*0.6):
        print('Adjust Face: Face not founddd')
        break
  print(len(collection_frames_fin))
  return v_fps,collection_frames_fin


################################################

def read_png_frames(folder_path):
    # Get all PNG files and sort them by the numeric part of the filename
    png_files = sorted([file for file in os.listdir(folder_path) if file.endswith('.png')],
                       key=lambda x: int(re.search(r'(\d+)', x).group(1)))

    frames = []

    for png_file in png_files:
        try:
            file_path = os.path.join(folder_path, png_file)
            print(file_path)
            image = cv2.imread(file_path)
            # print(type(image))
            # frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            frames.append(image)
        except:
            pass

    return frames

def reorder_files(userSessionUID):
    filenames = os.listdir(f'./{userSessionUID}')
    # print(filenames)
    # print(len(filenames))
    save_sorted_files(filenames, userSessionUID)
    

def save_sorted_files(filenames, userSessionUID):
    l = filenames
    for i in range(len(l)):
        print(f'\n\nfor filename {l[i]}')
        pre, post = l[i][12:].split('-')
        post =  post.split('.')[0]
        
        if int(post) == 100:
            os.rename(f'./{userSessionUID}/{l[i]}', f'./{userSessionUID}/{pre}{post[1:]}.png')
            # print(f'{pre}{post[1:]}.png')
            
        else:
            
            if len(post) == 1:
                if (int(pre) - 1) == 0:
                    os.rename(f'./{userSessionUID}/{l[i]}', f'./{userSessionUID}/{int(post)}.png')
                    print(f'{int(post)}.png')
                else:
                    os.rename(f'./{userSessionUID}/{l[i]}', f'./{userSessionUID}/{int(pre) - 1}0{int(post)}.png')
                    # print(f'{int(pre) - 1}0{int(post)}.png')
                
            else:
                if (int(pre) - 1) == 0:
                    os.rename(f'./{userSessionUID}/{l[i]}', f'./{userSessionUID}/{post}.png')
                    # print(f'{post}.png')
                    
                else:
                    os.rename(f'./{userSessionUID}/{l[i]}', f'./{userSessionUID}/{int(pre) - 1}{post}.png')
                    # print(f'{int(pre) - 1}{post}.png')


@app.route('/', methods=['POST'])
def receive_list():
    print('got something')
    if request.method == 'POST':
        try:
            data = request.get_json()
            # print(data)
            # print(len(str(data)))
            
            
            fileName = str(list(data.keys())[0])
            userSessionUID = fileName[8:]
            
            if not os.path.exists(f'./{userSessionUID}'):
                os.mkdir(f'{userSessionUID}')
            
            print(userSessionUID)
            arrImages = data[f'{fileName}']
            i = 1
            for b64ImageString in arrImages:
                image_data = base64.b64decode(b64ImageString)
                with open(f'./{userSessionUID}/output_image{fileName[4]}-{i}.png', 'wb') as file:
                    file.write(image_data)
                    i += 1
            
            if ( 'output_image9-100.png' in os.listdir(f'./{userSessionUID}') ):
                reorder_files(userSessionUID)
                
                frames = read_png_frames(f'./{userSessionUID}')
                _, frames = get_ROI(frames)

                sampling_rate = 30
                wave = POS_WANG(frames, sampling_rate)

                heart_rate_bpm = calc_hr_rr(wave, sampling_rate=sampling_rate)
                ibi, sdnn, rmssd, pnn20, pnn50, hrv, rr = calc_hp_metrics(wave,sampling_rate=30)
                sysbp, diabp, spo2 = pred_adv(wave)

                age = 21
                gender = 0
                weight = 71
                height = 172

                vo2max = calculate_cardio_fit(age,heart_rate_bpm)
                [mhr, hrr, thr, co, map, 
                hu, bv, tbw, bwp, bmi, bf] = calc_all_params(age, gender, weight, height, sysbp, diabp, heart_rate_bpm)
                si = calculate_Stress_Index(wave, sampling_rate)
                wave2 = POS_WANG2(frames, sampling_rate)
                asth_rs = get_asthama_riskscore(wave2)
                
                if ( sysbp < 80 or sysbp > 170 ):
                  sysbp = "N/A"
                
                
                if (diabp < 50 or diabp > 140):
                  diabp = "N/A"
                
                
                print({
                    'hr': (math.floor(heart_rate_bpm)),  
                    "ibi": (round(float(ibi), 1)), 
                    "sdnn": (round(float(sdnn), 1)), 
                    "rmssd": (round(float(rmssd), 1)), 
                    "pnn20": (round(float(pnn20) * 100, 1)), 
                    "pnn50": (round(float(pnn50) * 100, 1)), 
                    "hrv": (hrv),
                    "rr": (round(float(rr), 2)), 
                    "sysbp": (math.floor(sysbp[0, 0])), 
                    "diabp": (math.floor(diabp[0,0])), 
                    # "spo2": (math.floor(spo2[0 ,0])),
                    "vo2max": (round(vo2max, 1)), 
                    "si": (round(si, 1)), 
                    "mhr": (math.floor(float(mhr))), 
                    "hrr": (math.floor(float(hrr))), 
                    "thr": (math.floor(float(thr))), 
                    "co": (round(float(co), 1)),
                    "map": (round(float(map), 1)), 
                    "hu": (round(float(hu), 1)), 
                    "bv": (bv), 
                    "tbw": (float(tbw)), 
                    "bwp": (round(float(bwp), 1)), 
                    "bmi": (round(float(bmi), 1)), 
                    "bf": (round(float(bf), 1)), 
                    "asth_risk": round(float(asth_rs),1)
                
                })
                
                
                print(f'SPO2 is {spo2} and type is {spo2}')
                
                # # if client session directory in folder, deleted, else skip
                # if userSessionUID in os.listdir('./'):
                #     try:
                #         shutil.rmtree(f'./{userSessionUID}')
                #         print(f'Client directory removed successfully -> {userSessionUID}')
                #     except:
                #         print(f'Failed to remove client directory -> {userSessionUID}')
                
                
                if isinstance(spo2, np.ndarray):
                  
                  return json.dumps(
                      {
                          'hr': (math.floor(heart_rate_bpm)),  
                          "ibi": (round(float(ibi)), 1), 
                          "sdnn": (round(float(sdnn), 1)), 
                          "rmssd": (round(float(rmssd), 1)), 
                          "pnn20": (round(float(pnn20) * 100, 1)), 
                          "pnn50": (round(float(pnn50) * 100, 1)), 
                          "hrv": (hrv),
                          "rr": (round(float(rr), 2)), 
                          "sysbp": (math.floor(sysbp[0, 0])), 
                          "diabp": (math.floor(diabp[0,0])), 
                          "spo2": (math.floor(spo2[0,0])),
                          "vo2max": (round(vo2max, 1)), 
                          "si": (round(si, 1)), 
                          "mhr": (math.floor(float(mhr))), 
                          "hrr": (math.floor(float(hrr))), 
                          "thr": (math.floor(float(thr))), 
                          "co": (round(float(co), 1)),
                          "map": (round(float(map), 1)), 
                          "hu": (round(float(hu), 1)), 
                          "bv": (bv), 
                          "tbw": (float(tbw)), 
                          "bwp": (round(float(bwp), 1)), 
                          "bmi": (round(float(bmi), 1)), 
                          "bf": (round(float(bf), 1)), 
                          "asth_risk": round(float(asth_rs),1)
                      
                      }
                )
                elif isinstance(spo2, list):
                  return json.dumps(
                      {
                          'hr': (math.floor(heart_rate_bpm)),  
                          "ibi": (round(float(ibi)), 1), 
                          "sdnn": (round(float(sdnn), 1)), 
                          "rmssd": (round(float(rmssd), 1)), 
                          "pnn20": (round(float(pnn20) * 100, 1)), 
                          "pnn50": (round(float(pnn50) * 100, 1)), 
                          "hrv": (hrv),
                          "rr": (round(float(rr), 2)), 
                          "sysbp": (math.floor(sysbp[0, 0])), 
                          "diabp": (math.floor(diabp[0,0])), 
                          "spo2": (math.floor(spo2[0][0])),
                          "vo2max": (round(vo2max, 1)), 
                          "si": (round(si, 1)), 
                          "mhr": (math.floor(float(mhr))), 
                          "hrr": (math.floor(float(hrr))), 
                          "thr": (math.floor(float(thr))), 
                          "co": (round(float(co), 1)),
                          "map": (round(float(map), 1)), 
                          "hu": (round(float(hu), 1)), 
                          "bv": (bv), 
                          "tbw": (float(tbw)), 
                          "bwp": (round(float(bwp), 1)), 
                          "bmi": (round(float(bmi), 1)), 
                          "bf": (round(float(bf), 1)), 
                          "asth_risk": round(float(asth_rs),1)
                      
                      }
                )
                  
                  
                else:
                  return json.dumps(
                      {
                          'hr': (math.floor(heart_rate_bpm)),  
                          "ibi": (round(float(ibi)), 1), 
                          "sdnn": (round(float(sdnn), 1)), 
                          "rmssd": (round(float(rmssd), 1)), 
                          "pnn20": (round(float(pnn20) * 100, 1)), 
                          "pnn50": (round(float(pnn50) * 100, 1)), 
                          "hrv": (hrv),
                          "rr": (round(float(rr), 2)), 
                          "sysbp": (math.floor(sysbp[0, 0])), 
                          "diabp": (math.floor(diabp[0,0])), 
                          "spo2": (math.floor(spo2)),
                          "vo2max": (round(vo2max, 1)), 
                          "si": (round(si, 1)), 
                          "mhr": (math.floor(float(mhr))), 
                          "hrr": (math.floor(float(hrr))), 
                          "thr": (math.floor(float(thr))), 
                          "co": (round(float(co), 1)),
                          "map": (round(float(map), 1)), 
                          "hu": (round(float(hu), 1)), 
                          "bv": (bv), 
                          "tbw": (float(tbw)), 
                          "bwp": (round(float(bwp), 1)), 
                          "bmi": (round(float(bmi), 1)), 
                          "bf": (round(float(bf), 1)), 
                          "asth_risk": round(float(asth_rs),1)
                      
                      }
                )
            
            # return 200, till all image loads have not been sent to server
            return json.dumps({"opcode": 200})
        except Exception as e:
            print(traceback.format_exc())
            
            # # if client session directory in folder, deleted, else skip
            # if userSessionUID in os.listdir('./'):
            #     try:
            #         shutil.rmtree(f'./{userSessionUID}')
            #         print(f'Client directory removed successfully -> {userSessionUID}')
            #     except:
            #         print(f'Failed to remove client directory -> {userSessionUID}')
            return json.dumps({"opcode": 500})
    else:
        return jsonify({"error": "Only POST requests are allowed"}), 405

if __name__ == '__main__':
    app.run("0.0.0.0", port=5000)