from pywt import dwt
from scipy import signal
import numpy as np


def r_peaks(ecg,fs,maxiter=100):
  sos = signal.butter(15, [1, 20], btype='bandpass', fs=fs, output="sos")
  ecg_prep = signal.sosfilt(sos, ecg)
  y = signal.filtfilt([50/fs]*int(fs/50), 1, ecg_prep)
  t = np.arange(len(y))/fs

  [_, cD] = dwt(y,'sym3')
  fs_c = fs*len(cD)/len(y)
  t_c = np.linspace(0,max(t),len(cD))

  cD2 = signal.medfilt(cD*cD,11)
  cD2 /= max(cD2)
  
  win_l = int(fs_c)
  for j in range(0,len(cD2)-win_l,int(win_l/2)):
    cD2[j:j+win_l] = cD2[j:j+win_l]/max(cD2[j:j+win_l])

  
  locs = signal.find_peaks(cD2,height=0.5,distance=int(0.3*fs_c))
  locs = np.array([int(x/fs_c*fs) for x in locs[0]])

  locs_tmp = locs.copy()
  locs_array = []
  locs_array.append(locs.copy())

  eq = False
  win_l = int(np.min(locs[1:-1]-locs[0:-2])/1.5)
  for j in range(maxiter):
    for i, loc in enumerate(locs[1:]):
      if locs[i]-locs[i-1] > np.mean(locs[1:]-locs[0:-1])*1.5:
        locs = np.append(locs,int((locs[i]+locs[i-1])/2))
        locs = np.sort(locs)
    
    for locs_in_arr in locs_array:
      if np.array_equal(locs_in_arr,locs):
        eq = True
        break
    if eq:
      break
    if np.array_equal(locs_tmp,locs):
      break
    else:
      locs_array.append(locs.copy())
      locs_tmp = locs

  locs_tmp = locs.copy()
  locs_array = []
  locs_array.append(locs.copy())

  eq = False
  win_l = int(np.min(locs[1:-1]-locs[0:-2])/1.25)
  for j in range(maxiter):
    for i, loc in enumerate(locs):
      win_start = max([0,loc-int(win_l/2)])
      win_stop = min([len(ecg),loc+int(win_l/2)])
      locs[i] = np.argmax(ecg[win_start:win_stop])+win_start
    
    for locs_in_arr in locs_array:
      if np.array_equal(locs_in_arr,locs):
        eq = True
        break
    if eq:
      break
    if np.array_equal(locs_tmp,locs):
      break
    else:
      locs_array.append(locs.copy())
      locs_tmp = locs

  locs = np.sort(np.unique(locs))
  time_locs = locs/fs
  RR = time_locs[1:] - time_locs[0:-1]
  RR = np.round(RR,3)

  return locs, time_locs, RR
