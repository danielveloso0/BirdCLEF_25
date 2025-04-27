import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa 
import scipy.signal as signal
import torch
import torch.nn as nn
import matplotlib.cm as cm

class AudioProcessing():
    def __init__(self,n_ffts,fmin,fmax):
        '''
        Parâmetros do Espectrograma:
        '''
        self.ffts = n_ffts
        self.fmin = fmin
        self.fmax = fmax
    def shuffle_window(self,path):
        '''
        Escolhe uma janela da 5 segundos aleatórios na faixa de áudio.
        '''
        y, fs = librosa.load(path)
        chunk = 5
        N = np.random.choice(range(0,100,16))
        window = y[N:N + chunk*fs]
        return window
    def energy(self,y):
        '''
        Calcula a energia do audio na janela.
        '''
        window = y/np.max(y)
        return np.sum(np.abs(window**2))
    def select_window(self,path):
        '''
        Seleciona a janela com maior energia.
        '''
        size = 5
        y, fs = librosa.load(path)
        E = []
        for i in range(0,len(y),size*fs):
            window = y[i:i+5*fs]
            E.append(self.energy(window))
        k = np.argwhere(E==max(E)).reshape(1)[0]
        return y[k:k+size*fs],fs
    
    def mel_spectrogram(self,ys,fs):
        '''
        Calcula o espectograma
        '''
        spec = librosa.feature.melspectrogram(y=ys,
                                           sr=fs,
                                           n_fft=self.ffts,
                                           fmin=self.fmin,
                                           fmax=self.fmax)
        melspec = librosa.power_to_db(spec)
        return melspec                
    def bandpass_filter(self,y,low,high,Fs,order=4):
        '''
        Fitro passa-banda Butterworth
        Recebe: frequências de corte superior e inferior e a frequência da amostragem.
        Retorna: O áudio filtrado.
        '''
        nyquist = 0.5*Fs
        b,a = signal.butter(order,[low/nyquist,high/nyquist],btype='bandpass',analog=False)
        y_filter = signal.filtfilt(b, a, y)
        return y_filter
    def melspec_to_image(self,melspec, cmap_name='inferno'):
        '''
        Transforma o espectograma em um imagem RGB (torch.tensor) 
        '''
        
        melspec_norm = (melspec - melspec.min()) / (melspec.max() - melspec.min() + 1e-6)
        colormap = cm.get_cmap(cmap_name)
        melspec_colored = colormap(melspec_norm)  # shape: [H, W, 4]
        rgb_image = (melspec_colored[:, :, :3] * 255).astype(np.uint8)
        # Converte para tensor [3, H, W], normalizado entre 0 e 1
        rgb_tensor = torch.tensor(rgb_image).permute(2, 0, 1).float() / 255.0
    
        return rgb_tensor