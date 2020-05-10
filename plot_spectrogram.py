import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

name = '00b01445_nohash_0.wav'
cl = 'two'
path = 'training/'+cl+'/'+name
f,data = wavfile.read(path)
plt.title('Spectrogram '+name+" ("+cl+")")
plt.plot(data)
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()
plt.title('Specgram '+name+" ("+cl+")")
plt.specgram(data,Fs = f)
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()
