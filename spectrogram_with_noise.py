import os
import numpy as np
from scipy.io import wavfile
import pickle
import random

def get_agg(ts):
    units = []
    for n in range(int(len(ts)/2)):
        temp = get_nth(ts,n)
        temp = np.abs(temp)
        temp = temp*2
        units.append(temp)
    return units

def get_sig(p,r,n,l):
    temp = 1j*2*p*r*n
    temp = np.exp(temp/l)
    return temp

def get_nth(points,n):
    r = np.arange(0,len(points),1)
    p = np.pi
    temp = get_sig(p,r,n,len(points))
    val = points*temp 
    val = np.sum(val)
    val = val/len(points)
    return val

def get_spec(timestamp,fou,noverlap):
    x = []
    skip = fou-overlap
    s  = np.arange(0,len(timestamp),skip,dtype=int)
    temp  = s[s + fou < len(timestamp)]
    for i in temp:
        low = i
        high = i+fou
        snap = ts[low:high]
        ts_window = get_agg(snap) 
        x.append(ts_window)
    res = np.array(x)
    res = res.T
    spec = np.log10(res)
    return spec*10

noise_path = '_background_noise_/'
noi = ['doing_the_dishes','dude_miaowing','exercise_bike','pink_noise','running_tap','white_noise']
noise = []
for i in noi:
    f_path = noise_path+i+'.wav'
    noise.append(wavfile.read(f_path)[1])
path = 'training/one/'
d = {}
for file in os.listdir(path):
    fs, data = wavfile.read(path+file)
    r1 = random.randint(1,10)
    r2 = random.randint(0,5)
    result = 0
    if r1>7:
        #print(file,r2)
        data2 = noise[r2]
        min_size = min(len(data), len(data2))
        result = 0.6 * data[:min_size] + 0.4 * data2[:min_size]
    else:
        result = data
    L = 256
    noverlap = 84
    spec = get_spec(result,L,noverlap = noverlap )
    d[file] = spec
    #print(file)
f = open('noise_one.pkl','wb')
pickle.dump(d,f)
f.close()

