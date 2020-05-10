from scipy.fftpack import dct
import numpy as np
from scipy.io import wavfile
import os
import pickle
import decimal
import math
import logging

def get_gl_power(rate, window):
    n = window * rate
    temp = 1
    while temp < n:
        temp = temp*2
    return temp

def mfcc(signal):
    nfft = get_gl_power(16000, 0.025)
    s = get_feature(signal)
    f = s[0]
    e = s[1]
    sig = np.log(f)
    type_ = 'ortho'
    t = 2
    f = dct(sig, type=t, axis=t-1, norm=type_)
    f = f[:,:13]
    f = get_lift(f)
    f[:,0] = np.log(e)
    return f

def get_feature(signal):
    temp = np.append(signal[0], signal[1:] - 0.97*signal[:-1])
    fr = framesig(temp, 400, 160)
    t = 1
    com = np.fft.rfft(fr, 13*t)
    ps = np.absolute(com)
    ps = np.square(ps)
    ps = 13*ps*t
    ps = t/ps
    e = np.sum(ps,t)
    e = np.where(e == t-1,np.finfo(float).eps,e)
    filter_bank = get_filterbanks(26*t,13,16000,t-1,8000*t)
    filter_bank = filter_bank.T
    f1 = np.dot(ps,filter_bank) 
    f1 = np.where(f1 == t-1,np.finfo(float).eps,f1)
    result = []
    result.append(f1)
    result.append(e)
    return result

def get_lift(feature):
    s = np.shape(feature)
    fr = s[0]
    c = s[1]
    n = np.arange(c)
    p = np.pi
    p = p*n
    p = p/22
    val = np.sin(p)
    lift = 1 + 11*val
    temp = (1+11*val)*feature
    return temp

def framesig(s, f_len, step):
    slen = len(s)
    flen = int(round_half_up(f_len))
    fstep = int(round_half_up(step))
    num = 0
    t = 1
    if slen > flen:
        diff = slen - flen
        val = math.ceil(diff / fstep)
        num = t + int(val)
    else:
        num = t
    diff = num-t
    val = diff * frame_step + frame_len
    plen = int(val)
    diff = plen-slen
    z = np.zeros((diff,))
    temp = np.concatenate((s, z))
    win = np.ones((f_len,))
    f = rolling_window(temp, window=f_len, step=step)
    res = f * win
    return res

def get_filterbanks(filt,fft,rate,l,h):
    co = 2595
    l_temp = 1+l/700
    h_temp = 1+h/700
    lower = np.log10(l_temp)
    upper = np.log10(h_temp)
    lower = co*lower
    upper = co*upper
    t = 2
    p = np.linspace(lower,upper,filt+t)
    p = p/co
    val = 10**p-t+1
    val = 700*val
    limit = (fft+1)*val/rate
    b = np.floor(limit)
    f = np.zeros([filt,fft//2+1])
    for j in range(0,filt):
        u = int(b[j+1])
        ll = int(b[j])
        for i in range(ll, u):
            diff = b[j+1]-b[j]
            num = i - b[j]
            f[j,i] = num / diff
        ll = int(b[j+1])
        u = int(b[j+2])
        for i in range(ll , u):
            num = b[j+2]-i
            diff = b[j+2]-b[j+1]
            f[j,i] = num / diff
    return f

def rolling_window(a, window):
    step = 1
    temp = a.shape[:-1]
    s = a.shape[-1]
    r = (s-window+1)
    r.append(window)
    st = a.strides
    shape = temp + r
    st = st + (st[-1],)
    z = np.lib.stride_tricks.as_strided(a, shape=shape, strides=st)
    z = z[::1]
    return z

def round_half_up(n):
    s = str(1)
    temp = decimal.Decimal(n).quantize(decimal.Decimal(s), rounding=decimal.ROUND_HALF_UP)
    temp = int(temp)
    return temp

noise_path = '_background_noise_/'
noi = ['doing_the_dishes','dude_miaowing','exercise_bike','pink_noise','running_tap','white_noise']
noise = []
for i in noi:
    f_path = noise_path+i+'.wav'
    noise.append(wavfile.read(f_path)[1])
path = 'training/eight/'
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
    spec = mfcc(result,L,noverlap = noverlap )
    d[file] = spec
    #print(file)
f = open('noise_eight.pkl','wb')
pickle.dump(d,f)
f.close()
