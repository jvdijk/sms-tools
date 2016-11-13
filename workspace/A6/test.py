import os
import sys
import numpy as np
import math
from scipy.signal import get_window
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import utilFunctions as UF
import harmonicModel as HM
import stft

eps = np.finfo(float).eps

"""
A6Part2 - Segmentation of stable note regions in an audio signal

Complete the function segmentStableNotesRegions() to identify the stable regions of notes in a specific 
monophonic audio signal. The function returns an array of segments where each segment contains the 
starting and the ending frame index of a stable note.

The input argument to the function are the wav file name including the path (inputFile), threshold to 
be used for deciding stable notes (stdThsld) in cents, minimum allowed duration of a stable note (minNoteDur), 
number of samples to be considered for computing standard deviation (winStable), analysis window (window), 
window size (M), FFT size (N), hop size (H), error threshold used in the f0 detection (f0et), magnitude 
threshold for spectral peak picking (t), minimum allowed f0 (minf0) and maximum allowed f0 (maxf0). 
The function returns a numpy array of shape (k,2), where k is the total number of detected segments. 
The two columns in each row contains the starting and the ending frame indexes of a stable note segment. 
The segments must be returned in the increasing order of their start times. 

In order to facilitate the assignment we have configured the input parameters to work with a particular 
sound, '../../sounds/sax-phrase-short.wav'. The code and parameters to estimate the fundamental frequency 
is completed. Thus you start from an f0 curve obtained using the f0Detection() function and you will use 
that to obtain the note segments. 

All the steps to be implemented in order to solve this question are indicated in segmentStableNotesRegions() 
as comments. These are the steps:

1. In order to make the processing musically relevant, the f0 values should be converted first from 
Hertz to Cents, which is a logarithmic scale. 
2. At each time frame (for each f0 value) you should compute the standard deviation of the past winStable 
number of f0 samples (including the f0 sample at the current audio frame). 
3. You should then apply a deviation threshold, stdThsld, to determine if the current frame belongs 
to a stable note region or not. Since we are interested in the stable note regions, the standard 
deviation of the previous winStable number of f0 samples (including the current sample) should be less 
than stdThsld i.e. use the current sample and winStable-1 previous samples. Ignore the first winStable-1 
samples in this computation.
4. All the consecutive frames belonging to the stable note regions should be grouped together into 
segments. For example, if the indexes of the frames corresponding to the stable note regions are 
3,4,5,6,12,13,14, we get two segments, first 3-6 and second 12-14. 
5. After grouping frame indexes into segments filter/remove the segments which are smaller in duration 
than minNoteDur. Return the segment indexes in the increasing order of their start frame index.
                              
Test case 1: Using inputFile='../../sounds/cello-phrase.wav', stdThsld=10, minNoteDur=0.1, 
winStable = 3, window='hamming', M=1025, N=2048, H=256, f0et=5.0, t=-100, minf0=310, maxf0=650, 
the function segmentStableNotesRegions() should return 9 segments. Please use loadTestcases.load() 
to check the expected segment indexes in the output.

Test case 2: Using inputFile='../../sounds/cello-phrase.wav', stdThsld=20, minNoteDur=0.5, 
winStable = 3, window='hamming', M=1025, N=2048, H=256, f0et=5.0, t=-100, minf0=310, maxf0=650, 
the function segmentStableNotesRegions() should return 6 segments. Please use loadTestcases.load() 
to check the expected segment indexes in the output.

Test case 3: Using inputFile='../../sounds/sax-phrase-short.wav', stdThsld=5, minNoteDur=0.6, 
winStable = 3, window='hamming', M=1025, N=2048, H=256, f0et=5.0, t=-100, minf0=310, maxf0=650, 
the function segmentStableNotesRegions() should return just one segment. Please use loadTestcases.load() 
to check the expected segment indexes in the output. 

We also provide the function plotSpectogramF0Segments() to plot the f0 contour and the detected 
segments on the top of the spectrogram of the audio signal in order to visually analyse the outcome 
of your function. Depending on the analysis parameters and the capabilities of the hardware you 
use, the function might take a while to run (even half a minute in some cases). 

"""
import loadTestCases

testcase = loadTestCases.load(3, 1)

tc = testcase['input']
out = testcase['output']

# inputFile = tc['inputFile']
# window = tc['window']
# M = tc['M']
# N = tc['N']
# H = tc['H']
# t1 = tc['t1']
# t2 = tc['t2']

# stdThsld= tc['stdThsld']
# minNoteDur= tc['minNoteDur']
# winStable = tc['winStable']

# f0et= tc['f0et']
# t= tc['t'] 
# minf0= tc['minf0']
# maxf0= tc['maxf0']
# Test case 1: If you run your code with inputFile = '../../sounds/piano.wav', t1=0.2, t2=0.4, window='hamming', M=2047, N=2048, H=128, f0et=5.0, t=-90, minf0=130, maxf0=180, nH = 25, the returned output should be 1.4543. 

inputFile = '../../sounds/piano.wav'
t1=0.2
t2=0.4
window='hamming'
M=2047
N=2048
H=128
f0et=5.0
t=-90
minf0=130
maxf0=180
nH = 25

fs, x = UF.wavread(inputFile)                               #reading inputFile
w  = get_window(window, M)                                  #obtaining analysis window    

# 1. Use harmonic model to compute the harmonic frequencies and magnitudes
xhfreq, xhmag, xhphase = HM.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, 0.01, 0.0)

l1 = int(np.floor(fs*t1/float(H)))
l2 = int(np.ceil(fs*t2/float(H)))

frames = xhfreq[l1:l2 + 1]
I = np.zeros(frames.shape[0])

# i = 0
# for l in frames:
#     f0 = l[0]
#     hs = np.where(l > 0.0)[0]
#     inh = 0
    
#     for r in hs:
#         if r > 0:
#             fest = l[r]
#             f = (r+1)*f0
#             inh += abs(fest - f) / (r + 1)
#             # print(f0, fest, f, r, inh)
#     I[i] = inh / float(nH)
#     i += 1

i = 0
for l in frames:
    f0 = l[0]
    inh = 0
    
    r = 0
    for fest in l:
        if fest > 0.0:
            f = (r + 1) * f0
            inh += abs(fest - f) / (r + 1)
            print(f0, f, fest, r, inh)
            r += 1
    I[i] = inh / l.size
    i += 1

# if f0 > 0.0:
#     indices = np.where(l > 0.0)[0]
#     R = indices.size
#     inh = np.zeros(nH)
#     for r in indices:
#         inh[r] = abs(l[r] - r*f0) / (r + 1)
#         print(l[r], (r + 1)*f0)

#     Il[i] = sum(inh) / nH
#     i += 1

Im = sum(I) / (l2 - l1 + 1)




    