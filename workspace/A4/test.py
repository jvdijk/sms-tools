import numpy as np
import time, os, sys
from scipy.signal import get_window

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import stft as STFT
import utilFunctions as UF
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import math
import loadTestCases
testcase = loadTestCases.load(3, 1)

tc = testcase['input']
out = testcase['output']

inputFile = tc['inputFile']
window = tc['window']
M = tc['M']
N = tc['N']
H = tc['H']

def computeEngEnv(inputFile, window, M, N, H):
	(fs, x) = UF.wavread(inputFile)
	w = get_window(window, M)

	minBin = int(np.floor(3000 * float(N) / float(fs)) + 1)
	maxBin = int(np.floor(10000 * float(N) / float(fs)) + 1)

	mX, pX = STFT.stftAnal(x, w, N, H)

	low = np.array([])
	high = np.array([])

	for frame in mX:
		lb = frame[1:minBin]
		lowL = pow(10, lb / 20.0)	
		lE = sum(abs(lowL) * abs(lowL))
		ldB = 10 * np.log10(lE)
		low = np.append(low, ldB)

		hb = frame[minBin:maxBin]
		highL = pow(10, hb / 20.0)	
		hE = sum(abs(highL) * abs(highL))
		hdB = 10 * np.log10(hE)
		high = np.append(high, hdB)

	return np.transpose([low, high])

# plt.figure(1, figsize=(9.5, 6))

# plt.subplot(211)
# numFrames = int(mX[:,0].size)
# frmTime = H*np.arange(numFrames)/float(fs)                             
# binFreq = np.arange(N/2+1)*float(fs)/N                         
# plt.pcolormesh(frmTime, binFreq, np.transpose(mX))
# plt.title('mX (inputfile), M=1001, N=1024, H=256')
# plt.autoscale(tight=True)

# plt.subplot(212)
# numFrames = int(pX[:,0].size)
# frmTime = H*np.arange(numFrames)/float(fs)                             
# binFreq = np.arange(N/2+1)*float(fs)/N                         
# plt.pcolormesh(frmTime, binFreq, np.diff(np.transpose(pX),axis=0))
# plt.title('pX derivative (piano.wav), M=1001, N=1024, H=256')
# plt.autoscale(tight=True)

# plt.tight_layout()
# # plt.savefig('spectrogram.png')
# plt.show()
