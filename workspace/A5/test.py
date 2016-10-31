import numpy as np
from scipy.signal import get_window
import math
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import dftModel as DFT
import utilFunctions as UF

# inputFile = '../../sounds/sine-490.wav'
# f = 490

pX = [1.0, 1.2, 1.3, 1.4, 0.9, 0.8, 0.7, 0.6, 0.7, 0.8]
p = 3
x = pX[p-2:p+3]
std = np.std(pX)