import numpy as np
import math
from scipy.signal import lfilter
from audiolazy import lpc
from python_speech_features import mfcc

class FeatureExtractor():
    def __init__(self, debug=True):
        self.debug = debug

    def intensity(self,window):
        mean = np.mean(window*window)
        return (10*math.log(mean/(2*(10**-5)),10))
    
    def compute_min(self,window):
        return np.amin(window)
    
    def compute_max(self,window):
        return np.amax(window)
        
    def _compute_mfcc(self, window):

        mfccs = mfcc(window,8000,winstep=.0125)
        if self.debug:
            print("{} MFCCs were computed over {} frames.".format(mfccs.shape[1], mfccs.shape[0]))
        return mfccs
    
    def _compute_delta_coefficients(self, window, n=2):
        mfcc = self._compute_mfcc(window)
        deno = 0
        dt = []
        for i in range(1,n+1):
            deno += 2*(i**2)
        for t in range(n, 79-n):
            numerator = 0
            for i in range(1, n+1):
                numerator += i * (np.subtract(mfcc[t + i] ,mfcc[t - i]))
            dt.append(np.divide(numerator,deno))
        return np.array(dt).flatten()
        
    def extract_features(self, window, debug=True):
        x = []
        x = np.append(x, self.compute_min(window))
        x = np.append(x,self.compute_max(window))
        x = np.append(x, self.intensity(window))
        x = np.append(x, self._compute_delta_coefficients(window))
        return x  