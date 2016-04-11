import scipy
import numpy as np
from scipy import scipy.io.wavfile
from scikits.talkbox.features import mfcc

from song import *

def exctract_features(path, genre):
        sample_rate, X = scipy.io.wavfile.read(path)
        fft_features = abs(scipy.fft(X) [:1000])

        ceps, mspec, spec = mfcc(X)
        num_ceps = ceps.shape[0]
        mfcc_features = np.mean(ceps[int(num_ceps*1/10):int(num_ceps*9/10)], axis = 0)

        return song(fft_features, mfcc_features, genre)
