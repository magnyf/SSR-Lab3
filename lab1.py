# DT2119, Lab 1 Feature Extraction

import numpy as np
import scipy.signal
import scipy.fftpack
from scipy.fftpack.realtransforms import dct
import math

from tools1 import *

# Function given by the exercise ----------------------------------

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    mspec = logMelSpectrum(spec, samplingrate)
    ceps = cepstrum(mspec, nceps)
    return lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """    
    return  [samples[x:x+winlen] for x in range(0, len(samples)-winlen, winshift)]

    
def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    N = len(input[0])
    b = np.array([1.0 - p for i in range(N)])
    b[0] = 1.
    a = np.array([1. for i in range(N)])
    return [ scipy.signal.lfilter(b,a, input[i]) for i in range(len(input))]


def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """

    hammingWindow = np.array(scipy.signal.hamming(len(input[0]), sym=False))
    windowed =  [np.multiply(x, hammingWindow) for x in input]
    return windowed
    
def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """

    N = len(input)
    output = [ scipy.fftpack.fft(input[i], nfft) for i in range(N)]
    output = [ abs(output[i])**2 for i in range(N)]
    return output

    
def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in tools.py to calculate the filterbank shapes and
          nmelfilters
    """

    nfft = len(input[0])
    mspecFilters = trfbank(samplingrate, nfft)
    mspec = np.matmul(input, np.transpose(mspecFilters))
    for i in range(len(mspec)):
        for j in range(len(mspec[i])):
            mspec[i][j] = math.log(mspec[i][j])
    return mspec

            
def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """

    return  np.array([dct(input[i],type=2, norm='ortho', axis= -1)[ :nceps] for i in range(len(input))])

def dist(x, y):
    assert len(y) == len(x)
    return math.sqrt(np.sum([(y[i] - x[i])**2 for i in range(len(x))]))

def traceback(D):
    """
    Computes the path
    """
    i, j = np.array(D.shape) - 1
    p, q = [i], [j]
    while (i > 0 and j > 0):
        tb = np.argmin((D[i-1, j-1], D[i-1, j], D[i, j-1]))

        if (tb == 0):
            i = i - 1
            j = j - 1
        elif (tb == 1):
            i = i - 1
        elif (tb == 2):
            j = j - 1

        p.insert(0, i)
        q.insert(0, j)


    return [(p[i]-1, q[i]-1) for i in range(len(p)) ]

def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """
    assert len(x)
    assert len(y)
    N, M = len(x), len(y)
    D0 = np.zeros((N + 1,M+ 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:] # view
    for i in range(N):
        for j in range(M):
            D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    for i in range(N):
        for j in range(M):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
        path = np.zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), no.zeros(len(x))
    else:
        path = traceback(D0)
    return D1[-1, -1] / (N+M), C, D1, path





