import numpy as np
import matplotlib.pyplot as pl
from math import log, exp, fabs
import collections
import tools2
import copy
from sklearn.mixture import *




prondict = {} 
prondict['o'] = ['ow']
prondict['z'] = ['z', 'iy', 'r', 'ow']
prondict['1'] = ['w', 'ah', 'n']
prondict['2'] = ['t', 'uw']
prondict['3'] = ['th', 'r', 'iy']
prondict['4'] = ['f', 'ao', 'r']
prondict['5'] = ['f', 'ay', 'v']
prondict['6'] = ['s', 'ih', 'k', 's']
prondict['7'] = ['s', 'eh', 'v', 'ah', 'n']
prondict['8'] = ['ey', 't']
prondict['9'] = ['n', 'ay', 'n']


phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()
phones = sorted(phoneHMMs.keys())
nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]
#print(stateList)
#print(stateList.index('ay_2'))


def word2phones(wordList, pronDict, addSilence = True, addShortPause = False):
    newPhone = np.array([])
    if addSilence:
        newPhone =np.concatenate((newPhone, np.array(['sil'])), axis = 0)
    for i in wordList:
        newPhone = np.concatenate((newPhone ,np.array(pronDict[i])), axis = 0)
        if addShortPause:
            newPhone = np.concatenate((newPhone, np.array(['sp'])),axis = 0)
    if addSilence:
        newPhone =  np.concatenate((newPhone, np.array(['sil'])), axis = 0)
    return newPhone



print(word2phones(['z','4','3'], prondict))


