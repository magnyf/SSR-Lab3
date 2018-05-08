import numpy as np
import matplotlib.pyplot as pl
from math import log, exp, fabs
import collections
import tools2
import copy
from sklearn.mixture import *

phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()


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



def concatAnyHMM(hmmmodels, namelist):
    N = len(namelist)
    word = {}
    word['name']="I dont know"
    word['startprob'] = [0 for i in range(3 * (N-1) + 4)]
    word['means'] = [[0 for j in range(13)] for i in range(3*N)]
    word['covars'] = [[0 for j in range(13)] for i in range(3*N)]

    #compute the startprob
    ## TO CHECK
    ## not sure what should append if there is more than 2 phones
    retenu = 1.0
    for i in range(N):
        word['startprob'][3*i] = hmmmodels[namelist[i]]['startprob'][0] * retenu
        word['startprob'][3*i+1] = hmmmodels[namelist[i]]['startprob'][1] * retenu
        word['startprob'][3*i+2] = hmmmodels[namelist[i]]['startprob'][2] * retenu
        retenu  = retenu * hmmmodels[namelist[i]]['startprob'][3]
    word['startprob'][-1] = retenu

    
    #compute means and covars
    ## TO CHECK:
    ## I don't know how to concat the means and the covars, not discribed in the document
    ## For now it is the old way to concat
    for i in range(N):
        for j in range(13):
            word['means'][3*i][j] = hmmmodels[namelist[i]]['means'][0][j]
            word['covars'][3*i][j] = hmmmodels[namelist[i]]['covars'][0][j]
                        
        for j in range(13):
            word['means'][3*i+1][j] = hmmmodels[namelist[i]]['means'][1][j]
            word['covars'][3*i+1][j] = hmmmodels[namelist[i]]['covars'][1][j]

        for j  in range(13):
            word['means'][3*i+2][j] = hmmmodels[namelist[i]]['means'][2][j]
            word['covars'][3*i+2][j] = hmmmodels[namelist[i]]['covars'][2][j]

    #compute transmat
    ## TO DO
    return word


dicoTest = {}
dicoTest['1'] = {}
dicoTest['2'] = {}
dicoTest['1']['means'] = [[1.0 for i in range(13)] for j in range(3)]
dicoTest['1']['covars'] =  [[1.0 for i in range(13)] for j in range(3)]
dicoTest['1']['startprob'] = [1.0, 1.25,1.50,2.0]
dicoTest['2']['means'] =  [[2.0 for i in range(13)] for j in range(3)]
dicoTest['2']['covars'] =  [[2.0 for i in range(13)] for j in range(3)]
dicoTest['2']['startprob'] = [1.0, 1.25,1.50,2.0]

print(concatAnyHMM(dicoTest, ['1', '2', '1'])['startprob'])

