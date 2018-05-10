import numpy as np
import matplotlib.pyplot as pl
from math import log, exp, fabs
import collections
import copy
import lab1
import lab2
import lab3_tools
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


def words2phones(wordList, pronDict, addSilence = True, addShortPause = False):
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



print(words2phones(['z','4','3'], prondict))



def concatAnyHMM(hmmmodels, namelist):
    N = len(namelist)
    # Array which conctain the idex of the start of a new phone
    cumulLen = [0 for i in range(N)]
    totalLen = 0
    for i in range(N):
        cumulLen[i] = totalLen
        totalLen += len(hmmmodels[namelist[i]]['startprob'])-1
    totalLen += 1
    word = {}
    word['name']="I dont know"
    word['startprob'] = [0 for i in range(totalLen)]
    word['means'] = [[0 for j in range(13)] for i in range(totalLen)]
    word['covars'] = [[0 for j in range(13)] for i in range(totalLen)]
    word['transmat'] = [[0 for j in range(totalLen)] for i in range(totalLen)]
    #compute the startprob
    ## TO CHECK
    ## not sure what should append if there is more than 2 phones
    retenu = 1.0
    for i in range(N):
        for j in range(len(hmmmodels[namelist[i]]['startprob'])-1):
            word['startprob'][cumulLen[i]+j] = hmmmodels[namelist[i]]['startprob'][j] * retenu
        retenu  = retenu * hmmmodels[namelist[i]]['startprob'][-1]
    word['startprob'][-1] = retenu
                       
    
    #compute means and covars
    ## TO CHECK:
    ## I don't know how to concat the means and the covars, not discribed in the document
    ## For now it is the old way to concat
    for i in range(N):
        for k in range(len(hmmmodels[namelist[i]]['startprob'])-1):
                for j in range(13):
                       word['means'][cumulLen[i]+k][j] = hmmmodels[namelist[i]]['means'][k][j]
                       word['covars'][cumulLen[i]+k][j] = hmmmodels[namelist[i]]['covars'][k][j]
                       
    #compute transmat
    ## TO DO
    for i in range(N):
        # Fill the square unchanged
        nbState = len(hmmmodels[namelist[i]]['startprob'])-1
        for j in range(nbState):
            for k in range(nbState):
                word['transmat'][cumulLen[i] + j][cumulLen[i]+k] =  hmmmodels[namelist[i]]['transmat'][j][k]

        for k in range(nbState):
            retenu = [hmmmodels[namelist[i]]['transmat'][j][-1] for j in range(nbState)]
            for j in range(i+1,N):
                for l in range(len(hmmmodels[namelist[j]]['startprob'])-1):
                    word['transmat'][cumulLen[i]+k][cumulLen[j]+l] = retenu[k] * hmmmodels[namelist[j]]['startprob'][l]
                # update retenu
                for m in range(nbState):
                    retenu[m] = retenu[m] * hmmmodels[namelist[j]]['startprob'][-1]
            word['transmat'][cumulLen[i] + k][-1] = retenu[k]

    word['transmat'][-1][-1] = 1.0    
    return word


# TEST pour concatAllHMM
dicoTestSubject = {}
dicoTestSubject['1'] = {}
dicoTestSubject['2'] = {}
dicoTestSubject['3'] = {}
dicoTestSubject['1']['means'] = [[1.0 for i in range(13)] for j in range(3)]
dicoTestSubject['1']['covars'] =  [[1.0 for i in range(13)] for j in range(3)]
dicoTestSubject['1']['startprob'] = [1,0,0,0]
dicoTestSubject['2']['means'] =  [[2.0 for i in range(13)] for j in range(3)]
dicoTestSubject['2']['covars'] =  [[2.0 for i in range(13)] for j in range(3)]
dicoTestSubject['2']['startprob'] = [0.4,0.6]
dicoTestSubject['3']['means'] =  [[2.0 for i in range(13)] for j in range(3)]
dicoTestSubject['3']['covars'] =  [[2.0 for i in range(13)] for j in range(3)]
dicoTestSubject['3']['startprob'] = [1,0,0,0]
dicoTestSubject['1']['transmat'] = [[0.5,0.5,0,0],
                                    [0,0.5,0.5,0],
                                    [0,0,0.3,0.7],
                                    [0,0,0,1]]
dicoTestSubject['3']['transmat'] = [[0.25,0.75,0,0],
                                    [0,0.25,0.75,0],
                                    [0,0,0.25,0.75],
                                    [0,0,0,1]]
dicoTestSubject['2']['transmat'] = [[0.20,0.8],
                                    [0,1.0]]




wordTest  = concatAnyHMM(phoneHMMs, ['sil', 'ow', 'sil'])
#wordTest = concatAnyHMM(dicoTestSubject, ['1','2','3'])
print("check that the intit proba is equal to 1")
print(np.sum(wordTest['startprob']) == 1)
sumTot = 0.0
N = len(wordTest['transmat'])
for i in range(N):
    sumTot += np.sum(wordTest['transmat'][i])
#print("les deux nombres suivants doivent etre egaux")
print(sumTot)
print(N)

#print(wordTest['startprob'])
#print(wordTest['transmat'])



###########
filename = 'tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'
samples, samplingrate = lab3_tools.loadAudio(filename)
lmfcc = lab1.mfcc(samples)
wordTrans = list(lab3_tools.path2info(filename)[2])
print(wordTrans)
#should be ['z', '4', '3']

phoneTrans = words2phones(wordTrans, prondict, addShortPause=False)
print(phoneTrans)
#should be ['sil', 'z', 'iy', 'r', 'ow', 'f', 'ao', 'r', 'th', 'r', 'iy', 'sil']

stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans 
for stateid in range(nstates[phone])]
print(stateTrans)

wordHMMs = concatAnyHMM(phoneHMMs, phoneTrans)

obsloglik =  lab2.log_multivariate_normal_density_diag(np.array(lmfcc), 
       np.array(wordHMMs['means']), 
       np.array(wordHMMs['covars']))
pi = wordHMMs['startprob']
concatMat = wordHMMs['transmat']

viterbiState = lab2.viterbi(obsloglik, lab2.log_inf(pi), lab2.log_inf(concatMat)
)[1]

viterbiStateTrans = [(stateTrans)[x] for x in viterbiState[:-1]]

lab3_tools.frames2trans(viterbiStateTrans, outfilename='z43a.lab')