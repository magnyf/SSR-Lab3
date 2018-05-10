import numpy as np
import matplotlib.pyplot as pl
from math import log, exp, fabs
import collections
import copy


def logsumexp(arr, axis=0):
    """Computes the sum of arr assuming arr is in the log domain.
    Returns log(sum(exp(arr))) while minimizing the possibility of
    over/underflow.
    """
    arr = np.rollaxis(arr, axis)
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    vmax = arr.max(axis=0)
    if vmax.ndim > 0:
        vmax[~np.isfinite(vmax)] = 0
    elif not np.isfinite(vmax):
        vmax = 0
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(arr - vmax), axis=0))
        out += vmax
        return out

def log_multivariate_normal_density_diag(X, means, covars):
    """Compute Gaussian log-density at X for a diagonal model

    Args:
        X: array like, shape (n_observations, n_features)
        means: array like, shape (n_components, n_features)
        covars: array like, shape (n_components, n_features)

    Output:
        lpr: array like, shape (n_observations, n_components)
    From scikit-learn/sklearn/mixture/gmm.py
    """
    n_samples, n_dim = X.shape
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                  + np.sum((means ** 2) / covars, 1)
                  - 2 * np.dot(X, (means / covars).T)
                  + np.dot(X ** 2, (1.0 / covars).T))
    return lpr



# data = np.load('lab2_data.npz')['data']

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

modellist = {}
for digit in prondict.keys():
	modellist[digit] = ['sil'] + prondict[digit] + ['sil']

def concatHMMs(hmmmodels, namelist):
    N = len(namelist)
    word = {}
    word['name'] = 'o'
    word['startprob'] = [0 for i in range(3 * N + 4)]
    word['startprob'][0] = 1.0
    word['means'] = [[0 for j in range(13)] for i in range(3*N)]
    word['covars'] = [[0 for j in range(13)] for i in range(3*N)]

    #compute means and covars
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


    #compute the tranmat matrice, with your code
    transMat = [phoneHMMs[k]['transmat'] for k in namelist]
    #for k in namelist:
    #    tranMat += [phoneHMMs[k]['transmat']]
    n = 0
    m = 0
    for x in transMat:
        n += len(x) -1
    n += 1

    result = [[0 for y in range(n)] for x in range(n)]

    i0 = 0
    j0 = 0
    for x  in transMat:
        for i in range(len(x)):
            for j in range(len(x)):
                result[i0+i][j0+j] = x[i][j]
        i0 += len(x)-1
        j0 += len(x)-1
    result[-1][-1]  = 1
    word['transmat'] = result
    return word

def log_inf(x):
	y = copy.deepcopy(x)

	if not(isinstance(x[0], collections.Iterable)):
	    # not iterable
	    for i in range(len(x)):
	    	if (x[i] > 0):
	    		y[i] = log(x[i])
	    	else:
	    		y[i] = -float('Inf')
	else:
		# iterable
		for i in  range(len(x)):
			for j in  range(len(x[0])):
				if (x[i][j] > 0):
					y[i][j] = log(x[i][j]) 
				else:
					y[i][j] = -float('Inf')
	return y



def forward(log_emlik, log_startprob, log_transmat):
	N = len(log_emlik)
	M = len(log_emlik[0])

	logAlpha = [[0 for x in range(M)] for y in range(N)]

	for j in range(M):
		logAlpha[0][j] = log_startprob[j] + log_emlik[0][j]

	for n in range(1, N):
		for j in range(M):
			logAlpha[n][j] = log_emlik[n][j]
			# building the array of the log sum
			sumArray = []
			for i in range(M):
				sumArray += [logAlpha[n-1][i] + log_transmat[i][j]]
			logAlpha[n][j] +=  logsumexp(np.array(sumArray))			
	return logAlpha

def gmmloglik(logAlpha):
	return logsumexp(np.array(logAlpha[-1]))

def viterbi(log_emlik, log_startprob, log_transmat):
        N = len(log_emlik)
        M = len(log_emlik[0])
        V = [[0 for j in range(M)] for n in range(N)]
        B = [[0 for j in range(M)] for n in range(N)]
        viterbi_path = [0 for i in range(N)]
        for j in range(M):
                V[0][j] = log_startprob[j] + log_emlik[0][j]
        for n in range(1,N):
                for j in range(M):
                        current = V[n-1][0] + log_transmat[0][j]
                        precedent = current
                        for i in range(M):
                                current = max(current, V[n-1][i] + log_transmat[i][j])
                                if (current != precedent):
                                        B[n][j] = i
                                precedent = current
                        V[n][j] = current + log_emlik[n][j]

        viterbi_path[N-1] = np.argmax(V[N-1])
        for i in range(N-2, 0, -1):
                viterbi_path[i] = B[i+1][viterbi_path[i+1]]
        viterbi_loglik = max(V[N-1])
        return (viterbi_loglik, np.array(viterbi_path))

def backward(log_emlik, log_startprob, log_transmat):
	N = len(log_emlik)
	M = len(log_emlik[0])

	logBeta = [[0 for x in range(M)] for y in range(N)]

	for i in range(M):
		logBeta[N-1][i] = 0
	for n in range(N-2, -1, -1):
		for i in range(M):
			# building the array of the log sum
			sumArray = []
			for j in range(M):
				sumArray += [log_transmat[i][j] + log_emlik[n+1][j] + logBeta[n+1][j]]
			
			logBeta[n][i] =  logsumexp(np.array(sumArray))			
	return logBeta 

def statePosteriors(log_alpha, log_beta):
        N = len(log_alpha)
        M = len(log_alpha[0])
        sum = 0
        y = [[ 0 for j in range(M)] for i in range(N) ]
        sum = logsumexp(log_alpha[N-1])
        for n in range(N):
                for j in range(M):
                           y[n][j] = log_alpha[n][j] + log_beta[n][j] - sum
        log_gamma = y
        return log_gamma

def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
	
	N = len(log_gamma)
	M = len(log_gamma[0])
	D = len(X[0])

	means =  [[0 for j in range(D)] for i in range(M)]
	covars = [[0 for j in range(D)] for i in range(M)]

	for j in range(M):
		sumGammaBottom = 0
		for n in range(N):
			sumGammaBottom += exp(log_gamma[n][j])
		for i in range(D):
			sumGammaTop = 0
			for n in range(N):
				sumGammaTop += exp(log_gamma[n][j])*X[n][i]
				means[j][i] = sumGammaTop/sumGammaBottom
	for j in range(M):
		sumGammaBottom = 0
		for n in range(N):
			sumGammaBottom += exp(log_gamma[n][j])
		for i in range(D):
			sumGammaTop = 0
			for n in range(N):
				xMinusMean = np.subtract(X[n], means[j])
				covarVector = np.multiply(xMinusMean, xMinusMean)
				sumGammaTop += exp(log_gamma[n][j])*covarVector[i]
				covars[j][i] = sumGammaTop/sumGammaBottom
				if (covars[j][i] < varianceFloor):
					covars[j][i] = varianceFloor

	return means, covars



#######################################################################################
## TEST
#######################################################################################





#######################################################################################
## MAIN
#######################################################################################


# lets compute the loglikelihood for utterance 10 (digit 4)

# X = data[32]['lmfcc']
 
# wordHMMs = concatHMMs(phoneHMMs, modellist['9'])

# obsloglik =log_multivariate_normal_density_diag(np.array(X), 
# 		np.array(wordHMMs['means']), 
# 		np.array(wordHMMs['covars']))

# pi = wordHMMs['startprob']
# concatMat = wordHMMs['transmat']

# logAlpha = forward(obsloglik, log_inf(pi), log_inf(concatMat))
# loglik4 = gmmloglik(logAlpha)
# print(loglik4)
# for iterations in range(20):
# 	oldloglik = loglik4

# 	# expectaion
# 	alpha = forward(obsloglik, log_inf(pi), log_inf(concatMat))
# 	beta = backward(obsloglik, log_inf(pi), log_inf(concatMat))
# 	gamma = statePosteriors(np.array(alpha), np.array(beta))

# 	# Maximization
# 	wordHMMs['means'], wordHMMs['covars'] = updateMeanAndVar(X, gamma, varianceFloor=5.0)
	
# 	obsloglik =log_multivariate_normal_density_diag(np.array(X), 
# 		np.array(wordHMMs['means']), 
# 		np.array(wordHMMs['covars']))

# 	logAlpha = forward(obsloglik, log_inf(pi), log_inf(concatMat))
# 	loglik4 = gmmloglik(logAlpha)
# 	print(loglik4)

# 	if (fabs(loglik4 - oldloglik) < 1):
# 		print('breaking at step'+str(iterations))
# 		break