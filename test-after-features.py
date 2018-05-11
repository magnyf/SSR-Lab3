import numpy as np
import os
import matplotlib.pyplot as pl
from sklearn.mixture import *
from sklearn.model_selection import train_test_split

def path2info(path):
    """
    path2info: parses paths in the TIDIGIT format and extracts information
               about the speaker and the utterance

    Example:
    path2info('tidigits/disc_4.1.1/tidigits/train/man/ae/z9z6531a.wav')
    """
    rest, filename = os.path.split(path)
    rest, speakerID = os.path.split(rest)
    rest, gender = os.path.split(rest)
    digits = filename[:-5]
    repetition = filename[-5]
    return gender, speakerID, digits, repetition


traindata = np.load('traindata.npz')['traindata']

testdata = np.load('testdata.npz')['testdata']

################
### TRAINING SET
################

## in order to comply with the requirement of having the same proportion of men and women
## and having speakers in only one training set and not the other, we will do the split
## twice (once for men and once for women) and do the split on the speckers rather than on 
## the utterances.

listSpeakersTrain = [(path2info(x['filename'])[0], path2info(x['filename'])[1], list(traindata).index(x)) for x in traindata]

listSpeakersTrainMaleWithRepetitions = []
listSpeakersTrainFemaleWithRepetitions = []
listSpeakersTrainMale = []
listSpeakersTrainFemale = []

for x in listSpeakersTrain:
    if (x[0] == 'man'):
        listSpeakersTrainMaleWithRepetitions += [(x[1], x[2])]
        listSpeakersTrainMale += [x[1]]
    else:
        listSpeakersTrainFemaleWithRepetitions += [(x[1], x[2])]
        listSpeakersTrainFemale += [x[1]]


# removing repetitions
listSpeakersTrainMale = list(set(listSpeakersTrainMale))
listSpeakersTrainFemale = list(set(listSpeakersTrainFemale))

# computing the list of the utterances by each speaker:
listUtterancesSpeakerMale = []
listUtterancesSpeakerFemale = []

for x in listSpeakersTrainMale:
    temp = []
    for y in listSpeakersTrainMaleWithRepetitions:
        if (y[0] == x):
            temp += [y[1]]
    listUtterancesSpeakerMale += [temp]

for x in listSpeakersTrainFemale:
    temp = []
    for y in listSpeakersTrainFemaleWithRepetitions:
        if (y[0] == x):
            temp += [y[1]]
    listUtterancesSpeakerFemale += [temp]

# here we will use the list of utterences' indexes as X and the speaker's id as y
listUtterancesSpeakerMale_train, listUtterancesSpeakerMale_val, listSpeakersTrainMale_train, listSpeakersTrainMale_val = train_test_split(listUtterancesSpeakerMale, listSpeakersTrainMale, test_size=0.1)
listUtterancesSpeakerFemale_train, listUtterancesSpeakerFemale_val, listSpeakersTrainFemale_train, listSpeakersTrainFemale_val = train_test_split(listUtterancesSpeakerFemale, listSpeakersTrainFemale, test_size=0.1)

##Now that the speakers are separated with the right repartition, we can manually create X and y
mspec_train_x = []
lmfcc_train_x = []
mspec_val_x = []
lmfcc_val_x = []
train_y = []
val_y = []

## we create a list of each frame and its label
for i in range(len(listSpeakersTrainMale_train)):
    for j in listUtterancesSpeakerMale_train[i]:
        for m in range(len(traindata[j]['mspec'])-1): ## -1 because last frame not labeled by viterbi ??? pas sur que ca soit ca la raison mais en tout cas il y a un écart d'un dans la longueur du fichier
            mspec_train_x += [traindata[j]['mspec'][m]]
            lmfcc_train_x += [traindata[j]['lmfcc'][m]]
            train_y += [traindata[j]['targets'][m]]

for i in range(len(listSpeakersTrainFemale_train)):
    for j in listUtterancesSpeakerFemale_train[i]:
        for m in range(len(traindata[j]['mspec'])-1):
            mspec_train_x += [traindata[j]['mspec'][m]]
            lmfcc_train_x += [traindata[j]['lmfcc'][m]]
            train_y += [traindata[j]['targets'][m]]


for i in range(len(listSpeakersTrainMale_val)):
    for j in listUtterancesSpeakerMale_val[i]:
        for m in range(len(traindata[j]['mspec'])-1):
            mspec_val_x += [traindata[j]['mspec'][m]]
            lmfcc_val_x += [traindata[j]['lmfcc'][m]]
            val_y += [traindata[j]['targets'][m]]

for i in range(len(listSpeakersTrainFemale_val)):
    for j in listUtterancesSpeakerFemale_val[i]:
        for m in range(len(traindata[j]['mspec'])-1):
            mspec_val_x += [traindata[j]['mspec'][m]]
            lmfcc_val_x += [traindata[j]['lmfcc'][m]]
            val_y += [traindata[j]['targets'][m]]




################
### TEST SET
################


listSpeakersTest = [(path2info(x['filename'])[0], path2info(x['filename'])[1], list(testdata).index(x)) for x in testdata]

listSpeakersTestMaleWithRepetitions = []
listSpeakersTestFemaleWithRepetitions = []
listSpeakersTestMale = []
listSpeakersTestFemale = []

for x in listSpeakersTest:
    if (x[0] == 'man'):
        listSpeakersTestMaleWithRepetitions += [(x[1], x[2])]
        listSpeakersTestMale += [x[1]]
    else:
        listSpeakersTestFemaleWithRepetitions += [(x[1], x[2])]
        listSpeakersTestFemale += [x[1]]


# removing repetitions
listSpeakersTestMale = list(set(listSpeakersTestMale))
listSpeakersTestFemale = list(set(listSpeakersTestFemale))

# computing the list of the utterances by each speaker:
listUtterancesSpeakerMale = []
listUtterancesSpeakerFemale = []

for x in listSpeakersTestMale:
    temp = []
    for y in listSpeakersTestMaleWithRepetitions:
        if (y[0] == x):
            temp += [y[1]]
    listUtterancesSpeakerMale += [temp]

for x in listSpeakersTestFemale:
    temp = []
    for y in listSpeakersTestFemaleWithRepetitions:
        if (y[0] == x):
            temp += [y[1]]
    listUtterancesSpeakerFemale += [temp]


mspec_test_x = []
lmfcc_test_x = []
test_y = []
for i in range(len(listSpeakersTrainMale)):
    for j in listUtterancesSpeakerMale[i]:
        for m in range(len(testdata[j]['mspec'])-1): ## -1 because last frame not labeled by viterbi ??? pas sur que ca soit ca la raison mais en tout cas il y a un écart d'un dans la longueur du fichier
            mspec_test_x += [testdata[j]['mspec'][m]]
            lmfcc_test_x += [testdata[j]['lmfcc'][m]]
            test_y += [testdata[j]['targets'][m]]

for i in range(len(listSpeakersTrainFemale)):
    for j in listUtterancesSpeakerFemale[i]:
        for m in range(len(testdata[j]['mspec'])-1):
            mspec_test_x += [testdata[j]['mspec'][m]]
            lmfcc_test_x += [testdata[j]['lmfcc'][m]]
            test_y += [testdata[j]['targets'][m]]


print(len(mspec_test_x))
print(len(lmfcc_test_x))
print(len(test_y))