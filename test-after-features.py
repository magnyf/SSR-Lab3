import numpy as np
import os
import matplotlib.pyplot as pl
from sklearn.mixture import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation

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

phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()
phones = sorted(phoneHMMs.keys())
nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]

## Whole datasets
# traindata = np.load('traindata.npz')['traindata']
# testdata = np.load('testdata.npz')['testdata']
# ### To save memory it was better to save the index rather than the name of the state,
# ## I didn't do it when generating the file so I fix this here
# ## TODO save the file again after the changes to save time afterwards

# for x in traindata:
#     x['targets'] = [stateList.index(y) for y in x['targets']]
# for x in testdata:
#     x['targets'] = [stateList.index(y) for y in x['targets']]


### Smaller sets to improve speed (the dataset is lame after this, no diversity)
traindata = np.load('traindataSimple.npz')['traindata']
testdata = np.load('testdataSimple.npz')['testdata']



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

### Smaller sets to improve speed: idea divide by 100 the size
arrays = [mspec_train_x, lmfcc_train_x, 
mspec_val_x, lmfcc_val_x, 
mspec_test_x, lmfcc_test_x,
train_y, 
val_y,
test_y]

for array in arrays:
    array = array[::100]
print('test set size')
print(len(mspec_test_x))


print('Feature extraction DONE')



############
## Dynamic Features
############

dmspec_train_x = []
dlmfcc_train_x = []
dmspec_val_x = []
dlmfcc_val_x = []
dmspec_test_x = []
dlmfcc_test_x = []

for i in range(len(mspec_train_x)):
    tempMspec = []
    tempLmfcc = []
    n = len(mspec_train_x)
    for j in range(-3, 4):
        tempMspec += list(mspec_train_x[(i+j)%n])
        tempLmfcc += list(lmfcc_train_x[(i+j)%n])
    assert(len(tempMspec) == len(mspec_train_x[0])*7)
    assert(len(tempLmfcc) == len(lmfcc_train_x[0])*7)

    dmspec_train_x += [tempMspec]
    dlmfcc_train_x += [tempLmfcc]

for i in range(len(mspec_val_x)):
    tempMspec = []
    tempLmfcc = []
    n = len(mspec_val_x)
    for j in range(-3, 4):
        tempMspec += list(mspec_val_x[(i+j)%n])
        tempLmfcc += list(lmfcc_val_x[(i+j)%n])
    assert(len(tempMspec) == len(mspec_val_x[0])*7)
    assert(len(tempLmfcc) == len(lmfcc_val_x[0])*7)

    dmspec_val_x += [tempMspec]
    dlmfcc_val_x += [tempLmfcc]


for i in range(len(mspec_test_x)):
    tempMspec = []
    tempLmfcc = []
    n = len(mspec_test_x)
    for j in range(-3, 4):        
        tempMspec += list(mspec_test_x[(i+j)%n])
        tempLmfcc += list(lmfcc_test_x[(i+j)%n])
    assert(len(tempMspec) == len(mspec_test_x[0])*7)
    assert(len(tempLmfcc) == len(lmfcc_test_x[0])*7)

    dmspec_test_x += [tempMspec]
    dlmfcc_test_x += [tempLmfcc]

print('Dynamic features DONE')

#################
# Standardisation

## la plus simple, sur l'ensemble du dataset
scaler_lmfcc = StandardScaler()
scaler_dlmfcc = StandardScaler()
scaler_mspec = StandardScaler()
scaler_dmspec = StandardScaler()

lmfcc_train_x = scaler_lmfcc.fit_transform(lmfcc_train_x)
dlmfcc_train_x = scaler_dlmfcc.fit_transform(dlmfcc_train_x)
mspec_train_x = scaler_mspec.fit_transform(mspec_train_x)
dmspec_train_x = scaler_dmspec.fit_transform(dmspec_train_x)

## test to see if the mean and variance are normalized
# scalerTest_lmfcc = StandardScaler()
# scalerTest_lmfcc.fit(lmfcc_train_x)
# print(scalerTest_lmfcc.mean_)
# print(scalerTest_lmfcc.var_)

# transform for the other sets
lmfcc_val_x = scaler_lmfcc.transform(lmfcc_val_x)
dlmfcc_val_x = scaler_dlmfcc.transform(dlmfcc_val_x)
mspec_val_x = scaler_mspec.transform(mspec_val_x)
dmspec_val_x = scaler_dmspec.transform(dmspec_val_x)

lmfcc_test_x = scaler_lmfcc.transform(lmfcc_test_x)
dlmfcc_test_x = scaler_dlmfcc.transform(dlmfcc_test_x)
mspec_test_x = scaler_mspec.transform(mspec_test_x)
dmspec_test_x = scaler_dmspec.transform(dmspec_test_x)

#### Format change to use Keras and a GPU

lmfcc_train_x = lmfcc_train_x.astype('float32')
dlmfcc_train_x = dlmfcc_train_x.astype('float32')
mspec_train_x = mspec_train_x.astype('float32')
dmspec_train_x = dmspec_train_x.astype('float32')

lmfcc_val_x = lmfcc_val_x.astype('float32')
dlmfcc_val_x = dlmfcc_val_x.astype('float32')
mspec_val_x = mspec_val_x.astype('float32')
dmspec_val_x = dmspec_val_x.astype('float32')

lmfcc_test_x = lmfcc_test_x.astype('float32')
dlmfcc_test_x = dlmfcc_test_x.astype('float32')
mspec_test_x = mspec_test_x.astype('float32')
dmspec_test_x = dmspec_test_x.astype('float32')

output_dim = len(stateList)

print(output_dim)

train_y = np_utils.to_categorical(train_y, output_dim)
val_y = np_utils.to_categorical(val_y, output_dim)
test_y = np_utils.to_categorical(test_y, output_dim)

print('Standardisation DONE')


##### PART 5
## TODO
# Define the proper size for the input and output layers depending on your feature vectors 
# and number of states. Choose the appropriate activation function for the output layer, 
# given that you want to perform classification. Be prepared to explain why you chose the 
# specific activation and what alternatives there are. For the intermediate layers you can
# choose, for example, between relu and sigmoid activation functions.

#### toutes les options sont choisies au pif pour l'instant
def trainAndEvaluate(typeFeature, numberOfLayers=1, epoch=10):
    print()
    print()
    print('----------'+typeFeature+' with '+str(numberOfLayers)+' layer-----------')

    if (typeFeature == "lmfcc"):
        input_dim = 13
        train_x = lmfcc_train_x
        test_x = lmfcc_test_x
        val_x = lmfcc_val_x
    elif (typeFeature == "dlmfcc"):
        input_dim = 91
        train_x = dlmfcc_train_x
        test_x = dlmfcc_test_x
        val_x = dlmfcc_val_x
    elif (typeFeature == "mspec"):
        input_dim = 40
        train_x = mspec_train_x
        test_x = mspec_test_x
        val_x = mspec_val_x
    elif (typeFeature == "dmspec"):
        input_dim = 280
        train_x = dmspec_train_x
        test_x = dmspec_test_x
        val_x = dmspec_val_x

    model = Sequential()

    model.add(Dense(256, input_dim=input_dim))
    model.add(Activation('relu'))

    for i in range(numberOfLayers-1):
        model.add(Dense(256))
        model.add(Activation('relu'))

    model.add(Dense(output_dim))
    model.add(Activation('relu'))


    ## in the documentation these are the ones adapted to a multi-class classification problem
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    history = model.fit(train_x, train_y, epochs=10, batch_size=256, validation_data=(val_x, val_y), verbose=2)
    # validation_data: tuple (val_x, y_val) or tuple (val_x, y_val, val_sample_weights) on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. 

    print(history.history)

    evaluation = model.evaluate(x=test_x, y=test_y, batch_size=256, steps=None, verbose=1)

    print(evaluation)

for typeFeature in ["lmfcc","dlmfcc","mspec","dmspec"]:
    for numberOfLayers in range(1,5):
        trainAndEvaluate(typeFeature, numberOfLayers)