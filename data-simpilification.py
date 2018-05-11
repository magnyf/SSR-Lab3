import numpy as np

phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()
phones = sorted(phoneHMMs.keys())
nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]

## Whole datasets
traindata = np.load('traindata.npz')['traindata']
testdata = np.load('testdata.npz')['testdata']

### Smaller sets to improve speed (the dataset is lame after this, no diversity)
# traindata = np.load('traindata.npz')['traindata'][:1000]
# testdata = np.load('testdata.npz')['testdata'][:1000]


### To save memory it was better to save the index rather than the name of the state,
## I didn't do it when generating the file so I fix this here
## TODO save the file again after the changes to save time afterwards

for x in traindata:
    x['targets'] = [stateList.index(y) for y in x['targets']]
for x in testdata:
    x['targets'] = [stateList.index(y) for y in x['targets']]



traindataSimple = traindata[::10]
testdataSimple = testdata[::10]

print(len(traindataSimple))
print(len(testdataSimple))

np.savez('traindataSimple.npz', traindata=traindataSimple)
np.savez('testdataSimple.npz', testdata=testdataSimple)
