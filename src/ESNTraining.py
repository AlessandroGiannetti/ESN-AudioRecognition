from utils import load, get_samples_and_labels, get_spectrosamples_and_labels
import numpy as np
from easyesn import ClassificationESN, OneHotEncoder


def mostprobableclass(arr, classcounter, classindex, segments):
    """

    :param arr: list
        labels
    :param classcounter: list
        number of elements per class
    :param classindex: list
        indexes to jump to every class
    :param segments: int
        number of segments to evaluate every loop
    :return: list
        most likely class every x segments
    """
    t = 0
    ans = []
    for i, class_val in enumerate(classcounter):
        s = segments
        for h in range(class_val//s):
            s = t + segments
            hash_func = dict()
            for data_chunk in range(t, (s-1)):
                if arr[data_chunk] in hash_func.keys():
                    hash_func[arr[data_chunk]] += 1
                else:
                    hash_func[arr[data_chunk]] = 1
            max_count = 0
            ret = -1
            for g in hash_func:
                if max_count < hash_func[g]:
                    ret = g
                    max_count = hash_func[g]

            ans.append(ret)
            t += segments
        t = classindex[i] + 1
    return ans


trainP = load("../../pickles_small/testPickle4")
testP = load("../../pickles_modified/trainPickle4")
classes = ['Backhoe_JD50DCompact', 'Compactor_Ingersoll_Rand', 'Concrete_Mixer_0',
           'Excavator_CAT320E', 'Excavator_Hitachi50U']

Xtrain, Ytrain = get_spectrosamples_and_labels(trainP, 'log')
Xtest, Ytest = get_spectrosamples_and_labels(testP, 'log')

Xtrain = np.array(Xtrain)
Ytrain = np.array(Ytrain)
Xtest = np.array(Xtest)
Ytest = np.array(Ytest)

# hot encodings
midYtrain = Ytrain.reshape(-1)
oheYtrain = np.eye(5)[midYtrain]

midYtest = Ytest.reshape(-1)
oheYtest = np.eye(5)[midYtest]

testclasscounter = [np.count_nonzero(Ytest == i) for i in range(len(classes))]  # how many elements per class
testclassindex = []
ind = 0
for elem in testclasscounter:  # creating indexes to jump from class to class directly
    ind += elem
    testclassindex.append(ind)
print("number of elements per class: ", testclasscounter)
print('-'*80)
# ----------------ESN-------------------
training = True
if training:
    esn = ClassificationESN(1, 1200, 5, leakingRate=0.1)
    esn.fit(Xtrain, oheYtrain, verbose=1)

    oheres = esn.predict(Xtest, verbose=1)
    res = [np.where(r == 1)[0][0] for r in oheres]  # undo one hot encoding
if not training:
    res = []
    with open('predictions_best.txt', 'r') as predictions:
        for line in predictions:
            res.append(int(line))

print("                         Single segment predictions:\n")

for i in range(-1, 4):
    print('predictions for class {}: '.format(i+1))
    if i == -1:
        for a in range(testclassindex[0]):
            print(res[a], end='')
    else:
        for a in range(testclassindex[i], testclassindex[i+1]):
            print(res[a], end='')
    print('\n')

corrects = [0]*6
for index in range(len(res)):
    if res[index] == Ytest[index]:
        corrects[len(classes)] += 1
        corrects[res[index]] += 1
print("*"*40)
print("Single segment accuracy is: {} %".format(corrects[len(classes)]/(len(res))*100))
print("{}/{}".format(corrects[len(classes)], len(res)))
print("*"*40)
for ind in range(5):
    print("Segment accuracy for {} is: {} %".format(classes[ind], corrects[ind]/testclasscounter[ind]*100))
    print("{}/{}\n".format(corrects[ind], testclasscounter[ind]))

print("-"*80)

if training:
    with open('results.txt', 'w') as results:  # save the predictions in a .txt file
        for val in res:
            results.write('%s\n' % val)

segments_num = 1000  # 67 for 2 seconds, 200 for 6 seconds
res = mostprobableclass(res, testclasscounter, testclassindex, segments_num)
Ytest = mostprobableclass(Ytest, testclasscounter, testclassindex, segments_num)

newclasscounter = [Ytest.count(i) for i in range(5)]
newclassindex = [0]

ind = 0
for elem in newclasscounter:
    ind += elem
    newclassindex.append(ind)

# -------------------display predictions---------------------

print("         Results for {} seconds of audio before class prediction:  \n".format(30*segments_num/1000))
for i in range(5):
    print('predictions for class {}: '.format(i))
    for a in range(newclassindex[i], newclassindex[i+1]):
        print(res[a], end='')
    print('\n')

corrects = [0]*6
for index in range(len(res)):
    if res[index] == Ytest[index]:
        corrects[len(classes)] += 1
        corrects[res[index]] += 1

print("*"*25)
print("Final accuracy is: {} %".format(corrects[len(classes)]/(len(res))*100))
print("{}/{}".format(corrects[len(classes)], len(res)))
print("*"*25)
print('\n')
for ind in range(5):
    print("Final accuracy for {} is: {} %".format(classes[ind], corrects[ind]/newclasscounter[ind]*100))
    print("{}/{}\n".format(corrects[ind], newclasscounter[ind]))
