from utils import load, get_samples_and_labels, get_spectrosamples_and_labels
import os
import random
from Waver import Waver
from Spectrum import Spectrum
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# i = 0
# r = 0
p = load("../../pickles_small/testPickle1")
path = "../../segments/testing0"  #to generate just one directory of data
classes_dict = None

#print(path + "/" + "Backhoe_JD50DCompact)")

def gen_dataset(src_path, type, class_dict=None):
    """
    :param src_path: STR
        root of the dataset to read
    :param type: STR
        either spectro or signal, decides which kind of data you want from the audio
    :param class_dict: dict
        dict with the correspondence name-labels
    :return: dict, Object
        return a dict with the classes correspondence,
        and a list of (Objects, label)
    """

    def _read_aux(path, label, datatype):
        ret = []
        if os.path.isdir(path):
            folders = os.listdir(path)
            print(folders)
            for folder in folders:
                files = os.listdir(path + "/" + folder)
                i = 0
                if folder == "ConcreteMixer_onsite.wav":
                    i = -2000
                for file in files:
                    if i < 1000:
                        if datatype == "spectro":
                            val = Spectrum.compute_specgram_and_delta(path + "/" + folder + "/" + file)
                        elif datatype == "signal":
                            val = Waver.get_waveform(path + "/" + folder + "/" + file)
                        else:
                            print("valid input is either spectro or signal")
                        ret.append(val)
                    else:
                        break
                    i += 1
        return ret

    if class_dict is None:
        ret_dict = {}
        class_id = -1
    else:
        ret_dict = class_dict

    classes = os.listdir(src_path)
    dataset = []
    for class_type in classes:
        if class_dict is None:
            class_id += 1
            ret_dict[class_type] = class_id
        else:
            class_id = class_dict[class_type]
        new_data = _read_aux((src_path + "/" + class_type), class_id, type)
        random.shuffle(new_data)
        dataset = dataset+new_data

    random.shuffle(dataset)
    return ret_dict, dataset


type = "spectro"      # "signal" or "spectro"(gram) data generation

classes_dict, test_data = gen_dataset(path, type, classes_dict)

print(len(test_data))

if type == "spectro":
    only_delta_flat = []
    for i in range(len(test_data)):
        only_delta_flat.append([])
        for a in test_data[i][:, :, 1]:
            for b in a:
                only_delta_flat[i].append(b)

    print(only_delta_flat[3])
    print(only_delta_flat[0])


    count = 0
    for elem in test_data[0]:
        count += 1
    print("spectrogram elements: ", count, '\n')
    print('\n', "spectrogram log and delta: ")
    print('\n', test_data[0])
    print('\n', "first pixel spectrogram: ")
    print('\n', test_data[0][0])
    count2 = 0
    for elem in p[0][1]:
        count2 += 1
    count3 = 0
    for elem in p[0][0]:
        count3 += 1
    print("spectrogram elements: ", count2, '\n')
    print("pixel number: ", count3)

    log_spec = test_data[0][:, :, 0]
    delta_spec = test_data[0][:, :, 1]
    print("log: ", '\n')
    print(log_spec)
    print("delta: ", '\n')
    print(delta_spec)
    log_spec1 = test_data[1][:, :, 0]

    count4 = 0
    for elem in delta_spec:
        count4 += 1
    print("delta elements: ", count4)
    count5 = 0
    for elem in log_spec:
        count5 += 1
    print("log elements: ", count5)

    plt.figure()
    librosa.display.specshow(log_spec, y_axis='mel', x_axis='time')
    plt.show()
    plt.figure()
    librosa.display.specshow(log_spec1, y_axis='mel', x_axis='time')
    plt.show()

elif type == "signal":

    print("number of elements for each signal image: ", end=' ')
    count = 0
    for elem in test_data[0]:
        count += 1
    print(count, '\n')
    print("array of values for image0: ", '\n')
    print(test_data[0])
    count1 = 0

print(len(p))
Xtrain, Ytrain = get_spectrosamples_and_labels(p, 'delta')
Xtrain2, Ytrain2 = get_spectrosamples_and_labels(p, 'log')
Xtrain3, Ytrain3 = get_spectrosamples_and_labels(p, 'both')
print('\n', Xtrain[0])
print('\n', Xtrain2[0])
print('\n', Xtrain3[0])
