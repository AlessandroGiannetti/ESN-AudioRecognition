import os
import json
import shutil
import random
import numpy as np

from utils import save
from Waver import Waver
from Spectrum import Spectrum
from pydub import AudioSegment

dataPath = "../dataset/5Classes"
percSplit = 0.7
audioMS = 30
audioHop = 15

def extractFiles(dir):
    l = os.listdir(dir)
    d = {}
    # subd = {}
    l.remove('.DS_Store')
    for elem in l:
        #print(elem)
        sublist = []
        for directory in os.listdir('Utah Audio Data/'+ elem):
            if directory != ".DS_Store":
                sublist.append(directory)
                #print(sublist)
        #if elem == '[New] Concrete Mixer 3':
            #sublist.remove('.DS_Store')
        d[elem] = sublist
    print(d)

# see how to do recursive directories scan

    for elem in l:
        for directories in d[elem]:
            print(directories)
            if directories != '.DS_Store':
                h = os.listdir('Utah Audio Data/' + elem + "/" + directories)
                print(h)


    # print(d)
    # for d[elem] in d:
    #     for subdirectories in d[elem]:
    #         helplist = os.listdir('Utah Audio Data/'+ d[elem] + "/" + subdirectories)
    #         print(helplist)
    #         subd[d[elem]] = helplist
    #         print(subd)
    # return l

def partition_track(track_path, out_path, ms, hop=None, get_drop=False):
    """
    :param track_path: str
        path of the track to partition
    :param out_path: str
        directory in which save the chuncks, if it do not exist it is created
    :param ms: int
        duration of the chunk to generate
    :param hop: int
        step to start next chunck
    :param get_drop: bool
        generate a track with the discarded audio
    :return: None
    """

    if hop is None:
        hop = ms

    audio = AudioSegment.from_file(track_path)

    size = len(audio)
    segments = []
    i = 0
    while i + ms < size:
        next_segment = audio[i:i + ms]
        segments.append(next_segment)
        i = i + hop

    form = track_path.split(".")[-1]

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    rms = np.array([x.rms for x in segments])
    q1 = np.percentile(rms, .25)
    q3 = np.percentile(rms, .75)

    lf = q1 - 1.5*(q3-q1)

    lo_sil = AudioSegment.empty()

    for idx, segment in enumerate(segments):
        if segment.rms < lf:
            lo_sil = lo_sil + segment
        else:
            segment.export(os.path.join(out_path, str(str(idx) + "." + form)), format=form)

    if get_drop:
        lo_sil.export(os.path.join(out_path, 'lo_sil.wav'), format=form)


def partition_dataset(in_path, out_path, ms, hop):
    """
    :param in_path: str
        root of the dataset where to start
    :param out_path: str
        root of the new dataset
    :param ms: int
        length of the chuncks
    :param hop: int
        ms after start next chunck
    :return:
    """
    if not os.path.isdir(in_path):
        print(out_path)
        partition_track(in_path, str(out_path), ms, hop)

    else:
        folders = os.listdir(in_path)
        for folder in folders:
            new_in_path = os.path.join(in_path, str(folder))
            new_out_path = os.path.join(out_path, str(folder))
            partition_dataset(new_in_path, new_out_path, ms, hop)


def gen_dataset(src_path, class_dict=None):
    """
    :param src_path: STR
        root of the dataset to read
    :param class_dict: dict
        dict with the correspondence name-labels
    :return: dict, Object
        return a dict with the classes correspondence,
        and a list of (Objects, label)
    """

    def _read_aux(path, label):
        ret = []
        if (not os.path.isdir(path)) and path.endswith('.wav'):
            val = (Waver.get_waveform(path), Spectrum.compute_specgram_and_delta(path), label)
            ret.append(val)
        elif os.path.isdir(path):
            folders = os.listdir(path)
            for folder in folders:
                ret += _read_aux(os.path.join(path, str(folder)), label)
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
        print(class_type)
        new_data = _read_aux(os.path.join(src_path, class_type), class_id)
        random.shuffle(new_data)
        print('class size is: ', len(new_data))
        dataset = dataset+new_data

    random.shuffle(dataset)
    return ret_dict, dataset


def get_samples_and_labels(data):
    """
    :param data: data to unpack
    :return: X, Z, Y:
        X - list of objects to classify
        Z - secondary list of objects to classify
        Y - list of labels of the objects
    """
    X = []
    Z = []
    Y = []
    for x, z, y in data:
        X.append(x)
        Z.append(z)
        Y.append(y)
    return X, Z, Y


def generate_config(config_path='config.json', dataset_path=dataPath, percentage=percSplit, audio_ms=audioMS, audio_hop=audioHop, overwrite=False):
    """
    :param config_path: str
        path to config file
    :param dataset_path: str
        path to dataset root
    :param percentage:
    :param audio_ms:
    :param audio_hop:
    :param overwrite: bool
        whether to overwrite the old config with the new one
    :return: dict
        param dict
    """
    params = dict()

    try:
        with open(config_path, mode = 'r', encoding='utf-8') as fin:
            params = json.load(fin)
    except Exception as e:
        print(e)
        params['DATA_PATH'] = dataset_path
        params['PERCENTAGE'] = percentage
        params['AUDIO_MS'] = audio_ms
        params['HOP_MS'] = audio_hop

    params['PICKLES_FOLDER'] = "../dataset/pickles/ms" + str(params['AUDIO_MS']) + "_hop" + str(params['HOP_MS'])
    params['TRAIN_PICKLE'] = params['PICKLES_FOLDER'] + "/train.p"
    params['TEST_PICKLE'] = params['PICKLES_FOLDER'] + "/test.p"
    params['DICT_JSON'] = params['PICKLES_FOLDER'] + "/classes.json"

    if overwrite:
        with open(config_path, mode='w+', encoding='utf-8') as fout:
            json.dump(params, fout)

    return params


if __name__ == '__main__':
    list = extractFiles('Utah Audio Data')
    #print(list)
    #print(list[0])

