from irm2fl.utils import generate_folder

import matplotlib.pyplot as plt
import numpy as np
import os


def import_history(file):
    history_dict = dict()

    with open(file, "r") as file:
        for n, line in enumerate(file):
            if n==0:
                keys = line.replace('\n', '').split('\t')
                history_dict = {key: list() for key in keys}
            else:
                line = line.replace('\n', '').split('\t')
                for k, l in zip(keys, line):
                    history_dict[k].append(int(l) if k=='epoch' else float(l))

    return history_dict

def export_history(dir_model, history):

    history_dict = {'epoch': history.epoch, **history.history}

    generate_folder(dir_model)

    keys = list(history_dict.keys())

    ### Write txt file
    if not os.path.exists(os.path.join(dir_model, 'history.txt')):
        file = open(os.path.join(dir_model, 'history.txt'),"w")
        file.write("\t".join(keys))
        file.write("\n")
        file.close()

    file = open(os.path.join(dir_model, 'history.txt'),"a")
    for i in range(len(history_dict[keys[0]])):
        for j in range(len(keys)):
            file.write(str(history_dict[keys[j]][i])+"\t")
        file.write("\n")
    file.close()

    history_dict = import_history(os.path.join(dir_model, 'history.txt'))

    ### Plot figure
    plt.figure(figsize=(11, 9))
    keys = [key for key in keys if not 'val' in key and not 'epoch' in key]
    cols = 2 if len(keys) > 1 else 1
    rows = int(np.ceil(len(keys)/cols))
    x = history_dict['epoch']
    for k, key in enumerate(keys):
        plt.subplot(cols, rows, k + 1)
        y = [float(item) for item in history_dict[key]]
        plt.plot(x, y, '.-')
        key_val = 'val_' + key if 'val_' + key in history_dict.keys() else None
        if not key_val is None:
            y_val = [float(item) for item in history_dict[key_val]]
            plt.plot(x, y_val, '.-')
        plt.xlabel('Epochs'); plt.ylabel('Loss')
        plt.legend([key, key_val] if not key_val is None else [key])
    plt.savefig(os.path.join(dir_model, 'history.png'))
    plt.close()
