import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def read_train(fn):
    '''
    Read train from file, given filename
    :param arg:
    :return:
    '''
    # TODO: more elegant?
    def str2float(s):
        result = None
        try:
            result = float(s)
        except:
            pass
        return result

    with open(fn) as f:
        # Why not f.readlines() ?
        values = list(map(str2float, f.readlines()))
        # Remove empty rows/values:
        values = list(filter(None.__ne__, values))
    return values

valid_dir = Path(fr"\\139.91.162.90\cluster\stefanos\Documents\GitHub\prefrontal-micro\experiment\network\publication_validation\excitatory_validation_multidend")
sd_file = valid_dir.joinpath('vsoma_sd_normal_CCLAMP_6.0_1.0_1_0_1.75.txt')
md_file = valid_dir.joinpath('vsoma_md_normal_CCLAMP_6.0_1.0_1_0_1.75.txt')

sd_vsoma = read_train(sd_file)
md_vsoma = read_train(md_file)

plt.plot(sd_vsoma, label='Single/original')
plt.plot(md_vsoma, label='Multi/shord dend')
plt.legend()
plt.show()

print("Tutto pronto!")
