import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ===%% Pycharm debug: %%===
import pydevd_pycharm
sys.path.append("pydevd-pycharm.egg")
DEBUG = False
if DEBUG:
    pydevd_pycharm.settrace(
        '79.167.94.93',
        port=12345,
        stdoutToServer=True,
        stderrToServer=True
    )
# ===%% -------------- %%===

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

valid_dir = Path(f"/home/cluster/stefanos/Documents/GitHub/prefrontal-micro"
f"/experiment/network/publication_validation/excitatory_validation_multidend")
sd_file = valid_dir.joinpath('vsoma_sd_normal_CCLAMP_6.0_1.0_1_0_1.75_1_30_1'
                             '.txt')
md_file = valid_dir.joinpath('vsoma_md_normal_CCLAMP_6.0_1.0_1_0_1.75_1_30_1'
                            '.tx)

sd_vsoma = read_train(sd_file)
md_vsoma = read_train(md_file)

plt.plot(sd_vsoma, label='Single/original')
plt.plot(md_vsoma, label='Multi/shord dend')
plt.legend()
plt.savefig("INPUT_RESISTANCE_VALID.png")
#plt.show()

print("Tutto pronto!")
