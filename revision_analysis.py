# %% Imports

import pydevd_pycharm
import sys
sys.path.append("pydevd-pycharm.egg")
pydevd_pycharm.settrace('nestboxx.ddns.net', port=12345, stdoutToServer=True, stderrToServer=True)
import matplotlib.pyplot as plt

# %%

# Check convergence attractors for each parameter:
plt.plot([1,2,3,4])
plt.show()
print("BLAH@")
print("BLAH@")
