import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import pandas as pd
import h5py

def quick_spikes(voltage_trace=None, upper_threshold=None, lower_threshold=None, steps_per_ms=10, plot=False):
    #   ADVANCED_SPIKE_COUNT(vt, lt, ht) find spikes in voltage trace vt ( that
    #   first cross high threshold and again low threshold).
    #   ADVANCED_SPIKE_COUNT(vt, lt, ht, 'threshold',true) will return spikes
    #   detected where NEURON with similar High threshold (ht) would (WIP).
    #   ADVANCED_SPIKE_COUNT(vt, lt, ht, 'plot',true) also plot results.
    #
    #   This updated function can handle:
    #   > voltage train without spikes
    #   > noisy (high freq) voltage train
    #
    #   author stamatiad.st@gmail.com

    #Find values above high threshold:
    upper_crossings = np.greater(voltage_trace, upper_threshold)
    lower_crossings = np.logical_not(upper_crossings)
    # You want the points where the vt crosses the upper threshold and again the lower one.
    # Simply detect crossing the upper threshold, across vt:
    ts_1 = np.add(upper_crossings.astype(int), np.roll(lower_crossings, 1).astype(int))
    spikes_start = np.nonzero(np.greater(ts_1, 1))[0]
    # Simply detect crossing the lower threshold, across vt:
    ts_2 = np.add(upper_crossings.astype(int), np.roll(lower_crossings.astype(int), -1))
    spikes_end = np.nonzero(np.greater(ts_2, 1))[0]
    # Make sure that we have the same amount of starts/ends:
    if spikes_start.size != spikes_end.size:
        raise ValueError('Check algorithm. Why is this happening?')
    # Then, get the maximum voltage in this region.
    spike_timings = []
    for start, stop in zip(spikes_start, spikes_end):
        spike_timings.append((np.argmax(voltage_trace[start:stop+1]) + start) / steps_per_ms)
    # Plot if requested.
    if plot:
        #vt_reduced = voltage_trace.loc[::steps_per_ms]
        vt_reduced = voltage_trace[::steps_per_ms]
        # Re index it:
        #vt_reduced.index = (range(vt_reduced.size))
        fig, ax = plt.subplots()
        ax.plot(vt_reduced)
        for st in spike_timings:
            plt.scatter(st, 50, s=10, marker='v', c='tab:red')

        ax.set(xlabel='time (ms)', ylabel='voltage (mV)',
               title='Spike events')

        #fig.savefig("test.png")
        plt.show()



    return spike_timings

def q_generator(q_size=50, q_total=2):
    '''
    Generate starting/ending positions of q_total windows q of size q_size.
    :param q_size:
    :param q_total:
    :return:
    '''
    for q in range(1, q_total + 1):
        q_start = (q - 1) * q_size + 1
        q_end = (q - 1) * q_size + q_size
        # yield starting/ending positions of q (in ms)
        yield (q_start, q_end)
        q += 1


if __name__ == "__main__":

    print('success!')
