import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
from colorspacious import cspace_converter
from pathlib import Path
import pandas as pd
from sklearn import decomposition
from scipy import spatial
from scipy import sparse
import time
from functools import wraps
from collections import namedtuple
from scipy.signal import savgol_filter
from itertools import chain

from datetime import datetime
#TODO: This is some namespace mangling, rename NWBFile in your code please.
from pynwb import NWBFile
from pynwb import NWBHDF5IO
from pynwb.form.backends.hdf5.h5_utils import H5DataIO
from pynwb import TimeSeries
from collections import defaultdict
from functools import partial
import matrix_utils as matu

import load_raw_batch as load_raw
from sklearn.cluster import AffinityPropagation as AP
import copy, sys, os
from collections import OrderedDict

def print_exception_info():
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)

def append_results_to_array(array=None):
    '''
    This function appends results AND corresponding run attributes to an array.
    This array can later on converted to a dataframe for ease of access.
    :param array:
    :return:
    '''
    def callable(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            result_nwb_obj = function(*args, **kwargs)
            array.append(
                {
                    'NWBfile': result_nwb_obj,
                    **kwargs
                }
            )
            #print(f'{function.__name__} took {toc-tic} seconds.')
            return result_nwb_obj
        return wrapped
    return callable

def run_for_all_parameters(function, *args, **kwargs):
    '''
    This wrapper calls the function for all the parameters contained in the
    params dictionary, using the keys as kwargs and their values as the
    kwargs values. If the dictionary values are arrays it iterates through
    all the values.
    This is just for space savings since it omits the use of multiple nested
    for loops in succession for different simulations. Used nicely with a
    partial in order to reduce the params size.
    Also this should save the data in another wrapped function or please
    modify it to take an extra array argument and place the function results
    there.
    :param params:  Should be an ordered dict!
    :return:
    '''
    def pop_one(obj=None):
        if isinstance(obj, list):
            for item in obj:
                yield item
        else:
            yield obj

    def all_dict_combinations(*args, **kwargs):
        '''
        Iterate in a consumable (generator) and recursive way through a dictionary.
        Since I'm using recursion it is important that the dict maintains the order!
        :param dict:
        :return:
        '''
        # This is an OrderedDict
        auto_param_array = kwargs['auto_param_array']
        if len(auto_param_array) == 0:
            # remove our function specific kwarg:
            kwargs.pop('auto_param_array')
            # This is a hack, albeit a beautiful one:
            # if any of the values must depend on some other/s values wait at
            # the last moment and call it. You must do this for all the
            # callable in the kwargs.
            for k,v in kwargs.items():
                if hasattr(v, '__call__'):
                    kwargs[k] = v(**kwargs)
            # run the function:
            function(*args, **kwargs)
            return

        passable = copy.deepcopy(auto_param_array)
        item_tuple = passable.popitem()
        key = item_tuple[0]
        for value in pop_one(item_tuple[1]):
            all_dict_combinations(
                    *args,
                    **{
                        **kwargs,
                        **{key: value},
                        'auto_param_array': passable
                    }
                )

    # if this throws KeyError you should not use the function in the
    # first place! The existence of params is given.
    auto_param_array = OrderedDict(kwargs['auto_param_array'])
    print(f"Running for parameters:\n {auto_param_array}")
    # you also initialize the function run parameters:
    all_dict_combinations(
        *args,
        **{
            **kwargs,
            'auto_param_array': auto_param_array
        })

def time_it(function):
    @wraps(function)
    def runandtime(*args, **kwargs):
        tic = time.perf_counter()
        result = function(*args, **kwargs)
        toc = time.perf_counter()
        print(f'{function.__name__} took {toc-tic} seconds.')
        return result
    return runandtime

def nwb_unique_rng(function):
    @wraps(function)
    def seed_rng(*args, **kwargs):
        # Determine first the NWB file:
        if kwargs.get('input_NWBfile', None):
            NWBfile = kwargs.get('input_NWBfile', None)
        elif kwargs.get('NWBfile_array', None):
            NWBfile = kwargs.get('NWBfile_array', None)[0]
        else:
            raise ValueError('Input NWBfile is nonexistent!')

        #TODO: get another seed number from kwargs, in order to perform multiple
        # consecutive reproducible passes.
        rng_iter = kwargs.get('rng_iter', None)
        if not rng_iter:
            rng_iter = 0

        animal_model_id, learning_condition_id = \
            get_acquisition_parameters(
                input_NWBfile=NWBfile,
                requested_parameters=[
                    'animal_model_id', 'learning_condition_id'
                ]
            )
        new_seed = animal_model_id * 4 + learning_condition_id + rng_iter
        np.random.seed(new_seed)
        print(f'{function.__name__} reseeds the RNG.')
        return function(*args, **kwargs)
    return seed_rng

def with_reproducible_rng(class_method):
    '''
    This is a function wrapper that calls rng.seed before every method call. Therefore user is expected to get the exact
    same results every time between different method calls of the same class instance.
    Multiple method calls in main file will produce the exact same results. This is of course if model parameters are
    the same between calls; changing e.g. cell no, will invoke different number of rand() called in each case.
    As seed, the network serial number is used.
    A function wrapper is used to dissociate different function calls: e.g. creating stimulus will be the same, even if
    user changed the number of rand() was called in a previous class function, resulting in the same stimulated cells.
    Multiple method calls in main file will produce the exact same results. This is of course if model parameters are
    the same between calls; changing e.g. cell no, will invoke different number of rand() called in each case.
    :param func:
    :return:
    '''
    @wraps(class_method)
    def reset_rng(*args, **kwargs):
        # this translates to self.serial_no ar runtime
        np.random.seed(kwargs['serial_no'])
        print(f'{class_method.__name__} reseeds the RNG.')
        return class_method(*args, **kwargs)
    return reset_rng

def exception_logger(function):
    @wraps(function)
    def safe_run(*args, **kwargs):
        try:
            result = function(*args, **kwargs)
        except Exception as e:
            print(f'Exception in {function.__name__}.')
        return result
    return safe_run

experiment_config_filename = \
    'animal_model_{animal_model}_learning_condition_{learning_condition}_{type}_{experiment_config}.nwb'.format

# For simplicity, yet flexibility, change the format functions to normal ones:
def simulation_template_load_old(**kwargs):
    mystr = (f"{kwargs['prefix']}"
    f"SN{kwargs['animal_model']}"
    f"LC{kwargs['learning_condition']}"
    f"TR{kwargs['trial']}"
    f"_EB{kwargs['excitation_bias']:.3f}"
    f"_IB{kwargs['inhibition_bias']:.3f}"
    f"_GBF2.000"
    f"_NMDAb{kwargs['nmda_bias']:.3f}"
    f"_AMPAb{kwargs['ampa_bias']:.3f}"
    f"_{kwargs['experiment_config']}_simdur{kwargs['sim_duration']}"
    f"{kwargs['postfix']}")
    return mystr

def simulation_template_save_old(**kwargs):
    mystr = (f"{kwargs['prefix']}"
            f"SN{kwargs['animal_model']}"
            f"LC{kwargs['learning_condition']}"
            f"TR{kwargs['trial_first']}-{kwargs['trial_last']}"
            f"_EB{kwargs['excitation_bias']:.3f}"
            f"_IB{kwargs['inhibition_bias']:.3f}"
            f"_GBF2.000"
            f"_NMDAb{kwargs['nmda_bias']:.3f}"
            f"_AMPAb{kwargs['ampa_bias']:.3f}"
            f"_{kwargs['experiment_config']}_simdur{kwargs['sim_duration']}"
            f"{kwargs['postfix']}")
    return mystr

def simulation_template_load(**kwargs):
    mystr = (f"{kwargs['prefix']}"
    f"SN{kwargs['animal_model']}"
    f"LC{kwargs['learning_condition']}"
    f"TR{kwargs['trial']}"
    f"_EB{kwargs['excitation_bias']:.3f}"
    f"_IB{kwargs['inhibition_bias']:.3f}"
    f"_GBF2.000"
    f"_NMDAb{kwargs['nmda_bias']:.3f}"
    f"_AMPAb{kwargs['ampa_bias']:.3f}"
    f"_CP{kwargs['cp']}"
    f"_{kwargs['experiment_config']}_simdur{kwargs['sim_duration']}"
    f"{kwargs['postfix']}")
    return mystr

def simulation_template_save(**kwargs):
    mystr = (f"{kwargs['prefix']}"
    f"SN{kwargs['animal_model']}"
    f"LC{kwargs['learning_condition']}"
    f"TR{kwargs['trial_first']}-{kwargs['trial_last']}"
    f"_EB{kwargs['excitation_bias']:.3f}"
    f"_IB{kwargs['inhibition_bias']:.3f}"
    f"_GBF2.000"
    f"_NMDAb{kwargs['nmda_bias']:.3f}"
    f"_AMPAb{kwargs['ampa_bias']:.3f}"
    f"_CP{kwargs['cp']}"
    f"_{kwargs['experiment_config']}_simdur{kwargs['sim_duration']}"
    f"{kwargs['postfix']}")
    return mystr

def simulation_template_load_ri(**kwargs):
    # One must either provide the full experiment config string OR the
    # individual parameters:
    experiment_config = kwargs.get('experiment_config', None)
    if not experiment_config:
        try:
            dendlen = kwargs['dendlen']
            dendno = kwargs['dendno']
            connectivity_type = kwargs['connectivity_type']
            experiment_config = f'{connectivity_type}_{dendno}{dendlen}dend'
        except KeyError:
            print(f'You forgot to provide experiment configuration parameters!')
    mystr = (f"{kwargs['prefix']}"
             f"SN{kwargs['animal_model']}"
             f"LC{kwargs['learning_condition']}"
             f"TR{kwargs['trial']}"
             f"_EB{kwargs['excitation_bias']:.3f}"
             f"_IB{kwargs['inhibition_bias']:.3f}"
             f"_GBF2.000"
             f"_NMDAb{kwargs['nmda_bias']:.3f}"
             f"_AMPAb{kwargs['ampa_bias']:.3f}"
             f"_RI{kwargs['ri']}"
             f"_{experiment_config}_simdur{kwargs['sim_duration']}"
             f"{kwargs['postfix']}")
    return mystr

def simulation_template_save_ri(**kwargs):
    # One must either provide the full experiment config string OR the
    # individual parameters:
    experiment_config = kwargs.get('experiment_config', None)
    if not experiment_config:
        try:
            dendlen = kwargs['dendlen']
            dendno = kwargs['dendno']
            connectivity_type = kwargs['connectivity_type']
            experiment_config = f'{connectivity_type}_{dendno}{dendlen}dend'
        except KeyError:
            print(f'You forgot to provide experiment configuration parameters!')
    mystr = (f"{kwargs['prefix']}"
             f"SN{kwargs['animal_model']}"
             f"LC{kwargs['learning_condition']}"
             f"TR{kwargs['trial_first']}-{kwargs['trial_last']}"
             f"_EB{kwargs['excitation_bias']:.3f}"
             f"_IB{kwargs['inhibition_bias']:.3f}"
             f"_GBF2.000"
             f"_NMDAb{kwargs['nmda_bias']:.3f}"
             f"_AMPAb{kwargs['ampa_bias']:.3f}"
             f"_RI{kwargs['ri']}"
             f"_{experiment_config}_simdur{kwargs['sim_duration']}"
             f"{kwargs['postfix']}")
    return mystr

def simulation_template_load_iid(**kwargs):
    # This is a hack because I'm adding variables as I move along
    # for the reviews: I need to brake experiment configuration to different
    # dendritic lengths and number, as well as the network configuration.

    # One must either provide the full experiment config string OR the
    # individual parameters:
    experiment_config = kwargs.get('experiment_config', None)
    if not experiment_config:
        try:
            dendlen = kwargs['dendlen']
            dendno = kwargs['dendno']
            connectivity_type = kwargs['connectivity_type']
        except KeyError:
            print(f'You forgot to provide experiment configuration parameters!')
        experiment_config = f'{connectivity_type}_{dendno}{dendlen}dend'
    mystr = (f"{kwargs['prefix']}"
             f"SN{kwargs['animal_model']}"
             f"LC{kwargs['learning_condition']}"
             f"TR{kwargs['trial']}"
             f"_EB{kwargs['excitation_bias']:.3f}"
             f"_IB{kwargs['inhibition_bias']:.3f}"
             f"_GBF2.000"
             f"_NMDAb{kwargs['nmda_bias']:.3f}"
             f"_AMPAb{kwargs['ampa_bias']:.3f}"
             f"_CP{kwargs['cp']}"
             f"_DCP{kwargs['dend_clust_perc']}"
             f"_DCS{kwargs['dend_clust_seg']}"
             f"_{experiment_config}_simdur{kwargs['sim_duration']}"
             f"{kwargs['postfix']}")
    return mystr

def simulation_template_save_iid(**kwargs):
    # This is a hack because I'm adding variables as I move along
    # for the reviews: I need to brake experiment configuration to different
    # dendritic lengths and number, as well as the network configuration.

    # One must either provide the full experiment config string OR the
    # individual parameters:
    experiment_config = kwargs.get('experiment_config', None)
    if not experiment_config:
        try:
            dendlen = kwargs['dendlen']
            dendno = kwargs['dendno']
            connectivity_type = kwargs['connectivity_type']
        except KeyError:
            print(f'You forgot to provide experiment configuration parameters!')
        experiment_config = f'{connectivity_type}_{dendno}{dendlen}dend'
    mystr = (f"{kwargs['prefix']}"
             f"SN{kwargs['animal_model']}"
             f"LC{kwargs['learning_condition']}"
             f"TR{kwargs['trial_first']}-{kwargs['trial_last']}"
             f"_EB{kwargs['excitation_bias']:.3f}"
             f"_IB{kwargs['inhibition_bias']:.3f}"
             f"_GBF2.000"
             f"_NMDAb{kwargs['nmda_bias']:.3f}"
             f"_AMPAb{kwargs['ampa_bias']:.3f}"
             f"_CP{kwargs['cp']}"
             f"_DCP{kwargs['dend_clust_perc']}"
             f"_DCS{kwargs['dend_clust_seg']}"
             f"_{experiment_config}_simdur{kwargs['sim_duration']}"
             f"{kwargs['postfix']}")
    return mystr

def simulation_template_load_iid_ri(**kwargs):
    # This is a hack because I'm adding variables as I move along
    # for the reviews: I need to brake experiment configuration to different
    # dendritic lengths and number, as well as the network configuration.

    # One must either provide the full experiment config string OR the
    # individual parameters:
    experiment_config = kwargs.get('experiment_config', None)
    if not experiment_config:
        try:
            dendlen = kwargs['dendlen']
            dendno = kwargs['dendno']
            connectivity_type = kwargs['connectivity_type']
        except KeyError:
            print(f'You forgot to provide experiment configuration parameters!')
        experiment_config = f'{connectivity_type}_{dendno}{dendlen}dend'
    mystr = (f"{kwargs['prefix']}"
             f"SN{kwargs['animal_model']}"
             f"LC{kwargs['learning_condition']}"
             f"TR{kwargs['trial']}"
             f"_EB{kwargs['excitation_bias']:.3f}"
             f"_IB{kwargs['inhibition_bias']:.3f}"
             f"_GBF2.000"
             f"_NMDAb{kwargs['nmda_bias']:.3f}"
             f"_AMPAb{kwargs['ampa_bias']:.3f}"
             f"_RI{kwargs['ri']}"
             f"_DCP{kwargs['dend_clust_perc']}"
             f"_DCS{kwargs['dend_clust_seg']}"
             f"_{experiment_config}_simdur{kwargs['sim_duration']}"
             f"{kwargs['postfix']}")
    return mystr

def simulation_template_save_iid_ri(**kwargs):
    # This is a hack because I'm adding variables as I move along
    # for the reviews: I need to brake experiment configuration to different
    # dendritic lengths and number, as well as the network configuration.

    # One must either provide the full experiment config string OR the
    # individual parameters:
    experiment_config = kwargs.get('experiment_config', None)
    if not experiment_config:
        try:
            dendlen = kwargs['dendlen']
            dendno = kwargs['dendno']
            connectivity_type = kwargs['connectivity_type']
        except KeyError:
            print(f'You forgot to provide experiment configuration parameters!')
        experiment_config = f'{connectivity_type}_{dendno}{dendlen}dend'
    mystr = (f"{kwargs['prefix']}"
             f"SN{kwargs['animal_model']}"
             f"LC{kwargs['learning_condition']}"
             f"TR{kwargs['trial_first']}-{kwargs['trial_last']}"
             f"_EB{kwargs['excitation_bias']:.3f}"
             f"_IB{kwargs['inhibition_bias']:.3f}"
             f"_GBF2.000"
             f"_NMDAb{kwargs['nmda_bias']:.3f}"
             f"_AMPAb{kwargs['ampa_bias']:.3f}"
             f"_RI{kwargs['ri']}"
             f"_DCP{kwargs['dend_clust_perc']}"
             f"_DCS{kwargs['dend_clust_seg']}"
             f"_{experiment_config}_simdur{kwargs['sim_duration']}"
             f"{kwargs['postfix']}")
    return mystr

def simulation_template_load_sp_ri(**kwargs):
    # One must either provide the full experiment config string OR the
    # individual parameters:
    experiment_config = kwargs.get('experiment_config', None)
    if not experiment_config:
        try:
            dendlen = kwargs['dendlen']
            dendno = kwargs['dendno']
            connectivity_type = kwargs['connectivity_type']
            experiment_config = f'{connectivity_type}_{dendno}{dendlen}dend'
        except KeyError:
            print(f'You forgot to provide experiment configuration parameters!')
    mystr = (f"{kwargs['prefix']}"
             f"SN{kwargs['animal_model']}"
             f"LC{kwargs['learning_condition']}"
             f"TR{kwargs['trial']}"
             f"_EB{kwargs['excitation_bias']:.3f}"
             f"_IB{kwargs['inhibition_bias']:.3f}"
             f"_GBF2.000"
             f"_NMDAb{kwargs['nmda_bias']:.3f}"
             f"_AMPAb{kwargs['ampa_bias']:.3f}"
             f"_RI{kwargs['ri']}"
             f"_{experiment_config}"
             f"_SPLOC{kwargs['sploc']}"
             f"_SPCL{kwargs['spcl']}"
             f"_SPDCL{kwargs['spdcl']}"
             f"_WR{kwargs['wr']}"
             f"_simdur{kwargs['sim_duration']}"
             f"{kwargs['postfix']}")
    return mystr

def simulation_template_save_sp_ri(**kwargs):
    # One must either provide the full experiment config string OR the
    # individual parameters:
    experiment_config = kwargs.get('experiment_config', None)
    if not experiment_config:
        try:
            dendlen = kwargs['dendlen']
            dendno = kwargs['dendno']
            connectivity_type = kwargs['connectivity_type']
            experiment_config = f'{connectivity_type}_{dendno}{dendlen}dend'
        except KeyError:
            print(f'You forgot to provide experiment configuration parameters!')
    mystr = (f"{kwargs['prefix']}"
             f"SN{kwargs['animal_model']}"
             f"LC{kwargs['learning_condition']}"
             f"TR{kwargs['trial_first']}-{kwargs['trial_last']}"
             f"_EB{kwargs['excitation_bias']:.3f}"
             f"_IB{kwargs['inhibition_bias']:.3f}"
             f"_GBF2.000"
             f"_NMDAb{kwargs['nmda_bias']:.3f}"
             f"_AMPAb{kwargs['ampa_bias']:.3f}"
             f"_RI{kwargs['ri']}"
             f"_{experiment_config}"
             f"_SPLOC{kwargs['sploc']}"
             f"_SPCL{kwargs['spcl']}"
             f"_SPDCL{kwargs['spdcl']}"
             f"_WR{kwargs['wr']}"
             f"_simdur{kwargs['sim_duration']}"
             f"{kwargs['postfix']}")
    return mystr

def simulation_template_load_spu_ri(**kwargs):
    # One must either provide the full experiment config string OR the
    # individual parameters:
    experiment_config = kwargs.get('experiment_config', None)
    if not experiment_config:
        try:
            dendlen = kwargs['dendlen']
            dendno = kwargs['dendno']
            connectivity_type = kwargs['connectivity_type']
            experiment_config = f'{connectivity_type}_{dendno}{dendlen}dend'
        except KeyError:
            print(f'You forgot to provide experiment configuration parameters!')
    mystr = (f"{kwargs['prefix']}"
             f"SN{kwargs['animal_model']}"
             f"LC{kwargs['learning_condition']}"
             f"TR{kwargs['trial']}"
             f"_EB{kwargs['excitation_bias']:.3f}"
             f"_IB{kwargs['inhibition_bias']:.3f}"
             f"_GBF2.000"
             f"_NMDAb{kwargs['nmda_bias']:.3f}"
             f"_AMPAb{kwargs['ampa_bias']:.3f}"
             f"_RI{kwargs['ri']}"
             f"_{experiment_config}"
             f"_SPLOC{kwargs['sploc']}"
             f"_SPCL{kwargs['spcl']}"
             f"_SPDCL{kwargs['spdcl']}"
             f"_U"
             f"_simdur{kwargs['sim_duration']}"
             f"{kwargs['postfix']}")
    return mystr

def simulation_template_save_spu_ri(**kwargs):
    # One must either provide the full experiment config string OR the
    # individual parameters:
    experiment_config = kwargs.get('experiment_config', None)
    if not experiment_config:
        try:
            dendlen = kwargs['dendlen']
            dendno = kwargs['dendno']
            connectivity_type = kwargs['connectivity_type']
            experiment_config = f'{connectivity_type}_{dendno}{dendlen}dend'
        except KeyError:
            print(f'You forgot to provide experiment configuration parameters!')
    mystr = (f"{kwargs['prefix']}"
             f"SN{kwargs['animal_model']}"
             f"LC{kwargs['learning_condition']}"
             f"TR{kwargs['trial_first']}-{kwargs['trial_last']}"
             f"_EB{kwargs['excitation_bias']:.3f}"
             f"_IB{kwargs['inhibition_bias']:.3f}"
             f"_GBF2.000"
             f"_NMDAb{kwargs['nmda_bias']:.3f}"
             f"_AMPAb{kwargs['ampa_bias']:.3f}"
             f"_RI{kwargs['ri']}"
             f"_{experiment_config}"
             f"_SPLOC{kwargs['sploc']}"
             f"_SPCL{kwargs['spcl']}"
             f"_SPDCL{kwargs['spdcl']}"
             f"_U"
             f"_simdur{kwargs['sim_duration']}"
             f"{kwargs['postfix']}")
    return mystr


def simulation_template_load_spi_ri(**kwargs):
    # One must either provide the full experiment config string OR the
    # individual parameters:
    experiment_config = kwargs.get('experiment_config', None)
    if not experiment_config:
        try:
            dendlen = kwargs['dendlen']
            dendno = kwargs['dendno']
            connectivity_type = kwargs['connectivity_type']
            experiment_config = f'{connectivity_type}_{dendno}{dendlen}dend'
        except KeyError:
            print(f'You forgot to provide experiment configuration parameters!')
    mystr = (f"{kwargs['prefix']}"
             f"SN{kwargs['animal_model']}"
             f"LC{kwargs['learning_condition']}"
             f"TR{kwargs['trial']}"
             f"_EB{kwargs['excitation_bias']:.3f}"
             f"_IB{kwargs['inhibition_bias']:.3f}"
             f"_GBF2.000"
             f"_NMDAb{kwargs['nmda_bias']:.3f}"
             f"_AMPAb{kwargs['ampa_bias']:.3f}"
             f"_RI{kwargs['ri']}"
             f"_{experiment_config}"
             f"_WR{kwargs['wr']}"
             f"_simdur{kwargs['sim_duration']}"
             f"{kwargs['postfix']}")
    return mystr

def simulation_template_save_spi_ri(**kwargs):
    # One must either provide the full experiment config string OR the
    # individual parameters:
    experiment_config = kwargs.get('experiment_config', None)
    if not experiment_config:
        try:
            dendlen = kwargs['dendlen']
            dendno = kwargs['dendno']
            connectivity_type = kwargs['connectivity_type']
            experiment_config = f'{connectivity_type}_{dendno}{dendlen}dend'
        except KeyError:
            print(f'You forgot to provide experiment configuration parameters!')
    mystr = (f"{kwargs['prefix']}"
             f"SN{kwargs['animal_model']}"
             f"LC{kwargs['learning_condition']}"
             f"TR{kwargs['trial_first']}-{kwargs['trial_last']}"
             f"_EB{kwargs['excitation_bias']:.3f}"
             f"_IB{kwargs['inhibition_bias']:.3f}"
             f"_GBF2.000"
             f"_NMDAb{kwargs['nmda_bias']:.3f}"
             f"_AMPAb{kwargs['ampa_bias']:.3f}"
             f"_RI{kwargs['ri']}"
             f"_{experiment_config}"
             f"_WR{kwargs['wr']}"
             f"_simdur{kwargs['sim_duration']}"
             f"{kwargs['postfix']}")
    return mystr

def simulation_template_load_spinh_ri(**kwargs):
    # One must either provide the full experiment config string OR the
    # individual parameters:
    experiment_config = kwargs.get('experiment_config', None)
    if not experiment_config:
        try:
            dendlen = kwargs['dendlen']
            dendno = kwargs['dendno']
            connectivity_type = kwargs['connectivity_type']
            experiment_config = f'{connectivity_type}_{dendno}{dendlen}dend'
        except KeyError:
            print(f'You forgot to provide experiment configuration parameters!')
    mystr = (f"{kwargs['prefix']}"
             f"SN{kwargs['animal_model']}"
             f"LC{kwargs['learning_condition']}"
             f"TR{kwargs['trial']}"
             f"_EB{kwargs['excitation_bias']:.3f}"
             f"_IB{kwargs['inhibition_bias']:.3f}"
             f"_GBF2.000"
             f"_NMDAb{kwargs['nmda_bias']:.3f}"
             f"_AMPAb{kwargs['ampa_bias']:.3f}"
             f"_RI{kwargs['ri']}"
             f"_{experiment_config}"
             f"_syninh{kwargs['syninh']}"
             f"_dendinh{kwargs['dendinh']}"
             f"sw"
             f"_simdur{kwargs['sim_duration']}"
             f"{kwargs['postfix']}")
    return mystr

def simulation_template_save_spinh_ri(**kwargs):
    # One must either provide the full experiment config string OR the
    # individual parameters:
    experiment_config = kwargs.get('experiment_config', None)
    if not experiment_config:
        try:
            dendlen = kwargs['dendlen']
            dendno = kwargs['dendno']
            connectivity_type = kwargs['connectivity_type']
            experiment_config = f'{connectivity_type}_{dendno}{dendlen}dend'
        except KeyError:
            print(f'You forgot to provide experiment configuration parameters!')
    mystr = (f"{kwargs['prefix']}"
             f"SN{kwargs['animal_model']}"
             f"LC{kwargs['learning_condition']}"
             f"TR{kwargs['trial_first']}-{kwargs['trial_last']}"
             f"_EB{kwargs['excitation_bias']:.3f}"
             f"_IB{kwargs['inhibition_bias']:.3f}"
             f"_GBF2.000"
             f"_NMDAb{kwargs['nmda_bias']:.3f}"
             f"_AMPAb{kwargs['ampa_bias']:.3f}"
             f"_RI{kwargs['ri']}"
             f"_{experiment_config}"
             f"_syninh{kwargs['syninh']}"
             f"_dendinh{kwargs['dendinh']}"
             f"sw"
             f"_simdur{kwargs['sim_duration']}"
             f"{kwargs['postfix']}")
    return mystr

def simulation_template_load_spinh_wr_ri(**kwargs):
    # One must either provide the full experiment config string OR the
    # individual parameters:
    experiment_config = kwargs.get('experiment_config', None)
    if not experiment_config:
        try:
            dendlen = kwargs['dendlen']
            dendno = kwargs['dendno']
            connectivity_type = kwargs['connectivity_type']
            experiment_config = f'{connectivity_type}_{dendno}{dendlen}dend'
        except KeyError:
            print(f'You forgot to provide experiment configuration parameters!')
    mystr = (f"{kwargs['prefix']}"
             f"SN{kwargs['animal_model']}"
             f"LC{kwargs['learning_condition']}"
             f"TR{kwargs['trial']}"
             f"_EB{kwargs['excitation_bias']:.3f}"
             f"_IB{kwargs['inhibition_bias']:.3f}"
             f"_GBF2.000"
             f"_NMDAb{kwargs['nmda_bias']:.3f}"
             f"_AMPAb{kwargs['ampa_bias']:.3f}"
             f"_RI{kwargs['ri']}"
             f"_{experiment_config}"
             f"_syninh{kwargs['syninh']}"
             f"_dendinh{kwargs['dendinh']}"
             f"_WR{kwargs['weights_realization']}"
             f"_simdur{kwargs['sim_duration']}"
             f"{kwargs['postfix']}")
    return mystr

def simulation_template_save_spinh_wr_ri(**kwargs):
    # One must either provide the full experiment config string OR the
    # individual parameters:
    experiment_config = kwargs.get('experiment_config', None)
    if not experiment_config:
        try:
            dendlen = kwargs['dendlen']
            dendno = kwargs['dendno']
            connectivity_type = kwargs['connectivity_type']
            experiment_config = f'{connectivity_type}_{dendno}{dendlen}dend'
        except KeyError:
            print(f'You forgot to provide experiment configuration parameters!')
    mystr = (f"{kwargs['prefix']}"
             f"SN{kwargs['animal_model']}"
             f"LC{kwargs['learning_condition']}"
             f"TR{kwargs['trial_first']}-{kwargs['trial_last']}"
             f"_EB{kwargs['excitation_bias']:.3f}"
             f"_IB{kwargs['inhibition_bias']:.3f}"
             f"_GBF2.000"
             f"_NMDAb{kwargs['nmda_bias']:.3f}"
             f"_AMPAb{kwargs['ampa_bias']:.3f}"
             f"_RI{kwargs['ri']}"
             f"_{experiment_config}"
             f"_syninh{kwargs['syninh']}"
             f"_dendinh{kwargs['dendinh']}"
             f"_WR{kwargs['weights_realization']}"
             f"_simdur{kwargs['sim_duration']}"
             f"{kwargs['postfix']}")
    return mystr


# These are the different templates I use when simulating in NEURON.
# To avoid confusion, since these change based on sim parameters (checking them,
# reviewers comments, experimentation) I must pass the one I use in each
# load function.
simulation_templates = {
    "load_old": simulation_template_load_old,
    "save_old": simulation_template_save_old,
    "load": simulation_template_load,
    "save": simulation_template_save,
    "load_ri": simulation_template_load_ri,
    "save_ri": simulation_template_save_ri,
    "load_iid": simulation_template_load_iid,
    "save_iid": simulation_template_save_iid,
    "load_iid_ri": simulation_template_load_iid_ri,
    "save_iid_ri": simulation_template_save_iid_ri,
    "load_sp_ri": simulation_template_load_sp_ri,
    "save_sp_ri": simulation_template_save_sp_ri,
    "load_spu_ri": simulation_template_load_spu_ri,
    "save_spu_ri": simulation_template_save_spu_ri,
    "load_spi_ri": simulation_template_load_spi_ri,
    "save_spi_ri": simulation_template_save_spi_ri,
    "load_spinh_ri": simulation_template_load_spinh_ri,
    "save_spinh_ri": simulation_template_save_spinh_ri,
    "load_spinh_wr_ri": simulation_template_load_spinh_wr_ri,
    "save_spinh_wr_ri": simulation_template_save_spinh_wr_ri,
}

#TODO: name them correctly:
excitatory_validation_template = (
    '{location}_{condition}'
    '_{currents}'
    '_{nmda_bias:0.1f}'
    '_{ampa_bias:0.1f}'
    '_{synapse}'
    '_{segment}'
    '_{excitation_bias:0.2f}.txt').format

MD_params = namedtuple('MD_params', ['mu', 'S'])

def get_cell_type(cellid, pn_no):
    # simply determine the type, given that cells' ids are sorted with PN first:
    if cellid < pn_no:
        return 'PN'
    else:
        return 'PV'

def getargs(*argnames):
    '''getargs(*argnames, argdict)
    Convenience function to retrieve arguments from a dictionary in batch
    '''
    if len(argnames) < 2:
        raise ValueError('Must supply at least one key and a dict')
    if not isinstance(argnames[-1], dict):
        raise ValueError('last argument must be dict')
    kwargs = argnames[-1]
    if not argnames:
        raise ValueError('must provide keyword to get')
    if len(argnames) == 2:
        return kwargs.get(argnames[0])
    return [kwargs.get(arg) for arg in argnames[:-1]]

def nwb_iter(sequencelike):
    # This is a wrapper for NWB sequences. It returns something that you can iterate on.
    # Without this, iterating can be cumbersome (does not stop; you can't even create a list).
    # Although these types DO appear to support __len__ and __getitem__ :
    n = len(sequencelike)
    ctr = 0
    while ctr < n:
        yield sequencelike[ctr]
        ctr += 1

def named_serial_no(name, n=1):
    # Generates a size n iterable of strings, with name, followed by an
    # increasing serial number:
    ctr = 1
    while ctr < n + 1:
        yield f'{name} {ctr}'
        ctr += 1

def load_nwb_file(**kwargs):
    animal_model, learning_condition, experiment_config, data_path, type = \
    getargs('animal_model', 'learning_condition', 'experiment_config', \
            'data_path', 'type', kwargs)

    filename = data_path.joinpath(experiment_config_filename(
        animal_model=animal_model,
        learning_condition=learning_condition,
        experiment_config=experiment_config,
        type=type
    ))

    if not Path.is_file(filename):
        raise FileNotFoundError(f'The file {filename} was not found :(')

    print(f'Loading NWB file:\n\t{str(filename)}')
    nwbfile = NWBHDF5IO(str(filename), 'r').read()
    return nwbfile

def read_validation_potential(cellid=0, trialid=0, \
    inputdir=None, **kwargs
):
    # This function takes two required argumends and return a timeseries. To
    # be flexible, it accepts more specific kwargs, that are later removed
    # with the use of a partial.
    location, condition, currents, nmda_bias, ampa_bias, synapse_activation = \
        getargs(
            'location', 'condition', 'currents', 'nmda_bias', 'ampa_bias',
            'synapse_activation', kwargs
        )
    # Some args are specific/hardcoded, but always can be passed through the
    # kwargs list if wanted.
    inputfile = inputdir.joinpath(
        excitatory_validation_template(
            location=location,
            condition=condition,
            currents=currents,
            nmda_bias=nmda_bias,
            ampa_bias=ampa_bias,
            synapse=synapse_activation[trialid],
            segment=2,
            excitation_bias=1.75
        )
    )
    # In this instance just reads a txt file:
    #TODO: make try/except in case file is nonexistent!
    with open(inputfile, 'r') as f:
        timeseries = list(map(float, f.readlines()))
    return timeseries


def import_recordings_to_nwb(nwbfile=None, read_function=None, **kwargs):
    # Load files as different trials on a nwb file given.
    # Iteratively load each file and append it to the dataset. Create and
    # annotate trials and stimulus in the process.
    # The nwb file is created externally, so this function can be used multiple
    # times to add more acquisitions.
    if not read_function:
        raise NotImplementedError('read_function() is not implemented!')

    ncells, ntrials, timeseries_name, timeseries_description, \
    stim_start_offset, stim_stop_offset, trial_len, samples_per_ms, \
        samples2ms_factor = getargs(
        'ncells', 'ntrials', 'timeseries_name', 'timeseries_description', \
        'stim_start_offset', 'stim_stop_offset', 'trial_len',
        'samples_per_ms', 'samples2ms_factor',
        kwargs,
    )


    # the base unit of time is the ms:
    samples2ms_factor = 1 / samples_per_ms
    nsamples = trial_len * samples_per_ms


    # Create the base array, containing the cells x trials.
    membrane_potential = np.zeros((ncells, ntrials * nsamples))

    #TODO: add a descriptor for each trial, to filter them at load time:
    try:
        nwbfile.add_trial_column(
            'acquisition_name',
            'The name of the acquisition that these trials belong'
        )
    except Exception as e:
        print(str(e))

    # Iterate all cells and trials, loading each cell's activity and
    # concatenating trials:
    for trial, (trial_start, trial_end) in zip(range(ntrials),
            generate_slices(size=nsamples, number=ntrials)
    ):
        for cellid in range(ncells):
            # Take input filename externally as an array:
            # Search inputdir for files specified in the parameters

            # User should provide a function that returns a time series, given
            # the cellid and trial:
            try:
                voltage_trace = read_function(cellid=cellid, trialid=trial)
                membrane_potential[cellid, trial_start:trial_end] = \
                    voltage_trace[:nsamples]
                #TODO: WRONG! You can not add trials with every import!
                #TODO: How can you link trials with acquisitions?
                # Add trial:
                nwbfile.add_trial(
                    start_time=trial_start * samples2ms_factor,
                    stop_time=trial_end * samples2ms_factor,
                    acquisition_name=timeseries_name
                )
                # Add stimulus epoch for that trial in ms:
                nwbfile.add_epoch(
                    start_time=float((trial_start * samples2ms_factor) + stim_start_offset),
                    stop_time=float((trial_start * samples2ms_factor) + stim_stop_offset),
                    tags=f'trial {trial} stimulus'
                )
            except Exception as e:
                raise e('read_function error!')


        print(f'Trial {trial}, processed.')

    # Chunk and compress the data:
    wrapped_data = H5DataIO(
        data=membrane_potential,
        chunks=True,
        compression='gzip',
        compression_opts=9
    )
    # Add somatic voltage traces (all trials concatenated)
    timeseries = TimeSeries(
        timeseries_name,  # Name of the TimeSeries
        wrapped_data,  # Actual data
        'miliseconds',  # Base unit of the measurement
        starting_time=0.0,  # The timestamp of the first sample
        rate=10000.0,  # Sampling rate in Hz
        conversion=samples2ms_factor,  #  Scalar to multiply each element in data to convert it to the specified unit
        # Since we can only use strings, stringify the dict!
        description=timeseries_description
    )
    nwbfile.add_acquisition(timeseries)
    print('Time series acquired.')
    return

def create_nwb_validation_file(inputdir=None, outputdir=None,
                               synapse_activation=None, location=None,
                               **kwargs):
    # Create a NWB file from the results of the validation routines.

    print('Creating NWBfile.')
    nwbfile = NWBFile(
        session_description='NEURON validation results.',
        identifier='excitatory_validation',
        session_start_time=datetime.now(),
        file_create_date=datetime.now()
    )

    # Partially automate the loading with the aid of a reading function;
    # use a generic reading function that you make specific with partial and
    # then just load all the trials:
    #TODO: you should put somewhere the external kwargs!!
    if not synapse_activation:
        synapse_activation = list(range(1, 150, 5))
    if not location:
        location = 'vsoma'
    basic_kwargs = {'ncells': 1, 'ntrials': len(synapse_activation), \
                    'stim_start_offset': 100, 'stim_stop_offset': 140,
                    'trial_len': 700, 'samples_per_ms': 10}


    # 'Freeze' some portion of the function, for a simplified one:
    read_somatic_potential = partial(
        read_validation_potential,
        inputdir=inputdir,
        synapse_activation=synapse_activation,
        location='vsoma'
    )

    # Load first batch:
    import_recordings_to_nwb(
        nwbfile=nwbfile,
        read_function=partial(
            read_somatic_potential,
            condition='normal',
            currents='NMDA+AMPA',
            nmda_bias=6.0,
            ampa_bias=1.0,
        ),
        timeseries_name='normal_NMDA+AMPA',
        timeseries_description='Validation',
        **basic_kwargs
    )

    # Rinse and repeat:
    import_recordings_to_nwb(
        nwbfile=nwbfile,
        read_function=partial(
            read_somatic_potential,
            condition='normal',
            currents='AMPA',
            nmda_bias=0.0,
            ampa_bias=50.0,
        ),
        timeseries_name='normal_AMPA_only',
        timeseries_description='Validation',
        **basic_kwargs
    )

    # Rinse and repeat:
    import_recordings_to_nwb(
        nwbfile=nwbfile,
        read_function=partial(
            read_somatic_potential,
            condition='noMg',
            currents='NMDA+AMPA',
            nmda_bias=6.0,
            ampa_bias=1.0,
        ),
        timeseries_name='noMg_NMDA+AMPA',
        timeseries_description='Validation',
        **basic_kwargs
    )


    # Use partial to remove some of the kwargs that are the same:
    read_dendritic_potential = partial(
        read_validation_potential,
        inputdir=inputdir,
        synapse_activation=synapse_activation,
        location='vdend'
    )

    # Load dendritic potential:
    import_recordings_to_nwb(
        nwbfile=nwbfile,
        read_function=partial(
            read_dendritic_potential,
            condition='normal',
            currents='NMDA+AMPA',
            nmda_bias=6.0,
            ampa_bias=1.0,
        ),
        timeseries_name='vdend_normal_NMDA+AMPA',
        timeseries_description='Validation',
        **basic_kwargs
    )


    # write to file:
    output_file = outputdir.joinpath(
        'excitatory_validation.nwb'
    )
    print(f'Writing to NWBfile: {output_file}')
    with NWBHDF5IO(str(output_file), 'w') as io:
        io.write(nwbfile)

def create_nwb_validation_file_ampatest(inputdir=None, outputdir=None,
                               synapse_activation=None, location=None,
                               **kwargs):
    # Create a NWB file from the results of the validation routines.

    print('Creating NWBfile.')
    nwbfile = NWBFile(
        session_description='NEURON validation results.',
        identifier='excitatory_validation_ampatest',
        session_start_time=datetime.now(),
        file_create_date=datetime.now()
    )

    # Partially automate the loading with the aid of a reading function;
    # use a generic reading function that you make specific with partial and
    # then just load all the trials:
    #TODO: you should put somewhere the external kwargs!!
    if not synapse_activation:
        synapse_activation = list(range(1, 25, 1))
    if not location:
        location = 'vsoma'
    basic_kwargs = {'ncells': 1, 'ntrials': len(synapse_activation), \
                    'stim_start_offset': 100, 'stim_stop_offset': 140,
                    'trial_len': 700, 'samples_per_ms': 10}


    # 'Freeze' some portion of the function, for a simplified one:
    read_somatic_potential = partial(
        read_validation_potential,
        inputdir=inputdir,
        synapse_activation=synapse_activation,
        location='vsoma'
    )

    # Load first batch:
    import_recordings_to_nwb(
        nwbfile=nwbfile,
        read_function=partial(
            read_somatic_potential,
            condition='normal',
            currents='NMDA+AMPA',
            nmda_bias=6.0,
            ampa_bias=1.0,
        ),
        timeseries_name='normal_NMDA+AMPA',
        timeseries_description='Validation',
        **basic_kwargs
    )

    # Rinse and repeat:
    import_recordings_to_nwb(
        nwbfile=nwbfile,
        read_function=partial(
            read_somatic_potential,
            condition='normal',
            currents='AMPA',
            nmda_bias=0.0,
            ampa_bias=1.0,
        ),
        timeseries_name='normal_AMPA_only',
        timeseries_description='Validation',
        **basic_kwargs
    )

    '''
    # Use partial to remove some of the kwargs that are the same:
    read_dendritic_potential = partial(
        read_validation_potential,
        inputdir=inputdir,
        synapse_activation=synapse_activation,
        location='vdend'
    )

    # Load dendritic potential:
    import_recordings_to_nwb(
        nwbfile=nwbfile,
        read_function=partial(
            read_dendritic_potential,
            condition='normal',
            currents='NMDA+AMPA',
            nmda_bias=6.0,
            ampa_bias=1.0,
        ),
        timeseries_name='vdend_normal_NMDA+AMPA',
        timeseries_description='Validation',
        **basic_kwargs
    )

    # Load dendritic potential:
    import_recordings_to_nwb(
        nwbfile=nwbfile,
        read_function=partial(
            read_dendritic_potential,
            condition='normal',
            currents='AMPA',
            nmda_bias=0.0,
            ampa_bias=1.0,
        ),
        timeseries_name='vdend_normal_AMPA',
        timeseries_description='Validation',
        **basic_kwargs
    )
    '''

    # write to file:
    output_file = outputdir.joinpath(
        'excitatory_validation_ampatest_1.nwb'
    )
    print(f'Writing to NWBfile: {output_file}')
    with NWBHDF5IO(str(output_file), 'w') as io:
        io.write(nwbfile)

def load_nwb_from_neuron(
    data_folder,
    reload_raw=False,
    include_membrane_potential=False,
    **kwargs):
    # Why I pass params as a variable rather than as an kwargs dict??
    #TODO:  the reload_raw parameter should bleed into the import_raw neuron 
    # data, inside the create_nwb_file? Or add a second flag to do that?
    # Do I need that is the question?
    analysis_parameters = {
        'stim_start_offset': 50,
        'stim_stop_offset': 1050,
        'q_size': 50,
        'total_qs': None,
        'ntrials': 1,
        'ncells': 333,
        'spike_upper_threshold': 0,
        'spike_lower_threshold': -10,
        'data_dim': 2,
        'samples_per_ms': 10,
        'nmda_bias': 3.0,
        'ampa_bias': 1.0,
        'animal_model': 1,
        'learning_condition': 1,
        'prefix': '',
        'postfix': '',
        'trial': 0,
        'sps': 10,
    }
    # TODO: Where it gets its default value? Raise KeyError to notice it:
    #template_postfix = kwargs.get('template_postfix', '')
    template_postfix = kwargs['template_postfix']

    # Compute internal characteristics.
    all_input_params = {**analysis_parameters, **kwargs}
    all_input_params['trial_len'] = \
        all_input_params['sim_duration'] * 1000
    all_input_params['total_qs'] = \
        int(np.floor(all_input_params['trial_len'] / all_input_params['q_size']))
    #TODO: I'm so confused about how I THE USER insert trials and how PYTHON
    # interprets my input to read from neuron. Need to fix these indices.
    # I need different variables for these, to save correctly named NWB files.
    # Also like this I keep everything on input params only. create_nwb_file,
    # just needs to parse this.
    all_input_params['trial_first'] = 0
    all_input_params['trial_last'] = all_input_params['ntrials'] - 1

    outfn = simulation_templates[f'save{template_postfix}'](
        **all_input_params
    ) + '.nwb'
    filename = data_folder.joinpath(
        outfn
    )
    already_converted = filename.exists()

    if reload_raw or not already_converted:
        try:
            # TODO: should check the order of the args for any shadowing!
            create_nwb_file(
                **{
                'inputdir': data_folder,
                'outputdir': data_folder,
                'same_io_folder': True,
                'include_membrane_potential': include_membrane_potential,
                'reload_raw': reload_raw,
                **all_input_params
            }
            )
        except Exception as e:
            print(str(e))
            return None
    else:
        print(f'Run dir {data_folder} is already converted to NWB!')

    if not Path.is_file(filename):
        raise FileNotFoundError(f'The file {filename} was not found :(')

    print(f'Loading NWB file2:\n\t{str(filename)}')
    nwbfile = NWBHDF5IO(str(filename), 'r').read()

    return nwbfile


def load_raw_neuron(inputdir, rundir, reload_raw=False):
    '''
    This function will create HDF5 files from raw NEURON output (text files).
    :param inputdir:
    :param rundir:
    :param reload_raw: If set to true, will re-create HDF5 files even if
    already existing (if one re-ran NEURON on the same experiment).
    :return:
    '''
    try:
        inputfile = inputdir.joinpath(rundir).joinpath('vsoma.hdf5')
        # If input hdf file exists, just take the traces.
        if reload_raw or not inputfile.exists():
            #print(f'Converting data to HDF in dir:\n\t{rundir}')
            try:
                load_raw.main(inputdir.joinpath(rundir))
            except Exception as e:
                #print(f'Exception while reading voltage files:\n\t{str(e)}\n\t maybe simulation is not complete!')
                raise FileNotFoundError
        # Convert dataframe to ndarray:
        voltage_traces = pd.read_hdf(inputfile, key='vsoma').values
        return voltage_traces
    except FileNotFoundError as e:
        #print(f'NEURON file {rundir} not found! Returning None.')
        return None

def create_nwb_file(inputdir=None, outputdir=None, \
                    include_membrane_potential=False, **kwargs):
    '''
    This function will create a NWB file from neuron output.
    In will automatically detect if raw neuron output (text files) are
    converted to HDF5 format for faster loading. The callee is responsible
    for checking if the specific NWB is already created and not call this
    function.
    :param inputdir:
    :param outputdir:
    :param include_membrane_potential:
    :param kwargs:
    :return:
    '''
    #TODO: this should get automagicaly all the keys in the dict and create one same named variable for each one.
    # Finally to pass the value of the dict to the new variable.
    # Get parameters externally:
    experiment_config, animal_model, learning_condition, ntrials, trial_len, ncells, stim_start_offset, \
    stim_stop_offset, samples_per_ms, spike_upper_threshold, spike_lower_threshold, excitation_bias, \
        inhibition_bias, nmda_bias, ampa_bias, sim_duration, q_size, fn_prefix, fn_postfix, cp, dcp, dcs= \
        getargs('experiment_config', 'animal_model', 'learning_condition', 'ntrials', 'trial_len', 'ncells', 'stim_start_offset', \
                   'stim_stop_offset', 'samples_per_ms', 'spike_upper_threshold', 'spike_lower_threshold', \
                'excitation_bias', 'inhibition_bias', 'nmda_bias', 'ampa_bias', 'sim_duration', 'q_size', 'prefix', 'postfix', 'cp','dend_clust_perc','dend_clust_seg',kwargs)

    #THis is an override to force the function to re-create the HDF5 files
    # from raw NEURON output.
    no_need_to_reload_neuron = False
    if no_need_to_reload_neuron:
        reload_raw = False
        print("NOT RELOADING THE RAW NEURON DATA. POSSIBLE MISSING HDF5 WILL GENERATE ERROR!")
    else:
        reload_raw = kwargs.get('reload_raw', False)
        if reload_raw:
            print("RELOADING THE RAW NEURON DATA ONTO HDF5! TIME CONSUMING!")

    template_postfix = kwargs.get('template_postfix', '')

    # the base unit of time is the ms:
    samples2ms_factor = 1 / samples_per_ms
    nsamples = int(trial_len * samples_per_ms)

    # Expand the NEURON/experiment parameters in the acquisition dict:
    #TODO: change these:
    pn_no = 250
    pv_no = 83
    acquisition_description = {
        **kwargs,
        'pn_no': pn_no,
        'pv_no': pv_no
    }
    #TODO: make it a function later on
    experiment_config = kwargs.get('experiment_config', None)
    if not experiment_config:
        try:
            dendlen = kwargs['dendlen']
            dendno = kwargs['dendno']
            connectivity_type = kwargs['connectivity_type']
        except KeyError:
            print(f'You forgot to provide experiment configuration parameters!')
        experiment_config = f'{connectivity_type}_{dendno}{dendlen}dend'
    nwbfile = NWBFile(
        session_description='NEURON simulation results.',
        identifier=experiment_config,
        session_start_time=datetime.now(),
        file_create_date=datetime.now()
    )

    nwbfile.add_unit_column('cell_id', 'Id of the cell recorded')
    nwbfile.add_unit_column('cell_type', 'Type of the cell recorded (PN or PV)')
    nwbfile.add_trial_column('persistent_activity', 'If this trial has persistent activity')
    #nwbfile.add_epoch_column('stimulus')

    print('===================================================================')
    print('Loading files from NEURON output.')
    print('===================================================================')
    time_series_l = []
    spike_train_l = []
    spike_trains_d = defaultdict(partial(np.ndarray, 0))
    membrane_potential = np.array([])
    vsoma = np.zeros((ncells, nsamples), dtype=float)
    trial_offset_samples = 0
    trials_loaded = []
    # Visualize the location from which the function is loading runs:
    # Here I'm using the save template, because it mentions the total number of
    # trials requested to load and also the various parameters of the simulation
    # so an eye checking that this is the correct run can be performed as
    # loading.
    general_tmpl = simulation_templates[f'save{template_postfix}'](
        **{
            **kwargs,
            'trial_first': 0,
            'trial_last': ntrials - 1,
        }
    )
    print(f'Simulation directory is:\n{inputdir}')
    print('-------------------------------------------------------------------')
    print(f'Simulation template is:\n{general_tmpl}')
    print('-------------------------------------------------------------------')
    for trial, (trial_start_t, trial_end_t) in enumerate(generate_slices(size=nsamples, number=ntrials)):
        # Search inputdir for files specified in the parameters
        # get input dir and search for hdf file. if non existent create it.
        print(f'Trial {trial}. ', end='')
        rundir = inputdir.joinpath(
            simulation_templates[f'load{template_postfix}'](
                **{
                    **kwargs,
                    'trial': trial,
                }
            ))
        voltage_traces = load_raw_neuron(inputdir, rundir, reload_raw=reload_raw)
        if voltage_traces is not None:
            # Register trial as existing:
            trials_loaded.append(trial)
            vsoma = voltage_traces[:ncells, :nsamples]
            # Offset trial start, in case of any missing trials occuring.
            trial_start_t -= trial_offset_samples
            trial_end_t -= trial_offset_samples
            # Use a dict to save space:
            for cellid in range(ncells):
                spike_train = quick_spikes(
                    voltage_trace=voltage_traces[cellid],
                    upper_threshold=spike_upper_threshold,
                    lower_threshold=spike_lower_threshold,
                    plot=False
                )
                spike_trains_d[cellid] = np.append(
                    spike_trains_d[cellid], np.add(
                        spike_train, trial_start_t * samples2ms_factor
                    )
                )

            # Define the region of PA as the last 200 ms of the simulation:
            pa_stop = int(nsamples * samples2ms_factor) + \
                      trial_start_t * samples2ms_factor
            pa_start = int(pa_stop - 200)
            has_persistent = False
            for cellid, spike_train in spike_trains_d.items():
                if any(spike_train > pa_start) and any(spike_train < pa_stop):
                    #print(f'On trial:{trial}, cell:{cellid} has spikes, so PA.')
                    has_persistent = True
                    break
            # Add trial:
            nwbfile.add_trial(
                start_time=trial_start_t * samples2ms_factor,
                stop_time=trial_end_t * samples2ms_factor,
                persistent_activity=has_persistent
            )
            # Add stimulus epoch for that trial:
            #TODO: stimulus times are in ms or samples? Because _t is in samples!
            nwbfile.add_epoch(
                start_time=float(trial_start_t + stim_start_offset),
                stop_time=float(trial_start_t + stim_stop_offset),
                tags=f'trial {trial} stimulus'
            )

            if membrane_potential.size:
                membrane_potential = np.concatenate((membrane_potential, vsoma), axis=1)
            else:
                membrane_potential = vsoma
            print(f'Processed successfully.')
        else:
            print(f'Not found.')
            # Inform next trials to skip the ms of the missing file
            trial_offset_samples += nsamples

    if len(trials_loaded) == 0:
        print(f'ERROR no trials found for {general_tmpl}\n\tReturning '
              'None!')
        raise FileNotFoundError

    if include_membrane_potential:
        # Chunk and compress the data:
        wrapped_data = H5DataIO(
            data=membrane_potential,
            chunks=True,  # <---- Enable chunking (although I'm not sure if it will do any good in my huge dataset.
            compression='gzip',
            compression_opts=9
        )
        # Add somatic voltage traces (all trials concatenated)
        vsoma_timeseries = TimeSeries(
            'membrane_potential',  # Name of the TimeSeries
            wrapped_data,  # Actual data
            'miliseconds',  # Base unit of the measurement
            starting_time=0.0,  # The timestamp of the first sample
            rate=10000.0,  # Sampling rate in Hz
            conversion=samples2ms_factor,  # Scalar to multiply each element in data to convert it to the specified unit
            # Since we can only use strings, stringify the dict!
            description=str(acquisition_description)
        )
        nwbfile.add_acquisition(vsoma_timeseries)
        print('\tMembrane potential acquired.')

    for cellid in range(ncells):
        # Get each trial start/end in seconds rather than ms:
        trial_intervals = [
            [trial_start, trial_end]
            for trial_start, trial_end in \
            generate_slices(
                size=trial_len / 1000, number=ntrials, start_from=0
            )
        ]
        #TODO: These should be separate. If you don't have spikes in ANY
        # cell, pynwb throws error, uppon saving.
        if spike_trains_d[cellid].size > 0:
            nwbfile.add_unit(
                id=cellid,
                spike_times=spike_trains_d[cellid],
                obs_intervals=trial_intervals,
                cell_id=cellid,
                cell_type=get_cell_type(cellid, pn_no)
            )
        else:
            nwbfile.add_unit(
                id=cellid,
                spike_times=[],
                obs_intervals=trial_intervals,
                cell_id=cellid,
                cell_type=get_cell_type(cellid, pn_no)
            )
    print('\tSpikes acquired.')

    cells_with_spikes = spike_trains_d.keys()
    spike_trains = spike_trains_d.values()

    # Bin spiking activity for all trials/cells in total_qs bins of q_size size:
    # How many qs in all trials?
    #TODO: do not reassign! rename!
    # Also, since I might got LESS trials, I should reassign the ntrials
    # variable:
    trials_with_data = len(nwbfile.trials)
    total_qs = int(np.floor(trial_len / q_size)) * trials_with_data
    trial_qs = int(np.floor(trial_len / q_size))
    binned_activity = np.zeros((ncells, total_qs), dtype=int)
    # This is essentially what we are doing, but since python is so slow, we refactor it with some optimized code.
    #for cellid, spike_train in zip(cells_with_spikes, iter(spike_trains)):
    #    for q, (q_start, q_end) in enumerate(generate_slices(size=q_size, number=total_qs)):
    #        binned_activity[cellid][q] = sum([1 for spike in spike_train if q_start <= spike and spike < q_end])
    try:
        for cellid, spike_train in zip(cells_with_spikes, spike_trains):
            bins = np.floor_divide(spike_train, q_size).astype(int)
            np.add.at(binned_activity[cellid][:], bins, 1)
    except Exception as e:
        print(str(e))
        print_exception_info()
        raise e

    # Chunk and compress the data:
    wrapped_data = H5DataIO(
        data=binned_activity,
        chunks=True,  # <---- Enable chunking (although I'm not sure if it will do any good in my huge dataset.
        compression='gzip',
        compression_opts=9
    )
    # Add binned activity:
    network_binned_activity = TimeSeries(
        'binned_activity',  # Name of the TimeSeries
        wrapped_data,  # Actual data
        'miliseconds',  # Base unit of the measurement
        starting_time=0.0,  # The timestamp of the first sample
        rate=1000/q_size,  # Sampling rate in Hz
        conversion=float(q_size),  #  Scalar to multiply each element in data to convert it to the specified unit
        # Since we can only use strings, stringify the dict!
        description=str(acquisition_description)
    )
    nwbfile.add_acquisition(network_binned_activity)
    print('\tBinned activity acquired')

    # write to file:
    if include_membrane_potential:
        type = 'mp'
    else:
        type = 'bn'

    if kwargs.get('same_io_folder', False):
        outfn = simulation_templates[f'save{template_postfix}'](
            **{
                'trial_first': 0,
                'trial_last': ntrials - 1,
                **kwargs
            }
        ) + '.nwb'
        output_file = inputdir.joinpath(
            outfn
        )
    else:
        output_file = outputdir.joinpath(
            experiment_config_filename(
                animal_model=animal_model,
                learning_condition=learning_condition,
                experiment_config=experiment_config,
                type=type
            )
        )
    print(f'Writing to NWBfile:\n{outputdir}\n{output_file.name}')
    print('===================================================================')
    print('\n\n')
    with NWBHDF5IO(str(output_file), 'w') as io:
        io.write(nwbfile)

def get_acquisition_potential(NWBfile=None, acquisition_name=None, cellid=None, trialid=None):
    #TODO: make it trial informed and group the spike trains that way!
    '''
    Return a tuple of cellid and its spike train, per trial.
    :param NWBfile:
    :param acquisition_name:
    :param group_per_trial: If true groups spike trains per trial. Use false if
        you want to batch handle the spiketrains.
    :return:
    '''
    animal_model_id, learning_condition_id, ncells, pn_no, ntrials, \
    trial_len, q_size, trial_q_no, correct_trials_idx, correct_trials_no = \
        get_acquisition_parameters(
            input_NWBfile=NWBfile,
            requested_parameters=[
                'animal_model_id', 'learning_condition_id', 'ncells',
                'pn_no', 'ntrials', 'trial_len', 'q_size', 'trial_q_no',
                'correct_trials_idx', 'correct_trials_no'
            ]
        )

    total_trial_qs = trial_len / q_size
    one_sec_qs = 1000 / q_size
    start_q = total_trial_qs - one_sec_qs
    samples_per_ms = NWBfile.acquisition['membrane_potential'].rate / 1000  # Sampling rate (Hz) / ms

    trials = get_acquisition_trials(
        NWBfile=NWBfile
    )
    # Unpack trial start/stop.
    trialid, trial_start_ms, trial_stop_ms, *_ = trials[trialid]

    # cellid can also be a vector, describing the requested cells' voltages:
    potential = NWBfile.acquisition['membrane_potential']. \
                    data[cellid, int(trial_start_ms * samples_per_ms): \
                                 int(trial_stop_ms * samples_per_ms): \
                                 int(samples_per_ms)]

    # Return the transposed of the potential so you can matplotlib.pyplot the
    # result with a single command.
    return potential.T


def get_acquisition_spikes(NWBfile=None, acquisition_name=None, group_per_trial=True):
    #TODO: make it trial informed and group the spike trains that way!
    '''
    Return a tuple of cellid and its spike train, per trial.
    :param NWBfile:
    :param acquisition_name:
    :param group_per_trial: If true groups spike trains per trial. Use false if
        you want to batch handle the spiketrains.
    :return:
    '''
    animal_model_id, learning_condition_id, ncells, pn_no, ntrials, \
    trial_len, q_size, trial_q_no, correct_trials_idx, correct_trials_no = \
        get_acquisition_parameters(
            input_NWBfile=NWBfile,
            requested_parameters=[
                'animal_model_id', 'learning_condition_id', 'ncells',
                'pn_no', 'ntrials', 'trial_len', 'q_size', 'trial_q_no',
                'correct_trials_idx', 'correct_trials_no'
            ]
        )

    total_trial_qs = trial_len / q_size
    one_sec_qs = 1000 / q_size
    start_q = total_trial_qs - one_sec_qs

    # Convert iterators to list, to avoid exhausting them!
    cells_with_spikes = list(nwb_iter(NWBfile.units['cell_id']))
    spike_trains = list(nwb_iter(NWBfile.units['spike_times']))

    network_spiketrains = []
    if group_per_trial:
        # Create a list of lists:
        trials = get_acquisition_trials(
            NWBfile=NWBfile
        )
        network_spiketrains = [None] * ntrials
        for trial in trials:
            # Unpack trial start/stop.
            trialid, trial_start_ms, trial_stop_ms, *_ = trial

            network_spiketrains[trialid] = []
            for cellid, spike_train in zip(cells_with_spikes, spike_trains):
                # Get only spikes in the trial interval:
                cell_spikes = [
                    spike - trial_start_ms
                    for spike in list(spike_train)
                    if trial_start_ms <= spike and spike < trial_stop_ms
                ]
                network_spiketrains[trialid].append((cellid, cell_spikes))
    else:
        network_spiketrains=[
            (cellid, spike_train)
            for cellid, spike_train in zip(cells_with_spikes, spike_trains)
        ]
    return network_spiketrains



def bin_activity(input_NWBfile, **kwargs):
    q_size = getargs('q_size', kwargs)

    #Get ncells without load the whole aquisition dataset
    nwbfile_description_d = eval(
        input_NWBfile.acquisition['membrane_potential'].description
    )
    ncells = nwbfile_description_d['ncells']

    ntrials = len(input_NWBfile.trials)
    # This is the current experiment identifier (structured, random etc):
    experiment_id = input_NWBfile.identifier
    # Here I am using the same trial length for all my trials, because its a simulation,
    # so I safely grab the first one only.
    trial_len = input_NWBfile.trials['stop_time'][0] - \
                input_NWBfile.trials['start_time'][0]  # in ms
    samples_per_ms = input_NWBfile.acquisition['membrane_potential'].rate / 1000  # Sampling rate (Hz) / ms
    conversion_factor = input_NWBfile.acquisition['membrane_potential'].conversion

    # CAUTION: these appear to return similar objects, where they don't. Be very
    # careful on how you use them together (e.g. zip() etc).
    # Also these appear to not behave like iterables. So create some out of
    # them:
    cells_with_spikes = nwb_iter(input_NWBfile.units['cell_id'])
    spike_trains = nwb_iter(input_NWBfile.units['spike_times'])

    # Bin spiking activity for all trials/cells in total_qs bins of q_size size:
    #TODO: THIS REQUIRES that each trial is divided by q_size exactly! Impose that constraint!
    # How many qs in all trials?
    total_qs = int(np.floor(trial_len / q_size)) * ntrials
    trial_qs = int(np.floor(trial_len / q_size))
    binned_activity = np.zeros((ncells, total_qs), dtype=int)
    # This is essentially what we are doing, but since python is so slow, we refactor it with some optimized code.
    #for cellid, spike_train in zip(cells_with_spikes, iter(spike_trains)):
    #    for q, (q_start, q_end) in enumerate(generate_slices(size=q_size, number=total_qs)):
    #        binned_activity[cellid][q] = sum([1 for spike in spike_train if q_start <= spike and spike < q_end])
    try:
        for cellid, spike_train in zip(cells_with_spikes, spike_trains):
            #TODO: this is a serious bug!
            if spike_train.max() >= trial_len * ntrials:
                print('having spikes outside of trial! How is this possible?')
                #spike_train = spike_train[:-1]
                raise IndexError('Error in spike trains!')
            bins = np.floor_divide(spike_train, q_size).astype(int)
            np.add.at(binned_activity[cellid][:], bins, 1)
    except Exception as e:
        print(str(e))
        raise e

    # Chunk and compress the data:
    wrapped_data = H5DataIO(
        data=binned_activity,
        chunks=True,  # <---- Enable chunking (although I'm not sure if it will do any good in my huge dataset.
        compression='gzip',
        compression_opts=9
    )
    # Add binned activity:
    network_binned_activity = TimeSeries(
        'binned_activity',  # Name of the TimeSeries
        wrapped_data,  # Actual data
        'miliseconds',  # Base unit of the measurement
        starting_time=0.0,  # The timestamp of the first sample
        rate=20.0,  # Sampling rate in Hz
        conversion=float(q_size)  #  Scalar to multiply each element in data to convert it to the specified unit
    )
    input_NWBfile.add_acquisition(network_binned_activity)

    # Reshape before returning:
    return binned_activity.reshape(ncells, ntrials, trial_qs)


#===================================================================================================================
def read_network_activity(fromfile=None, dataset=None, **kwargs):
    # Parse keyword arguments:
    ntrials = kwargs.get('ntrials', None)
    total_qs = kwargs.get('total_qs', None)
    cellno = kwargs.get('cellno', None)

    data = None
    # Read spiketrains and plot them
    df = pd.read_hdf(fromfile, key=dataset)
    if df is not None:
        data = df.values.T
        # Also reshape the data into a 3d array:
        data = data.reshape(data.shape[0], ntrials, total_qs, order='C')

    return data

def get_acquisition_parameters(input_NWBfile=None, requested_parameters=[],
                               **kwargs):
    # Dig into NWB file and return the requested parameters. If not found,
    # raise (EAFP).
    # Takes as granted that ONE of the acquisitions will have a description
    # associated with it.
    #TODO: It would constitute a good practice to have the same annotations
    # in different parts of the NWB file, if for any reason someone needs to
    # trim it down i.e. I could have the description also in the spikes and
    # binned sections, rather than only in membrane_potential.

    try:
        # Search NWB file for a description, containing the strtingified dict:
        for k in input_NWBfile.acquisition.keys():
            if k in input_NWBfile.acquisition:
                nwbfile_description_d = \
                    eval(input_NWBfile.acquisition[k].description)
            else:
                raise ValueError('No description found in NWB!')

        # TODO: this is a little bit problematic, as some of the parameters need
        # to be computed in advance. So compute them first and then filter them:
        animal_model_id = nwbfile_description_d['animal_model']
        learning_condition_id = nwbfile_description_d['learning_condition']
        ncells = nwbfile_description_d['ncells']
        pn_no = nwbfile_description_d['pn_no']
        ntrials = len(input_NWBfile.trials)
        # Here I am using the same trial length for all my trials, because its a simulation,
        # so I safely grab the first one only.
        trial_len = input_NWBfile.trials['stop_time'][0] - input_NWBfile.trials['start_time'][0]  # in ms
        q_size = input_NWBfile.acquisition['binned_activity'].conversion
        trial_q_no = int(np.floor(trial_len / q_size))
        #TODO: rename to Valid trials!
        correct_trials_idx = list(
            nwb_iter(input_NWBfile.trials['persistent_activity'])
        )
        correct_trials_no = sum(correct_trials_idx)

        parameters = {
            'animal_model_id': animal_model_id,
            'learning_condition_id': learning_condition_id,
            'ncells': ncells,
            'pn_no': pn_no,
            'ntrials': ntrials,
            'trial_len': trial_len,
            'q_size': q_size,
            'trial_q_no': trial_q_no,
            'correct_trials_idx': correct_trials_idx,
            'correct_trials_no': correct_trials_no
        }

        #Pass the requested parameters through getargs():
        returned_parameters = getargs(*requested_parameters, parameters)
        return returned_parameters
        #TODO: implement this check:
        #if len(returned_parameters) != len(requested_parameters):
        #    raise ValueError('(Some) Requested parameter not found!')
        #else:
        #    return returned_parameters
    except Exception as e:
        print(f"EXCEPTION! No NWB file! ({str(e)})")
        #raise e
        return None

def calculate_stimulus_isi(NWBfile=None):

    animal_model_id, learning_condition_id, ncells, pn_no, ntrials, \
    trial_len, q_size, trial_q_no, correct_trials_idx, correct_trials_no = \
        get_acquisition_parameters(
            input_NWBfile=NWBfile,
            requested_parameters=[
                'animal_model_id', 'learning_condition_id', 'ncells',
                'pn_no', 'ntrials', 'trial_len', 'q_size', 'trial_q_no',
                'correct_trials_idx', 'correct_trials_no'
            ]
        )

    # Compute ISI histograms of all the structured learning conditions' trials.
    cells_with_spikes = nwb_iter(NWBfile.units['cell_id'])
    spike_trains = nwb_iter(NWBfile.units['spike_times'])

    stim_ISIs = []
    stim_ISIs_CV = []
    for trial in range(ntrials):
        # Unpack stimulus start/stop.
        _, stim_start_ms, stim_stop_ms, *_ = NWBfile.epochs[trial]

        for cellid, spike_train in zip(cells_with_spikes, spike_trains):
            # Get only spikes in the stimulus interval:
            stim_spikes = [
                spike
                for spike in list(spike_train)
                if stim_start_ms <= spike and spike < stim_stop_ms
            ]
            ISIs = np.diff(stim_spikes)
            stim_ISIs.append(list(ISIs))
            mu = ISIs.mean()
            sigma = ISIs.std()
            stim_ISIs_CV.append(sigma / mu)

    return list(chain(*stim_ISIs)), stim_ISIs_CV

def calculate_delay_isi(NWBfile=None):

    animal_model_id, learning_condition_id, ncells, pn_no, ntrials, \
    trial_len, q_size, trial_q_no, correct_trials_idx, correct_trials_no = \
    get_acquisition_parameters(
        input_NWBfile=NWBfile,
        requested_parameters=[
            'animal_model_id', 'learning_condition_id', 'ncells',
            'pn_no', 'ntrials', 'trial_len', 'q_size', 'trial_q_no',
            'correct_trials_idx', 'correct_trials_no'
        ]
    )

    # Compute ISI histograms of all the structured learning conditions' trials.
    cells_with_spikes = nwb_iter(NWBfile.units['cell_id'])
    spike_trains = nwb_iter(NWBfile.units['spike_times'])

    delay_ISIs = []
    delay_ISIs_CV = []
    for trial in range(ntrials):
        # Unpack trial start/stop.
        _, trial_start_ms, trial_stop_ms, *_= NWBfile.trials[trial]
        # Unpack stimulus start/stop.
        _, stim_start_ms, stim_stop_ms, *_ = NWBfile.epochs[trial]
        # Get delay start/stop
        delay_start_ms = stim_stop_ms
        delay_stop_ms = trial_stop_ms

        for cellid, spike_train in zip(cells_with_spikes, spike_trains):
            # Get only spikes in the stimulus interval:
            stim_spikes = [
                spike
                for spike in list(spike_train)
                if delay_start_ms <= spike and spike < delay_stop_ms
            ]
            ISIs = np.diff(stim_spikes)
            delay_ISIs.append(list(ISIs))
            mu = ISIs.mean()
            sigma = ISIs.std()
            delay_ISIs_CV.append(sigma / mu)

    return list(chain(*delay_ISIs)), delay_ISIs_CV

def get_acquisition_trials(NWBfile=None, acquisition_name=None):
    '''
    Return the trials for the requested acquisition.
    :param NWBfile:
    :param acquisition_name: Optional, only if you want trials for a specific
        acquisition.
    :return:
    '''
    if acquisition_name:
        acquisition_trials = [
            trial
            for idx, trial in enumerate(nwb_iter(NWBfile.trials))
            if NWBfile.trials['acquisition_name'][idx] == acquisition_name
        ]
    else:
        acquisition_trials = [
            trial
            for idx, trial in enumerate(nwb_iter(NWBfile.trials))
        ]

    return acquisition_trials


def separate_trials(input_NWBfile=None, acquisition_name=None):
    # Return an iterable of each trial acrivity.

    #TODO: check if wrapped and unwrap:
    raw_acquisition = input_NWBfile.acquisition[acquisition_name].data

    trials = get_acquisition_trials(
        NWBfile=input_NWBfile,
        acquisition_name=acquisition_name
    )

    #TODO: get samples_per_ms
    f = 10
    trial_activity = [
        raw_acquisition[:, int(trial_start_t*f):int(trial_end_t*f) - 1]
        for _, trial_start_t, trial_end_t, *_ in trials
    ]
    return trial_activity

def energy(data=None):
    # 50 ms.
    ntrials = data.shape[1]
    #TODO: Get tne q_size correctly and NOT hard coded:

    # TODO: check again the values!
    dt = 1000.0 / 50.0
    # Get frobenius norm:
    #TODO: na balw swsta to 50 (q_size):
    energy = np.array([
        np.abs(np.diff(np.sum(data[:, trial, :] * dt, axis=0) /250 ))
        for trial in range(ntrials)
    ])
    return energy

def velocity(data=None):
    # Should I do it in the original data and NOT only in the PCA/reduced ones?
    # This function calculates the multidimensional velocity in distances of
    # 50 ms.
    ntrials = data.shape[1]
    #TODO: Get tne q_size correctly and NOT hard coded:

    # TODO: check again the values!
    dt = 1000.0 / 50.0
    # Get frobenius norm:
    #TODO: DOUBLE CHECK
    #TODO: na balw swsta to 50 (q_size):
    velocity = np.array([
        np.linalg.norm(np.diff(data[:, trial, :] * dt, axis=1), axis=0) / 50
        for trial in range(ntrials)
    ])

    #dt = 1000/p.ws;
    #fws = 4;
    #b = (1/fws)*ones(1,fws);
    #for c=1:size(scores,2)
    #    if ~isempty(scores{r,c})
    #        % calculate velocity (in all PC dims):
    #        tmpdiff = diff(scores{r,c},1,1);
    #        score_vel{gri,c} = zeros(size(tmpdiff,1),1);
    #        for kk=1:size(tmpdiff,1)
    #            % giati kanw norm edw?? (gia to euclidean distance
    #            score_vel{gri,c}(kk,1) = norm(tmpdiff(kk,:))./ (dt);
    #        end
    #        score_vel{gri,c} = filter(b,1,score_vel{gri,c});
    #    end
    #end
    return velocity


#TODO: prepei na 3eka8arisw mia fora oti ta correct trials na ta lew VALILD trials.
def get_nwb_list_valid_ntrials(NWBarray):
    '''
    Return the number of valid trials (having PA) in the NWBarray
    This is DIFFERENT than the number of trials that the NWB has as
    acquisition!
    :param NWBarray:
    :return:
    '''
    total_valid_ntrials = 0
    for NWBfile in NWBarray:
        correct_trials_no = \
            get_acquisition_parameters(
                input_NWBfile=NWBfile,
                requested_parameters=[
                    'correct_trials_no'
                ]
            )
        total_valid_ntrials += correct_trials_no

    return total_valid_ntrials

def get_binned_activity(NWBfile, custom_range=None):
    #TODO: this is not what id does. Decide and keep only few of these funcs.
    # Return only trials with persistent activity (see text).
    animal_model_id, learning_condition_id, ncells, pn_no, ntrials, \
    trial_len, q_size, trial_q_no, correct_trials_idx, correct_trials_no = \
        get_acquisition_parameters(
            input_NWBfile=NWBfile,
            requested_parameters=[
                'animal_model_id', 'learning_condition_id', 'ncells',
                'pn_no', 'ntrials', 'trial_len', 'q_size', 'trial_q_no',
                'correct_trials_idx', 'correct_trials_no'
            ]
        )

    if correct_trials_no < 1:
        raise ValueError('No correct trials were found in the NWBfile!')

    # Use custom_range to compute PCA only on a portion of the original data:
    if custom_range is not None:
        if not isinstance(custom_range, tuple):
            raise ValueError('Custom range must be a tuple!')
        trial_slice_start = int(custom_range[0])
        trial_slice_stop = int(custom_range[1])
        duration = trial_slice_stop - trial_slice_start
    else:
        trial_slice_start = 0
        trial_slice_stop = trial_q_no
        duration = trial_q_no

    # Load binned acquisition (all trials together)
    binned_network_activity = NWBfile. \
                                  acquisition['binned_activity']. \
                                  data[:pn_no, :]. \
        reshape(pn_no, ntrials, trial_q_no)

    return binned_network_activity, trial_slice_start, trial_slice_stop


def get_correct_trials(NWBfile, custom_range=None):
    # Return only trials with persistent activity (see text).
    binned_network_activity, trial_slice_start, trial_slice_stop = \
        get_binned_activity(
            NWBfile,
            custom_range=custom_range
        )

    animal_model_id, learning_condition_id, ncells, pn_no, ntrials, \
    trial_len, q_size, trial_q_no, correct_trials_idx, correct_trials_no = \
        get_acquisition_parameters(
            input_NWBfile=NWBfile,
            requested_parameters=[
                'animal_model_id', 'learning_condition_id', 'ncells',
                'pn_no', 'ntrials', 'trial_len', 'q_size', 'trial_q_no',
                'correct_trials_idx', 'correct_trials_no'
            ]
        )

    if correct_trials_no < 1:
        raise ValueError('No correct trials were found in the NWBfile!')

    # Slice out non correct trials and unwanted trial periods:
    response_array = binned_network_activity[
                     :,
                     correct_trials_idx,
                     trial_slice_start:trial_slice_stop
                     ]

    return response_array


def sparsness(NWBfile, custom_range=None):
    # Treves - Rolls metric of population sparsness:
    # function sparsness = S_tr(r)
    # r is the population rates: a matrix MxN of M neurons and N trials
    # stamatiad.st@gmail.com
    # S_T = @(r,N) (sum(r/N))^2 / (sum(r.^2/N));

    animal_model_id, learning_condition_id, ncells, pn_no, ntrials, \
    trial_len, q_size, trial_q_no, correct_trials_idx, correct_trials_no = \
        get_acquisition_parameters(
            input_NWBfile=NWBfile,
            requested_parameters=[
                'animal_model_id', 'learning_condition_id', 'ncells',
                'pn_no', 'ntrials', 'trial_len', 'q_size', 'trial_q_no',
                'correct_trials_idx', 'correct_trials_no'
            ]
        )

    # Make delay period the default period:
    if custom_range is None:
        custom_range = (20, int(trial_len / 50))

    tmp = get_correct_trials(NWBfile, custom_range=custom_range)

    trial_rates = np.median(tmp, axis=2)

    M, N = trial_rates.shape
    # N = size(r,1);
    # M = size(r,2);
    S_TR = np.power(np.sum(trial_rates.mean(axis=1) / pn_no), 2) / \
        np.sum(np.power(trial_rates.mean(axis=1), 2) / pn_no)

    #TODO: check that it works!
    return 1 - S_TR

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def plot_pca_in_3d(NWBfile=None, custom_range=None, smooth=False, \
                   plot_axes=None, klabels=None):
    #TODO: is this deterministic? Because some times I got an error in some
    # matrix.
    animal_model_id, learning_condition_id, ncells, pn_no, ntrials, \
    trial_len, q_size, trial_q_no, correct_trials_idx, correct_trials_no = \
        get_acquisition_parameters(
            input_NWBfile=NWBfile,
            requested_parameters=[
                'animal_model_id', 'learning_condition_id', 'ncells',
                'pn_no', 'ntrials', 'trial_len', 'q_size', 'trial_q_no',
                'correct_trials_idx', 'correct_trials_no'
            ]
        )
    tmp = get_correct_trials(NWBfile)
    # TODO: Duration equals trial_q_no, if no custom_range is used:
    # Reshape in array with m=cells, n=time bins.
    duration = tmp.shape[2]
    pool_array = tmp.reshape(pn_no, correct_trials_no * duration)

    # how many PCA components
    L = 3
    pca = decomposition.PCA(n_components=L)
    t_L = pca.fit_transform(pool_array.T).T
    # Reshape PCA results into separate trials for plotting.
    #TODO: do a more elegant way of splitting into trials:
    total_data_trials = int(pool_array.shape[1]/duration)
    t_L_per_trial = t_L.reshape(L, total_data_trials, duration, order='C')
    #TODO: somewhere here I get a warning about a non-tuple sequence for
    # multi dim indexing. Why?
    # Smooth more, since this is more for visualization purposes.
    if smooth:
        for trial in range(total_data_trials):
            for l in range(L):
                t_L_per_trial[l, trial, :] = savgol_filter(t_L_per_trial[l, trial, :], 31, 3)


    # If not part of a subfigure, create one:
    if not plot_axes:
        fig = plt.figure()
        plt.ion()
        plot_axes = fig.add_subplot(111, projection='3d')
    #plot_axes.set_title(f'Model {animal_model_id}, learning condition {learning_condition_id}')
    # Stylize the 3d plot:
    plot_axes.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    plot_axes.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    plot_axes.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    plot_axes.set_xlabel('PC1')
    plot_axes.set_ylabel('PC2')
    plot_axes.set_zlabel('PC3')
    # Also for visualization purposes, tight the axis limits
    pc_lims_max = []
    pc_lims_min = []
    #TODO: this is the second not intended behaviour that I capture list
    # comprehensions do. Is this deliberate? am I missing something here?
    #UPDATE: running them interactively in the debugger, gives the error. BUT,
    # running past by them (through the debugger again), removes the problem.
    # So I lean towards a PyCharm issue, rather than undocumented python.
    for pc in range(3):
        pc_lims_max.append(np.array([
            t_L_per_trial[pc][trial][:].max()
            for trial in range(correct_trials_no)
        ]).max())
        pc_lims_min.append(np.array([
            t_L_per_trial[pc][trial][:].min()
            for trial in range(correct_trials_no)
        ]).min())

    pca_axis_lims = list(zip(pc_lims_min, pc_lims_max))
    pca_xaxis_limits = pca_axis_lims[0]
    pca_yaxis_limits = pca_axis_lims[1]
    pca_zaxis_limits = pca_axis_lims[2]
    plot_axes.set_xlim(pca_xaxis_limits)
    plot_axes.set_ylim(pca_yaxis_limits)
    plot_axes.set_zlim(pca_zaxis_limits)
    plot_axes.set_xticks(pca_xaxis_limits)
    plot_axes.set_yticks(pca_yaxis_limits)
    plot_axes.set_zticks(pca_zaxis_limits)
    plot_axes.elev = 22.5
    plot_axes.azim = 52.4

    if klabels is not None:
        # If you have cluster information for the data:
        # Create custom colormaps for the stimulus and trial period, per label.
        # Utilize normalized trial times, to create the colormaps.
        #TODO: make colormap gradient more steep after stimulus.
        #TODO: Make it read the proper value!
        stim_stop = 21  # Stim stop in q=50ms
        stim_stop_norm = stim_stop / duration

        c = mcolors.ColorConverter().to_rgb
        stim_start_color = 'limegreen'
        stim_stop_color = 'darkgreen'
        cluster_a_cm = make_colormap(
            [
                c(stim_start_color), c(stim_stop_color), stim_stop_norm,
                c('orange'), c('red'), stim_stop_norm + 0.01,
                c('red'), c('darkred'), 0.99,
                c('darkred')
            ]
        )
        cluster_b_cm = make_colormap(
            [
                c(stim_start_color), c(stim_stop_color), stim_stop_norm,
                c('orange'), c('violet'), stim_stop_norm + 0.01,
                c('violet'), c('darkviolet'), 0.99,
                c('darkviolet')
            ]
        )
        cluster_c_cm = make_colormap(
            [
                c(stim_start_color), c(stim_stop_color), stim_stop_norm, 
                c('orange'), c('gold'), stim_stop_norm + 0.01,
                c('gold'), c('goldenrod'), 0.99,
                c('goldenrod')
            ]
        )
        cluster_d_cm = make_colormap(
            [
                c(stim_start_color), c(stim_stop_color), stim_stop_norm, 
                c('orange'), c('blue'), stim_stop_norm + 0.01,
                c('blue'), c('darkblue'), 0.99,
                c('darkblue')
            ]
        )
        colors = [
            cluster_a_cm, cluster_b_cm, cluster_c_cm, cluster_d_cm
        ]

        labels = klabels.tolist()
        nclusters = np.unique(klabels).size
        #colors_old = cm.Set2(np.linspace(0, 1, nclusters))
        _, key_labels = np.unique(labels, return_index=True)
        handles = []
        for i, (trial, label) in enumerate(zip(range(total_data_trials), labels)):
            # Transform colormap to color values:
            color_values = colors[label - 1](np.linspace(0, 1, duration))
            for t, c in zip(range(duration - 1), color_values):
                handle = plot_axes.plot(
                    t_L_per_trial[0][trial][t:t+2],
                    t_L_per_trial[1][trial][t:t+2],
                    t_L_per_trial[2][trial][t:t+2],
                    color=c,
                    linewidth=3,
                    label=f'Cluster {label}'
                )
            if i in key_labels:
                handles.append(handle)
        # Youmust group handles based on unique labels.
        plot_axes.legend(
            handles=handles,
            labels=named_serial_no('State', len(key_labels)),
            loc='upper right'
        )
    else:
        #TODO: Here cycle through sequential colormaps to point every diferent trial, but in time.
        colors = cm.summer(np.linspace(0, 1, duration - 1))
        for trial in range(total_data_trials):
            #TODO: plot stimulus stop and final point
            #_, stim_start_ms, stim_stop_ms, *_ = NWBfile.epochs[trial]
            for t, c in zip(range(duration - 1), colors):
                plot_axes.plot(
                    t_L_per_trial[0][trial][t:t+2],
                    t_L_per_trial[1][trial][t:t+2],
                    t_L_per_trial[2][trial][t:t+2],
                    color=c,
                    linewidth=3
                )
    if not plot_axes:
        plt.show()

    return t_L_per_trial


def q2sec(q_size=50, q_time=0):
    return np.divide(q_time, (1000 / q_size))

@nwb_unique_rng
def stim_variance_captured(
        input_NWBfile=None, S=None, T=None, T_all=None, plot_2d=False, plot_3d=False,
        stim_and_delay_range=None, delay_range=None, **kwargs
):
    '''
    Custom and all range must end at the same time and custom must also contain
    the stimulus
    :param NWBfile_array:
    :param plot:
    :param custom_range:
    :param klabels:
    :param smooth:
    :param kwargs:
    :return:
    '''

    # you need to check that ranges are as requested:
    if stim_and_delay_range[1] != delay_range[1]:
        raise ValueError('Ranges must end at the same time point!')

    # Data are by default the delay data, as on these we compute the mnemonic
    # and dynamic subspaces.
    data = get_correct_trials(input_NWBfile, custom_range=delay_range)
    # Eliminate time:
    data_over_stimuli = data.mean(axis=2)
    # Eliminate stimuli:
    data_over_time = data.mean(axis=1)
    # number of correct trials (stimuli):
    # Neurons
    N = data.shape[0]
    # Stimuli
    M = data.shape[1]
    # Remove the average activity of each neuron.
    r_bar = data_over_stimuli.mean(axis=1)
    max_delta_t = 80

    # Analog of Murray et al, 2017 figure 3A-B:
    fig2, ax2 = plt.subplots()

    # Stim variance captured for mnemonic subspace as a function of time:
    # This figure uses the stimulus and delay data, but the spaces defined only
    # on the delay data:
    data_all = get_correct_trials(
        input_NWBfile,
        custom_range=stim_and_delay_range
    )
    r_bar_all = data_all.mean(axis=2).mean(axis=1)

    V_t = np.full((data_all.shape[2], 1), np.nan)
    for t in range(data_all.shape[2]):
        X = data_all[:, :, t].T - r_bar_all
        C = (X.T @ X) / (M - 1)
        V_t[t, 0] = np.trace(S @ C @ S.T) / N

    # Stim variance captured for dynamic subspace as a function of time:
    # We use one iteration less, since T is evaluated between two adjacent time
    # points and has indices up to total_time - 1:
    V_Dtt = np.full((data_all.shape[2] - 1, 1), np.nan)
    for t in range(data_all.shape[2] - 1):
        X = data_all[:,:,t].T - r_bar_all
        C = (X.T @ X) / (M - 1)
        # C and T are for the same t:
        #TODO: again is the T for a time bin? maybe I need to correct something.
        V_Dtt[t, 0] = np.trace(T_all[t] @ C @ T_all[t].T) / N

    ax2.plot(np.nanmean(V_t, axis=1), color='blue')
    ax2.plot(np.nanmean(V_Dtt, axis=1), color='red')
    ax2.set_xlabel('Time (50ms bins)')
    ax2.set_ylabel('Stimulus variance captured')

    # Analog of Murray et al, 2017 figure 3C-D:
    fig3, ax3 = plt.subplots()

    # Stim variance captured for mnemonic subspace as a function of time
    # separation:
    V_t_delta_t = np.full((max_delta_t, data.shape[2]), np.nan)
    for delta_t in range(1, max_delta_t):
        for t in range(data.shape[2]-delta_t-1):
            X = data[:, :, t+delta_t].T - r_bar
            C = (X.T @ X) / (M - 1)
            V_t_delta_t[delta_t, t] = np.trace(S @ C @ S.T) / N

    # Stim variance captured for dynamic subspace as a function of time
    # separation:
    V_D_t_delta_t = np.full((max_delta_t, data.shape[2]), np.nan)
    for delta_t in range(1, max_delta_t):
        for t in range(data.shape[2]-delta_t-1):
            X = data[:, :, t+delta_t].T - r_bar
            C = (X.T @ X) / (M - 1)
            #TODO: To T den einai gia binned activity? H gia ka8e t time point?
            V_D_t_delta_t[delta_t, t] = np.trace(T[t] @ C @ T[t].T) / N

    ax3.plot(np.nanmean(V_t_delta_t, axis=1), color='blue')
    ax3.plot(np.nanmean(V_D_t_delta_t, axis=1), color='red')
    ax3.set_xlabel('Time Separation (50ms bins)')
    ax3.set_ylabel('Stimulus variance captured')


@nwb_unique_rng
def pcaL2_with_time_variance(
        input_NWBfile=None, plot_2d=False, plot_3d=False, stim_and_delay_range=None,
        delay_range=None,
        klabels=None, pca_components=20, smooth=False, plot_axes=None,
        axis_label_font_size=12, tick_label_font_size=12, labelpad_x=10,
        labelpad_y=10, **kwargs
):
    #TODO: HAS BUG: if klabels are few, it plots only them, concealing data!
    '''
    This function reads binned activity from a list of files and performs PCA
    with L=2 on it.
    :param NWBfile_array:
    :param plot:
    :param custom_range:
    :param klabels:
    :param smooth:
    :param kwargs:
    :return:
    '''
    # The subspaces estimation is done with the delay data only!
    data = get_correct_trials(input_NWBfile, custom_range=delay_range)
    data_over_stimuli = data.mean(axis=2)
    data_over_time = data.mean(axis=1)
    r_bar = data_over_stimuli.mean(axis=1)

    # Get mnemonic subspace, S:
    # Do the two (stim/time) PCAs and then orthogonalize the time axis:
    # how many PCA components over different stimuli?
    # Use max pca components, then decide how many to keep based on threshold.
    L = 2
    pca = decomposition.PCA(n_components=L)
    t_L_stimuli = pca.fit_transform(data_over_stimuli.T - r_bar).T
    components_stimuli = pca.components_
    latent_stimuli = pca.explained_variance_
    t_L_time = pca.fit_transform(data_over_time.T - r_bar).T
    components_time = pca.components_
    latent_time = pca.explained_variance_

    #todo: cross product?
    time_component = \
        components_time[0] \
        - (components_stimuli[0].T @ components_time[0]) * components_stimuli[0]\
        - (components_stimuli[1].T @ components_time[0]) * components_stimuli[1]
    # project data to components:
    S = np.concatenate((components_stimuli,time_component.reshape(-1,1).T), axis=0)

    # Get dynamic subspace T:
    L = 2
    pca = decomposition.PCA(n_components=L)
    T = []
    # My t time is in 50ms units (and the data in 50ms bins).
    for t in range(data.shape[2] -1):
        _ = pca.fit_transform(data[:,:,t].T - r_bar).T
        components_stimuli = pca.components_
        _ = pca.fit_transform(data_over_time[:,t:t+2].T - r_bar).T
        components_time = pca.components_

        #todo: cross product?
        time_component = \
            components_time[0] \
            - (components_stimuli[0].T @ components_time[0]) * components_stimuli[0] \
            - (components_stimuli[1].T @ components_time[0]) * components_stimuli[1]
        T.append(np.concatenate((components_stimuli,time_component.reshape(-1,1).T), axis=0))

    # Get also a dynamic subspace for ALL the data, not only the delay period:
    data_all = get_correct_trials(
        input_NWBfile,
        custom_range=stim_and_delay_range
    )
    data_over_time_all = data_all.mean(axis=1)
    r_bar_all = data_all.mean(axis=2).mean(axis=1)
    T_all = []
    # My t time is in 50ms units (and the data in 50ms bins).
    for t in range(data_all.shape[2] - 1):
        _ = pca.fit_transform(data_all[:, :, t].T - r_bar_all).T
        components_stimuli = pca.components_
        _ = pca.fit_transform(data_over_time_all[:, t:t+2].T - r_bar_all).T
        components_time = pca.components_

        #todo: cross product?
        time_component = \
            components_time[0] \
            - (components_stimuli[0].T @ components_time[0]) * components_stimuli[0] \
            - (components_stimuli[1].T @ components_time[0]) * components_stimuli[1]
        T_all.append(np.concatenate((components_stimuli, time_component.reshape(-1, 1).T), axis=0))

    correct_trials_no = data.shape[1]
    duration = data.shape[2]
    t_L_per_trial = np.empty([3,correct_trials_no,duration])
    for trial in range(correct_trials_no):
        #TODO: have you forgot to remove the mean?
        blah = S @ data[:,trial,:]
        if t_L_per_trial.size:
            t_L_per_trial[:,trial,:] = blah
            pass
        else:
            t_L_per_trial[:,0,:] = blah

    if False:
        # If not part of a subfigure, create one:
        if not plot_axes:
            fig = plt.figure()
            plt.ion()
            plot_axes = fig.add_subplot(111, projection='3d')
        #TODO: set title outside (you have the axis handle)
        #plot_axes.set_title(f'Synaptic Reconfiguration {learning_condition_id}')
        # Stylize the 3d plot:
        plot_axes.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        plot_axes.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        plot_axes.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        plot_axes.set_xlabel(
            'PC1', fontsize=axis_label_font_size, labelpad=labelpad_x
        )
        plot_axes.set_ylabel(
            'PC2', fontsize=axis_label_font_size, labelpad=labelpad_x
        )
        plot_axes.set_zlabel(
            'Time', fontsize=axis_label_font_size, labelpad=labelpad_x
        )
        pc1_max = int(np.max(blah[0, :].reshape(-1, 1)))
        pc1_min = int(np.min(blah[0, :].reshape(-1, 1)))
        pc2_max = int(np.max(blah[1, :].reshape(-1, 1)))
        pc2_min = int(np.min(blah[1, :].reshape(-1, 1)))
        pc3_max = int(np.max(blah[2, :].reshape(-1, 1)))
        pc3_min = int(np.min(blah[2, :].reshape(-1, 1)))
        pc1_axis_limits = (pc1_min, pc1_max)
        pc2_axis_limits = (pc2_min, pc2_max)
        pc3_axis_limits = (pc3_min, pc3_max)
        #TODO: change the 20 with a proper variable (do I have one?)
        time_axis_ticks = np.linspace(0, duration, (duration / 20) + 1)
        time_axis_ticklabels = q2sec(q_time=time_axis_ticks)  #np.linspace(0, time_axis_limits[1], duration)
        plot_axes.set_xlim(pc1_axis_limits)
        plot_axes.set_ylim(pc2_axis_limits)
        plot_axes.set_zlim(pc3_axis_limits)
        plot_axes.set_xticks(pc1_axis_limits)
        plot_axes.set_yticks(pc2_axis_limits)
        plot_axes.set_zticks(pc3_axis_limits)
        plot_axes.set_zticklabels(
            time_axis_ticklabels, fontsize=tick_label_font_size
        )
        plot_axes.elev = 22.5
        plot_axes.azim = 52.4

        colors = cm.viridis(np.linspace(0, 1, duration - 1))
        for t, c in zip(range(duration - 1), colors):
            plot_axes.plot(
                blah[0][t:t+2],
                blah[1][t:t+2],
                blah[2][t:t+2],
                color=c,
                linewidth=3
            )
        plt.show()


    total_data_trials = correct_trials_no
    if plot_2d:
        # Plots the t_L_r as 2d timeseries per trial. Also to ease the cluster
        # identification in the case of multiple learning conditions plots in
        # addition a 2d scatterplot of the data.

        # Scatterplot:
        if klabels is not None:
            # If not part of a subfigure, create one:
            if not plot_axes:
                fig = plt.figure()
                plt.ion()
                plot_axes = fig.add_subplot(111)

            plot_axes.set_title(f'Model {animal_model_id}, learning condition {learning_condition_id}')
            plot_axes.set_xlabel('PC1')
            plot_axes.set_ylabel('PC2')
            # Format 3d plot:
            #plot_axes.axhline(linewidth=4)  # inc. width of x-axis and color it green
            #plot_axes.axvline(linewidth=4)
            for axis in ['top', 'bottom', 'left', 'right']:
                plot_axes.spines[axis].set_linewidth(2)

            labels = klabels.tolist()
            nclusters = np.unique(klabels).size
            colors = cm.Set3(np.linspace(0, 1, nclusters))
            _, key_labels = np.unique(labels, return_index=True)
            handles = []
            for i, (trial, label) in enumerate(zip(range(total_data_trials), labels)):
                #print(f'Curently plotting trial: {trial}')
                for t in range(duration - 1):
                    handle, = plot_axes.plot(
                        t_L_per_trial[0, trial, t:t+2],
                        t_L_per_trial[1, trial, t:t+2],
                        label=f'Cluster {label}',
                        color=colors[label - 1],
                        alpha=t / duration
                    )
                if i in key_labels:
                    handles.append(handle)

            #TODO: in multiple NWB files case, with external labels (afto paei
            # ston caller k oxi edw ston callee, alla anyways) discarded
            # trials have labels and loops get out of index.

        else:
            plot_axes.set_title(f'Model {animal_model_id}, learning condition {learning_condition_id}')
            colors = cm.Greens(np.linspace(0, 1, duration - 1))
            for trial in range(total_data_trials):
                for t, c in zip(range(duration - 1), colors):
                    plot_axes.plot(
                        t_L_per_trial[0, trial, t:t+2],
                        t_L_per_trial[1, trial, t:t+2],
                        color=c,
                        alpha=t / duration
                    )
                mean_point = np.mean(np.squeeze(t_L_per_trial[:2, trial, :]), axis=1)
                plot_axes.scatter(
                    mean_point[0], mean_point[1], s=70, c='r', marker='+',
                    zorder=200
                )

    if plot_3d:
        # If not part of a subfigure, create one:
        if not plot_axes:
            fig = plt.figure()
            plt.ion()
            plot_axes = fig.add_subplot(111, projection='3d')
        #TODO: set title outside (you have the axis handle)
        #plot_axes.set_title(f'Synaptic Reconfiguration {learning_condition_id}')
        # Stylize the 3d plot:
        plot_axes.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        plot_axes.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        plot_axes.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        plot_axes.set_xlabel(
            'PC1', fontsize=axis_label_font_size, labelpad=labelpad_x
        )
        plot_axes.set_ylabel(
            'PC2', fontsize=axis_label_font_size, labelpad=labelpad_x
        )
        plot_axes.set_zlabel(
            'Time PC1', fontsize=axis_label_font_size, labelpad=labelpad_x
        )
        pc1_max = int(np.max(t_L_per_trial[0, :, :].reshape(-1, 1)))
        pc1_min = int(np.min(t_L_per_trial[0, :, :].reshape(-1, 1)))
        pc2_max = int(np.max(t_L_per_trial[1, :, :].reshape(-1, 1)))
        pc2_min = int(np.min(t_L_per_trial[1, :, :].reshape(-1, 1)))
        pc1_axis_limits = (pc1_min, pc1_max)
        pc2_axis_limits = (pc2_min, pc2_max)
        time_axis_limits = (0, duration)
        #TODO: change the 20 with a proper variable (do I have one?)
        time_axis_ticks = np.linspace(0, duration, (duration / 20) + 1)
        time_axis_ticklabels = q2sec(q_time=time_axis_ticks)  #np.linspace(0, time_axis_limits[1], duration)
        plot_axes.set_xlim(pc1_axis_limits)
        plot_axes.set_ylim(pc2_axis_limits)
        plot_axes.set_zlim(time_axis_limits)
        plot_axes.set_xticks(pc1_axis_limits)
        plot_axes.set_yticks(pc2_axis_limits)
        plot_axes.set_zticks(time_axis_ticks)
        plot_axes.set_zticklabels(
            time_axis_ticklabels, fontsize=tick_label_font_size
        )
        plot_axes.elev = 22.5
        plot_axes.azim = 52.4


        if klabels is not None:
            # EDIT: decide about that:
            progressive_colors = True
            #TODO: you need some more variables for the colors.
            if progressive_colors:
                # If you have cluster information for the data:
                # Create custom colormaps for the stimulus and trial period, per label.
                # Utilize normalized trial times, to create the colormaps.
                #TODO: make colormap gradient more steep after stimulus.
                #TODO: Make it read the proper value!
                stim_stop = 21  # Stim stop in q=50ms
                stim_stop_norm = stim_stop / duration

                c = mcolors.ColorConverter().to_rgb
                stim_start_color = 'limegreen'
                stim_stop_color = 'darkgreen'
                color_values = []
                if kwargs.get('plot_stim_color', False):
                    color_values.append(make_colormap(
                        [
                            c(stim_start_color), c(stim_stop_color), stim_stop_norm,
                            c('red'), c('red'), stim_stop_norm + 0.01,
                            c('red'), c('darkred'), 0.99,
                            c('darkred')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                        [
                            c(stim_start_color), c(stim_stop_color), stim_stop_norm,
                            c('violet'), c('violet'), stim_stop_norm + 0.01,
                            c('violet'), c('darkviolet'), 0.99,
                            c('darkviolet')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                        [
                            c(stim_start_color), c(stim_stop_color), stim_stop_norm,
                            c('gold'), c('gold'), stim_stop_norm + 0.01,
                            c('gold'), c('goldenrod'), 0.99,
                            c('goldenrod')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                        [
                            c(stim_start_color), c(stim_stop_color), stim_stop_norm,
                            c('blue'), c('blue'), stim_stop_norm + 0.01,
                            c('blue'), c('darkblue'), 0.99,
                            c('darkblue')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                        [
                            c(stim_start_color), c(stim_stop_color), stim_stop_norm,
                            c('deepskyblue'), c('deepskyblue'), stim_stop_norm + 0.01,
                            c('deepskyblue'), c('dodgerblue'), 0.99,
                            c('dodgerblue')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                        [
                            c(stim_start_color), c(stim_stop_color), stim_stop_norm,
                            c('lightgray'), c('lightgray'), stim_stop_norm + 0.01,
                            c('lightgray'), c('darkgray'), 0.99,
                            c('darkgray')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                        [
                            c(stim_start_color), c(stim_stop_color), stim_stop_norm,
                            c('magenta'), c('magenta'), stim_stop_norm + 0.01,
                            c('magenta'), c('darkmagenta'), 0.99,
                            c('darkmagenta')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                        [
                            c(stim_start_color), c(stim_stop_color), stim_stop_norm,
                            c('darkturquoise'), c('darkturquoise'), stim_stop_norm + 0.01,
                            c('darkturquoise'), c('teal'), 0.99,
                            c('teal')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                        [
                            c(stim_start_color), c(stim_stop_color), stim_stop_norm,
                            c('lime'), c('lime'), stim_stop_norm + 0.01,
                            c('lime'), c('limegreen'), 0.99,
                            c('limegreen')
                        ]
                    )
                    )
                else:
                    color_values.append(make_colormap(
                        [
                            c('red'), c('red'), 0.0,
                            c('red'), c('darkred'), 0.99,
                            c('darkred')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                        [
                            c('violet'), c('violet'), 0.0,
                            c('violet'), c('darkviolet'), 0.99,
                            c('darkviolet')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                        [
                            c('gold'), c('gold'), 0.0,
                            c('gold'), c('goldenrod'), 0.99,
                            c('goldenrod')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                        [
                            c('blue'), c('blue'), 0.0,
                            c('blue'), c('darkblue'), 0.99,
                            c('darkblue')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                        [
                            c('deepskyblue'), c('deepskyblue'), 0.0,
                            c('deepskyblue'), c('dodgerblue'), 0.99,
                            c('dodgerblue')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                        [
                            c('lightgray'), c('lightgray'), 0.0,
                            c('lightgray'), c('darkgray'), 0.99,
                            c('darkgray')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                        [
                            c('magenta'), c('magenta'), 0.0,
                            c('magenta'), c('darkmagenta'), 0.99,
                            c('darkmagenta')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                        [
                            c('darkturquoise'), c('darkturquoise'), 0.0,
                            c('darkturquoise'), c('teal'), 0.99,
                            c('teal')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                        [
                            c('lime'), c('lime'), 0.0,
                            c('lime'), c('limegreen'), 0.99,
                            c('limegreen')
                        ]
                    )
                    )

            labels = klabels.tolist()
            nclusters = np.unique(klabels).size
            colors = cm.Set2(np.linspace(0, 1, nclusters))
            _, key_labels = np.unique(labels, return_index=True)
            handles = []
            for i, (trial, label) in enumerate(zip(range(total_data_trials), labels)):
                if progressive_colors:
                    colors = color_values[label - 1 + kwargs.get('color_skip', 0)](np.linspace(0, 1, duration))
                    for t, c in zip(range(duration - 1), colors):
                        if t <= 0:
                            continue
                        handle = plot_axes.plot(
                            t_L_per_trial[0][trial][t:t+2],
                            t_L_per_trial[1][trial][t:t+2],
                            range(t,t+2),
                            color=c,
                            linewidth=3,
                            label=f'Cluster {label}'
                        )
                else:
                    x = t_L_per_trial[0][trial][:]
                    y = t_L_per_trial[1][trial][:]
                    handle = plot_axes.plot(x, y,
                                            range(duration),
                                            color=colors[label - 1],
                                            linewidth=3,
                                            label=f'Cluster {label}'
                                            )
                if i in key_labels:
                    handles.append(handle[0])
            # Youmust group handles based on unique labels.
            plot_axes.legend(
                handles=handles,
                labels=named_serial_no('State', len(key_labels)),
                loc='upper right'
            )
        else:
            colors = cm.viridis(np.linspace(0, 1, duration - 1))
            for trial in range(total_data_trials):
                for t, c in zip(range(duration - 1), colors):
                    plot_axes.plot(
                        t_L_per_trial[0][trial][t:t+2],
                        t_L_per_trial[1][trial][t:t+2],
                        [t, t+1], color=c,
                        linewidth=3
                    )
        if not plot_axes:
            plt.show()

    return t_L_per_trial, L, S, T, T_all


@nwb_unique_rng
def pcaL2(
        NWBfile_array=[], plot_2d=False, plot_3d=False, custom_range=None,
        klabels=None, pca_components=20, smooth=False, plot_axes=None,
        axis_label_font_size=12, tick_label_font_size=12, labelpad_x=10,
        labelpad_y=10, only_trials=None, **kwargs
):
    '''
    This function reads binned activity from a list of files and performs PCA
    with L=2 on it.
    :param NWBfile_array:
    :param plot:
    :param custom_range:
    :param klabels:
    :param smooth:
    :param kwargs:
    :return:
    '''
    legend_labels = kwargs.get('legend_labels', None)

    #TODO: make more readable the whole function:
    nfiles = len(NWBfile_array)
    if nfiles < 1:
        raise ValueError('Got empty NWBfile_array array!')

    # Initialize pool_array:
    pool_array = np.array([])
    for input_NWBfile in NWBfile_array:
        #TODO: is this deterministic? Because some times I got an error in some
        # matrix.
        animal_model_id, learning_condition_id, ncells, pn_no, ntrials, \
        trial_len, q_size, trial_q_no, correct_trials_idx, correct_trials_no = \
        get_acquisition_parameters(
            input_NWBfile=input_NWBfile,
            requested_parameters=[
                'animal_model_id', 'learning_condition_id', 'ncells',
                'pn_no', 'ntrials', 'trial_len', 'q_size', 'trial_q_no',
                'correct_trials_idx', 'correct_trials_no'
            ]
        )

        if correct_trials_no < 1:
            #TODO: Make it more elegant: check if you don't have at ALL data.
            #raise ValueError('No correct trials were found in the NWBfile!')
            pass

        # Use custom_range to compute PCA only on a portion of the original data:
        #TODO: bug: what if custom_range is None? trial_slice are not defined!
        if custom_range is not None:
            if not isinstance(custom_range, tuple):
                raise ValueError('Custom range must be a tuple!')
            trial_slice_start = int(custom_range[0])
            trial_slice_stop = int(custom_range[1])
            duration = trial_slice_stop - trial_slice_start
        else:
            duration = trial_q_no

        # Load binned acquisition (all trials together)
        binned_network_activity = input_NWBfile. \
            acquisition['binned_activity']. \
            data[:pn_no, :]. \
            reshape(pn_no, ntrials, trial_q_no)
        #If you care only about specific trials, get them:
        requested_trials = kwargs.get('only_trials', None)
        if requested_trials:
            # Slice out non correct trials and unwanted trial periods:
            tmp = binned_network_activity[:, requested_trials,
            trial_slice_start:trial_slice_stop]
            # Reshape in array with m=cells, n=time bins.
            tmp = tmp.reshape(pn_no, len(requested_trials) * duration)
        else:
            # Slice out non correct trials and unwanted trial periods:
            tmp = binned_network_activity[:, correct_trials_idx, trial_slice_start:trial_slice_stop]
            # Reshape in array with m=cells, n=time bins.
            tmp = tmp.reshape(pn_no, correct_trials_no * duration)
        # Concatinate it the pool array:
        if pool_array.size:
            pool_array = np.concatenate((pool_array, tmp), axis=1)
            pass
        else:
            pool_array = tmp


    # how many PCA components
    # Use max pca components, then decide how many to keep based on threshold.
    L = np.min(pool_array.shape)  # pca_components
    pca = decomposition.PCA(n_components=L)
    t_L = pca.fit_transform(pool_array.T).T
    latent = pca.explained_variance_

    #total_variance = np.sum(latent)
    #explained_variance = latent / total_variance
    #cum_explained_variance = np.cumsum(explained_variance)
    #L = np.nonzero(np.less(cum_explained_variance, 0.8))[0].size

    #norm_eigenvals = latent / latent.max()
    #eigenval_diff = np.abs(np.diff(norm_eigenvals))
    #L = np.nonzero(np.greater(eigenval_diff, 0.02))[0].size
    L = np.nonzero(np.greater(latent, 1.0))[0].size
    if L < 2:
        if plot_2d or plot_3d:
            # assuming that user wants a plot only, continue:
            L = 2
            print('No eigenvalue larger that 1! Plotting with L=2.')
        else:
            raise ValueError('No eigenvalue larger that 1!')

    print(f'L found to be: {L}')
    # Reshape PCA results into separate trials for plotting.
    #t_L_per_trial = t_L.reshape(L, correct_trials_no, duration, order='C')
    #TODO: do a more elegant way of splitting into trials:
    total_data_trials = int(pool_array.shape[1]/duration)
    t_L_per_trial = t_L[:L, :].reshape(L, total_data_trials, duration, order='C')
    #TODO: somewhere here I get a warning about a non-tuple sequence for
    # multi dim indexing. Why?
    if smooth:
        for trial in range(total_data_trials):
            for l in range(L):
                t_L_per_trial[l, trial, :] = savgol_filter(t_L_per_trial[l, trial, :], 11, 3)

    #TODO: prepei na ftia3w to plotting mia k kalh...se ligo 8a 8ewreitai bug
    if plot_2d:
        # Plots the t_L_r as 2d timeseries per trial. Also to ease the cluster
        # identification in the case of multiple learning conditions plots in
        # addition a 2d scatterplot of the data.

        # Set custom colors:
        c = mcolors.ColorConverter().to_rgb
        color_values = []
        color_values.append(make_colormap(
            [
                c('red'), c('red'), 0.0,
                c('red'), c('darkred'), 0.99,
                c('darkred')
            ]
        )
        )
        color_values.append(make_colormap(
            [
                c('blue'), c('blue'), 0.0,
                c('blue'), c('darkblue'), 0.99,
                c('darkblue')
            ]
        )
        )
        color_values.append(make_colormap(
            [
                c('gold'), c('gold'), 0.0,
                c('gold'), c('goldenrod'), 0.99,
                c('goldenrod')
            ]
        )
        )
        color_values.append(make_colormap(
            [
                c('violet'), c('violet'), 0.0,
                c('violet'), c('darkviolet'), 0.99,
                c('darkviolet')
            ]
        )
        )
        color_values.append(make_colormap(
            [
                c('deepskyblue'), c('deepskyblue'), 0.0,
                c('deepskyblue'), c('dodgerblue'), 0.99,
                c('dodgerblue')
            ]
        )
        )
        color_values.append(make_colormap(
            [
                c('lightgray'), c('lightgray'), 0.0,
                c('lightgray'), c('darkgray'), 0.99,
                c('darkgray')
            ]
        )
        )
        color_values.append(make_colormap(
            [
                c('magenta'), c('magenta'), 0.0,
                c('magenta'), c('darkmagenta'), 0.99,
                c('darkmagenta')
            ]
        )
        )
        color_values.append(make_colormap(
            [
                c('darkturquoise'), c('darkturquoise'), 0.0,
                c('darkturquoise'), c('teal'), 0.99,
                c('teal')
            ]
        )
        )
        color_values.append(make_colormap(
            [
                c('lime'), c('lime'), 0.0,
                c('lime'), c('limegreen'), 0.99,
                c('limegreen')
            ]
        )
        )
        color_values.append(make_colormap(
            [
                c('olivedrab'), c('olivedrab'), 0.0,
                c('olivedrab'), c('yellowgreen'), 0.99,
                c('yellowgreen')
            ]
        )
        )
        # Add more colors in case there is need for them:
        color_values = [
                *color_values,
                *color_values,
                *color_values,
                *color_values,
                *color_values,
                ]

        # Scatterplot:
        if klabels is not None:
            # If not part of a subfigure, create one:
            if not plot_axes:
                fig = plt.figure()
                plt.ion()
                plot_axes = fig.add_subplot(111)

            plot_axes.set_title(f'Model {animal_model_id}, learning condition {learning_condition_id}')
            plot_axes.set_xlabel('PC1')
            plot_axes.set_ylabel('PC2')
            # Format 3d plot:
            #plot_axes.axhline(linewidth=4)  # inc. width of x-axis and color it green
            #plot_axes.axvline(linewidth=4)
            for axis in ['top', 'bottom', 'left', 'right']:
                plot_axes.spines[axis].set_linewidth(2)

            labels = klabels.tolist()
            nclusters = np.unique(klabels).size

            # Check that your custom label strings are of the right number.
            if legend_labels is not None:
                if len(legend_labels) != nclusters:
                    raise ValueError(
                        'Number of trials and number of labels must match!')

            #colors = cm.Set3(np.linspace(0, 1, nclusters))
            _, key_labels = np.unique(labels, return_index=True)
            handles = []
            for i, (trial, label) in enumerate(zip(range(total_data_trials), labels)):
                colors = color_values[label - 1](np.linspace(0, 1, duration))
                #print(f'Curently plotting trial: {trial}')
                for t in range(duration - 1):
                    handle, = plot_axes.plot(
                        t_L_per_trial[0, trial, t:t+2],
                        t_L_per_trial[1, trial, t:t+2],
                        label=f'Cluster {label}',
                        color=colors[label - 1],
                        alpha=t / duration
                    )
                if i in key_labels:
                    handles.append(handle)

                # You must group handles based on unique labels.
                if (legend_labels is not None):
                    legend_str_tags = legend_labels
                else:
                    legend_str_tags = named_serial_no('State', len(key_labels))

                '''
                plot_axes.legend(
                    handles=handles,
                    labels=legend_str_tags,
                    loc='upper right'
                )
                '''

            #TODO: in multiple NWB files case, with external labels (afto paei
            # ston caller k oxi edw ston callee, alla anyways) discarded
            # trials have labels and loops get out of index.
        else:
            plot_axes.set_title(f'Model {animal_model_id}, learning condition {learning_condition_id}')
            colors = cm.Greens(np.linspace(0, 1, duration - 1))
            for trial in range(total_data_trials):
                for t, c in zip(range(duration - 1), colors):
                    plot_axes.plot(
                        t_L_per_trial[0, trial, t:t+2],
                        t_L_per_trial[1, trial, t:t+2],
                        color=c,
                        alpha=t / duration
                    )
                mean_point = np.mean(np.squeeze(t_L_per_trial[:2, trial, :]), axis=1)
                plot_axes.scatter(
                    mean_point[0], mean_point[1], s=70, c='r', marker='+',
                    zorder=200
                )

    if plot_3d:
        # If not part of a subfigure, create one:
        if not plot_axes:
            fig = plt.figure()
            plt.ion()
            plot_axes = fig.add_subplot(111, projection='3d')
        #TODO: set title outside (you have the axis handle)
        #plot_axes.set_title(f'Synaptic Reconfiguration {learning_condition_id}')
        # Stylize the 3d plot:
        plot_axes.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        plot_axes.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        plot_axes.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        plot_axes.set_xlabel(
            'PC1', fontsize=axis_label_font_size, labelpad=labelpad_x
        )
        plot_axes.set_ylabel(
            'PC2', fontsize=axis_label_font_size, labelpad=labelpad_x
        )
        plot_axes.set_zlabel(
            'Time', fontsize=axis_label_font_size, labelpad=labelpad_x
        )
        pc1_max = int(np.max(t_L_per_trial[0, :, :].reshape(-1, 1)))
        pc1_min = int(np.min(t_L_per_trial[0, :, :].reshape(-1, 1)))
        pc2_max = int(np.max(t_L_per_trial[1, :, :].reshape(-1, 1)))
        pc2_min = int(np.min(t_L_per_trial[1, :, :].reshape(-1, 1)))
        pc1_axis_limits = (pc1_min, pc1_max)
        pc2_axis_limits = (pc2_min, pc2_max)
        time_axis_limits = (0, duration)
        #TODO: change the 20 with a proper variable (do I have one?)
        time_axis_ticks = np.linspace(0, duration, (duration / 20) + 1)
        time_axis_ticklabels = q2sec(q_time=time_axis_ticks)  #np.linspace(0, time_axis_limits[1], duration)
        plot_axes.set_xlim(pc1_axis_limits)
        plot_axes.set_ylim(pc2_axis_limits)
        plot_axes.set_zlim(time_axis_limits)
        plot_axes.set_xticks(pc1_axis_limits)
        plot_axes.set_yticks(pc2_axis_limits)
        plot_axes.set_zticks(time_axis_ticks)
        plot_axes.set_zticklabels(
            time_axis_ticklabels, fontsize=tick_label_font_size
        )
        plot_axes.elev = 22.5
        plot_axes.azim = 52.4

        if klabels is not None:
            # EDIT: decide about that:
            progressive_colors = True
            #TODO: you need some more variables for the colors.
            if progressive_colors:
                # If you have cluster information for the data:
                # Create custom colormaps for the stimulus and trial period, per label.
                # Utilize normalized trial times, to create the colormaps.
                #TODO: make colormap gradient more steep after stimulus.
                #TODO: Make it read the proper value!
                stim_stop = 21  # Stim stop in q=50ms
                stim_stop_norm = stim_stop / duration

                c = mcolors.ColorConverter().to_rgb
                stim_start_color = 'limegreen'
                stim_stop_color = 'darkgreen'
                color_values = []
                #TODO: kati prepei na kanw me afta ta xrwmata..
                if kwargs.get('plot_stim_color', False):
                    color_values.append(make_colormap(
                            [
                                #red
                                c(stim_start_color), c(stim_stop_color), stim_stop_norm,
                                c('ivory'), c('ivory'), stim_stop_norm + 0.01,
                                c('ivory'), c('darkred'), 0.99,
                                c('darkred')
                            ]
                        )
                    )
                    color_values.append(make_colormap(
                            [
                                #violet
                                c(stim_start_color), c(stim_stop_color), stim_stop_norm,
                                c('ivory'), c('ivory'), stim_stop_norm + 0.01,
                                c('ivory'), c('darkviolet'), 0.99,
                                c('darkviolet')
                            ]
                        )
                    )
                    color_values.append(make_colormap(
                            [
                                #gold
                                c(stim_start_color), c(stim_stop_color), stim_stop_norm,
                                c('ivory'), c('ivory'), stim_stop_norm + 0.01,
                                c('ivory'), c('goldenrod'), 0.99,
                                c('goldenrod')
                            ]
                        )
                    )
                    color_values.append(make_colormap(
                            [
                                c(stim_start_color), c(stim_stop_color), stim_stop_norm,
                                c('blue'), c('blue'), stim_stop_norm + 0.01,
                                c('blue'), c('darkblue'), 0.99,
                                c('darkblue')
                            ]
                        )
                    )
                    color_values.append(make_colormap(
                            [
                                c(stim_start_color), c(stim_stop_color), stim_stop_norm,
                                c('deepskyblue'), c('deepskyblue'), stim_stop_norm + 0.01,
                                c('deepskyblue'), c('dodgerblue'), 0.99,
                                c('dodgerblue')
                            ]
                        )
                    )
                    color_values.append(make_colormap(
                            [
                                c(stim_start_color), c(stim_stop_color), stim_stop_norm,
                                c('lightgray'), c('lightgray'), stim_stop_norm + 0.01,
                                c('lightgray'), c('darkgray'), 0.99,
                                c('darkgray')
                            ]
                        )
                    )
                    color_values.append(make_colormap(
                        [
                            c(stim_start_color), c(stim_stop_color), stim_stop_norm,
                            c('magenta'), c('magenta'), stim_stop_norm + 0.01,
                            c('magenta'), c('darkmagenta'), 0.99,
                            c('darkmagenta')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                        [
                            c(stim_start_color), c(stim_stop_color), stim_stop_norm,
                            c('darkturquoise'), c('darkturquoise'), stim_stop_norm + 0.01,
                            c('darkturquoise'), c('teal'), 0.99,
                            c('teal')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                            [
                                c(stim_start_color), c(stim_stop_color), stim_stop_norm,
                                c('lime'), c('lime'), stim_stop_norm + 0.01,
                                c('lime'), c('limegreen'), 0.99,
                                c('limegreen')
                            ]
                        )
                    )
                else:
                    color_values.append(make_colormap(
                        [
                            c('red'), c('red'), 0.0,
                            c('red'), c('darkred'), 0.99,
                            c('darkred')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                        [
                            c('violet'), c('violet'), 0.0,
                            c('violet'), c('darkviolet'), 0.99,
                            c('darkviolet')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                        [
                            c('gold'), c('gold'), 0.0,
                            c('gold'), c('goldenrod'), 0.99,
                            c('goldenrod')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                        [
                            c('blue'), c('blue'), 0.0,
                            c('blue'), c('darkblue'), 0.99,
                            c('darkblue')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                        [
                            c('deepskyblue'), c('deepskyblue'), 0.0,
                            c('deepskyblue'), c('dodgerblue'), 0.99,
                            c('dodgerblue')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                        [
                            c('lightgray'), c('lightgray'), 0.0,
                            c('lightgray'), c('darkgray'), 0.99,
                            c('darkgray')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                        [
                            c('magenta'), c('magenta'), 0.0,
                            c('magenta'), c('darkmagenta'), 0.99,
                            c('darkmagenta')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                        [
                            c('darkturquoise'), c('darkturquoise'), 0.0,
                            c('darkturquoise'), c('teal'), 0.99,
                            c('teal')
                        ]
                    )
                    )
                    color_values.append(make_colormap(
                        [
                            c('lime'), c('lime'), 0.0,
                            c('lime'), c('limegreen'), 0.99,
                            c('limegreen')
                        ]
                    )
                    )

            labels = klabels.tolist()
            nclusters = np.unique(klabels).size
            colors = cm.Set2(np.linspace(0, 1, nclusters))
            _, key_labels = np.unique(labels, return_index=True)
            handles = []

            # Check that your custom label strings are of the right number.
            if legend_labels is not None:
                if len(legend_labels) != nclusters:
                    raise ValueError('Number of trials and number of labels must match!')

            #Block zip() from accidentally skipping trials.
            if (total_data_trials != len(labels)):
                raise ValueError('Number of trials and number of labels must match!')

            for i, (trial, label) in enumerate(zip(range(total_data_trials), labels)):
                if progressive_colors:
                    colors = color_values[label - 1 + kwargs.get('color_skip', 0)](np.linspace(0, 1, duration))
                    for t, c in zip(range(duration - 1), colors):
                        if t <= 0:
                            continue
                        handle = plot_axes.plot(
                            t_L_per_trial[0][trial][t:t+2],
                            t_L_per_trial[1][trial][t:t+2],
                            range(t,t+2),
                            color=c,
                            linewidth=3,
                            label=f'Cluster {label}'
                        )
                else:
                    x = t_L_per_trial[0][trial][:]
                    y = t_L_per_trial[1][trial][:]
                    handle = plot_axes.plot(x, y,
                                      range(duration),
                                      color=colors[label - 1],
                                      linewidth=3,
                                      label=f'Cluster {label}'
                                      )
                if i in key_labels:
                    handles.append(handle[0])
            # Youmust group handles based on unique labels.
            if (legend_labels is not None):
                legend_str_tags = legend_labels
            else:
                legend_str_tags = named_serial_no('State', len(key_labels))

            plot_axes.legend(
                handles=handles,
                labels=legend_str_tags,
                loc='upper right'
            )
        else:
            colors = cm.viridis(np.linspace(0, 1, duration - 1))
            for trial in range(total_data_trials):
                for t, c in zip(range(duration - 1), colors):
                    plot_axes.plot(
                        t_L_per_trial[0][trial][t:t+2],
                        t_L_per_trial[1][trial][t:t+2],
                        [t, t+1], color=c,
                        linewidth=3
                    )
        if not plot_axes:
            plt.show()

    return t_L_per_trial, L

#TODO: I want this also to implement a Cross validation routine.
# So: I need a list of M binary matrices that include a randomly sampled, sparse
# part from the orignal matrix, A (with the constraints that apply).
class NMF_HALS(object):

    """ Base class for NMF algorithms
    Specific algorithms need to be implemented by deriving from this class.
    #TODO: this code is from https://github.com/kimjingu/nonnegfac-python/
    under the BSD 3 license.

    """
    default_max_iter = 100
    default_max_time = np.inf

    def __init__(self, default_max_iter=100, default_max_time=np.inf):
        self.eps = 1e-16
        self.set_default(default_max_iter, default_max_time)

    def initializer(self, W, H):
        W, H, weights = matu.normalize_column_pair(W, H)
        return W, H

    def iter_solver(self, A, W, H, k, it):
        AtW = A.T @ W
        WtW = W.T @ W
        for kk in iter(range(0, k)):
            temp_vec = H[:, kk] + AtW[:, kk] - H @ WtW[:, kk]
            H[:, kk] = np.maximum(temp_vec, self.eps)

        AH = A @ H
        HtH = H.T @ H
        for kk in iter(range(0, k)):
            temp_vec = W[:, kk] * HtH[kk, kk] + AH[:, kk] - W @ HtH[:, kk]
            W[:, kk] = np.maximum(temp_vec, self.eps)
            ss = np.linalg.norm(W[:, kk])
            if ss > 0:
                W[:, kk] = W[:, kk] / ss

        return (W, H)

    def set_default(self, default_max_iter, default_max_time):
        self.default_max_iter = default_max_iter
        self.default_max_time = default_max_time

    def run(self, A, k, M=None, init=None, max_iter=None, max_time=None, verbose=0):
        """ Run a NMF algorithm
        Parameters
        ----------
        A : numpy.array or scipy.sparse matrix, shape (m,n)
        M : numpy.array of the same size as A used as mask.
        k : int - target lower rank
        Optional Parameters
        -------------------
        init : (W_init, H_init) where
                    W_init is numpy.array of shape (m,k) and
                    H_init is numpy.array of shape (n,k).
                    If provided, these values are used as initial values for NMF iterations.
        max_iter : int - maximum number of iterations.
                    If not provided, default maximum for each algorithm is used.
        max_time : int - maximum amount of time in seconds.
                    If not provided, default maximum for each algorithm is used.
        verbose : int - 0 (default) - No debugging information is collected, but
                                    input and output information is printed on screen.
                        -1 - No debugging information is collected, and
                                    nothing is printed on screen.
                        1 (debugging/experimental purpose) - History of computation is
                                        returned. See 'rec' variable.
                        2 (debugging/experimental purpose) - History of computation is
                                        additionally printed on screen.
        Returns
        -------
        (W, H, rec)
        W : Obtained factor matrix, shape (m,k)
        H : Obtained coefficient matrix, shape (n,k)
        rec : dict - (debugging/experimental purpose) Auxiliary information about the execution
        """
        #TODO: Give params like train/valid/test M:
        info = {'k': k,
                'alg': str(self.__class__),
                'A_dim_1': A.shape[0],
                'A_dim_2': A.shape[1],
                'A_type': str(A.__class__),
                'max_iter': max_iter if max_iter is not None else self.default_max_iter,
                'verbose': verbose,
                'max_time': max_time if max_time is not None else self.default_max_time}
        if init != None:
            W = init[0].copy()
            H = init[1].copy()
            info['init'] = 'user_provided'
        else:
            W = np.random.rand(A.shape[0], k)
            H = np.random.rand(A.shape[1], k)
            info['init'] = 'uniform_random'

        if verbose >= 0:
            print ('[NMF] Running: ')
            #print (json.dumps(info, indent=4, sort_keys=True))

        norm_A = matu.norm_fro(A)
        total_time = 0

        if verbose >= 1:
            his = {'iter': [], 'elapsed': [], 'rel_error': []}

        start = time.time()
        # algorithm-specific initilization
        (W, H) = self.initializer(W, H)
        # If M binary mask provided, use it on A:
        if M is not None:
            A = A * M

        for i in range(1, info['max_iter'] + 1):
            start_iter = time.time()
            # algorithm-specific iteration solver
            (W, H) = self.iter_solver(A, W, H, k, i)
            elapsed = time.time() - start_iter

            if verbose >= 1:
                rel_error = matu.norm_fro_err(A, W, H, norm_A) / norm_A
                his['iter'].append(i)
                his['elapsed'].append(elapsed)
                his['rel_error'].append(rel_error)
                if verbose >= 2:
                    print ('iter:' + str(i) + ', elapsed:' + str(elapsed) + ', rel_error:' + str(rel_error))

            total_time += elapsed
            if total_time > info['max_time']:
                break

        W, H, weights = matu.normalize_column_pair(W, H)

        final = {}
        final['norm_A'] = norm_A
        final['rel_error'] = matu.norm_fro_err(A, W, H, norm_A) / norm_A
        final['iterations'] = i
        final['elapsed'] = time.time() - start

        rec = {'info': info, 'final': final}
        if verbose >= 1:
            rec['his'] = his

        if verbose >= 0:
            print ('[NMF] Completed: ')
            #print (json.dumps(final, indent=4, sort_keys=True))
        return (W, H, rec)


@nwb_unique_rng
def nnmf(A, k, M_train=None, **kwargs):
    '''
    Applies Non Negative Matrix Factorization in matrix A, with k components.
    If optional M_train binary matrix is given, alternating minimization is performed
    with zero values of M_train as missing values of A.

    This function also gets (through kwargs) the NWB file and a RNG iteration
    integer to be reproducible.

    :param A: numpy.ndarray
    :param k: int
    :param M_train: numpy.ndarray, optional
    :return:
    '''
    # maximum multiplicative rules iterations
    max_iters = 100

    # Initialize W and H:
    W = np.random.rand(A.shape[0], k) * 70
    H = np.random.rand(k, A.shape[1])

    if M_train is not None:
        A = M_train * A
    else:
        M_train = np.ones(A.shape)

    #W_dist = np.zeros((max_iters))
    #H_dist = np.zeros((max_iters))

    for iter in range(max_iters):
        '''
        W_previous = W
        H_previous = H
        '''
        W = np.maximum(W * ((A @ H.T) / ((M_train * (W @ H)) @ H.T)), 1e-16)
        H = np.maximum(H * ((W.T @ A) / (W.T @ (M_train * (W @ H)))), 1e-16)
        '''
        W_dist[iter] = \
            np.linalg.norm((W_previous - W), ord='fro') / \
            np.sqrt(A.shape[0] * A.shape[1])
        H_dist[iter] = \
            np.linalg.norm((H_previous - H), ord='fro') / \
            np.sqrt(A.shape[0] * A.shape[1])
        '''

        # Break for efficiency if converged:
        #if W_dist[iter] < 1e-12 and H_dist[iter] < 1e-12:
            #break

    return W, H#, W_dist, H_dist

def NNMF(
        NWBfile_array=[], plot_2d=False, plot_3d=False, custom_range=None,
        klabels=None, n_components=5, smooth=False, plot_axes=None,
        M=None, **kwargs
):
    '''
    This function reads binned activity from a list of files and performs NNMF
    with k on it.
    If optional M binary matrix, then it applies it during alternating
    minimization.
    :param NWBfile_array:
    :param plot:
    :param custom_range:
    :param klabels:
    :param smooth:
    :param kwargs:
    :return:
    '''
    #TODO: make more readable the whole function:
    nfiles = len(NWBfile_array)
    if nfiles < 1:
        raise ValueError('Got empty NWBfile_array array!')

    # Initialize pool_array:
    pool_array = np.array([])
    for input_NWBfile in NWBfile_array:
        #TODO: WHAT IS THIS MESS!
        animal_model_id, learning_condition_id, ncells, pn_no, ntrials, \
        trial_len, q_size, trial_q_no, correct_trials_idx, correct_trials_no = \
            get_acquisition_parameters(
                input_NWBfile=input_NWBfile,
                requested_parameters=[
                    'animal_model_id', 'learning_condition_id', 'ncells',
                    'pn_no', 'ntrials', 'trial_len', 'q_size', 'trial_q_no',
                    'correct_trials_idx', 'correct_trials_no'
                ]
            )
        # Get the new duration from the data (if custom_range is given):
        tmp = get_correct_trials(input_NWBfile, custom_range=custom_range)
        duration = tmp.shape[2]
        # Reshape in array with m=cells, n=time bins.
        tmp = tmp.reshape(pn_no, correct_trials_no * duration)
        # Concatinate it the pool array:
        if pool_array.size:
            pool_array = np.concatenate((pool_array, tmp), axis=1)
            pass
        else:
            pool_array = tmp

    #TODO: replace randomization with proper cross-validation:
    #M = np.random.rand(pool_array.shape[0], pool_array.shape[1])
    #TODO: make analysis functions reproducable with rng wrappers. In what field
    # to base the seed?
    if n_components > np.min(pool_array.shape):
        print('Must be k < min(n,m)!')
        n_components = np.min(pool_array.shape)

    W, H = nnmf(
        pool_array.T, n_components,
        input_NWBfile=NWBfile_array[0]
    )
    #W, H, info = NMF_HALS().run(pool_array.T, n_components, M=M)
    H = H.T
    #TODO: return the fit error and test error!
    error_bar = None
    error_train = None
    print('Cross-validated!')
    ##TODO: check that this is NOT RANDOM and thus REPRODUCABLE!
    ## Randomize original data by permuting each row (observation).
    #if randomize:
    #    for row in range(pool_array.shape[1]):
    #        row_data = pool_array[:, row]
    #        pool_array[:, row] = np.random.permutation(row_data)

    #estimator = decomposition.NMF(n_components=n_components, init='nndsvd',
    #                              tol=5e-3)
    #W = estimator.fit_transform(pool_array.T).T
    #H = estimator.components_
    #dist_from_original = np.linalg.norm(pool_array - (W.T @ H).T, ord='fro') / \
    #                     np.sqrt(pool_array.size)

    #TODO: do a more elegant way of splitting into trials:
    total_data_trials = int(pool_array.shape[1]/duration)
    components_per_trial = W.T.reshape(n_components, total_data_trials, duration, order='C')
    #TODO: somewhere here I get a warning about a non-tuple sequence for
    # multi dim indexing. Why?
    if smooth:
        for trial in range(total_data_trials):
            for l in range(n_components):
                components_per_trial[l, trial, :] = \
                    savgol_filter(components_per_trial[l, trial, :], 11, 3)

    if plot_2d:
        # Plots the t_L_r as 2d timeseries per trial. Also to ease the cluster
        # identification in the case of multiple learning conditions plots in
        # addition a 2d scatterplot of the data.

        # Scatterplot:
        if klabels is not None:
            # If not part of a subfigure, create one:
            if not plot_axes:
                fig = plt.figure()
                plt.ion()
                plot_axes = fig.add_subplot(111)

            plot_axes.set_title(f'Model {animal_model_id}, learning condition {learning_condition_id}')
            plot_axes.set_xlabel('PC1')
            plot_axes.set_ylabel('PC2')
            # Format 3d plot:
            #plot_axes.axhline(linewidth=4)  # inc. width of x-axis and color it green
            #plot_axes.axvline(linewidth=4)
            for axis in ['top', 'bottom', 'left', 'right']:
                plot_axes.spines[axis].set_linewidth(2)

            labels = klabels.tolist()
            nclusters = np.unique(klabels).size
            colors = cm.Set2(np.linspace(0, 1, nclusters))
            _, key_labels = np.unique(labels, return_index=True)
            handles = []
            for i, (trial, label) in enumerate(zip(range(total_data_trials), labels)):
                #print(f'Curently plotting trial: {trial}')
                for t in range(duration - 1):
                    handle, = plot_axes.plot(
                        components_per_trial[0, trial, t:t+2],
                        components_per_trial[1, trial, t:t+2],
                        label=f'Cluster {label}',
                        color=colors[label - 1],
                        alpha=t / duration
                    )
                if i in key_labels:
                    handles.append(handle)

            for clusterid in range(nclusters):
                #TODO: This comprehension is problematic, why?
                # Plot each cluster mean (average of last second activity):
                #cluster_trials = [
                #    idx
                #    for idx, label in enumerate(labels)
                #    if label == clust + 1
                #]
                cluster_trials = []
                for idx, label in enumerate(labels):
                    if label == clusterid + 1:
                        cluster_trials.append(idx)
                mean_point = np.mean(
                    components_per_trial[:2, cluster_trials, :]. \
                        reshape(2, len(cluster_trials) * duration),
                    axis=1
                )
                plot_axes.scatter(
                    mean_point[0], mean_point[1], s=70, c='k', marker='+',
                    zorder=20000
                )

        else:
            plot_axes.set_title(f'Model {animal_model_id}, learning condition {learning_condition_id}')
            colors = cm.Greens(np.linspace(0, 1, duration - 1))
            for trial in range(total_data_trials):
                for t, c in zip(range(duration - 1), colors):
                    plot_axes.plot(
                        components_per_trial[0, trial, t:t+2],
                        components_per_trial[1, trial, t:t+2],
                        color=c,
                        alpha=t / duration
                    )
                mean_point = np.mean(np.squeeze(components_per_trial[:2, trial, :]), axis=1)
                plot_axes.scatter(
                    mean_point[0], mean_point[1], s=70, c='r', marker='+',
                    zorder=200
                )

    if plot_3d:
        # If not part of a subfigure, create one:
        if not plot_axes:
            fig = plt.figure()
            plt.ion()
            plot_axes = fig.add_subplot(111, projection='3d')
        plot_axes.set_title(f'Model {animal_model_id}, learning condition {learning_condition_id}')
        # Stylize the 3d plot:
        plot_axes.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        plot_axes.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        plot_axes.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        plot_axes.set_xlabel('PC1')
        plot_axes.set_ylabel('PC2')
        plot_axes.set_zlabel('Time')
        pca_axis_limits = (-10, 10)
        time_axis_limits = (0, duration)
        #TODO: change the 20 with a proper variable (do I have one?)
        time_axis_ticks = np.linspace(0, duration, (duration / 20) + 1)
        time_axis_ticklabels = q2sec(q_time=time_axis_ticks)  #np.linspace(0, time_axis_limits[1], duration)
        plot_axes.set_xlim(pca_axis_limits)
        plot_axes.set_ylim(pca_axis_limits)
        plot_axes.set_zlim(time_axis_limits)
        plot_axes.set_xticks(pca_axis_limits)
        plot_axes.set_yticks(pca_axis_limits)
        plot_axes.set_zticks(time_axis_ticks)
        plot_axes.set_zticklabels(time_axis_ticklabels)
        plot_axes.elev = 22.5
        plot_axes.azim = 52.4

        if klabels is not None:
            labels = klabels.tolist()
            nclusters = np.unique(klabels).size
            colors = cm.Set2(np.linspace(0, 1, nclusters))
            _, key_labels = np.unique(labels, return_index=True)
            handles = []
            for i, (trial, label) in enumerate(zip(range(total_data_trials), labels)):
                x = components_per_trial[0][trial][:]
                y = components_per_trial[1][trial][:]
                handle = plot_axes.plot(x, y,
                                        range(duration),
                                        color=colors[label - 1],
                                        label=f'Cluster {label}'
                                        )
                if i in key_labels:
                    handles.append(handle)
            # Youmust group handles based on unique labels.
            plt.legend(handles)
        else:
            colors = cm.viridis(np.linspace(0, 1, duration - 1))
            for trial in range(total_data_trials):
                for t, c in zip(range(duration - 1), colors):
                    plot_axes.plot(
                        components_per_trial[0][trial][t:t+2],
                        components_per_trial[1][trial][t:t+2],
                        [t, t+1], color=c
                    )
        if not plot_axes:
            plt.show()

    return (components_per_trial, error_bar, error_train)

def from_zero_to(x):
    '''
    Range wrapper, counting from zero to x excluded.
    :param x:
    :return:
    '''
    return range(x)

def from_one_to(x):
    '''
    Range wrapper, counting from one to x included.
    :param x:
    :return:
    '''
    return range(1, x + 1)


def trial_instantaneous_frequencies(NWBfile=None, trialid=None, smooth=False):
    # Divide binned activity with the bin size:

    animal_model_id, learning_condition_id, ncells, pn_no, ntrials, \
    trial_len, q_size, trial_q_no, correct_trials_idx, correct_trials_no = \
        get_acquisition_parameters(
            input_NWBfile=NWBfile,
            requested_parameters=[
                'animal_model_id', 'learning_condition_id', 'ncells',
                'pn_no', 'ntrials', 'trial_len', 'q_size', 'trial_q_no',
                'correct_trials_idx', 'correct_trials_no'
            ]
        )

    total_trial_qs = trial_len / q_size
    one_sec_qs = 1000 / q_size
    start_q = total_trial_qs - one_sec_qs

    network_spiketrains = get_acquisition_spikes(
        NWBfile=NWBfile,
        acquisition_name='membrane_potential',
    )

    # Unpack cell ids and their respective spike trains:
    cell_ids, cell_spiketrains = zip(*[
        cell_spiketrain
        for cell_spiketrain in network_spiketrains[trialid]
    ])

    #trials = get_acquisition_trials(
    #    NWBfile=NWBfile
    #)
    ## Unpack trial start/stop.
    #_, trial_start_ms, trial_stop_ms, *_ = trials[trialid]

    trial_inst_ff = []
    for cell_id, cell_spiketrain in zip(cell_ids, cell_spiketrains):
        if len(cell_spiketrain) > 1:
            # Add a spike at the end to prolong the last isi:
            spiketrain = np.array(cell_spiketrain)
            isi = np.diff(spiketrain)
            instantaneous_lambda = np.divide(1000.0, isi)
            x_points = spiketrain[:-1] + isi / 2
            instantaneous_firing_frequency = np.interp(
                np.linspace(0, trial_len, int(trial_len)),
                x_points,
                instantaneous_lambda,
                left=0.0,
                right=0.0
            )
            if smooth:
                instantaneous_firing_frequency = \
                    savgol_filter(instantaneous_firing_frequency, 501, 3)
                # Smoothing can introduce negative values:
                instantaneous_firing_frequency[
                    instantaneous_firing_frequency < 0
                ] = 0

            trial_inst_ff.append((cell_id, instantaneous_firing_frequency))

    return trial_inst_ff


def quick_spikes(
        voltage_trace=None, upper_threshold=None, lower_threshold=None,
        samples_per_ms=10, plot=False
):
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

    #Find values from low, crossing high threshold:
    upper_crossings = np.greater(voltage_trace, upper_threshold)
    not_upper_crossings = np.logical_not(upper_crossings)
    # You want the points where the vt crosses the upper threshold and again the lower one.
    # Simply detect crossing the upper threshold, across vt:
    ts_1 = np.add(
        upper_crossings.astype(int), np.roll(not_upper_crossings, 1).astype(int)
    )
    # These are the time points where the upper threshold is crossed.
    lower2upper_crossings = np.nonzero(np.greater(ts_1, 1))[0]
    # Simply detect crossing the lower threshold, across vt:
    lower_crossings = np.less(voltage_trace, lower_threshold)
    not_lower_crossings = np.logical_not(lower_crossings)
    # greater than before and then roll, to detect lower threshold crossings.
    ts_2 = np.add(
        not_lower_crossings.astype(int),
        np.roll(lower_crossings.astype(int), -1)
    )
    # Avoid having a spike JUST at the very last moment. This could happen if a
    # spike was ongoing when the simulation was stopped. And could blow up
    # something downstream if ever the spike times are used as array indices.
    # Im my case this can happen, so this is implementation specific.
    ts_2[-1] = 1
    # This is the time points where the lower threshold is called
    upper2lower_crossings = np.nonzero(np.greater(ts_2, 1))[0]

    spike_timings = []
    if lower2upper_crossings.size > 0 and upper2lower_crossings.size > 0:
        # Make sure that we start from lower2upper (e.g due to noise)!!!:
        tmpidx = np.greater(upper2lower_crossings, lower2upper_crossings[0])
        upper2lower_crossings = upper2lower_crossings[tmpidx]

        # Make sure that each upward threshold crossing is matched with a downward:
        all_crossings = np.hstack(
            (lower2upper_crossings, upper2lower_crossings)
        )
        binary_crossings = np.hstack(
            (np.ones((lower2upper_crossings.size)),
             np.ones((upper2lower_crossings.size)) * -1)
        )
        idx = np.argsort(all_crossings)
        # Detect spike crossings (first cross the upper, then cross the lower):
        spike_crossings = np.nonzero(
            np.less(np.diff(binary_crossings[idx]), 0))[0]
        spikes_start = all_crossings[idx[spike_crossings]]
        spikes_end = all_crossings[idx[spike_crossings + 1]]

        #TODO: remove old code after commit:
        #tmp = np.not_equal(np.diff(binary_crossings[idx]), 0)
        # Make sure we end the sequence with a upper2lower crossing:
        #if binary_crossings[idx[-1]] < 0:
        #    unique_crossings = idx[np.hstack((tmp, [True]))]
        #else:
        #    unique_crossings = idx[np.hstack((tmp, [False]))]

        # Make sure that we have the same amount of starts/ends:
        #if np.mod(unique_crossings.size, 2):
        #    raise ValueError('crossnigs are not unique!')
        #all_crossings_sorted = all_crossings[unique_crossings]. \
        #    reshape(-1, 2)
        #spikes_start = all_crossings_sorted[:, 0]
        #spikes_end = all_crossings_sorted[:, 1]

        # Then, get the maximum voltage in this region.
        for start, stop in zip(spikes_start, spikes_end):
            spike_timings.append(
                (np.argmax(voltage_trace[start:stop+1]) + start) /
                samples_per_ms
            )

    # Plot if requested.
    if plot:
        #vt_reduced = voltage_trace.loc[::samples_per_ms]
        vt_reduced = voltage_trace[::samples_per_ms]
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

def generate_slices(size=50, number=2, start_from=0, to_slice = True):
    '''
    Generate starting/ending positions of q_total windows q of size q_size.
    TODO: MAJOR BUG in slicing! To recheck whenever I use this function!
    TODO: accomondate the user case starting from idx zero, rather than one.
    If the toslice is True, you got one extra included unit at the end, so the qs can be used to slice a array.
    :param q_size:
    :param q_total:
    :return:
    '''
    for q in range(number):
        q_start = q * size
        q_end = q_start + size
        # yield starting/ending positions of q (in ms)
        if to_slice:
            yield (q_start + start_from, q_end + start_from)
        else:
            yield (q_start + start_from, q_end + start_from - 1)

class NDPoint():
    def __init__(self, ndarray=None):
        assert ndarray.shape[0] == 1
        self.ndarray = ndarray
        self.m, self.n = ndarray.shape
        # this is point dimension
        self.dim = self.n


class MeanShiftCentroid(NDPoint):
    # This is specific to mean shift
    @property
    def shift(self):
        return self._shift

    @shift.setter
    def shift(self, shift):
        self._shift = shift

    def __init__(self, ndarray=None):
        NDPoint.__init__(self, ndarray)
        self.shift = 100000

    def distance_from(self, pts_array):
        m, n = pts_array.shape
        assert self.dim == n, 'Arrays have different dimensions'
        distance = spatial.distance.pdist(
            np.concatenate((self.ndarray, pts_array), axis=0)
        )
        #TODO: do I start from 0 or 1: this is NOT squareform!
        return (distance[:m]).reshape(m, 1)

    def update(self, ndarray):
        # Updates/moves centroid. Also calculates norm of movement
        m, n = ndarray.shape
        assert m == 1, 'Nd array must be row vector!'
        self.shift = np.linalg.norm(self.ndarray - ndarray)
        self.ndarray = ndarray

    def to_point(self):
        return NDPoint(self.ndarray)
        pass





def K_rbf(distance=None, sigma=None):
    # Calculate K RBF for one or multiple points
    # distance is ||x-x`||
    return np.exp(-np.divide(np.power(distance, 2), 2 * (sigma ** 2)))

def mu_bar(k_rbf=None, xs=None):
    # calculate new mu_bar based on K_rbf:
    # K can be np.ndarray of one or many scalars
    m, n = xs.shape
    m2, n2 = k_rbf.shape
    assert m == m2, 'mu_bar: Arrays have different dims!'
    x_mu_bar = np.multiply(k_rbf, xs).sum(axis=0) / k_rbf.sum()
    return x_mu_bar.reshape(1, -1)

def mean_shift(data=None, k=None, plot=False, **kwargs):

    dims, ntrials, duration = data.shape
    #return [density_pts, sigma_hat]
    # k for the kmeans (how many clusters)
    # N are the number of trials (meanshift initial points)
    #N = properties.ntrials
    # Collapse data to ndim points (dims):
    pts = data.reshape(dims, ntrials * duration, order='C')
    # Take the average STD the cells in PCA space:
    sigma_hat = pts.std(axis=1)
    # TODO: make slices smaller, we need a rational size window (1sec?) to account for activity drifting.
    # TODO: to slicing den einai swsto gia C type arrays; prepei na to ftia3w, kai na bebaiw8w gia opou allou to xrisimopoiw!
    std_array = np.array([
        pts[:, slice_start:slice_end].std(axis=1)
        for slice_start, slice_end in generate_slices(size=duration, number=ntrials)
    ])
    sigma_hat = std_array.mean(axis=0).mean()

    if plot:
        fig, ax = plt.subplots()
        plt.ion()
        ax.scatter(pts[0, :], pts[1, :], s=2, c='black')
        plt.xlabel('Principal component 1')
        plt.ylabel('Principal component 2')
        plt.show()

    # Initialize k centroids:
    init_centroids = data.mean(axis=2)
    centroids = [
        MeanShiftCentroid(init_centroids[:, [x]].T)
        for x in range(ntrials)
    ]
    #rw_mus = squeeze(mean(X,1))'
    while any([centroid.shift for centroid in centroids] > sigma_hat/10):
        for centroid in centroids:
            k_rbf = K_rbf(
                distance=centroid.distance_from(pts.T),
                sigma=sigma_hat
            )
            mu_x_bar = mu_bar(k_rbf=k_rbf, xs=pts.T)
            #This centroid movement also updates the shift (movement distance):
            if plot:
                plt.plot(
                    # TypeError: only integer scalar arrays can be converted to a scalar index. Why I get this:
                    #np.concatenate(centroid.ndarray[0, [0]], mu_x_bar[0, [0]]),
                    [centroid.ndarray[0, 0], mu_x_bar[0, 0]],
                    [centroid.ndarray[0, 1], mu_x_bar[0, 1]],
                    c='lime'
                )
                plt.scatter(centroid.ndarray[:,0], centroid.ndarray[:,1], s=20, c='lime')
                plt.scatter(mu_x_bar[:,0], mu_x_bar[:,1], s=20, c='red', zorder=1000)
                plt.pause(0.0001)
            centroid.update(mu_x_bar)

    # Cast to a list of points:
    points = [
        centroid.to_point()
        for centroid in centroids
    ]

    return (points, sigma_hat)

def itriu(size=None, idx=None):
    # transform index (idx) to i,j pair in pdist, without getting its squareform:
    # Only for square (pdist)matrices of side size
    start = 1
    ctr = 0
    for i in range(size):
        for j in range(start, size):
            if ctr == idx:
                return (i, j)
            ctr += 1
        start += 1

def ndpoints2array(points=None, **kwargs):
    # TODO: works as expected??
    # convert a batch of same dimension points to an np.array, for handling:
    #Optionally keep only one dim (useful for plotting):
    assert len(points), 'Points array is empty!'
    only_dim = kwargs.get('only_dim', range(points[0].dim))
    pts_array = np.concatenate([
        point.ndarray[:, only_dim]
        for point in points
    ])
    return pts_array
    pass

def initialize_algorithm(data=None, k=None, plot=None, **kwargs):
    '''
    This function initializes the k-means, performing a mean-shift first and,
    clustering initially with the euclidean distance of the points from the
    closest cluster center.
    :param data:
    :param k:
    :param plot:
    :param kwargs:
    :return:
    '''
    dims, ntrials, duration = data.shape

    mean_shift_points, sigma_hat = mean_shift(data=data, k=k, plot=False, **kwargs)
    # TODO: check if I get this error and handle it:
    #if any(isnan(density_pts))
    #    error('I got NaNs inside density points!');
    #end

    # Sort density points by their distance and merge them, pair-wise, until have only k of them:
    while len(mean_shift_points) > k:
        # Calculate distances between mean shift points:
        mean_shift_pts_inter_distance = spatial.distance.pdist(
            ndpoints2array(mean_shift_points)
        )
        # Locate the two points (i,j) having the smallest distance:
        idx = np.argsort(mean_shift_pts_inter_distance, axis=0)
        i, j = itriu(size=len(mean_shift_points), idx=idx[0])
        # Replace them with their mean:
        # TODO:this wont work
        tmp_point = NDPoint(
            np.mean(
                ndpoints2array([mean_shift_points[i], mean_shift_points[j]]),
            axis=0).reshape(1, -1)
        )
        mean_shift_points[i] = tmp_point
        mean_shift_points.pop(j)

    if plot:
        fig, ax = plt.subplots()
        plt.ion()
        ax.scatter(
            ndpoints2array(points=mean_shift_points, only_dim=0),
            ndpoints2array(points=mean_shift_points, only_dim=1),
        s=10, c='red')
        # Collapse data to ndim points (2d):
        duration = data.shape[2]
        pts = data.reshape(dims, ntrials * duration, order='C')
        ax.scatter(pts[0], pts[1], s=2, c='black')
        plt.xlabel('Principal component 1')
        plt.ylabel('Principal component 2')
        plt.show()
        pass

    # Initialize the clusters with simple euclidean distance:
    ed_array = np.zeros((k, ntrials))
    for cluster in range(k):
        for trial in range(ntrials):
            ed_array[cluster, trial] = point2points_average_euclidean(
                point=mean_shift_points[cluster].ndarray, points=data[:, trial, :].T
            )
    klabels = ed_array.argmin(axis=0)
    return klabels

def create_agregate_dataset(klabels=None, k=None):
    # Group the trials in clusters, creating an aggregate dataset:
    aggregate_dataset = [
        np.nonzero(cluster == klabels)[0]
        for cluster in range(k)
    ]
    return aggregate_dataset

def mahalanobis_distance(idx_a=None, idx_b=None):
    # Compute the MD of a trial average (idx_b) from the cluster centroid (idx_a):
    # Return also the mu and S for later use.
    #TODO: check again how to handle e.g. mean to return a matrix not a vector. It really mess up the multiplications..
    try:
        dim, trials, t = idx_a.shape
        if any(np.array(idx_a.shape) < 1):
            return np.nan, MD_params(np.nan, np.nan)

        cluster_data = idx_a.reshape(dim, t * trials, order='C').T
        point_data = idx_b[:, 0, :].mean(axis=1).reshape(1, -1)
        mu = cluster_data.mean(axis=0).reshape(1, -1)
        S = np.cov(cluster_data.T)
        # Debug/scatter
        if False:
            fig, ax = plt.subplots(1,1)
            ax.scatter(point_data[:,0], point_data[:,1], s=50, c='r', marker='+')
            ax.scatter(np.squeeze(idx_b)[0,:], np.squeeze(idx_b)[1,:],s=20, c='r', marker='.')
            ax.scatter(cluster_data[:,0], cluster_data[:,1],s=5, c='g', marker='.')
            ax.scatter(mu[:,0], mu[:,1],s=50, c='k', marker='+')

        tmp = point_data - mu
        # Covariance matrix must be positive, semi definite. Check that:
        try:
            if S.shape == ():
                MD = np.sqrt(tmp * (1/S) * tmp)
            else:
                MD = np.sqrt(tmp @ np.linalg.inv(S) @ tmp.T)
        except np.linalg.LinAlgError as e:
            raise e('Covariance matrix must be positive semi definite!')
    except Exception as e:
        print(str(e))
        raise(e)
    return MD, MD_params(mu, S)

def point2points_average_euclidean(point=None, points=None):
    # return the average euclidean distance between the point a and points b
    # run dist of a single point against a list of points:
    m, n = points.shape
    distance = spatial.distance.pdist(
        np.concatenate((point, points), axis=0)
    )
    return np.mean(distance[:m])

#@time_it
def kmeans_clustering(data=None, k=2, max_iterations=100, plot=False, **kwargs):
    #return (labels_final, J_k_final, S_i)
    # Perform kmeans clustering.
    # Input:
    #   X: n x d data matrix
    #   m: initialization parameter (k)
    # Adapted by Stefanos Stamatiadis (stamatiad.st@gmail.com).
    def rearrange_labels(klabels, distance_array, params):
        # Make the klabels unique and monotonically incrementing:
        # Change also the md_array row ordering to reflect that.
        key_labels = np.unique(klabels)
        # Labels start from 1:
        new_klabels = np.zeros(klabels.shape, dtype=int)
        new_params = {}
        for new_label, original_label in enumerate(key_labels):
            new_klabels[klabels == original_label] = new_label + 1
            # Swap distance array rows, utilizing slice copy and unpacking:
            distance_array[original_label, :],\
            distance_array[new_label, :] = \
                distance_array[new_label, :].copy(),\
                distance_array[original_label, :].copy()
            # Update parameters dictionary:
            new_params[new_label] = params[original_label]
        return new_klabels, distance_array, new_params

    def compare_arrays(arr1=None, arr2=None):
        arr_a = arr1.tolist()
        arr_b = arr2.tolist()
        assert len(arr_a) == len(arr_b), 'Arrays must have the same length!'
        for i, j in zip(arr_a, arr_b):
            if i != j:
                return False
        return True

    dims, ntrials, total_qs = data.shape
    # Initiallization step:
    klabels = initialize_algorithm(data=data, k=k, plot=plot, **kwargs)
    aggregate_dataset = create_agregate_dataset(klabels=klabels, k=k)
    for iteration in from_one_to(max_iterations):
        # Assignment step:
        md_array = np.zeros((k, ntrials))
        md_params_d = {}
        for cluster in range(k):
            for trial in range(ntrials):
                md_array[cluster, trial], md_params_d[cluster] = \
                    mahalanobis_distance(
                    idx_a=data[:, aggregate_dataset[cluster], :],
                    idx_b=data[:, [trial], :]
                )
        klabels_old = klabels
        klabels = np.nanargmin(md_array, axis=0)
        # Update step:
        aggregate_dataset = create_agregate_dataset(klabels=klabels, k=k)
        # Termination step:
        if k == 1:
            break
        #TODO: change name to something better:
        if compare_arrays(klabels_old, klabels):
            break
    # Calculate the within cluster distance:
    cumulative_dist = np.zeros((1, ntrials))
    for cluster_i in range(k):
        cumulative_dist[0, cluster_i] = np.nansum(
            md_array[cluster_i, [aggregate_dataset[cluster_i]]]
        )
    J_k = cumulative_dist.sum() / ntrials
    klabels, md_array, md_params_d = rearrange_labels(klabels, md_array, md_params_d)
    # Debug plot the results for quick inspection:
    if False:
        dim, trials, t = data.shape
        flatten_data = data.reshape(dim, t * trials, order='C').T
        fig, ax = plt.subplots(1,1)
        colors = cm.Set2(np.linspace(0, 1, len(klabels)))
        for i, l in enumerate(klabels):
            this_label = klabels == i
            len_label = sum(this_label)
            label_data = data[:, this_label, :].reshape(dim, t * len_label, order='C').T
            label_data_mean = label_data.mean(axis=0).reshape(1, -1)
            ax.scatter(label_data[:,0], label_data[:,1], s=20, c=colors[i,:], marker='.')
            ax.scatter(label_data_mean[:,0], label_data_mean[:,1], s=50, c=colors[i,:], marker='+')

        ax.scatter(flatten_data[:,0], flatten_data[:,1],s=5, c='k', marker='.')

    return klabels, J_k, md_array, md_params_d

def evaluate_clustering(
        klabels=None,
        md_array=None,
        md_params=None,
        no_optimal_L=None,
        **kwargs
):
    # Calculate likelihood of each trial, given the cluster centroid:
    nclusters, ntrials = md_array.shape

    ln_L = np.zeros((1, ntrials))
    ln_L.fill(np.nan)
    for trial in range(ntrials):
        #TODO: is the labels with the md_array values aligned? Or they change with the unique labeling code?
        #TODO: decide in using start from 0 or 1...
        cluster = klabels[trial] - 1
        mdist = md_array[cluster, trial]
        S = md_params[cluster].S
        #TODO: to nclusters einai la8os. To k sto original einai to dimensionality
        # tou multivariate distribution (katalabainw oti einai to k number of PCA
        # components). Na bebaiw8w kai oti o S den einai o antistrofos pou xrhsimopoiw
        # sto MD.
        # Remove clusters without any points:
        try:
            if S.shape == ():
                ln_L[0, trial] = \
                    np.exp(-1/2 * mdist) / \
                    np.sqrt((2*np.pi)**no_optimal_L * float(S))
            else:
                ln_L[0, trial] = \
                    np.exp(-1/2 * mdist) / \
                    np.sqrt((2*np.pi)**no_optimal_L * np.linalg.det(S))
        except Exception as e:
            print('Something went wrong while evaluating BIC!')
            print(str(e))
    L_ln_hat = np.nansum(np.log(ln_L * 0.0001))

    if L_ln_hat > 0:
        print('L_ln_hat is wrong! Possibly near singular S!')
        BIC = np.nan
    else:
        BIC = np.log(ntrials)*nclusters - 2 * L_ln_hat
    return BIC


def test_for_overfit(klabels=None, data_pca=None, S=None, threshold=None):
    '''
    Test for overfit by computing the MD between centroids, but utilizing the
    overall data covariance. If k means overfits, return True.
    :param klabels:
    :param data_pca:
    :param S:
    :param threshold:
    :return:
    '''
    unique_labels = np.unique(klabels)
    k = len(unique_labels)
    if k == 1:
        #print('\tk=1, so skipping test for overfit.')
        return False

    dim, ntrial, duration = data_pca.shape
    # Attention, labels must start from 0:
    aggregate_datasets = create_agregate_dataset(klabels=klabels-1, k=k)
    cluster_mu = []
    for dataset in aggregate_datasets:
        cluster_mu.append(data_pca[:, dataset, :].reshape(dim, -1).mean(axis=1))

    # I only need the triangular upper portion of the distance matrix:
    offset = 1
    for i in range(k):
        for j in range(offset, k):
            # Calculate the distance between clusters:
            xy_diff = cluster_mu[i] - cluster_mu[j]
            if S.shape == ():
                MD = np.sqrt(xy_diff * (1/S) * xy_diff)
            else:
                MD = np.sqrt(xy_diff @ np.linalg.inv(S) @ xy_diff.T)
            # If MD is less that the threshold, then k-means overfits
            if MD < threshold:
                return True
        offset += 1

    return False


@nwb_unique_rng
def determine_number_of_clusters(
        NWBfile_array=[], max_clusters=None, custom_range=None,
        **kwargs
    ):
    '''
    Return the optimal number of clusters, as per BIC:

    :param NWBfile_array:
    :param max_clusters:
    :param custom_range:
    :param kwargs:
    :return:
    '''

    nfiles = len(NWBfile_array)
    if nfiles < 1:
        raise ValueError('Got empty NWBfile_array array!')

    #TODO: This is python's EAFP, but tide up a little bit please:
    animal_model_id, learning_condition_id, ncells, pn_no, ntrials, \
    trial_len, q_size, trial_q_no, correct_trials_idx, correct_trials_no = \
    get_acquisition_parameters(
        input_NWBfile=NWBfile_array[0],
        requested_parameters=[
            'animal_model_id', 'learning_condition_id', 'ncells',
            'pn_no', 'ntrials', 'trial_len', 'q_size', 'trial_q_no',
            'correct_trials_idx', 'correct_trials_no'
        ]
    )
    #TODO: Need to have in the code the distinction of valid trials:
    if correct_trials_no == 0:
        return 0, None, None, None

    #TODO: this custom range must be given by the caller!
    total_trial_qs = trial_len / q_size
    one_sec_qs = 1000 / q_size
    start_q = total_trial_qs - one_sec_qs

    # Perform PCA to the binned network activity:
    # Analyze only each trial's last second:
    data_pca, no_optimal_L = pcaL2(
        NWBfile_array=NWBfile_array,
        custom_range=(start_q, total_trial_qs),
        plot=False
    )
    #TODO: Apparently this CAN have no activity in the last second?? Is not
    # this the definition of PA?
    if data_pca.size == 0:
        return 0, None, None, None

    # Keep only prominent principal components from the data set:
    data_pca = data_pca[:no_optimal_L, :, :]

    dims, ntrials, duration = data_pca.shape

    try:
        # move the means to the origin for each trial:
        #TODO: plot this, to see it:
        all_datapoints = np.zeros((dims, duration, ntrials))
        for trial in range(ntrials):
            trial_datapoints = np.squeeze(data_pca[:, trial, :]).T
            mu = trial_datapoints.mean(axis=0)
            all_datapoints[:, :, trial] = np.transpose(trial_datapoints - mu)

        # Compute this 'average' S (covariance).
        S_all = np.cov(all_datapoints.reshape(dims, duration * ntrials))
    except Exception as e:
        print(f'Exception when calculating S!')
        raise e

    try:
        #assert max_clusters <= ntrials, 'Cannot run kmeans with greater k than the data_pcapoints!'
        if max_clusters > ntrials:
            print('Cannot run kmeans with greater k than the data_pcapoints!')
            max_clusters = ntrials

        kmeans_labels = np.zeros((max_clusters, ntrials), dtype=int)
        J_k_all = [0] * max_clusters
        BIC_all = [0] * max_clusters
        md_params_all = [0] * max_clusters
        # Calculate BIC for up to max_clusters:
        for i, k in enumerate(range(1, max_clusters + 1)):
            #print(f'Clustering with {k} clusters.')
            try:
                klabels, J_k, md_array, md_params_d = kmeans_clustering(
                    data=data_pca, k=k, max_iterations=100, **kwargs
                )
            except Exception as e:
                print(f'something hapened {str(e)}')
            # I need to check overfitting (clustering into multiple subclusters).
            #  Since
            # the BIC will BE better moving over greater k, we need to calculate a
            # maximum k - over that some k cluster centroids will be so close,
            # essentially belonging to the same cluster.
            # So after each k means I need to check if centroids are so close.
            #TODO: 8elw ena function pou 8a ypologizei ola me ola ta centroids
            # kai 8a entopizei afta pou exoun MD mikroterh apo ena threshold.
            # Afto 8a shmatodotei k to telos tou increase k, ka8ws einai quasi-
            # Deterministic to k-means init (mean-shift) kai oso kai na synexizw
            # den yparxei periptwsh na parw 'diaforetiko' syndyasmo apo labels.
            #TODO: afto pou me apasxolei einai, 8a einai to BIC kalo ean apla to
            # krathsw se afth th sta8erh timh?
            k_means_overfit = test_for_overfit(
                klabels=klabels, data_pca=data_pca, S=S_all, threshold=3.0
            )
            if k_means_overfit:
                print(f'@k:{k} k_means Overfit!!!')
                # Stop searching for fit of greater ks.
                BIC_all[i:] = [BIC_all[i - 1]] * (max_clusters - i)
                kmeans_labels[i:, :] = kmeans_labels[i - 1, :]
                J_k_all[i:] = [J_k_all[i - 1]] * (max_clusters - i)
                md_params_all[i:] = [md_params_all[i - 1]] * (max_clusters - i)
                break
            else:
                BIC = evaluate_clustering(
                    klabels=klabels, md_array=md_array, md_params=md_params_d,
                    no_optimal_L=no_optimal_L, **kwargs
                )
                BIC_all[i] = BIC
                kmeans_labels[i, :] = klabels.T
                J_k_all[i] = J_k
                md_params_all[i] = md_params_d

        #TODO: remove the y since you dont use it anymore!
        if len(np.unique(BIC_all)) == 1:
            # If all BIC values are the same, we overfitted on k=2. So we get
            # the labels from k=1.
            K_s_labelidx = 0
            # and the K* is one:
            optimal_k = 1
        else:
            # Compute K* as a variant of the rate distortion function, utilizing BIC:
            #K_s = np.argmax(np.diff(np.power(BIC_all, -y)))
            #TODO: I argue that the BIC is already done, since I just have
            # pick the smallest value:
            K_s = np.argmin(BIC_all)
            # The idx of the kmeans_labels array (starts from 0 = one cluster):
            K_s_labelidx = K_s
            # This directly corresponds to how many clusters:
            optimal_k = K_s_labelidx + 1

        # Add 1 to start counting from 1, then another, since we diff above:
        # This is the optimal no of clusters:
        K_star = optimal_k
        # Store the klabels corresponding to each K*:
        K_labels = kmeans_labels[K_s_labelidx, :]

        # Get BIC value:
        BIC_value = np.min(BIC_all)
        # Calculate no of PC that cross the 70% variance:

        #principal_components_no = np.nonzero(np.greater(np.cumsum(explained_variance), 0.7))[0][0] + 1

        return K_star, K_labels, BIC_value, no_optimal_L
    except Exception as e:
        raise e

@nwb_unique_rng
def determine_number_of_ensembles(
        NWBfile_array, K_max=None, custom_range=None,
        K_cv=10, rng_max_iters=20, **kwargs
):
    '''
    Same as deternime_number_of_clusters, but utilizes NNMF in search for
    optimal number of distinct neuronal ensembles.
    This function needs to perform NNMF. To decide about the number of
    components a cross-validation method is used.
    Parts of the A array are removed, randomly and then their reconstruction
    error is assessed.
    This function returns the best  estimate.

    :param NWBfile_array:
    :param K_max:
    :param custom_range:
    :param kwargs:
    :return:
    '''

    nfiles = len(NWBfile_array)
    if nfiles < 1:
        raise ValueError('Got empty NWBfile_array array!')

    #TODO: This is python's EAFP, but tide up a little bit please:
    animal_model_id, learning_condition_id, ncells, pn_no, ntrials, \
    trial_len, q_size, trial_q_no, correct_trials_idx, correct_trials_no = \
        get_acquisition_parameters(
            input_NWBfile=NWBfile_array[0],
            requested_parameters=[
                'animal_model_id', 'learning_condition_id', 'ncells',
                'pn_no', 'ntrials', 'trial_len', 'q_size', 'trial_q_no',
                'correct_trials_idx', 'correct_trials_no'
            ]
        )

    # Check if you need to continue:
    fn_str = (
        'cross_validation_errors_structured'
        '_AM{animal_model_id}'
        '_LC{learning_condition_id}'
        '_Kmax{K_max}'
        '_Kcv{K_cv}'
        '_RI{rng_max_iters}.hdf').format
    filename = Path(fn_str(
        animal_model_id=animal_model_id,
        learning_condition_id=learning_condition_id,
        K_max=K_max,
        K_cv=K_cv,
        rng_max_iters=rng_max_iters
    ))
    #filename = Path(f'cross_validation_errors_structured_AM{animal_model_id}_LC{learning_condition_id}_RI{rng_max_iters}.hdf')
    if filename.is_file():
        #return (1, 1, 1)
        pass

    total_trial_qs = trial_len / q_size
    one_sec_qs = 1000 / q_size
    start_q = total_trial_qs - one_sec_qs

    tmp = get_correct_trials(NWBfile_array[0])
    # Reshape in array with m=cells, n=time bins.
    duration = tmp.shape[2]
    data = tmp.reshape(pn_no, correct_trials_no * duration)
    norm_factor = data.shape[0] * data.shape[1]

    # This is now handled by the decorator!
    #np.random.seed(animal_model_id * 10 + learning_condition_id)
    # Training error:
    error_train = np.zeros((K_max, K_cv, rng_max_iters))
    error_train_rm = np.zeros((K_max, K_cv, rng_max_iters))
    # Test error:
    error_test = np.zeros((K_max, K_cv, rng_max_iters))
    error_test_rm = np.zeros((K_max, K_cv, rng_max_iters))
    for k in range(1, K_max + 1):
        print(f'No of components {k}')
        for rng_iteration in range(rng_max_iters):
            # Divide data into K_cv partitions:
            #TODO: make sure this is always doable!
            rnd_idx = np.random.permutation(data.size)
            K_idx = np.zeros(data.size)
            partition_size = int(np.floor(data.size / K_cv))
            for partition_id, (start, end) in enumerate(generate_slices(size=partition_size, number=K_cv)):
                K_idx[rnd_idx[start:end]] = partition_id
            K_idx = K_idx.reshape(data.shape)

            # Run NNMF in a K_cv-1 fashion:
            for k_fold in range(K_cv):
                print(f'k_fold is {k_fold}')
                #tic = time.perf_counter()
                # M_test einai to M_test, einai ta liga, ta left outs.
                M_test = K_idx == k_fold
                # Afto einai to M_train
                M_train = np.logical_not(M_test)
                #toc = time.perf_counter()
                #print(f'M matrices took {toc-tic} seconds.')
                #TODO: You need to perform NNMF multiple times, to avoid (must you?) the
                # variations due to the random initialization.
                #tic = time.perf_counter()
                W, H = nnmf(
                    data, k, M_train=M_train,
                    input_NWBfile=NWBfile_array[0], rng_iter=rng_iteration
                )
                #toc = time.perf_counter()
                #print(f'NNMF took {toc-tic} seconds.')

                if False:
                    # ready made NNMF algo:
                    estimator = decomposition.NMF(
                        n_components=k, init='nndsvd', tol=5e-3
                    )
                    W_rm = estimator.fit_transform(M_train * data)
                    H_rm = estimator.components_
                    #TODO: to error bar rm ypologizetai kala, opote gia poio logo
                    #xreiazetai pragmatika na xrhsimopoihsw gradient descent?
                    #TODO: Deftero: blepw oti ta data mou exoun para poly variation
                    # otan xrhsimopoiw diaforetiko random sto GD. Mhpws afto
                    # shmainei oti den xrhsimopoiw polla iterations? Den 8a eprepe
                    # (efoson to partition einai to idio) na mou dinei k_fold arketa
                    # paromoia apotelesmata?
                    error_train_rm[k-1, k_fold, rng_iteration] = \
                        np.linalg.norm(M_train * (W_rm @ H_rm - data), ord='fro') / \
                        np.sqrt(norm_factor)
                    error_test_rm[k-1, k_fold, rng_iteration] = \
                        np.linalg.norm(M_test * (W_rm @ H_rm - data), ord='fro') / \
                        np.sqrt(norm_factor)

                #tic = time.perf_counter()
                error_train[k-1, k_fold, rng_iteration] = \
                    np.linalg.norm(M_train * (W @ H - data), ord='fro') / \
                    (partition_size * (K_cv - 1))
                error_test[k-1, k_fold, rng_iteration] = \
                    np.linalg.norm(M_test * (W @ H - data), ord='fro') / \
                    partition_size
                #toc = time.perf_counter()
                #print(f'Frobenius norm took {toc-tic} seconds.')

                # This runs.
                #W, H, info = NMF_HALS().run(data.T, k)

        # Save the errors:

    # Serialize them before saving, and remember to deserialize them after.
    print(f'Saving CV errors for animal {animal_model_id}, lc {learning_condition_id}')
    df = pd.DataFrame({
        'K_max': [K_max],
        'K_cv': [K_cv],
        'rng_max_iters': [rng_max_iters],
        'dim_order': 'K_max, K_cv, rng_max_iters'
    })
    df.to_hdf(str(filename), key='attributes', mode='w')
    df = pd.DataFrame(error_train.reshape(-1, 1))
    df.to_hdf(str(filename), key='error_train')
    #df = pd.DataFrame(error_train_rm.reshape(-1, 1))
    #df.to_hdf(str(filename), key='error_train_rm')
    df = pd.DataFrame(error_test.reshape(-1, 1))
    df.to_hdf(str(filename), key='error_test')
    #df = pd.DataFrame(error_test_rm.reshape(-1, 1))
    #df.to_hdf(str(filename), key='error_test_rm')

    ## Plot the errors to get the picture:
    #fig = plt.figure()
    #plt.plot(error_train.T, color='C0', alpha=0.2)
    #plt.plot(error_train.T, color='C1', alpha=0.2)
    #plt.plot(error_train.T.mean(axis=1), color='C0')
    #plt.plot(error_train.T.mean(axis=1), color='C1')
    #np.argmin(error_train.mean(axis=0))
    #plt.show()

    return 0

# Clustering with Affinity propagation:
#TODO: need to make one for normal files also (no only NWB ones)!
# make reproducible rng decorator
@with_reproducible_rng
def apclusterk(s, requested_k, prc=0, serial_no=None):
    max_bifurcation_tries = 100
    N = s.shape[0]
    # Add a little noise:
    S = s + (np.finfo(float).eps * s + np.finfo(float).tiny * 100) * np.random.rand(N,N)

    # Find limits:
    dpsim1 = np.sum(S, axis=0).max()
    if dpsim1 == -np.inf:
        print(f'Could not find pmin')
    elif N > 1000:
        for k in range(N):
            S[k, k] = -np.inf

        m = np.max(S, axis=1)
        tmp = m.sum()
        yy = m.min()
        ii = np.argmin(m)
        # SKATA EINAI AFTO:
        tmp = tmp - yy - np.min(m[ii[0] - 1, ii[0] + 1: N])
        #tmp = tmp - yy - min(m([1:ii(1) - 1, ii(1) + 1: N]));
        pmin = dpsim1 - tmp
    else:
        dpsim2 = -np.inf
        for j21 in range(N - 1):
            for j22 in range(j21+1, N):
                tmp = np.sum(np.max(S[:, [j21, j22]], axis=1))
                if tmp > dpsim2:
                    dpsim2=tmp
                    k21=j21
                    k22=j22
        pmin = dpsim1 - dpsim2

    for k in range(N):
        S[k, k] = -np.inf

    pmax = S.max().max()
    highpref = pmax
    highk = N
    lowpref = pmin
    lowk = 1

    for k in range(N):
        S[k, k] = 0

    print(f'Bounds: lowp={lowpref}  highp={highpref}  lowk={lowk}  highk={highk}\n')
    print(f'Attempting to improve lower bound:\n')

    # Run AP several times to find better lower bound:
    ap = AP()
    ap.affinity = 'precomputed'
    ap.max_iter = 2000
    ap.convergence_iter = 200
    ap.damping = 0.9
    ap.verbose = True

    i = -4
    dn = False
    while not dn:
        tmppref = highpref - 10 ** i * (highpref - lowpref)
        print(f'  Trying p={tmppref}\n')
        ap.preference = tmppref
        ap_result = ap.fit(S)
        idx = ap_result.labels_
        # Maximum cluster no returned by AP:
        tmpk = idx.max() + 1
        if tmpk <= requested_k:
            dn = True
        elif i == -1:
            tmpk = lowk
            tmppref = lowpref
            dn = True
        else:
            i = i + 1

    # Use bisection method to find k:
    if abs(tmpk - requested_k) / requested_k * 100 > prc:
        idx = None
        print('Applying bisection method:\n')
        lowk = tmpk
        lowpref = tmppref
        ntries = 0
        while (abs(tmpk - requested_k) / requested_k * 100 > prc) and (ntries < max_bifurcation_tries):
            print(f'  lowp={lowpref}  highp={highpref}   lowk={lowk}  highk={highk}\n')
            tmppref = 0.5 * highpref + 0.5 * lowpref
            ap.preference = tmppref
            ap_result = ap.fit(S)
            idx = ap_result.labels_
            tmpk = idx.max() + 1

            if requested_k > tmpk:
                lowpref=tmppref
                lowk=tmpk
            else:
                highpref=tmppref
                highk=tmpk
            ntries = ntries + 1

    pref = tmppref
    print(f'Found {tmpk} clusters using a preference of {pref}\n')
    return idx

@with_reproducible_rng
def get_random_subset(stim_cells_list, max_stim_size=50, **kwargs):
    # Kwargs must contain serial_no for reproducible rng.
    new_stim_cells = []
    # Max no of trials is 50, comparable for all cluster configs.
    for cluster, anatomical_cluster in enumerate(stim_cells_list):
        new_stim_cells.append([])
        for trial in range(int(np.ceil(50/len(stim_cells_list)))):
            rnd_idx = np.random.permutation(anatomical_cluster.size)
            new_stim_cells[cluster].append(
                np.sort(anatomical_cluster[rnd_idx[:max_stim_size]], axis=None)
            )

    return new_stim_cells


def compare_dend_params(NWBarray_of_arrays, dataset_names, **kwargs):
    '''
    Compares NWBarrays with qualitatively different runs.
    Just a helper function.
    :param NWBarray_of_arrays: This contains the different datasets.
    :return:
    '''
    trial_len = get_acquisition_parameters(
        input_NWBfile=NWBarray_of_arrays[0][0],
        requested_parameters=['trial_len']
    )
    delay_range = (20, int(trial_len / 50))
    all_range = (0, int(trial_len / 50))

    # To properly visualize the simulations corresponding to each dendritic
    # configuration, get the valid (having PA) trials and flag them with a
    # unique int for each dend configuration. So this is only for plotting.
    k_labels_arrays = [
        [i+1] * get_nwb_list_valid_ntrials(NWBarray_of_arrays[i])
        for i in range(len(NWBarray_of_arrays))
    ]
    K_labels = np.array(list(chain(*k_labels_arrays)))
    NWBfile_array = list(chain(*NWBarray_of_arrays))

    '''
    #The old way (this is what the function does for more than 2 NWBs):
    K_labels_2 = np.array(list(chain(
        *[ [1] * get_nwb_list_valid_ntrials(NWBarray_1), \
           [2] * get_nwb_list_valid_ntrials(NWBarray_2)] \
        )))
    NWBfile_array_2 = list(chain(*[
        NWBarray_1,
        NWBarray_2
    ]))
    '''
    K_star_array = []
    for i in range(len(NWBarray_of_arrays)):
        K_star, * _ = determine_number_of_clusters(
            NWBfile_array=NWBarray_of_arrays[i],
            max_clusters=20,
            custom_range=delay_range
        )
        K_star_array.append(K_star)

    label_tags = [
        f'{name}: K* = {k_star}'
        for k_star, name in zip(K_star_array, dataset_names)
    ]

    plot_3d = kwargs.get('plot_3d', False)
    plot_2d = not plot_3d
    plot_axis = kwargs.get('plot_axis', None)

    if plot_axis is None:
        fig = plt.figure()
        if plot_2d:
            plot_axis = fig.add_subplot(111)
        else:
            plot_axis = fig.add_subplot(111, projection='3d')

    pcaL2(
        NWBfile_array=NWBfile_array,
        custom_range=delay_range,
        klabels=K_labels,
        smooth=True,
        plot_2d=plot_2d,
        plot_3d=plot_3d,
        plot_stim_color=True,
        plot_axes=plot_axis,
        legend_labels=label_tags
    )

def query_and_add_pa_column(dataframe, **kwargs):
    '''
    Queries the dataframe NWB objects and calculates the PA for each one (
    supports multiple runs on a single NWB file).
    Saves the results on a new column on the dataframe named 'PA'.
    ATTENTION: kwargs should be only the dataframe query terms! So the kwargs is
    the equivalent of the 'filter' dictionary for the dataframe!
    '''
    #TODO: implement something to filter out nonexistend keys/columns.
    #df_keys = set(df.columns.values.astype(str))
    #query_keys = set(filter_d.keys())
    ## Remove nonexistent query keys:
    #query_keys - df_keys

    #dataframe.loc[(dataframe[list(kwargs)] == pd.Series(kwargs)).all(axis=1)]
    nwb_index = (dataframe[list(kwargs)] == pd.Series(kwargs)).all(axis=1)
    # this should be a single entry!
    if nwb_index.values.astype(int).nonzero()[0].size > 1:
        try:
            raise ValueError
        except ValueError:
            print('Multiple dataframe rows correspond to this query! PA '
                  'values will be wrong! Exiting...')
    try:
        df_index = nwb_index.values.astype(int).nonzero()[0][0]
        # Get the corresponding NWB file:
        NWBfile_array = dataframe[
            nwb_index
        ]['NWBfile'].values.tolist()
        if NWBfile_array[0] is not None:
            # Get NWB's trials PA:
            trials_pa = list(nwb_iter(NWBfile_array[0].trials[
                                                   'persistent_activity']))
            # Save back to dataframe the PA percentage:
            dataframe.at[df_index, 'PA'] = np.array(trials_pa).astype(int).mean()
    except Exception:
        print(f'Parameters were not found on dataframe! \n \t {kwargs}')

def query_and_add_attractors_len_column(dataframe, **kwargs):
    '''
    This should group the NWB rows by their CP value, so same CP belong to
    the same group. Then computes the number of attractors for each group.
    Saves the results on a new column on the dataframe named 'attractors_len'.
    This way reusability is maximised: if in need to plot the result just
    query for the specific CP group and get all the NWB files return as a
    list and feed them to the plotting function.

    ATTENTION: kwargs should be only the dataframe query terms! So the kwargs is
    the equivalent of the 'filter' dictionary for the dataframe!
    '''
    #TODO: implement something to filter out nonexistend keys/columns.

    # Get the index of the NWB files that equal the given parameters:
    nwb_index = (dataframe[list(kwargs)] == pd.Series(kwargs)).all(axis=1)
    try:
        df_index = nwb_index.values.astype(int).nonzero()[0]

        # If entries in the dataframe exist:
        if df_index.size == 1:
            # Get the corresponding NWB file:
            NWBfile = dataframe[
                nwb_index
            ]['NWBfile'].values[0]

            try:
                # Calculate for the grouped NWB files their number of attractors:
                trial_len = get_acquisition_parameters(
                    input_NWBfile=NWBfile,
                    requested_parameters=['trial_len']
                )
                delay_range = (20, int(trial_len / 50))

                K_star, * _ = determine_number_of_clusters(
                    NWBfile_array=[NWBfile],
                    max_clusters=20,
                    custom_range=delay_range
                )

                # Now Write the number of attractors on 'attractors_len' on the dataframe.
                dataframe.loc[df_index, 'attractors_len'] = K_star
            except Exception:
                print("Skipping NWBfile with no valid trials.")
    except Exception:
        print(f'Parameters were not found on dataframe! \n \t {kwargs}')


def query_and_add_sparsness_column(dataframe, **kwargs):
    '''
    ATTENTION: kwargs should be only the dataframe query terms! So the kwargs is
    the equivalent of the 'filter' dictionary for the dataframe!
    '''
    #TODO: implement something to filter out nonexistend keys/columns.

    # Get the index of the NWB files that equal the given parameters:
    nwb_index = (dataframe[list(kwargs)] == pd.Series(kwargs)).all(axis=1)
    try:
        df_index = nwb_index.values.astype(int).nonzero()[0]

        # If entries in the dataframe exist:
        if df_index.size == 1:
            # Get the corresponding NWB file:
            NWBfile = dataframe[
                nwb_index
            ]['NWBfile'].values[0]

            overall_sparsness = -1.0
            try:
                # Calculate for the grouped NWB files their number of attractors:
                trial_len = get_acquisition_parameters(
                    input_NWBfile=NWBfile,
                    requested_parameters=['trial_len']
                )
                delay_range = (20, int(trial_len / 50))

                overall_sparsness = sparsness(
                    NWBfile=NWBfile,
                    custom_range=delay_range
                )

                # Now Write the number of attractors on 'attractors_len' on the dataframe.
                dataframe.loc[df_index, 'sparsness'] = overall_sparsness
            except Exception as e:
                print(f"Skipping NWBfile with no valid trials. ({str(e)})")
    except Exception:
        print(f'Parameters were not found on dataframe! \n \t {kwargs}')

if __name__ == "__main__":

    print('success!')
