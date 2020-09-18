# %% Check the plots
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
from scipy.stats import invgamma, lognorm
from pathlib import Path
import pandas as pd
import network_tools as nt
from itertools import chain
from load_synaptic_patterns import load_synaptic_patterns
import analysis_tools as analysis
from functools import partial, wraps

# ===%% Pycharm debug: %%===
import pydevd_pycharm
sys.path.append("pydevd-pycharm.egg")
DEBUG = False
if DEBUG:
    pydevd_pycharm.settrace(
        '79.167.87.207',
        port=12345,
        stdoutToServer=True,
        stderrToServer=True
    )
# ===%% -------------- %%===

def reproducible_weights_rng(class_method):
    '''
    Reproducible weights matrix realization
    :param class_method:
    :return:
    '''
    @wraps(class_method)
    def reset_rng(*args, **kwargs):
        # this translates to self.serial_no ar runtime
        np.random.seed(kwargs['weights_realization'])
        print(f'{class_method.__name__} reseeds the RNG.')
        return class_method(*args, **kwargs)
    return reset_rng

def reproducible_learning_condition_rng(class_method):
    '''
    Reproducible weights matrix realization
    :param class_method:
    :return:
    '''
    @wraps(class_method)
    def reset_rng(*args, **kwargs):
        # this translates to self.serial_no ar runtime
        np.random.seed(kwargs['learning_condition'])
        print(f'{class_method.__name__} reseeds the RNG.')
        return class_method(*args, **kwargs)
    return reset_rng

def get_connectivity_matrix_pc():
    # load connectivity matrix:
    # Load original network dataset:
    export_path = Path('/home/cluster/stefanos/Documents/GitHub'
                       '/prefrontal_analysis/subset_files/')
    filename_prefix = ''
    configuration_alias = 'structured'
    serial_no = 1
    filename_postfix = ''
    filename = export_path.joinpath(
        f'{filename_prefix}{configuration_alias}_network_SN'
        f'{serial_no}{filename_postfix}.hdf5'
    )
    blah = pd.read_hdf(filename, key='attributes')
    serial_no = 1
    connectivity_mat = pd.read_hdf(filename, key='connectivity_mat').values
    # weights_mat = pd.read_hdf(filename, key='weights_mat').values
    return connectivity_mat[:250, :250]


@reproducible_weights_rng
def create_weights(**kwargs):
    # THis is optional and more of a verbose error:
    weights_realization = kwargs.get('weights_realization', None)
    if not weights_realization:
        raise ValueError('You have not provided weights_realization for RNG '
                     'reproducibility!')

    max_weight = 5 #mV

    connectivity_mat_pc = get_connectivity_matrix_pc()
    tmpnet = nt.Network(serial_no=1, pc_no=250, pv_no=83)
    ncn = tmpnet.common_neighbors(connectivity_mat_pc)

    b = 1.0
    a_step = 10.0 / (ncn.max() + 1.0)
    invg_array = []
    for i in range(int(ncn.max() + 1)):
        print(f"alpha = {10.0 - (i * a_step)}")
        #invg_array.append(invgamma(2.0 - (i * a_step), scale=b))
        # Replace with lognormal and check resulting overall distribution:
        invg_array.append(lognorm(10.0 - (i * a_step)))
    '''
    fig, ax = plt.subplots(1, 1)
    for k in range(int(ncn.max() + 1)):
        x = np.linspace(0, max_weight, num=1000)
        ax.plot(x, invg_array[k].pdf(x), 'k-', lw=1)
    fig.savefig(f"WR{weights_realization}_validation_alldists.png")
    '''

    weights_mat_pc = np.zeros(ncn.shape)
    for i in range(250):
        for j in range(250):
            if connectivity_mat_pc[i,j]:
                tmp = invg_array[int(ncn[i, j])].rvs()
                while tmp > max_weight:
                    tmp = invg_array[int(ncn[i, j])].rvs()
                weights_mat_pc[i, j] = tmp


    #plot the resulting distribution as debugging:
    # Check similarity with Brunel's data (figure 3D).
    ncn_bins = ncn.astype(int)
    #TODO: You are not checking the weights histo correctly! you need only
    # unique weights (pairs)!
    # YET: the reciprocal pairs have non symmetric weights! Which is correct,
    # but I must note that.
    hist, *_ = np.histogram(np.triu(weights_mat_pc,k=1),
                            np.arange(0.1, max_weight, 0.1))
    #hist[0] = hist[0] - (250*250/2 + 250)
    #fig, ax = plt.subplots(1, 1)
    #ax.plot(np.arange(0, hist.size), hist / hist.sum(), 'b-', lw=1)
    #fig.savefig(f"WR{weights_realization}_Brunel_Fig3d_updated.png")
    blah = np.where(np.triu(weights_mat_pc,k=1))
    print(f" Mean weight {weights_mat_pc[blah].mean()}")

    if False:
        #Plot all distributions from where we sample:
        fig, ax = plt.subplots(1, 1)
        for k in range(int(ncn.max() + 1)):
            x = np.linspace(0, max_weight, num=1000)
            ax.plot(x, invg_array[k].pdf(x), 'k-', lw=1)
        fig.savefig(f"WR{weights_realization}_validation_alldists.png")

        # validate the weights with respect to common neighbors:
        # This can be bypassed, by adding 1 and converting to int:
        # ncn_bins = np.digitize(ncn, np.arange(0,ncn.max()+1,1))
        ncn_bins = ncn.astype(int)
        max_somatic_depol = 10

        for k in range(int(ncn.max() + 1)):
            tmparr = []
            for i, idx in enumerate(tuple(ncn_bins)):
                tmparr.append(weights_mat_pc[i, idx == k])
            weights_valid = np.array(list(chain(*tmparr)))
            hist, *_ = np.histogram(weights_valid,
                                    np.arange(0, max_somatic_depol, 0.01))
            try:
                x = np.linspace(0, max_somatic_depol, num=1000)
                fig, ax = plt.subplots(1, 1)
                ax.plot(x, invg_array[k].pdf(x), 'k-', lw=1)
                ax.plot(x[:999], hist / hist.sum(), 'b-', lw=1)
                fig.savefig(f"WR{weights_realization}_validation_{k}.png")
            except Exception:
                # suck it up
                pass

        # Check similarity with Brunel's data (figure 3D).
        hist, *_ = np.histogram(weights_mat_pc,
                                np.arange(0, max_somatic_depol, 0.01))
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.arange(0, hist.size), hist / hist.sum(), 'b-', lw=1)
        fig.savefig(f"Brunel_Fig3d.png")

    # Here you should check for reciprocal connections and increase the weights:
    # Kawaguchi & Kubota, elife, 2015 say that the weights are not significantly
    # larger in the reciprocal case. Also even if larger weights exist they were
    # not symmetrical. So I am not increasing any weights.
    return weights_mat_pc


@reproducible_learning_condition_rng
def generate_patterns(
        export_path=Path.cwd(), weights_mat_pc=None,
        inhomogeneous=False, export2file=True, **kwargs
):
    # THis is optional and more of a verbose error:
    learning_condition = kwargs.get('learning_condition', None)
    weights_realization = kwargs.get('weights_realization', None)
    if not kwargs.get('learning_condition', None):
        raise ValueError('You have not provided serial_no for RNG '
                     'reproducibility!')

    synaptic_inhomogeneity = kwargs.get('synaptic_inhomogeneity', None)
    dendritic_inhomogeneity = kwargs.get('dendritic_inhomogeneity', None)
    # Read NEURON pattern data and classify it in a dataframe:
    patterns_path = Path(r'/home/cluster/stefanos/Documents/GitHub/prefrontal-micro'
                         r'/experiment/network/synaptic_patterns')

    dendlen = kwargs['dendlen']
    dendno = kwargs['dendno']
    nseg = int(dendlen/30.0)
    iters = kwargs.get('iters',100000)
    max_depol_val = kwargs.get('max_depol_val',5)
    min_w = kwargs.get('min_w', -10)
    max_w = kwargs.get('max_w',10)
    # The synapses are correlated ONLY in the synaptic homogeneous case:
    if synaptic_inhomogeneity is 0:
        postfix='_corr'
        min_w = 0
        max_w = 3
    else:
        postfix = ''
    patterns_df = load_synaptic_patterns(
        patterns_path,
        new_methodology=True,
        dendlen=dendlen,
        dendno=dendno,
        nseg=nseg,
        nsyn=5,
        max_depol_val=max_depol_val,
        min_w=min_w,
        max_w=max_w,
        iters=iters,
        postfix=postfix
    )

    #TODO: I need to generate different filters, one for each inhomogeneity
    # case.
    # Label patterns based on their synaptic inhomogeneity:
    case_w_idx = np.array([
        len(np.where(i)[0])
        for i in patterns_df['W']
    ])
    syn_inhomo_idx = [[],[],[]]
    syn_inhomo_idx[0] = np.full(patterns_df.shape[0],True,dtype=bool)
    # No easier way?
    syn_inhomo_idx[1] = \
        np.where(
            case_w_idx == 3,
            np.ones((1,case_w_idx.size),dtype=bool),
            np.zeros((1,case_w_idx.size),dtype=bool),
            )[0]
    syn_inhomo_idx[2] = \
        np.where(
            case_w_idx == 1,
            np.ones((1,case_w_idx.size),dtype=bool),
            np.zeros((1,case_w_idx.size),dtype=bool),
            )[0]

    # Label patterns based on their dendritic inhomogeneity:
    case_d_idx = np.array([
        len(np.unique(i))
        for i in patterns_df['D']
    ])
    dend_inhomo_idx = [[],[],[]]
    # No easier way?
    dend_inhomo_idx[0] = \
        np.where(
            case_d_idx == 3,
            np.ones((1,case_d_idx.size),dtype=bool),
            np.zeros((1,case_d_idx.size),dtype=bool),
            )[0]
    dend_inhomo_idx[1] = \
        np.where(
            case_d_idx == 2,
            np.ones((1,case_d_idx.size),dtype=bool),
            np.zeros((1,case_d_idx.size),dtype=bool),
            )[0]
    dend_inhomo_idx[2] = \
        np.where(
            case_d_idx == 1,
            np.ones((1,case_d_idx.size),dtype=bool),
            np.zeros((1,case_d_idx.size),dtype=bool),
            )[0]


    # Change to inhomogeneous patterns if not found!
    if (synaptic_inhomogeneity is not None) and \
        (dendritic_inhomogeneity is not None):

        # Include the pattern labels into the dataframe:
        patterns_df['synaptic_inhomogeneity'] = \
            syn_inhomo_idx[synaptic_inhomogeneity]
        patterns_df['dendritic_inhomogeneity'] = \
            dend_inhomo_idx[dendritic_inhomogeneity]

        filters = {
            'synaptic_inhomogeneity': True,
            'dendritic_inhomogeneity': True,
        }

        # Provide the boilerplate to select class preset (e.g. distal)
        case_df = patterns_df.loc[(patterns_df[list(filters)] == pd.Series(
            filters)).all(axis=1)]
    else:
        inhomogeneous = True
        case_df = patterns_df

    # if is a case we don't want, return None:
    if case_df.shape[0] == 0:
        return None


    # OTHER DEBUG CODE:
    if False:
        #TODO: make sure that it works!
        W = case_df['W']
        blah = []
        for i in range(W.shape[0]):
            if np.where(W[i])[0].size > 3:
                blah.append(i)
        #histo, *_ = np.histogram(blah, np.arange(0, 7))
        #histo = histo / histo.sum()
        #Keep only the synaptic patterns with 4 or 5 synapses in total:
        case_df2 = case_df.iloc[blah]
        histo, *_ = np.histogram(case_df2['somatic_depolarization_bin'].values,
                                 np.arange(0, 51))
        histo = histo / histo.sum()

    # Generate NEURON arrays for each PC cell containing PID and W of each synapse:
    # This file will be addressible/identifiable by the filters above and used
    # along with the importNetworkParams files. Should change when the filters
    # change. Filters should exist as keywords in NEURON and saved in simulations
    # metadata.
    # I also need RNG seed here, as a SN of the pattern file.

    # If I provide a weights matrix for PCs use it. Else, choose a medial
    # value of somatic depolarization and assign 'uniform' patterns that
    # comply to the filters, yet are homegeneous in their depolarization.
    depol_bins = case_df['somatic_depolarization_bin'].values

    # TODO: compute the histogram to be sure that there are enough
    #  patterns for each depolarization bin:
    histo, *_ = np.histogram(depol_bins, np.arange(0,50,1))
    histo =histo / histo.sum()

    connectivity_mat_pc = get_connectivity_matrix_pc()
    depol_bin_mat_pc = np.zeros(connectivity_mat_pc.shape)
    if weights_mat_pc is None:
        real_weights = False
        #TODO: this depol bin should be GLOBAL to the df, not after the
        # filters! So the same patterns file, generates consistent no-real
        # weights across realizations.
        depol_bins_global = patterns_df['somatic_depolarization_bin'].values
        # I decide the average depolarization that I want by the average
        # depolarization weight that I get from the weights distribution.
        average_weight = 2.5
        '''
        histo, *_ = np.histogram(
            depol_bins_global,
            np.arange(0, int(depol_bins_global.max()) + 1, 1)
        )
        depol_bin = np.argmax(histo) + 1
        '''

        # create a depol matrix, with 'bin' values:
        depol_bin_mat_pc[connectivity_mat_pc] = np.around(average_weight*10)

    else:
        real_weights = True
        # Scale down the weights as per the depolarization maximum:
        scaled_weights = (weights_mat_pc / weights_mat_pc.max()) * \
        depol_bins.max()
        #scaled_weights = connectivity_mat_pc.astype(int) * scaled_weights
        depol_bin_mat_pc = np.ceil(scaled_weights)

    # sample the patterns:
    pair_d = {}
    sampled_indices = []
    # For the dataframe, if you choose to export it:
    SRCs=[]
    TRGs=[]
    for m in range(250):
        for n in range(250):
            if connectivity_mat_pc[m, n]:
                query_bin = int(depol_bin_mat_pc[m, n])
                # sample the depol values:
                subdf_idx, *_ = np.where(depol_bins == query_bin)
                if subdf_idx.size == 0:
                    # Get the closest one, not the min/max one!!
                    bin_diff = np.absolute(depol_bins - query_bin)
                    query_bin = depol_bins[np.argmin(bin_diff)]
                    print(f'\t replaced with {query_bin}')
                    subdf_idx, *_ = np.where(depol_bins == query_bin)
                max_ind = subdf_idx.size -1
                if max_ind == 0:
                    print(f"Single pattern meets criteria! {m},{n}, "
                          f"query_bin: {query_bin}")
                    rnd_idx = 0
                else:
                    # THis is equivalent to a random permutation
                    rnd_idx = np.random.randint(0, max_ind)
                case_idx = case_df.index.values
                sampled_indices.append(case_idx[subdf_idx[rnd_idx]])
                pair_PID, pair_W, pair_D,soma_depol = case_df.loc[
                    case_idx[subdf_idx[rnd_idx]]
                ][['PID','W', 'D', 'somatic_depolarization']]
                # Do a check that no PIDs are border cases 0/1:
                #THIS IS DONE ON load_synaptic_patterns():
                pair_PID[pair_PID==0] = 0.1
                pair_PID[pair_PID==1] = 0.99
                pair_d[(m,n)] = (pair_PID, pair_W, pair_D, soma_depol)
                # for the dataframe:
                SRCs.append(m)
                TRGs.append(n)


        # create a depol matrix, with 'bin' values:

        #depol_bin_mat_pc[connectivity_mat_pc] = 1

    if False: #if DEBUG:
        nseg_step = 1/nseg
        #DEBUG: mine the statistics of the dend attributes to see which of them
        # produces what no of atractors.
        # 1) distal proximal (real, not what I report).
        dendpatt_df = case_df.loc[sampled_indices]
        synpatt_arr = np.zeros((dendpatt_df.shape[0],nseg*dendno), dtype=float)
        for index in range(dendpatt_df.shape[0]):
            entry = dendpatt_df.iloc[index]
            #sort_idx = np.argsort(entry.PID)
            #W = entry.W[sort_idx]
            #D = entry.D[sort_idx]
            syn_seg = np.digitize(entry.PID + entry.D, np.arange(0, dendno +
                                                                 nseg_step,
                                                                 nseg_step)) - 1
            for seg_id in range(nseg * dendno):
                synpatt_arr[index,seg_id] = entry.W[np.where(seg_id ==
                                                             syn_seg)].sum()

        syn_seg = np.zeros((dendpatt_df.shape[0],nseg), dtype=float)
        for index in range(dendpatt_df.shape[0]):
            entry = dendpatt_df.iloc[index]
            syn_seg[index,:] = np.digitize(
                entry.PID + entry.D,
                np.arange(0, dendno + nseg_step, nseg_step)
            ) - 1

        histo_array = np.empty((synpatt_arr.shape[0],nseg*dendno))
        histo_array[:] = np.nan
        for i in range(synpatt_arr.shape[0]):
            tmp_histo = \
                np.histogram(
                    syn_seg[i,:],
                    bins=np.arange(0, nseg*dendno+1, 1)
                )[0]
            histo_array[i,:] = tmp_histo

        syn_perc = np.full((5+1,nseg*dendno),0)
        for seg in range(nseg*dendno):
            for syn in range(5+1):
                syn_perc[syn, seg] = np.sum(histo_array[:, seg] == syn)

        syn_perc_cumsum = np.cumsum(syn_perc, axis=0)


        # 5 segment case:
        labels = ['1', '2', '3', '4', '5','6','7','8','9','10','11','12',
                  '13','14','15']
        width = 0.35  # the width of the bars: can also be len(x) sequence

        fig, ax = plt.subplots()

        ax.bar(labels, syn_perc[0, :].tolist(), width, label=f'syn#0')
        for i in range(1,5+1):
            ax.bar(labels, syn_perc[i,:].tolist(), width, label=f'syn#{i}',
                   bottom=syn_perc_cumsum[i-1,:].tolist())

        ax.set_ylabel('# synapses')
        ax.set_title('Synapse distribution per segment')
        ax.legend()
        plt.savefig(f'syn_dist_rev_5.png')

        # plot in 3d the synapses in each dend compartment:
        patterns_covar = np.cov(synpatt_arr, rowvar=False)
        fig = plt.figure()
        plot_axes = fig.add_subplot(111)
        plt.matshow(patterns_covar)
        cb = plt.colorbar()
        plt.savefig(f"splogn_rev_6_corr.png")

        patt_average = synpatt_arr.mean(axis=0)
        fig = plt.figure()
        plot_axes = fig.add_subplot(111)
        plt.plot(patt_average)
        plt.savefig(f"splogn_rev_6_average.png")


        new_patterns_df = load_synaptic_patterns(
            patterns_path,
            dendlen=dendlen,
            dendno=dendno,
            nseg=nseg,
            nsyn=5,
            max_depol_val=max_depol_val,
            min_w=min_w,
            max_w=max_w,
            iters=iters,
            new_methodology=True
        )
        new_dendpatt_df = new_patterns_df.loc[sampled_indices]

        # 2) Is there any dendritic clustering, and if yes, how much

        # 3) What happens in the network.


    if export2file:
        #TODO: There is no serial number here: only learning condition i.e.
        # different reshuffling of the synapses. But I use serial_no, in order to
        # reuse the RNG wrapper. The SN is defined from the network I read/import.
        #CORRECTED I consideer the RW case as the only one
        if inhomogeneous:
            filename = export_path.joinpath(
                f"sp_inhomogeneous_dendno_{dendno}_dendlen_{dendlen}_"
                f"wr_{int(weights_realization)}_LC{learning_condition}_lognorm.hoc"
            )
        else:
            # If no weights mat provided, use the uniform case: normalized
            # postsynaptic depolarizations for each pair.
            if weights_mat_pc is None:
                filename = export_path.joinpath(
                    f"sp_syninh_"
                    f"{int(synaptic_inhomogeneity)}_dendinh"
                    f"_{int(dendritic_inhomogeneity)}_"
                    f"dendno_{dendno}_dendlen_{dendlen}_"
                    f"LC{learning_condition}_singleweight_rev1.hoc"
                )
            else:
                filename = export_path.joinpath(
                    f"sp_synloc_{synapse_location}_syncl_"
                    f"{int(synapse_clustering)}_dendcl_{int(dendritic_clustering)}_"
                    f"dendno_{dendno}_dendlen_{dendlen}_"
                    f"wr_{int(weights_realization)}_LC"
                    f"{learning_condition}_lognorm_rev6.hoc"
                )

        with open(filename, 'w') as f:
            f.write(
                f'// This HOC file was generated with generate_patterns python '
                f'module.\n')
            f.write(f'objref synaptic_patterns[250][250]\n')
            f.write(f'objref synaptic_weights[250][250]\n')
            f.write(f'objref synaptic_dends[250][250]\n')

            #in each obj location that conn mat has a connection, create a vector:
            for afferent in range(250):
                for efferent in range(250):
                    if connectivity_mat_pc[afferent, efferent]:
                        f.write(
                            f'synaptic_patterns[{afferent}][{efferent}]=new Vector'
                            f'(5)\n'
                        )
                        f.write(
                            f'synaptic_weights[{afferent}][{efferent}]=new Vector'
                            f'(5)\n'
                        )
                        f.write(
                            f'synaptic_dends[{afferent}][{efferent}]=new Vector'
                            f'(5)\n'
                        )

                        PID, W, D, *_ = pair_d[(afferent,efferent)]
                        for i, (pid, w, d) in enumerate(zip(PID,W, D)):
                            f.write(
                                f'synaptic_patterns[{afferent}][{efferent}].x[{i}]'
                                f'={pid}\n')
                            f.write(
                                f'synaptic_weights[{afferent}][{efferent}].x[{i}]'
                                f'={w}\n')
                            f.write(
                                f'synaptic_dends[{afferent}][{efferent}].x[{i}]'
                                f'={d}\n')

            # Network connectivity:
            f.write('//EOF\n')
    else:
        # Combine into a dataframe:
        export_df = case_df.loc[sampled_indices]
        export_df['SRC'] = SRCs
        export_df['TRG'] = TRGs
        return export_df

    # Make changes in NEURON to run the new simulations:
    print("Pronto!")

if __name__ == '__main__':
    # test weights matrix RNG:
    #TODO: let me choose the different levels of projection clustering/dendritic
    # inhomogeneity. So clustering is determined by that amount. Also
    # proximal/distal will not be a parameter here, since I want max
    # inhomogeneity each time (later I can constrain some of the more
    # inhomogenous runs only on the proximal/distal part and see what happens).

    if True:
        # This is the uniform case (No weight matrix) with the inhomogeneity
        # slider parameter only:
        for learning_condition in range(1, 6):
            gp = partial(
                generate_patterns,
                weights_mat_pc=None,
                learning_condition=learning_condition,
                min_w=-10,
                max_w=10,
                iters=100000,
                dendlen=150,
                dendno=3,
                max_depol_val=5,
            )

            params = {
                # These are the inhomogeneity sliders. In this example span
                # from 0 to 2.
                'synaptic_inhomogeneity': [0,1,2],
                'dendritic_inhomogeneity': [0,1,2],
            }

            analysis.run_for_all_parameters(
                gp,
                **{'auto_param_array': params}
            )

    elif False:
        # This is the uniform case (No weight matrix):
        for learning_condition in range(2, 3):
            gp = partial(
                generate_patterns,
                weights_mat_pc=None,
                learning_condition=learning_condition,
                min_w=-10,
                max_w=10,
                iters=100000,
                dendlen=30,
                dendno=1,
                max_depol_val=5,
                #postfix='_corr'
            )

            params = {
                'synapse_location': ['distal'],
                'synapse_clustering': [True],
                'dendritic_clustering': [True],
                'dendlen': [150],
                'dendno': [3]
            }

            analysis.run_for_all_parameters(
                gp,
                **{'auto_param_array': params}
            )

    else:
        #I will keep the same weights matrix instantiation as this is a different
        # level of inhomogeneity
        # I will keep this 1 for now:
        for weights_realization in range(1, 2):
            # This is the synaptic pattern learning condition.
            for learning_condition in range(1, 2):

                gp = partial(
                    generate_patterns,
                    weights_mat_pc=create_weights(
                        weights_realization=weights_realization
                    ),
                    learning_condition=learning_condition,
                    weights_realization = weights_realization,
                    min_w=-10,
                    max_w=10,
                    iters=100000,
                    dendlen=30,
                    dendno=1,
                    max_depol_val=5,
                    export2file=False
                )

                params = {
                    'synapse_location': ['proximal', 'distal'],
                    'synapse_clustering': [True, False],
                    'dendritic_clustering': [True, False],
                    'dendlen': [150],
                    'dendno': [2,3]
                }
                '''
                # Inhomogeneous patterns!
                params = {
                    'dendlen': [30, 150],
                    'dendno': [2,3]
                }
                '''

                analysis.run_for_all_parameters(
                    gp,
                    **{'auto_param_array': params}
                )
