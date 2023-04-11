import sys
import os
import numpy as np
from scipy.stats import norm
import datetime
from tqdm import tqdm

from ..topology.graph import Graph
from ..analysis.distance import check_neg, check_bound, check_min_max
from ..main import load_states, set_state_name, load
from ..preparation.epn import LIMS, LIMS_DICT

from matplotlib import pyplot as plt
from matplotlib.figure import figaspect
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.offsetbox import AnchoredText
import matplotlib.colorbar as cbar
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.optimize import curve_fit

from matplotlib import rc
rc('font', size=16)
rc('axes', titlesize=18)
rc('axes', labelsize=18)

ATNF_PARAMS=[
    'JNAME',
    'BNAME',
    'RAJ',
    'DECJ',
    'P0',
    'P1',
    'ASSOC',
    'BINARY',
    'TYPE',
    'P1_I',
    'DM',
    'age',
    'R_Lum',
    'R_Lum14',
    'BSurf',
    'Edot',
    'PMTot',
    'W50',
    'W10'
]

def save_figure(plt, metric, stokes_to_include, function_name, freq_ids_to_include=None, save_to='images/', extension='pdf', dpi=300, state_prefix=None):
    plt.tight_layout()
    if freq_ids_to_include is None:
        outpath = '%s/%s' % (save_to, function_name)
        if state_prefix is None:
            outfile = '%s_%s_%s.%s' % (metric, stokes_to_include, function_name, extension)
        else:
            outfile = '%s_%s.%s' % (state_prefix, function_name, extension)
    else:
        s_freqs = ''
        for f in freq_ids_to_include:
            if s_freqs == '':
                s_freqs += "%d" % f
            else:
                s_freqs += "_%d" % f
        outpath = '%s/%s' % (save_to, function_name)

        if state_prefix is None:
            outfile = '%s_%s_%s_%s.%s' % (metric, stokes_to_include, s_freqs, function_name, extension)
        else:
            outfile = '%s_%s.%s' % (state_prefix, function_name, extension)

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    print ('Saving to %s/%s' % (outpath, outfile))

    plt.savefig('%s/%s' % (outpath, outfile))

def add_at(ax, t, loc=2):
    fp = dict(size=13, alpha=0.7, )
    _at = AnchoredText(t, loc=loc, prop=fp)
    ax.add_artist(_at)
    return _at

def set_fig_dims(direction, data_arr, spectrum=False):
    if direction == 'horizontal':
        ncols = len(data_arr)
        nrows = 1
    elif direction == 'vertical':
        ncols = 1
        nrows = len(data_arr)

    return ncols, nrows

def set_multi_axes(ax, direction, xticks, xtick_labels, yticks, ytick_labels):
    """Set axes ticks and tick labels

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Array of axes
    direction : str
        General direction onto which append subplots
    xticks : list
        List of ticks for x axis
    xticks_labels : list
        List of tick labels for x axis
    yticks : list
        List of ticks for y axis
    yticks_labels : list
        List of tick labels for y axis

    """
    for i, axi in enumerate(ax):
        if len(xticks) > 0 and len(xtick_labels) > 0:
            if (direction == 'vertical' and i == len(ax)-1) or direction == 'horizontal':
                axi.set_xlabel(r'$\phi$')
                axi.set_xticks(xticks)
                axi.set_xticklabels(xtick_labels)
            else:
                plt.setp(axi.get_xticklabels(), visible=False)
        else:
            axi.set_xlabel('X Index')

        if len(yticks) > 0 and len(ytick_labels) > 0:
            if (direction == 'horizontal' and i == 0) or direction == 'vertical':
                axi.set_ylabel('profile index')
            else:
                plt.setp(axi.get_yticklabels(), visible=False)
        else:
            axi.set_ylabel('S/N')

def plot_profile_sequence(sequence_arr,
                          labels=[],
                          direction='vertical',
                          figtype='imshow',
                          scale_factor=1,
                          # color=['#f7f7f7', '#cccccc', '#969696', '#636363', '#252525'],
                          color=['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0'],
                          draw_line=True,
                          draw_background=False,
                          linewidth=1,
                          step=1,
                          figsize=(20,10),
                          loc=4,
                          save_fig=False,
                          fig_name="profile",
                          fig_ext='png',
                          dpi=150):
    ncols, nrows = set_fig_dims(direction, sequence_arr)

    fig, ax = plt.subplots(
        figsize=figsize,
        ncols=ncols,
        nrows=nrows,
        sharex=False if nrows == 1 else True,
        sharey=False if ncols == 1 else True,
    )

    if figtype == 'imshow':
        if ncols > 1 or nrows > 1:
            for i, axi in enumerate(ax):
                axi.imshow(sequence_arr[i], origin='lower')

                axi.set_yticklabels([])
                axi.set_yticks([])
                axi.axes.get_yaxis().set_visible(False)

                if len(labels) > 0:
                    axi.set_title(labels[i])
                #     add_at(axi, labels[i], loc=loc)
        else:
            ax.imshow(sequence_arr[0], origin='lower')

            # ax.set_yticklabels([])
            # ax.set_yticks([])
            # ax.axes.get_yaxis().set_visible(False)


            if len(labels) > 0:
                ax.set_title(labels[0])
            #     add_at(ax, labels[0], loc=loc)

    elif figtype == 'plot':
        if ncols > 1 or nrows > 1:
            for i, axi in enumerate(ax):
                for line in sequence_arr[i]:
                    axi.plot(line + i)

                axi.set_yticklabels([])
                axi.set_yticks([])
                axi.axes.get_yaxis().set_visible(False)

                if len(labels) > 0:
                    add_at(axi, labels[i], loc=loc)
        else:
            # for i, line in list(enumerate(sequence_arr[0])):
            print (sequence_arr[0].shape[0])
            idx = [i for i in range(0, sequence_arr[0].shape[0]*2, 2)]
            for i, line in tqdm(reversed(list(enumerate(sequence_arr[0])))):
                xs = np.linspace(-.5, .5, line.shape[0])
                _line = scale_factor * line + idx[i]

                # Plot curve
                if draw_line:
                    ax.plot(
                        xs,
                        _line,
                        c='black',
                        linewidth=linewidth,
                    )

                # Fill below the curve
                # cmap = plt.cm.viridis
                # normalize = mpl.colors.DivergingNorm(
                #     vmin=_line.min(),
                #     vcenter=_line.min()+0.15,
                #     vmax=_line.max()
                # )
                if draw_background:
                    plt.fill_between(
                        xs,
                        y1=np.min(_line),
                        y2=_line,
                        color=color[i%len(color)],
                        interpolate=True,
                        # alpha=0.5
                        # color=cmap(normalize(_line))
                    )

            # ax.set_yticklabels([])
            # ax.set_yticks([])
            # ax.axes.get_yaxis().set_visible(False)

            if len(labels) > 0:
                add_at(ax, labels[0], loc=loc)

    print ("save_fig", save_fig)
    if save_fig:
        plt.tight_layout()
        print ("%s.%s" % (fig_name, fig_ext))
        plt.savefig("%s.%s" % (fig_name, fig_ext), dpi=dpi)

    # except:
    #     ax.imshow(sequence_arr[0], origin='lower')
    #

def plot_1Ds(sequence_arr,
             labels=[],
             direction='vertical',
             figsize=(20,10),
             loc=4,
             sharey=False,
             save_fig=False,
             fig_name="profile",
             fig_ext='png',
             dpi=150):
    ncols, nrows = set_fig_dims(direction, sequence_arr)

    fig, ax = plt.subplots(
        figsize=figsize,
        ncols=ncols,
        nrows=nrows,
        sharex=False, # if nrows == 1 else True,
        sharey=sharey# if ncols == 1 else True,
    )

    # try:
    if ncols > 1 or nrows > 1:
        for i, axi in enumerate(ax):
            axi.plot(sequence_arr[i])

            if len(labels) > 0:
                add_at(axi, labels[i], loc=loc)
    else:
        ax.plot(sequence_arr[0])

        # ax.set_yticklabels([])
        # ax.set_yticks([])
        # ax.axes.get_yaxis().set_visible(False)

        if len(labels) > 0:
            add_at(ax, labels[0], loc=loc)

    print ("save_fig", save_fig)
    if save_fig:
        plt.tight_layout()
        print ("%s.%s" % (fig_name, fig_ext))
        plt.savefig("%s.%s" % (fig_name, fig_ext), dpi=dpi)

def plot_results(original_image, new_image, save=False, output_filename='output.png', dpi=300):
    fig, ax = plt.subplots(figsize=(10, 10), ncols=2, nrows=1)
    ax[0].imshow(original_image)
    ax[1].imshow(new_image)
    fig.tight_layout()
    fig.show()

    if save:
        plt.savefig(output_filename, dpi=dpi)

def plot_centroid(observation, prop='stokes_I', centroid=None):
    plt.plot(observation.get_property(prop))
    if centroid is None:
        if 'I' in prop:
            plt.axvline(observation.centroid)
        else:
            plt.axvline(observation.get_centroid(prop))
    else:
        plt.axvline(centroid)

def plot_sequence(epn_metadata, population_sequence, mst, min_snr=10, state_prefix=''):
    for j, pulsar in enumerate(population_sequence):
        weight = -1
        if j > 0:
            try:
                weight = mst.loc[
                    (mst['u'].astype(int) == population_sequence[j].index) &
                    (mst['v'].astype(int) == population_sequence[j-1].index),
                    'w'
                ].values[0]
            except IndexError:
                weight = mst.loc[
                    (mst['u'].astype(int) == population_sequence[j-1].index) &
                    (mst['v'].astype(int) == population_sequence[j].index),
                    'w'
                ].values[0]

        i = 0
        z = 0
        for freq in pulsar.observations.keys():
            obs = pulsar.observations[freq]
            if obs.snr >= min_snr or min_snr is None:

                meta = epn_metadata.loc[
                    (epn_metadata['jname'] == pulsar.jname) &
                    (epn_metadata['frequency (MHz)'] == obs.frequency)
                ]

                xs = np.linspace(-.5, .5, obs.stokes_I.shape[0])

                ax = plt.subplot(1,1,1)

                ax.plot(
                    xs,
                    obs.stokes_I + i,
                    label="%.0fMHz, S/N:%d, w=%.2f" % (
                        obs.frequency,
                        obs.snr if not np.isnan(obs.snr) else -1,
                        float(weight)
                    ),
                    c='black',
                    linewidth=0.5
                )

                try:
                    ax.plot(
                        xs,
                        obs.stokes_Q + i,
                        c='blue',
                        linewidth=0.2
                    )

                    ax.plot(
                        xs,
                        obs.stokes_L + i,
                        c='red',
                        linewidth=0.2
                    )
                except TypeError:
                    pass

                try:
                    bname = meta['bname'].values[0]
                    ax.set_title("%s (%s)" % (pulsar.jname, bname) if bname != 'n/a' else pulsar.jname)
                except IndexError:
                    ax.set_title("%s" % (pulsar.jname) if bname != 'n/a' else pulsar.jname)

                i += 2

def sequence_to_pdf(epn_metadata, population_sequence, mst, population_graph_indices, min_snr=10, state_prefix='', folder='images/'):
    print ('Outputting sequence to ' + folder + state_prefix + '_sequence.pdf')

    if not os.path.exists(folder):
        os.makedirs(folder)

    with PdfPages(folder + state_prefix + '_sequence.pdf') as pdf:
        for j, pulsar in enumerate(population_sequence):
            weight = -1
            if j > 0:
                try:
                    weight = mst.loc[
                        (mst['u'].astype(int) == population_graph_indices[population_sequence[j].index]) &
                        (mst['v'].astype(int) == population_graph_indices[population_sequence[j-1].index]),
                        'w'
                    ].values[0]
                except IndexError:
                    weight = mst.loc[
                        (mst['u'].astype(int) == population_graph_indices[population_sequence[j-1].index]) &
                        (mst['v'].astype(int) == population_graph_indices[population_sequence[j].index]),
                        'w'
                    ].values[0]

            i = 0
            for freq in pulsar.observations.keys():
                obs = pulsar.observations[freq]
                if obs.snr >= min_snr or min_snr is None:

                    meta = epn_metadata.loc[
                        (epn_metadata['jname'] == pulsar.jname) &
                        (epn_metadata['frequency (MHz)'] == obs.frequency)
                    ]

        #             colors = [plt.cm.plasma(i) for i in np.linspace(0, 1, len(pulsar.observations)*2)]
                    xs = np.linspace(-.5, .5, obs.stokes_I.shape[0])

                    ax = plt.subplot(1,1,1)

                    ax.plot(
                        xs,
                        obs.stokes_I + i,
                        label="%.0fMHz, S/N:%d, w=%.2f" % (
                            obs.frequency,
                            obs.snr if not np.isnan(obs.snr) else -1,
                            float(weight)
                        ),
                        c='black',
                        linewidth=0.5
                    )

                    try:
                        ax.plot(
                            xs,
                            obs.stokes_Q + i,
                            c='blue',
                            linewidth=0.2
                        )

                        ax.plot(
                            xs,
                            obs.stokes_L + i,
                            c='red',
                            linewidth=0.2
                        )
                    except TypeError:
                        pass

                    try:
                        bname = meta['bname'].values[0]
                        ax.set_title("%s (%s)" % (pulsar.jname, bname) if bname != 'n/a' else pulsar.jname)
                    except IndexError:
                        ax.set_title("%s" % (pulsar.jname))

                    i += 2
            if i > 0:
                handles, labels = ax.get_legend_handles_labels()
                plt.gca().axes.get_yaxis().set_visible(False)
                plt.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 10 if i//2 < 16 else 6})
                plt.tight_layout()
                try:
                    pdf.attach_note("%s (%s)" % (pulsar.jname, bname) if bname != 'n/a' else pulsar.jname)
                except UnboundLocalError:
                    pdf.attach_note("%s" % (pulsar.jname))
                pdf.savefig()
                plt.close()

        # We can also set the file's metadata via the PdfPages object:
        d = pdf.infodict()
        d['Title'] = 'European Pulsar Network database multi-frequency profiles'
        d['Author'] = 'Dany Vohl, Yogesh Maan, Joeri van Leeuwen'
        d['Subject'] = 'EPN multi-frequency pulse profile catalogue'
        d['Keywords'] = 'EPN pulsar profile frequency-evolution'
        d['CreationDate'] = datetime.datetime.today()
        d['ModDate'] = datetime.datetime.today()

def solution_space_pruning(distances, state_prefix='', folder='images/'):

    graph = Graph()
    for u, v, w in distances:
        graph.add_edge(u, v, w)

    graph.spanning_tree(reverse=False)
    mst = graph.as_dataframe()
    sequence_indices = graph.get_longest_path()

    fig, ax = plt.subplots(1,3, figsize=(15,4), sharex=True, sharey=True)

    # Complete graph wedge
    ax[0].set_title('Compete undirected weighted graph wedge', size=10)
    graph.graph_as_dataframe().plot.scatter('u', 'v', marker='s', c='w', ax=ax[0], s=1, cmap='viridis')
    add_at(ax[0],
           '|V|={:,}'.format(graph.graph_as_dataframe()['u'].size),
           loc='lower right')

    # Minimum spanning tree wedge
    ax[1].set_title('Minimum spanning tree wedge', size=10)
    mst.plot('u', 'v', c='grey', ax=ax[1], linewidth=0.2, alpha=0.4)
    mst.plot.scatter('u', 'v', c='w', ax=ax[1], cmap='viridis')
    ax[1].set_ylabel('')
    ax[1].get_legend().remove()
    add_at(ax[1],
           '|E|={:,}'.format(mst['u'].size),
           loc='lower right')

    # Sequence wedge
    xs, ys, zs = [], [], []
    for row in zip(
        [sequence_indices[i] for i in range(len(sequence_indices)-1)],
        [sequence_indices[i+1] for i in range(len(sequence_indices)-1)],
        [i for i in range(len(sequence_indices)-1)]
    ):
        x, y, si = row

        xs.append(x if x < y else y)
        ys.append(y if x < y else x)
        zs.append(si)

    ax[2].set_title('Directed longuest sequence_indices wedge', size=10)
    ax[2].plot(xs, ys, c='black', linewidth=0.2, alpha=0.4)
    cb = ax[2].scatter(xs, ys, c=zs, cmap='cividis')
    add_at(ax[2],
           '|V|={:,}'.format(len(xs)),
           loc='lower right')

    a_x = [sequence_indices[i] for i in range(len(sequence_indices)-1)][0]
    a_y = [sequence_indices[i+1] for i in range(len(sequence_indices)-1)][0]
    z_x = [sequence_indices[i] for i in range(len(sequence_indices)-1)][-1]
    z_y = [sequence_indices[i+1] for i in range(len(sequence_indices)-1)][-1]
    ax[2].plot(
        a_x if a_x < a_y else a_y,
        a_y if a_x < a_y else a_x,
        marker='+', c='white', label='start'
    )
    ax[2].plot(
        z_x if z_x < z_y else z_y,
        z_y if z_x < z_y else z_x,
        marker='+', c='red', label='finish'
    )

    plt.colorbar(cb, ax=ax[2], label='Sequence index')
    ax[2].set_ylabel('')
    ax[2].set_xlabel('u')

    plt.tight_layout()

    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.savefig(folder + state_prefix + '_search.pdf')
    plt.savefig(folder + state_prefix + '_search.png', dpi=300)
    print ('saved as', folder + state_prefix + '_search.[png, pdf]')

def plot_MST_elongation(verbose=False):
    if verbose:
        print ('Plotting elongation')
    elongation_dict = load('elongation_dict')
    u=0.
    v=0.
    t = np.linspace(0, 2*np.pi, 100)

    fig, ax = plt.subplots(1,1, figsize=(10, 5))

    freqs = {
            0: u'0-200', # in MHz
            1: u'200-400',
            2: u'400-700',
            3: u'700-1000',
            4: u'1000-1500',
            5: u'1500-2000',
            6: u'2000-'
        }

    colors = ['', '', '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']

    xx, yy = [], []
    for metric in ['DTW', 'L2', 'shape']:
        for k in elongation_dict.keys():
            if metric in k:
                a=elongation_dict[k]['half_width']
                b=elongation_dict[k]['length']

                freq_id = int(k.split('freq_')[-1].split('_')[0])
                try:
                    k_cluster = int(k.split('kcluster_')[-1].split('_')[0])
                    cluster_id = int(k.split('cluster_id_')[-1].split('_')[0])
                except:
                    k_cluster, cluster_id = None, None

                xx.append(u+a*np.cos(t))
                yy.append(v+b*np.sin(t))
                ax.plot(u+a*np.cos(t),
                        v+b*np.sin(t),
                        '--' if metric == 'shape' else '.' if metric == 'L2' else '-',
                        label=u"{}  {} MHz kCluster={} cluster_id={} $\eta$={:.2f}  $\eta'$={:.2f}  n={}".format(metric,
                                                           freqs[freq_id],
                                                           k_cluster,
                                                           cluster_id,
                                                           elongation_dict[k]['elongation'],
                                                           elongation_dict[k]['normalized_elongation'],
                                                           elongation_dict[k]['N']) if k_cluster is not None else \
                               u"{}  {} MHz $\eta$={:.2f}  $\eta'$={:.2f}  n={}".format(metric,
                                                           freqs[freq_id],
                                                           elongation_dict[k]['elongation'],
                                                           elongation_dict[k]['normalized_elongation'],
                                                           elongation_dict[k]['N']),
                        linewidth=1,
                        markersize=1,
                        # color=colors[freq_id]
                        )

    ax.set_xlabel('Half-width')
    ax.set_ylabel('Length')
    # print (yy)
    ax.set_xlim([np.min(yy), np.max(yy)])

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})
    plt.tight_layout()

    print ('Saving to images/elongation/elongations_linear_polarization_degree.pdf')
    if not os.path.exists('images/elongation'):
        os.makedirs('images/elongation')
    plt.savefig('images/elongation/elongations_linear_polarization_degree.pdf')

def set_frequency_text(freq_bin):
    freqs = {
        0: u'$0-200$', # in MHz
        1: u'$200-400$',
        2: u'$400-700$',
        3: u'$700-1000$',
        4: u'$1000-1500$',
        5: u'$1500-2000$',
        6: u'$2000-$'
    }

    return freqs[freq_bin]

def plot_sequence_colormap(population, sequence_indices, mst, metric, stokes_to_include, freq_id_to_include, show_stokes, population_graph_indices, graph_population_indices, facecolor='white', stokes_linewidth = 0.2):
    fs = [[0,  200],
          [200,400],
          [400,700],
          [700,1000],
          [1000,1500],
          [1500,2000],
          [2000,1199169832000000]]
    fs_idx = np.asarray([i for i, v in enumerate(fs)])

    # if freq_id_to_include is not None and len(freq_id_to_include) > 1:
    #     colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d']
    # else:
    if facecolor == 'white':
        colors = ['#000000' for i in range(10)]
    else:
        colors = ['#ffffff' for i in range(10)]

    fig, ax = plt.subplots(1,1)

    plt.title("Metric: %s   Freq. range (MHz): %s    Stokes %s" % (metric, set_frequency_text(freq_id_to_include), stokes_to_include))

    gap = 70
    k = 0
    pop = population.as_array()
    colors = plt.cm.viridis(np.linspace(0,1,len(sequence_indices)))

    for j in range(len(sequence_indices)):
        p_i = graph_population_indices[sequence_indices[j]]
        pulsar_freqs = list(pop[p_i].observations.keys())
        obs = pop[p_i].observations[pulsar_freqs[0] if freq_id_to_include is None else freq_id_to_include]
        shape = obs.stokes_I.shape[0]
        ax.plot(np.linspace(0, obs.linear_polarization_degree.size, obs.linear_polarization_degree.size), obs.linear_polarization_degree, color=colors[j])

def plot_sequence_w_neighbours(population, sequence_indices, distances, elongation_dict, mst, metric, stokes_to_include, freq_id_to_include, show_stokes, population_graph_indices, graph_population_indices, facecolor='white', stokes_linewidth = 0.2, inc=1.2, gap = 140, figsize=(30, 20), show_title=True, state_prefix=None):

    if facecolor == 'white':
        colors = ['#000000' for i in range(10)]
    else:
        colors = ['#ffffff' for i in range(10)]

    fig, ax = plt.subplots(1,1, figsize=figsize)

    ds = np.array(distances)
    cmap = mpl.cm.get_cmap('cubehelix').reversed()
    w_set_sequence, w_set_neighbours = [], []

    # ESTABLISH COLORMAP RANGE
    # Collect distance between pairs to bound the coloring with min and max
    for j in range(len(sequence_indices)):
        p_i = graph_population_indices[sequence_indices[j]]
        if j > 0:
            w_set_sequence.append(
                ds[
                    (ds.T[0] == sequence_indices[j-1]) & (ds.T[1] == sequence_indices[j]) |
                    (ds.T[0] == sequence_indices[j]) & (ds.T[1] == sequence_indices[j-1])
                ][0, -1]
            )
            match = mst.loc[(mst['u'] == population_graph_indices[p_i]) | (mst['v'] == population_graph_indices[p_i])]
            for i, row in match.iterrows():
                idx = row['v'] if row['u'] == population_graph_indices[p_i] else row['u']
                w_set_neighbours.append(
                    ds[
                        (ds.T[0] == idx) & (ds.T[1] == sequence_indices[j]) |
                        (ds.T[0] == sequence_indices[j]) & (ds.T[1] == idx)
                    ][0, -1]
                )
    w_set = np.array(w_set_sequence + w_set_neighbours)
    norm = mpl.colors.Normalize(vmin=w_set.min(), vmax=w_set.max())

    # MAIN PLOTTIING LOOP
    w=0
    k = 0
    max_stop = 0
    pop = population.as_array()
    for j in range(len(sequence_indices)):
        p_i = graph_population_indices[sequence_indices[j]]
        # pulsar_freqs = list(pop[p_i].observations.keys())
        obs = pop[p_i].observations[freq_id_to_include]
        shape = obs.stokes_I.shape[0]
        start = 0
        stop = shape

        # Check distance between the two pulsars in sequence
        if j > 0:
            w = ds[
                (ds.T[0] == sequence_indices[j-1]) & (ds.T[1] == sequence_indices[j]) |
                (ds.T[0] == sequence_indices[j]) & (ds.T[1] == sequence_indices[j-1])
            ][0, -1]
        else:
            w = np.nan

        if j > 0:
            # Print colored square for distance between nodes
            ax.fill_between(np.linspace(shape, start, shape),
                            y1=k,
                            y2=k-(inc-1),
                            color=cmap(norm(w)),
                            alpha=0.5)

        file_location = obs.file_location.split('/')[-2] + "/" + obs.file_location.split('/')[-1]
        central_freq = obs.file_location.split('/')[-1].split('_')[-1].split('.')[0]

        ax.text(start, k+(inc/2)-0.1, "{}".format(pop[p_i].jname), fontsize=11)
        ax.text(start, k + 0.2, "L[%]={:.0f}".format(obs.mean_linear_polarization_degree), fontsize=11)
        ax.text(stop, k+(inc/2)-0.1, "{}".format(obs.epn_reference_code), fontsize=11, horizontalalignment='right')
        ax.text(stop, k+0.2, "{} MHz".format(central_freq), fontsize=11, horizontalalignment='right')

        # ax.text(stop-(70 * 3.7), k + 0.2, "w={:.2f}".format(w), fontsize=11)

        ax.plot(np.linspace(start, stop, shape),
                k+obs.stokes_I,
                c=colors[0], linewidth=0.4)
        if show_stokes:
            ax.plot(np.linspace(start, stop, shape), k+obs.stokes_L, c='red', linewidth=stokes_linewidth-0.2)
            # ax.plot(np.linspace(start, stop, shape), k+obs.stokes_V, c='blue', linewidth=stokes_linewidth-0.2)

        if 'linear_polarization_degree' in stokes_to_include:
            ax.fill_between(np.linspace(start, stop, shape),
                            y1=k+obs.linear_polarization_degree,
                            y2=k,
                            color='green',
                            linewidth=stokes_linewidth,
                            alpha=0.4)

        if metric in ['shape', 'DTW']:
            # Check bounds on which the distance was computed
            a_s, a_e = int(obs.centroid - (5 * (obs.fwhm // 2))), \
                       int(obs.centroid + (5 * (obs.fwhm // 2)))

            a_s, as_pad = check_neg(a_s)
            a_e, ae_pad = check_neg(a_e)
            a_s, a_e = check_bound(a_s, a_e)
            a_s, a_e = check_min_max(a_s, a_e, obs.stokes_I.size)
            fb = ax.fill_between(np.linspace(a_s, a_e, a_e-a_s),
                            y1=k,
                            y2=k+1,
                            color='yellow',
                            linewidth=stokes_linewidth,
                            alpha=0.2)
            # fb.set_zorder(0)

        # Go through sequence vertex neighbours that are not part of the sequence
        match = mst.loc[(mst['u'] == population_graph_indices[p_i]) |
                        (mst['v'] == population_graph_indices[p_i])]
        for i, row in match.iterrows():
            # if i < 10:
            idx = row['v'] if row['u'] == population_graph_indices[p_i] else row['u']
            # pulsar_freqs = list(pop[graph_population_indices[int(idx)]].observations.keys())

            ii = 0
            obs = pop[graph_population_indices[int(idx)]].observations[freq_id_to_include]

            # Fetch distance with neighbour
            if j > 0:
                w = ds[
                    (ds.T[0] == idx) & (ds.T[1] == sequence_indices[j]) |
                    (ds.T[0] == sequence_indices[j]) & (ds.T[1] == idx)
                ][0, -1]
            else:
                w = np.nan

            if j > 0 and j < len(sequence_indices)-1:
                if idx not in [sequence_indices[j-1], sequence_indices[j+1]]:
                    _plot_it = True
                else:
                    _plot_it = False
            elif j == 0:
                if idx not in [sequence_indices[j+1]]:
                    _plot_it = True
                else:
                    _plot_it = False
            elif j == len(sequence_indices)-1:
                if idx not in [sequence_indices[j-1]]:
                    _plot_it = True
                else:
                    _plot_it = False
            else:
                _plot_it = True

            if _plot_it:
                start = stop + gap
                stop = start + shape
                if max_stop < stop:
                    max_stop = stop

                file_location = obs.file_location.split('/')[-2] + "/" + obs.file_location.split('/')[-1]
                central_freq = obs.file_location.split('/')[-1].split('_')[-1].split('.')[0]

                ax.plot(np.linspace(start, stop, shape), k+obs.stokes_I, c=colors[0], linewidth=0.4)
                ax.text(start, k+(inc/2)-0.1, "{}".format(pop[graph_population_indices[int(idx)]].jname), fontsize=11)
                ax.text(start, k + 0.2, "L[%]={:.0f}".format(obs.mean_linear_polarization_degree), fontsize=11)
                ax.text(stop, k+0.2, "{} MHz".format(central_freq), fontsize=11, horizontalalignment='right')
                ax.text(stop, k+(inc/2)-0.1, "{}".format(obs.epn_reference_code), fontsize=11, horizontalalignment='right')
                # ax.text(stop-(    70 * 3.7), k + 0.2, "w={:.2f}".format(w), fontsize=11)
                if show_stokes:
                    # Plot Stokes L
                    ax.plot(np.linspace(start, stop, shape), k+obs.stokes_L, c='red', linewidth=stokes_linewidth-0.2)
                    # ax.plot(np.linspace(start, stop, shape), k+obs.stokes_V, c='blue', linewidth=stokes_linewidth-0.2)

                if 'linear_polarization_degree' in stokes_to_include:
                    ax.fill_between(np.linspace(start, stop, shape),
                                    y1=k+obs.linear_polarization_degree,
                                    y2=k,
                                    color='green',
                                    linewidth=stokes_linewidth,
                                    alpha=0.4)

                if metric in ['shape', 'DTW']:
                    # Check bounds on which the distance was computed
                    a_s, a_e = int(obs.centroid - (5 * (obs.fwhm // 2))), \
                               int(obs.centroid + (5 * (obs.fwhm // 2)))

                    a_s, as_pad = check_neg(a_s)
                    a_e, ae_pad = check_neg(a_e)
                    a_s, a_e = check_bound(a_s, a_e)
                    a_s, a_e = check_min_max(a_s, a_e, obs.stokes_I.size)

                    fb = ax.fill_between(np.linspace(start+a_s, start+a_e, a_e-a_s),
                                    y1=k,
                                    y2=k+1,
                                    color='yellow',
                                    linewidth=stokes_linewidth,
                                    alpha=0.2)

                fb = ax.fill_between(np.linspace(start-(gap-50), start-30, gap),
                                y1=k,
                                y2=k+1,
                                color=cmap(norm(w)),
                                alpha=0.5)
                fb.set_zorder(0)

        k+=inc

    ax.plot([0, shape], [k, k], color='black')
    ax.plot([shape+gap, max_stop], [k, k], color='black')
    ax.text(shape//2, k+0.2, 'Sequence', horizontalalignment='center')
    ax.text((shape+gap)+(max_stop-(shape+gap))//2, k+0.2, 'Sequence vertex direct neighbour(s)', horizontalalignment='center')

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 label=u'$w$',
                 shrink=0.2)

    if facecolor != 'white':
        ax.set_facecolor('white')

    plt.axis('off')

    if show_title:
        plt.title(r"(Metric: %s) %s MHz  %s %s $\eta$=%.2f $\eta'$=%.2f" % (
            metric,
            set_frequency_text(freq_id_to_include),
            'Stokes' if stokes_to_include != 'linear_polarization_degree' else '',
            stokes_to_include if stokes_to_include != 'linear_polarization_degree' else stokes_to_include.replace('_', ' ').replace('degree', 'fraction'),
            elongation_dict['elongation'],
            elongation_dict['normalized_elongation'],

        ), fontsize=30)

    save_figure(plt, metric, stokes_to_include, 'tree_sequence', freq_ids_to_include=[freq_id_to_include], state_prefix=state_prefix)

def plot_multifreq_sequences(metrics = ['L2'],
                            stokes_to_include=['I', 'Q', 'U', 'L', 'V',
                                               'IQ', 'IU', 'IL', 'IV'
                                               'IQL', 'IUL', 'ILV',
                                               'IQUL',
                                               'IQULV',],
                            freq_ids_to_include=None,
                            reference=None,
                            min_snr=10,
                            stokes=False,
                            min_freq_id = 2,
                            skip=0,
                            colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d'],
                            verbose = False):

    if freq_ids_to_include is not None and len(freq_ids_to_include) == 1:
        colors = ['#000000' for i in range(10)]

    lims = {
        0: u'[0-200)', # in MHz
        1: u'[200-400)',
        2: u'[400-700)',
        3: u'[700-1000)',
        4: u'[1000-1500)',
        5: u'[1500-2000)',
        6: u'[2000,)'
    }

    k = 0
    for metric in metrics:
        for _stokes_to_include in stokes_to_include:
            fig, ax = plt.subplots(1,1, figsize=(10, 10))

            state_prefix = set_state_name('', [metric, _stokes_to_include, freq_ids_to_include, reference])
            epn_metadata, population, distances, mst, \
            sequence_indices, sequence_population, elongation_dict, \
            population_graph_indices, graph_population_indices, \
            models_clustering, models_vertices_indices = load_states(state_prefix)

            pop = population.as_array()
            del population

            shape=0
            if freq_ids_to_include is not None and len(freq_ids_to_include) == 1:
                min_freq_id = freq_ids_to_include[0]

            for pulsar in sequence_population:
                freqs = list(pulsar.observations.keys())
                for freq in pulsar.observations.keys() if freq_ids_to_include is None else freq_ids_to_include:
                    try:
                        obs = pulsar.observations[freq]
                        if obs.snr > 10:
                            shape = obs.stokes_I.shape[0]
                            base  = freq - min_freq_id
                            if verbose:
                                print (freq, base)
                            start = (base * shape) + (base * skip)
                            stop  = start+shape

                            ax.plot(np.linspace(start, stop, shape), k+obs.stokes_I, c=colors[freq])

                            if stokes:
                                if 'L' in stokes_to_include:
                                    plt.plot(np.linspace(start, stop, shape), k+obs.stokes_L, c='red', linewidth=0.2)
                                if 'V' in stokes_to_include:
                                    plt.plot(np.linspace(start, stop, shape), k+obs.stokes_V, c='blue', linewidth=0.2)
                    except KeyError:
                        pass

                k+=1.2
                if verbose:
                    print ()

            for freq in pulsar.observations.keys() if freq_ids_to_include is None else freq_ids_to_include:
                base  = freq - min_freq_id
                start = (base * shape) + (base * skip)
                stop  = start+shape
                x = start + (stop-start)/2 - len(lims[freq])
                y = k + 1.2
                ax.text(x, y, lims[freq], color=colors[freq],
                        horizontalalignment='center', verticalalignment='center')

            ax.text(x + 550, y, 'MHz', horizontalalignment='center', verticalalignment='center')

            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            fig.patch.set_visible(False)
            ax.axis('off')

            save_figure(plt, metric, _stokes_to_include, 'multifreq_sequence')

def plot_correlations(metric = 'shape',
                      stokes_to_include='linear_polarization_degree',
                      freq_ids_to_include=None,
                      reference=None,
                      min_snr=10,
                      stokes=False,
                      folder = 'images/',
                      param1 = 'EDOT',
                      param2 = 'P1',
                      figsize=(7, 7),
                      verbose = False):
    """Plot param1 against sequence, and color by param2
    """

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(str(stokes_to_include) + ' ' + str(freq_ids_to_include))

    state_prefix = set_state_name('', [metric, _stokes_to_include, freq_ids_to_include, reference])
    epn_metadata, population, distances, mst, \
    sequence_indices, sequence_population, elongation_dict, \
    population_graph_indices, graph_population_indices, \
    models_clustering, models_vertices_indices  = load_states(state_prefix)

    xs, _params = [], []
    total_distance = 0
    for j, pulsar in enumerate(sequence_population[::-1]):
        row = epn_metadata.loc[epn_metadata['jname'] == pulsar.jname]
        if j > 0:
            u, v = sequence_indices[j-1], sequence_indices[j]
            total_distance += mst.loc[
                ((mst['u'] == u) & (mst['v'] == v)) |
                ((mst['u'] == v) & (mst['v'] == u)),
                'w'
            ]
        try:
            _param = row[param].values[0]
        except IndexError:
            _param = -1

        if _param != -1:
            xs.append(total_distance)
            _params.append(_param)

    lims = [0.1, 1, 10] # ms, normal, and slow pulsars
    xs = np.asarray(xs)
    _params = np.asarray(_params)

    ms_lim = np.where(p0s < lims[0])
    s_lim = np.where((p0s >= lims[0]) & (p0s < lims[1]))
    slow_lim = np.where(p0s >= lims[1])

    # Set current axis to plot in
    _ax = ax[k+1, i] if ncols > 1 else ax[k+1]

    if _ys == 'P0':
        ys = p0s
        _ax.set_ylabel(r'$P$')
        _ax.set_yscale('log')
    if _ys == 'P1':
        ys = np.log10(p1s)
        _ax.set_ylabel(r'$log\ \dot{P}$')
    if _ys == 'age':
        ys = ages
        _ax.set_ylabel(r'$P/2\dot{P}$')
        _ax.set_yscale('log')
    if _ys == 'DM':
        ys = dms
        _ax.set_ylabel(r'DM (pc/cc)')
        _ax.set_yscale('log')
    if _ys == 'P0_DM':
        ys = p0s/dms
        _ax.set_ylabel(r'$P/$DM')
        _ax.set_yscale('log')
    if _ys == 'P1_DM':
        ys = np.log10(p1s)/np.log10(dms)
        _ax.set_ylabel(r'$\log\ \dot{P}/\log$ DM')
    if _ys == 'BSurf':
        ys = bsurfs
        _ax.set_ylabel(r'$BSurf$')
        _ax.set_yscale('log')

#             _ax.set_yscale('log')

    scat = _ax.scatter(
        xs[ms_lim],
        ys[ms_lim],
        c=np.log10(zs[ms_lim]),
#             c=dms[ms_lim],
        marker='^',
        label=r'$< 0.1$'
    )
    scat = _ax.scatter(
        xs[s_lim],
        ys[s_lim],
        c=np.log10(zs[s_lim]),
#             c=dms[s_lim],
        label=r'$0.1 \leq \dot{P} < 1$'
    )
    scat = _ax.scatter(
        xs[slow_lim],
        ys[slow_lim],
        c=np.log10(zs[slow_lim]),
#             c=dms[slow_lim],
        marker='v',
        label=r'$\geq 1$'
    )

    _ax.set_xlabel('Sequence index')
    if 'P1' not in _ys:
        _ax.set_ylim(10**(np.log10(np.nanmin(ys)) - 1), 10**(np.log10(np.nanmax(ys)) + 1))

    cbar = fig.colorbar(scat, ax=_ax)
    cbar.ax.set_ylabel(r'$\log\ \dot{E}$ (ergs/s)')

    ys = np.log10(p1s)
    _ax = ax[0, i] if ncols > 1 else ax[0]
    scat = _ax.scatter(
        p0s[ms_lim],
        ys[ms_lim],
        c=(xs[ms_lim]),
        marker='^',
        label=r'$< 0.1$'
    )
    scat = _ax.scatter(
        p0s[s_lim],
        ys[s_lim],
        c=(xs[s_lim]),
        label=r'$0.1 \leq \dot{P} < 1$'
    )
    scat = _ax.scatter(
        p0s[slow_lim],
        ys[slow_lim],
        c=(xs[slow_lim]),
        marker='v',
        label=r'$\geq 1$'
    )
    _ax.set_xlabel(r'$P$')
    _ax.set_ylabel(r'$\log\ \dot{P}$')
    _ax.set_xscale('log')

    _ax.legend(
            loc='lower left',
            title=r'Period (s)',
            title_fontsize='x-small',
            fontsize='xx-small'
        )

    _ax.set_ylim(-21.5, -10.3)
    _ax.set_xlim(10**-3, 10)

#     _ax.grid(True)
    cbar = fig.colorbar(scat, ax=_ax)
    cbar.ax.set_ylabel(r'`Sequence index')

    add_at(_ax, _stokes_to_include, loc='lower right')
    add_at(_ax, 'n=%d' % p1s.shape[0], loc='upper left')


    save_figure(plt, metric, _stokes_to_include, 'correlation_psrcat', freq_ids_to_include=freq_ids_to_include)

def plot_correlations_multipanel(metric = 'shape',
                                 stokes_to_include=['linear_polarization_degree',
                                                    'I', 'Q', 'U', 'L', 'V',
                                                    'IQ', 'IU', 'IL', 'IV'
                                                    'IQL', 'IUL', 'ILV',
                                                    'IQUL',
                                                    'IQULV',],
                                 freq_ids_to_include=None,
                                 reference=None,
                                 min_snr=10,
                                 stokes=False,
                                 folder = 'images/',
                                 params = ['P0', 'P1', 'BSurf', 'age', 'EDOT'], #'DM', 'P0_DM', 'P1_DM'],
                                 figsize=(10, 10),
                                 verbose = False):

    nrows = 2
    # ncols  = len(stokes_to_include)
    ncols = int(np.ceil(len(params)+1)/2)

    # This should go somewhere central
    lims = {
        0: u'[0-200)', # in MHz
        1: u'[200-400)',
        2: u'[400-700)',
        3: u'[700-1000)',
        4: u'[1000-1500)',
        5: u'[1500-2000)',
        6: u'[2000,)'
    }

    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    title = "Metric: %s; %s MHz" % (metric, lims[freq_ids_to_include[0]])
    if nrows == 1 and ncols == 1:
        ax.set_title(title)
    elif nrows == 1 or ncols == 1:
        ax[0].set_title(title)
    else:
        ax[0,0].set_title(title)

    ii = 0
    for i, _stokes_to_include in enumerate(stokes_to_include):
        state_prefix = set_state_name('', [metric, _stokes_to_include, freq_ids_to_include, reference])
        epn_metadata, population, distances, mst, \
        sequence_indices, sequence_population, elongation_dict, \
        population_graph_indices, graph_population_indices, \
        models_clustering, models_vertices_indices = load_states(state_prefix)

        for k, _ys in enumerate(params):
            xs, ages, zs, p0s, p1s, dms, bsurfs, edots, mu_lin_pol_degs = [], [], [], [], [], [], [], [], []
            total_distance = 0
            for j, pulsar in enumerate(sequence_population[::-1]):
                row = epn_metadata.loc[epn_metadata['jname'] == pulsar.jname]
                try:
                    age = row['AGE'].values[0]
                except IndexError:
                    age = -1

                try:
                    e_dot = row['EDOT'].values[0]
                except IndexError:
                    e_dot = np.nan

                try:
                    p0 = row['P0'].values[0]
                except IndexError:
                    p0 = -1

                try:
                    p1 = row['P1'].values[0]
                except IndexError:
                    p1 = -1

                try:
                    dm = row['DM'].values[0]
                except IndexError:
                    dm = -1

                try:
                    bsurf = row['BSURF'].values[0]
                except IndexError:
                    bsurf = -1

                if age != -1 and p1 != 1:
                    if j > 0:
                        u, v = sequence_indices[j-1], sequence_indices[j]
                        total_distance += mst.loc[
                            ((mst['u'] == u) & (mst['v'] == v)) |
                            ((mst['u'] == v) & (mst['v'] == u)),
                            'w'
                        ].values[0]
                    xs.append(total_distance)
                    ages.append(age)
                    edots.append(e_dot)
                    p0s.append(p0)
                    p1s.append(p1)
                    dms.append(dm)
                    bsurfs.append(bsurf)
                    mu_lin_pol_degs.append(pulsar.observations[freq_ids_to_include[0]].mean_linear_polarization_degree)

            lims = [0.1, 1, 10]
            xs = np.asarray(xs)
            zs = np.asarray(zs)
            ages = np.asarray(ages)
            edots = np.asarray(edots)
            p0s = np.asarray(p0s)
            p1s = np.asarray(p1s)
            dms = np.asarray(dms)
            bsurfs = np.asarray(bsurfs)
            mu_lin_pol_degs = np.asarray(mu_lin_pol_degs)
            # mu_lin_pol_degs = (mu_lin_pol_degs - mu_lin_pol_degs.min()) / (mu_lin_pol_degs.max() - mu_lin_pol_degs.min())

            ms_lim = np.where(p0s < lims[0])
            s_lim = np.where((p0s >= lims[0]) & (p0s < lims[1]))
            slow_lim = np.where(p0s >= lims[1])

            if (k > 0) and ((k % 2) == 0):
                ii += 1
            _ax = ax[k % 2, ii] if ncols > 1 else ax[k+1]

            # _ax.set_xscale('log')
            if _ys == 'P0':
                ys = p0s
                _ax.set_ylabel(r'$P$')
                _ax.set_yscale('log')
            if _ys == 'P1':
                ys = np.log10(p1s)
                _ax.set_ylabel(r'$log\ \dot{P}$')
            if _ys == 'age':
                ys = ages
                _ax.set_ylabel(r'$P/2\dot{P}$')
                _ax.set_yscale('log')
            if _ys == 'DM':
                ys = dms
                _ax.set_ylabel(r'DM [pc/cc]')
                _ax.set_yscale('log')
            if _ys == 'P0_DM':
                ys = p0s/dms
                _ax.set_ylabel(r'$P/$DM')
                _ax.set_yscale('log')
            if _ys == 'P1_DM':
                ys = np.log10(p1s)/np.log10(dms)
                _ax.set_ylabel(r'$\log\ \dot{P}/\log$ DM')
            if _ys == 'BSurf':
                ys = bsurfs
                _ax.set_ylabel(r'$BSurf$')
                _ax.set_yscale('log')
            if _ys == 'EDOT':
                ys = edots
                _ax.set_ylabel(r'$\dot{E}$')
                _ax.set_yscale('log')

            def line(x, a, b):
                return a * x + b

            # Millisecond pulsars
            # best_fit, cov_matrix = curve_fit(line, xs[ms_lim], ys[ms_lim])
            scat = _ax.scatter(
                xs[ms_lim],
                ys[ms_lim],
                c=mu_lin_pol_degs[ms_lim],
    #             c=dms[ms_lim],
                marker='^',
                label=r'$< 0.1$'
            )
            # _ax.plot(xs[ms_lim], line(xs[ms_lim], best_fit[0], best_fit[1]), 'r-')
            # Normal pulsars
            best_fit, cov_matrix = curve_fit(line, xs[s_lim], ys[s_lim])
            scat = _ax.scatter(
                xs[s_lim],
                ys[s_lim],
                #c=np.log10(zs[s_lim]) if _ys != 'P1' else mu_lin_pol_degs[s_lim],
                c=mu_lin_pol_degs[s_lim],
    #             c=dms[s_lim],
                label=r'$0.1 \leq \dot{P} < 1$'
            )
            _ax.plot(xs[s_lim], line(xs[s_lim], best_fit[0], best_fit[1]), 'r-')

            # Slow pulsars
            # best_fit, cov_matrix = curve_fit(line, xs[slow_lim], ys[slow_lim])
            scat = _ax.scatter(
                xs[slow_lim],
                ys[slow_lim],
                c=mu_lin_pol_degs[slow_lim],
    #             c=dms[slow_lim],
                marker='v',
                label=r'$\geq 1$'
            )
            # _ax.plot(xs[ms_lim], line(xs[ms_lim], best_fit[0], best_fit[1]), 'r-')

            _ax.set_xlabel('Sequence distance')
            if 'P1' not in _ys:
                _ax.set_ylim(10**(np.log10(np.nanmin(ys)) - 1), 10**(np.log10(np.nanmax(ys)) + 1))

            cbar = fig.colorbar(scat, ax=_ax)
            # cbar.ax.set_ylabel(r'$\log\ \dot{E}$ (ergs/s)' if _ys != 'P1' else 'L (%)')
            cbar.ax.set_ylabel('L [%]')

        # Starting P-Pdot diagram
        ys = np.log10(p1s)
        _ax = ax[-1, -1] if ncols > 1 else ax[-1]
        scat = _ax.scatter(
            p0s[ms_lim],
            ys[ms_lim],
            c=(xs[ms_lim]),
            marker='^',
            label=r'$< 0.1$'
        )
        scat = _ax.scatter(
            p0s[s_lim],
            ys[s_lim],
            c=(xs[s_lim]),
            label=r'$0.1 \leq \dot{P} < 1$'
        )
        scat = _ax.scatter(
            p0s[slow_lim],
            ys[slow_lim],
            c=(xs[slow_lim]),
            marker='v',
            label=r'$\geq 1$'
        )
        _ax.set_xlabel(r'$P$')
        _ax.set_ylabel(r'$\log\ \dot{P}$')
        _ax.set_xscale('log')

        _ax.legend(
                loc='lower left',
                title=r'Period (s)',
                title_fontsize='x-small',
                fontsize='xx-small'
            )

        _ax.set_ylim(-21.5, -10.3)
        _ax.set_xlim(10**-3, 10)

        cbar = fig.colorbar(scat, ax=_ax)
        cbar.ax.set_ylabel(r'`Sequence index')

        add_at(_ax, _stokes_to_include, loc='lower right')
        add_at(_ax, 'n=%d' % p1s.shape[0], loc='upper left')


    save_figure(plt, metric, _stokes_to_include, 'correlation_multipanel_psrcat', freq_ids_to_include=freq_ids_to_include)
