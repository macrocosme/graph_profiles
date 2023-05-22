import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import matplotlib.patheffects as PathEffects
from mpl_toolkits import axes_grid1
cmap = cm.get_cmap('cubehelix').reversed()

import os

from .meta import freqs_filename, palette, palette_morph_class, pink, blue_full
from .positioning import set_level_metrics, set_xy_positions
from .prediction import predict
from ..utils.io import check_slash

'''
Start from least connected vertex

Plot profile,

'''

def set_mst_vertices_properties(population, mst, phase_thres = .5, verbose=False):
    # Create children dict
    vertices = np.concatenate((np.unique(mst.as_dataframe()['u']),
                               np.unique(mst.as_dataframe()['v'])))

    levels = mst.farness[mst.centrality.argmax()].astype('int')
    level_counter = {l:0 for l in np.unique(levels)}
    level_metrics = {l:[] for l in np.unique(levels)}
    n_per_level = np.histogram(mst.farness[mst.centrality.argmax()],
                               bins=[i for i in range(np.unique(mst.farness[mst.centrality.argmax()]).size+1)],
                               range=[0, mst.farness[mst.centrality.argmax()].max()+1])
    node_coordinates = {}
    connections = {
        'parent' : [],
        'child' : [],
    }

    children = {}
    neighbours = {}
    for current in vertices:
        for u,v,w in mst.mst:
            if current in [u,v]:
                val = v if u == current else u

                # Set children
                try:
                    if val not in children[current] and levels[val] > levels[current]:
                        children[current].append(val)
                except:
                    children[current] = []
                    if levels[val] > levels[current]:
                        children[current].append(val)

                # Set neighbours
                try:
                    if val not in neighbours[current]:
                        neighbours[current].append(val)
                except:
                    neighbours[current] = []
                    neighbours[current].append(val)

    normalish = lambda arr: np.append(arr[arr.size % 2::2], arr[::-2])
    n_neighbours = {}
    neighbours_n = {}
    for k in children.keys():
        children[k] = np.array(children[k])
        neighbours_n[k] = []
        n_neighbours[k] = 0
        for kk in children[k]:
            n_neighbours[k] += 1

    for k in children.keys():
        for kk in children[k]:
            neighbours_n[k].append(n_neighbours[kk])
        neighbours_n[k] = np.array(neighbours_n[k])
        children[k] = children[k][np.argsort(neighbours_n[k])]
        neighbours_n[k] = neighbours_n[k][np.argsort(neighbours_n[k])]
        children[k] = normalish(children[k])
        neighbours_n[k] = normalish(neighbours_n[k])

    current = mst.centrality.argmax() #mst.find_least_connected_vertex()

    set_level_metrics(current, [current], children, levels, level_metrics)
    for k in level_metrics.keys():
        level_metrics[k] = np.array(level_metrics[k])

    annotate = True
    if np.abs(phase_thres) < 0.5:
        cond = np.where(np.abs(population.as_array()[0].observations[2].phase) < phase_thres)
        x_scaler = population.as_array()[0].observations[2].phase[cond].size
    else:
        x_scaler = population.as_array()[0].observations[2].phase.size
        cond = None
    if verbose:
        print ('v\tparent\tlevel\tx_start\tmax_x\tchild\tx\tchildren')
    max_x = set_xy_positions(current, children, mst, levels, n_per_level,
                             level_counter,
                             x_scaler=x_scaler,
                             node_coordinates=node_coordinates,
                             connections=connections,)

    # print (v, levels[v], level_counter[levels[v]], x_start, last_x)
    return node_coordinates, connections, neighbours, children, levels, x_scaler, max_x, cond

# ++++++++++++++++++++++++++++++++++++++++
#   Main plotting section
# ++++++++++++++++++++++++++++++++++++++++
def plot(population,
         mst,
         freq_ids_to_include = [2],
         colour_by='branch',
         contracted=False,
         contracted_mst_indices=None,
         phase_thres=0.5,
         plot_stokes_I=False,
         plot_pa=False,
         plot_polarization=False,
         annotate=False,
         print_bname=False,
         verbose=False,
         file_location='images/',
         filename='mst',
         file_extension='pdf',
         return_population=False):

    assert colour_by in ['branch', 'rankin_class'], "Current values accepted for `color_by` are ['branch', 'rankin_class']."

    # Prepare tree structure
    node_coordinates, connections, neighbours, children, levels, x_scaler, _, cond = \
        set_mst_vertices_properties(population, mst, phase_thres )

    _cond_ = np.abs(phase_thres) < 0.5
    _4freqs_ = True if len(freq_ids_to_include) > 1 else False

    base_color = 'white' if colour_by != 'rankin_class' else 'black'
    data_color = blue_full if colour_by != 'rankin_class' else pink
    annotation_color = 'white' if colour_by != 'rankin_class' else 'black'

    # Main code
    initialized = True if 'initialized' in globals() else False
    if not initialized:
        # Initialize all when needed...
        for i in range(len(population.as_array())):
            population.as_array()[i].predicted_class = 'N/A'
        initialized = True

    x_coords, y_coords, pulsars, facecolors, edgecolors = [], [], [], [], []
    for k in node_coordinates.keys():
        x_coords.append(node_coordinates[k][0])
        y_coords.append(node_coordinates[k][1])
        kk = k if not contracted else contracted_mst_indices[k]
        pulsars.append(population.as_array()[kk])

        if population.as_array()[kk].morphological_class != 'N/A':
            predicted = False
            morphological_class = population.as_array()[kk].morphological_class
        else:
            predicted = True
            morphological_class = predict(population, kk, neighbours, mst, verbose)
            population.as_array()[kk].predicted_class = morphological_class
            print (kk, population.as_array()[kk].predicted_class, neighbours[kk])

        if  colour_by == 'branch':
            facecolors.append(palette[node_coordinates[k][2]])
            edgecolors.append('red' if kk in mst.longest_path else palette[node_coordinates[k][2]])
        else: # assumes there are only two cases...
            facecolors.append(palette_morph_class[morphological_class])
            edgecolors.append('dimgrey' if predicted else 'red' \
                              if kk in mst.longest_path else palette_morph_class[morphological_class])


    _x = 10.5 if len(population.as_array()) < 100 else 21
    _y = 8 if len(population.as_array()) < 100 else 16

    fontsize = 1.1 if len(population.as_array()) < 100 else 0.1
    profile_linewidth = .2 if _cond_ else 0.8
    connection_linewidth = 1.5

    if contracted:
        _x = 10.5
        _y = 5
        fontsize = 1.5



    fig = plt.figure(figsize=(_x, _y))
    _ax = axes_grid1.AxesGrid(fig, 111,
                              nrows_ncols=(1, 1),
                              axes_pad=0.05,
                              cbar_mode='single',
                              cbar_location='right',
                              cbar_pad=0.1,
                              cbar_size="1%",
                              aspect=False)

    ax = _ax[0]

    y_box_height = 0.5 / 2 if len(population.as_array()) != 2 else 0.5

    multi_names_to_annotate = [
        # Leaves
        'J2321+6024', 'J2113+4644', #'J1740+1311', J0332+5434',
        # Level 1
        'J2219+4754', 'J1820-0427', 'J1917+1353',
        'J1607-0032', 'J2055+3630', 'J0629+2415']
    multi_annotation = {
        # Leaves
        'J2321+6024': {'text':'a', 'location':'top', 'color':'black'},
        'J2113+4644': {'text':'b', 'location':'top', 'color':'black'},
        # 'J0332+5434': {'text':'', 'location':'top', 'color':'black'},
        # Level 1
        'J2219+4754': {'text':'6', 'location':'left_to_edge', 'color':'black'},
        'J1820-0427': {'text':'5', 'location':'left_to_edge', 'color':'black'},
        'J1917+1353': {'text':'4', 'location':'left_to_edge', 'color':'black'},
        'J1607-0032': {'text':'3', 'location':'left_to_edge', 'color':'black'},
        'J2055+3630': {'text':'2', 'location':'left_to_edge', 'color':'black'},
        'J0629+2415': {'text':'1', 'location':'left_to_edge', 'color':'black'}
    }

    single_names_to_annotate = [
        # Level 1
        'J1607-0032',
        'J1917+1353',
        'J1822-2256',
        'J2313+4253',
        'J1820-0427',
        'J2219+4754',
        'J2354+6155',
        'J2055+3630']
    single_annotation = {
        # Level 1
        'J1607-0032': {'text':'1', 'location':'left_to_edge'},
        'J1917+1353': {'text':'2', 'location':'left_to_edge'},
        'J1822-2256': {'text':'3', 'location':'left_to_edge'},
        'J2313+4253': {'text':'4', 'location':'left_to_edge'},
        'J1820-0427': {'text':'5', 'location':'left_to_edge'},
        'J2219+4754': {'text':'6', 'location':'left_to_edge'},
        'J2354+6155': {'text':'7', 'location':'left_to_edge'},
        'J2055+3630': {'text':'8', 'location':'left_to_edge'}
    }


    for pulsar, x, y, facecolor, edgecolor in zip(pulsars, x_coords, y_coords, facecolors, edgecolors):
        for i, (yy, f) in enumerate(
            zip(
                [y_box_height/2, 0, -y_box_height/2, -y_box_height] if len(freq_ids_to_include) == 4 else [-y_box_height],
                freq_ids_to_include
            )):
            if _cond_:
                phase = pulsar.observations[f].phase[cond]
                if plot_stokes_I:
                    profile = pulsar.observations[f].stokes_I[cond]
                if plot_polarization:
                    stokes_V = pulsar.observations[f].stokes_V[cond]
                    stokes_L = pulsar.observations[f].stokes_L[cond]
                if plot_pa:
                    try:
                        cond_pa = np.where(np.abs(pulsar.observations[f].position_angle_phase) < phase_thres)
                        pa = pulsar.observations[f].position_angle[cond_pa]
                        pa_err = pulsar.observations[f].position_angle_yerr_high[cond_pa]
                        pa_phase = pulsar.observations[f].position_angle_phase[cond_pa]
                    except TypeError:
                        pa = None

                model = pulsar.observations[f].model[cond]
            else:
                phase = pulsar.observations[f].phase
                if plot_stokes_I:
                    profile = pulsar.observations[f].stokes_I
                if plot_polarization:
                    stokes_V = pulsar.observations[f].stokes_V
                    stokes_L = pulsar.observations[f].stokes_L
                if plot_pa:
                    try:
                        pa = pulsar.observations[f].position_angle
                        pa_err = pulsar.observations[f].position_angle_yerr_high
                        pa_phase = pulsar.observations[f].position_angle_phase
                    except TypeError:
                        pa = None

                model = pulsar.observations[f].model

            if i == 0:
                ax.fill_between(
                    np.linspace(x - phase.size / 2,
                                x + phase.size / 2,
                                phase.size),
                    y - y_box_height,
                    y + y_box_height,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    lw=1.4 if not contracted else 2.6,
                    alpha=1,
                    zorder=100
                )

            txt = ax.text(#x, y,
                     x - phase.size / 2 + x_scaler*0.005,
                     y + y_box_height + (y_box_height/30),
                     "%s" % (pulsar.jname),
                     horizontalalignment='left',
                     verticalalignment='top',
                     fontsize=fontsize,
                     c=annotation_color,
                     zorder=500)
            txt.set_path_effects([PathEffects.withStroke(linewidth=0.25, foreground='#ffffff88')])

            # Annotation (Top)
            if annotate:
                if len(freq_ids_to_include) == 1:
                    if pulsar.jname in single_names_to_annotate:
                        if single_annotation[pulsar.jname]['location'] == 'top':
                            txt = ax.text(#x, y,
                                    x - phase.size / 2 + x_scaler*0.25,
                                    y + y_box_height + (1.5 * y_box_height),
                                    "%s" % (single_annotation[pulsar.jname]['text']),
                                    horizontalalignment='center',
                                    verticalalignment='top',
                                    fontsize=20,
                                    c=annotation_color,
                                    zorder=500)
                            txt.set_path_effects([PathEffects.withStroke(linewidth=0.25, foreground='#ffffff88')])
                else:
                    if pulsar.jname in multi_names_to_annotate:
                        if multi_annotation[pulsar.jname]['location'] == 'top':
                            txt = ax.text(#x, y,
                                    x - phase.size / 2 + x_scaler*0.25,
                                    y + y_box_height + (1.5 * y_box_height),
                                    "%s" % (multi_annotation[pulsar.jname]['text']),
                                    horizontalalignment='left',
                                    verticalalignment='top',
                                    fontsize=12,
                                    c=multi_annotation[pulsar.jname]['color'],
                                    zorder=500)
                            txt.set_path_effects([PathEffects.withStroke(linewidth=0.25, foreground='#ffffff88')])

            if print_bname:
                txt = ax.text(#x, y,
                         x - phase.size / 2 + x_scaler*0.005,
                         y + y_box_height - (y_box_height/10),
                         "%s" % (pulsar.bname),
                         horizontalalignment='left',
                         verticalalignment='top',
                         fontsize=fontsize,
                         c=annotation_color,
                         zorder=500)
                txt.set_path_effects([PathEffects.withStroke(linewidth=0.25, foreground='#ffffff88')])
            txt = ax.text(#x, y,
                     x + phase.size / 2 + x_scaler*0.075,
                     y + y_box_height + (y_box_height/30),
                     "$\phi\in\pm%.1f$" % (phase_thres),
                     horizontalalignment='right',
                     verticalalignment='top',
                     fontsize=fontsize,
                     c=annotation_color,
                     zorder=500)
            txt.set_path_effects([PathEffects.withStroke(linewidth=0.25, foreground='#ffffff88')])

            if colour_by != 'branch':
                txt = ax.text(#x, y,
                         x + phase.size / 2  + x_scaler*0.005,
                         y + y_box_height - (y_box_height/10),
                         "$%s$" % (pulsar.morphological_code),
                         horizontalalignment='right',
                         verticalalignment='top',
                         fontsize=fontsize,
                         c=annotation_color,
                         zorder=500)
                txt.set_path_effects([PathEffects.withStroke(linewidth=0.25, foreground='#ffffff88')])

            ax.plot(
                np.linspace(x - phase.size / 2,
                            x + phase.size / 2,
                            phase.size
                ),
                (model * (0.1 if _4freqs_ else 0.5)) + y + yy,
                linewidth=profile_linewidth,
                zorder=600,
                color=base_color
            )

            if plot_stokes_I:
                ax.plot(
                    np.linspace(x - phase.size / 2,
                                x + phase.size / 2,
                                phase.size
                    ),
                    ((profile - pulsar.observations[f].central) * (0.1 if _4freqs_ else 0.5)) + y + yy,
                    linewidth=profile_linewidth,
                    zorder=700,
                    color=data_color,
                    linestyle=':'
                )

            if plot_polarization:
                ax.plot(
                    np.linspace(x - phase.size / 2,
                                x + phase.size / 2,
                                phase.size
                    ),
                    (stokes_L * (0.1 if _4freqs_ else 0.5)) + y + yy,
                    linewidth=profile_linewidth,
                    zorder=100,
                    color=pink
                )

                ax.plot(
                    np.linspace(x - phase.size / 2,
                                x + phase.size / 2,
                                phase.size
                    ),
                    (stokes_V * (0.1 if _4freqs_ else 0.5)) + y + yy,
                    linewidth=profile_linewidth,
                    zorder=100,
                    color=blue_full
                )

            if plot_pa:
                if pa is not None:
                    ax.errorbar(x + (pa_phase*phase.size)/(np.abs(phase.max())+np.abs(phase.min())),
                                 (((pa + 90) / 180) * (0.1 if _4freqs_ else 0.5)) + y + yy,
                                 yerr=(pa_err / 180) * (0.1 if _4freqs_ else 0.5),
                                 fmt='.',
                                 capthick=0.1,
                                 color='#F8BA00',
                                 ms=0.01,
                                 ecolor='#F8BA00',
                                 elinewidth=profile_linewidth,
                                 zorder=500)


    for c in connections['parent']:
        ax.plot(c[0], c[1],
                c='lightgrey',
                alpha=1,
                linewidth=connection_linewidth,
                zorder=50)

    _distances, _cumulative_distances = [], []
    for c in connections['child']:
        _distances.append(c[-3])
        _cumulative_distances.append(c[-1])

    _distances = np.array(_distances)
    _cumulative_distances = np.array(_cumulative_distances)

    norm = colors.LogNorm(vmin=np.nanmin(_distances), vmax=np.nanmax(_distances[np.where(_distances != np.inf)[0]]))

    if annotate:
        b_n = 1
    for c in connections['child']:
        # print (c[-3], np.nanmin(_distances), np.nanmax(_distances[np.where(_distances != np.inf)[0]]))
        ax.plot(c[0], c[1],
                c=cmap(norm(c[-3])),
                linewidth=connection_linewidth,
                zorder=10)

        # Annotation
        if annotate:
            if c[-1] == 1:
                ax.text(#x, y,
                        c[0][0] - phase.size / 2,
                        c[1][1] - 1.295 * (c[1][1]-c[1][0])/2,
                        "%s" % (b_n),
                        horizontalalignment='left',
                        verticalalignment='top',
                        fontsize=6,
                        c='black',
                        zorder=500)
                b_n += 1

    ax.set_yticks(np.arange(np.unique(levels).min(),
                            np.unique(levels).max()+1,
                            1.0))
    ax.set_ylabel('Vertex level')
    ax.axes.xaxis.set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    _ax.cbar_axes[0].colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        label=u'$w$',
    )

    if colour_by != 'branch':
        for i, k in enumerate(palette_morph_class.keys()):
            txt = ax.text(0.01, 0.95 - (0.05 * i),
                    '%s' % (k),
                    color=palette_morph_class[k],
                    horizontalalignment='left',
                    verticalalignment='center',
                    transform=ax.transAxes)
            if k != 'N/A':
                txt.set_path_effects([PathEffects.withStroke(linewidth=0.5, foreground='black')])

    plt.tight_layout()

    if not os.path.exists(file_location):
        os.makedirs(file_location)
    save_to = f'{check_slash(file_location)}{filename}.{file_extension}'
    plt.savefig(save_to)
    print (f'Saved to {save_to}')

    if return_population:
        return population
