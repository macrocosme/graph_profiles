'''
Start from least connected vertex

Plot profile,

'''
import copy
import numpy as np
from scipy.stats import norm, linregress, gaussian_kde
from matplotlib import style, collections as mc, colors, cm
cmap = cm.get_cmap('cubehelix').reversed()

from .meta import freqs_filename, palettes

def get_weight(graph, v, n):
    w = graph.as_dataframe().loc[
                ((graph.as_dataframe()['u'] == v) & (graph.as_dataframe()['v'] == n)) |
                ((graph.as_dataframe()['u'] == n) & (graph.as_dataframe()['v'] == v)),
                'w'
            ].values[0]
    return w

def set_direct_connections(node_coordinates, mst, v, n, x_parent, connections):
        connections.append(
            [ # First trial method
                [node_coordinates[v][0], node_coordinates[n][0]],
                [node_coordinates[v][1], node_coordinates[n][1]],
                cmap(norm(get_weight(mst, v, n))),
                '-' if x_parent and ((n in mst.longest_path) and (v in mst.longest_path)) else 'dotted',
                2 if n in mst.longest_path else 0.7,
            ]
        )

def set_square_connections(node_coordinates, mst, v, n, x_parent, connections, w_start, level):
    w = get_weight(mst, v, n)

    connections['parent'].append(
        [ # parent to middle line
            [node_coordinates[v][0], node_coordinates[v][0]],
            [node_coordinates[v][1], node_coordinates[v][1] + .5],
            '-' if x_parent and ((n in mst.longest_path) and (v in mst.longest_path)) else 'dotted',
            2 if n in mst.longest_path else 0.7,
        ]
    )

    if node_coordinates[v][0] != node_coordinates[n][0]:
        connections['parent'].append(
            [ # middle line from parent to child
                [node_coordinates[v][0], node_coordinates[n][0]],
                [node_coordinates[n][1]-0.5, node_coordinates[n][1]-0.5],
                '-' if x_parent and ((n in mst.longest_path) and (v in mst.longest_path)) else 'dotted',
                2 if n in mst.longest_path else 0.7,
            ]
        )

    connections['child'].append(
        [ # middle line to child
            [node_coordinates[n][0], node_coordinates[n][0]],
            [node_coordinates[n][1]-0.5, node_coordinates[n][1],],
            '-' if x_parent and ((n in mst.longest_path) and (v in mst.longest_path)) else 'dotted',
            2 if n in mst.longest_path else 0.7,
            w,
            w_start + w,
            level
        ],
    )

def set_level_metrics(v, visited, children, levels, level_metrics):
    i = 0
    for n in children[v]:
        if n not in visited:
            i += 1
    level_metrics[levels[v]].append(i)

    for n in children[v]:
        if n not in visited:
            visited.append(n)
            set_level_metrics(n, visited, children, levels, level_metrics)

def set_xy_positions(v,
                     children,
                     graph,
                     levels,
                     n_per_level,
                     level_counter,
                     x_start = 0,
                     w_start = 0,
                     node_coordinates = {},
                     connections = [],
                     bin_size = 1,
                     x_scaler = 1,
                     y_scaler = 1,
                     x_parent=False):

    max_x = copy.deepcopy(x_start)
    level_counter[levels[v]] += 1

    for li, n in enumerate(children[v]):
        max_x = set_xy_positions(n, children, graph,
                                  levels, n_per_level,
                                  level_counter,
                                  x_start = max_x if li == 0 else max_x + x_scaler + (0.1*x_scaler),
                                  w_start = w_start + graph.as_dataframe().loc[
                                        ((graph.as_dataframe()['u'] == v) & (graph.as_dataframe()['v'] == n)) |
                                        ((graph.as_dataframe()['u'] == n) & (graph.as_dataframe()['v'] == v)),
                                        'w'
                                    ].values[0],
                                  node_coordinates = node_coordinates,
                                  connections = connections,
                                  x_scaler = x_scaler,
                                  y_scaler = y_scaler,
                                  x_parent = v, )
    x = x_start + ((max_x - x_start) / 2) if len(children[v]) > 0 else x_start
    if verbose:
        print ('%d\t%d\t%d\t%.2f\t%.2f\t%d\t%.2f\t%d' %
               (v, x_parent, levels[v], x_start, max_x, len(children[v]), x, x_scaler), children[v])

    y = levels[v]

    node_coordinates[v] = [x, y, level_counter[1] if levels[v] > 0 else 0] # level_counter[1] to cluster
                                                                           # per branch from the root



    for n in children[v]:
#         set_direct_connections(node_coordinates, graph, mst, v, n, x_parent, connections)
        set_square_connections(node_coordinates, graph, v, n, x_parent, connections, w_start, levels[n])

    return max_x



def prep_nodes(population, mst):
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

    _cond_ = True
    annotate = True
    if _cond_:
        phase_thres = .1
        cond = np.where(np.abs(population.as_array()[0].observations[2].phase) < phase_thres)
        x_scaler = population.as_array()[0].observations[2].phase[cond].size
    else:
        x_scaler = population.as_array()[0].observations[2].phase.size

    verbose = False
    if verbose:
        print ('v\tparent\tlevel\tx_start\tmax_x\tchild\tx\tchildren')
    max_x = set_xy_positions(current, children, mst, levels, n_per_level,
                             level_counter,
                             x_scaler=x_scaler,
                             node_coordinates=node_coordinates,
                             connections=connections,)

    # print (v, levels[v], level_counter[levels[v]], x_start, last_x)
    return max_x



# ++++++++++++++++++++++++++++++++++++++++
#   Main plotting section
# ++++++++++++++++++++++++++++++++++++++++

contracted = False
plot_polarization = True
plot_pa = False
plot_stokes_I = True
colour_by = 'branch'
# colour_by = 'rankin_class'
print_bname = False
annotate = True

palette = palettes['Greys']

morphological_classes = np.array([p.morphological_class for p in population.as_array()])
morphological_codes = np.array([p.morphological_code for p in population.as_array()])

palette_morph_class = {
    c:palettes['rankin_class'][i] for i, c in enumerate(
        np.unique(morphological_classes[(morphological_classes != 'N/A')])
    )
}
palette_morph_class['N/A'] = '#000000'

print ('palette_morph_class', palette_morph_class.keys())

# palette_morph_code = {
#     c:palette_rankin_class[i] for i, c in enumerate(
#         np.unique(morphological_codes[morphological_codes != 'nan'])
#     )
# }


# Prediction section
def predict(population, k, neighbours, mst):
    def del_na(_classes:list):
        where_na = lambda _classes: (c == 'N/A' for c in _classes)
        for i, d in enumerate(where_na(_classes)):
            if d:
                del _classes[i]
        return _classes

    def majority(classes, verbose=False):
        # u: unique classes, c: unique counts
        u, c = np.unique(classes, return_counts=True)
        uu, cc = np.unique(c, return_counts=True)
#         print (uu, c)
        if cc[np.where(uu == np.max(c))] == 1:
            maj = u[np.argmax(c)]
            if verbose:
                print (maj)
        else:
            maj = None
            if verbose:
                print ('No majority')
        return maj

#     classes = del_na([population.as_array()[c].morphological_class for c in neighbours[k]])
#     for c in neighbours[k]:
#         population.as_array()[c].predicted_class = 'N/A'
    classes = del_na([population.as_array()[c].morphological_class \
               if population.as_array()[c].morphological_class != 'N/A' \
               else population.as_array()[c].predicted_class \
               for c in neighbours[k]])

    print (len(classes), classes)

    if len(classes) > 0:
        if majority(classes) is not None:
            return majority(classes)
            print (k, majority(classes))
        else:
            # find nearest neighbour
            nn = neighbours[k][np.argmin([get_weight(mst, k, c) for c in neighbours[k]])]
            return population.as_array()[nn].morphological_class
    else:
        nn = neighbours[k][np.argmin([get_weight(mst, k, c) for c in neighbours[k]])]
        return population.as_array()[nn].morphological_class

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
        morphological_class = predict(population, kk, neighbours, mst)
        print (morphological_class)
        population.as_array()[kk].predicted_class = morphological_class
        print(population.as_array()[kk].jname,
              population.as_array()[kk].bname,
              population.as_array()[kk].predicted_class)

    if  colour_by == 'branch':
        facecolors.append(palette[node_coordinates[k][2]])
        edgecolors.append('orange' if kk in mst.longest_path else palette[node_coordinates[k][2]])
    else: # assumes there are only two  cases...
        facecolors.append(palette_morph_class[morphological_class])
        edgecolors.append('black' if predicted else 'orange' \
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

top_cond = np.where((y_coords == np.max(y_coords)))[0][0]
max_y_top_level = y_coords[top_cond]
min_x_top_level = x_coords[top_cond]
min_x = np.min(x_coords)


multi_names_to_annotate = [
    # Leaves
    'J1740+1311', 'J0332+5434',
    # Level 1
    'J2219+4754', 'J1820-0427', 'J1917+1353',
    'J1607-0032', 'J2055+3630', 'J0629+2415']
multi_annotation = {
    # Leaves
    'J1740+1311': {'text':'a', 'location':'top'},
    'J0332+5434': {'text':'b', 'location':'top'},
    # Level 1
    'J2219+4754': {'text':'6', 'location':'left_to_edge'},
    'J1820-0427': {'text':'5', 'location':'left_to_edge'},
    'J1917+1353': {'text':'4', 'location':'left_to_edge'},
    'J1607-0032': {'text':'3', 'location':'left_to_edge'},
    'J2055+3630': {'text':'2', 'location':'left_to_edge'},
    'J0629+2415': {'text':'1', 'location':'left_to_edge'}
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

        ax.text(#x, y,
                 x - phase.size / 2 + x_scaler*0.02,
                 y + y_box_height + (y_box_height/30),
                 "%s" % (pulsar.jname),
                 horizontalalignment='left',
                 verticalalignment='top',
                 fontsize=fontsize,
                 c='white',
                 zorder=500)

        # Annotation (Top)
        if annotate:
            if len(freq_ids_to_include) == 1:
                if pulsar.jname in single_names_to_annotate:
                    if single_annotation[pulsar.jname]['location'] == 'top':
                        ax.text(#x, y,
                                x - phase.size / 2 + x_scaler*0.25,
                                y + y_box_height + (1.5 * y_box_height),
                                "%s" % (single_annotation[pulsar.jname]['text']),
                                horizontalalignment='center',
                                verticalalignment='top',
                                fontsize=20,
                                c='black',
                                zorder=500)
            else:
                if pulsar.jname in multi_names_to_annotate:
                    if multi_annotation[pulsar.jname]['location'] == 'top':
                        ax.text(#x, y,
                                x - phase.size / 2 + x_scaler*0.25,
                                y + y_box_height + (1.5 * y_box_height),
                                "%s" % (multi_annotation[pulsar.jname]['text']),
                                horizontalalignment='left',
                                verticalalignment='top',
                                fontsize=12,
                                c='black',
                                zorder=500)

        if print_bname:
            ax.text(#x, y,
                     x - phase.size / 2 + x_scaler*0.02,
                     y + y_box_height - (y_box_height/10),
                     "%s" % (pulsar.bname),
                     horizontalalignment='left',
                     verticalalignment='top',
                     fontsize=fontsize,
                     c='white',
                     zorder=500)
        ax.text(#x, y,
                 x + phase.size / 2 + x_scaler*0.02,
                 y + y_box_height + (y_box_height/30),
                 "$\phi\in\pm%.1f$" % (phase_thres),
                 horizontalalignment='right',
                 verticalalignment='top',
                 fontsize=fontsize,
                 c='white',
                 zorder=500)
        if colour_by != 'branch':
            ax.text(#x, y,
                     x + phase.size / 2,
                     y + y_box_height - (y_box_height/10),
                     "$%s$" % (pulsar.morphological_code),
                     horizontalalignment='right',
                     verticalalignment='top',
                     fontsize=fontsize,
                     c='white',
                     zorder=500)

        ax.plot(
            np.linspace(x - phase.size / 2,
                        x + phase.size / 2,
                        phase.size
            ),
            (model * (0.1 if _4freqs_ else 0.5)) + y + yy,
            linewidth=profile_linewidth,
            zorder=200,
            color='white'
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
                color=blue_full,
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
#                              marker=None,
#                              ms=0.001,
#                              mfc='black',
                             elinewidth=profile_linewidth,
#                              markersize=0.5,
#                              color='white',
                             zorder=500)


#         if x == min_x_top_level and y == max_y_top_level:
#             print ('in')
#             # Plot zoomed in section
#             axins.plot(
#                 phase,
#                 (profile * (0.1 if _4freqs_ else 0.5)) + y + yy,
# #                 linewidth=profile_linewidth,
# #                 zorder=200,
#                 color='black'
#             )
#             # sub region of to be zoomed
#             x1 = x - phase.size / 2
#             x2 = x + phase.size / 2
#             y1 = y - y_box_height - 0.5
#             y2 = y + y_box_height
#             print (x1, x2, y1, y2)
#             print ()
#             axins.set_xlim(x1, x2)
#             axins.set_ylim(y1, y2)



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

# norm = colors.Normalize(vmin=0, vmax=_distances.max())
norm = colors.LogNorm(vmin=np.nanmin(_distances), vmax=np.nanmax(_distances[np.where(_distances != np.inf)[0]]))


# norm = colors.LogNorm(vmin=_cumulative_distances.min(), vmax=_cumulative_distances.max())

if annotate:
    b_n = 1
for c in connections['child']:
    print (c[-3], np.nanmin(_distances), np.nanmax(_distances[np.where(_distances != np.inf)[0]]))
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
        ax.text(0.01, 0.95 - (0.05 * i),
                '%s' % (k),
                color=palette_morph_class[k],
                horizontalalignment='left',
                verticalalignment='center',
                transform=ax.transAxes)

plt.tight_layout()

s_p = step_pattern if type(step_pattern) == str else 'rabinerJuang'

if colour_by == 'branch':
    if _4freqs_:
        if not contracted:
            if _cond_:
                print ("%s" % phase_thres)
                filename = 'images/tree_sequence_4freqs_phase_%s_%s_%s' % (phase_thres, s_p, stokes_to_include[0])
            else:
                filename = 'images/tree_sequence_4freqs_full_phase_%s_%s' % (s_p, stokes_to_include[0])
        else:
            if _cond_:
                print ("%s" % phase_thres)
                filename = 'images/contr_tree_sequence_4freqs_phase_%s_%s_%s' % (phase_thres, s_p, stokes_to_include[0])
            else:
                filename = 'images/contr_tree_sequence_4freqs_full_phase_%s_%s' % (s_p, stokes_to_include[0])
    else:
        if not contracted:
            if _cond_:
                print ("%s" % phase_thres)
                filename = 'images/tree_sequence_%s_phase_%s_%s_%s' % (freqs_filename[freq_ids_to_include[0]],
                                                                          phase_thres,
                                                                          s_p, stokes_to_include[0])
            else:
                filename = 'images/tree_sequence_%s_full_phase_%s_%s' % (freqs_filename[freq_ids_to_include[0]],
                                                                           step_pattern, stokes_to_include[0])
        else:
            if _cond_:
                print ("%s" % phase_thres)
                filename = 'images/contr_tree_sequence_%s_phase_%s_%s_%s' % (freqs_filename[freq_ids_to_include[0]],
                                                                          phase_thres,
                                                                          s_p, stokes_to_include[0])
            else:
                filename = 'images/contr_tree_sequence_%s_full_phase_%s_%s' % (freqs_filename[freq_ids_to_include[0]],
                                                                           step_pattern, stokes_to_include[0])

if colour_by == 'rankin_class':
    if _4freqs_:
        if not contracted:
            if _cond_:
                print ("%s" % phase_thres)
                filename = 'images/tree_sequence_4freqs_phase_%s_%s_%s_%s' % (phase_thres, s_p, stokes_to_include[0], colour_by)
            else:
                filename = 'images/tree_sequence_4freqs_full_phase_%s_%s_%s' % (s_p, stokes_to_include[0], colour_by)
        else:
            if _cond_:
                print ("%s" % phase_thres)
                filename = 'images/contr_tree_sequence_4freqs_phase_%s_%s_%s_%s' % (phase_thres, s_p, stokes_to_include[0], colour_by)
            else:
                filename = 'images/contr_tree_sequence_4freqs_full_phase_%s_%s_%s' % (s_p, stokes_to_include[0], colour_by)
    else:
        if not contracted:
            if _cond_:
                print ("%s" % phase_thres)
                filename = 'images/tree_sequence_%s_phase_%s_%s_%s_%s' % (freqs_filename[freq_ids_to_include[0]],
                                                                          phase_thres,
                                                                          s_p, stokes_to_include[0], colour_by)
            else:
                filename = 'images/tree_sequence_%s_full_phase_%s_%s_%s' % (freqs_filename[freq_ids_to_include[0]],
                                                                           step_pattern, stokes_to_include[0], colour_by)
        else:
            if _cond_:
                print ("%s" % phase_thres)
                filename = 'images/contr_tree_sequence_%s_phase_%s_%s_%s_%s' % (freqs_filename[freq_ids_to_include[0]],
                                                                          phase_thres,
                                                                          s_p, stokes_to_include[0], colour_by)
            else:
                filename = 'images/contr_tree_sequence_%s_full_phase_%s_%s_%s' % (freqs_filename[freq_ids_to_include[0]],
                                                                           step_pattern, stokes_to_include[0], colour_by)

# plt.savefig(filename + '_%s.png' % state_prefix, dpi=250)
plt.savefig(filename + '_%s.pdf' % state_prefix)
print (filename)

