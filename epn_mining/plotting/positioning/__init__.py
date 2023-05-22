import copy
from scipy.stats import norm
from matplotlib import cm
cmap = cm.get_cmap('cubehelix').reversed()

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
                     x_scaler = 1,
                     y_scaler = 1,
                     x_parent=False,
                     verbose=False):

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
        set_square_connections(node_coordinates, graph, v, n, x_parent, connections, w_start, levels[n])

    return max_x
