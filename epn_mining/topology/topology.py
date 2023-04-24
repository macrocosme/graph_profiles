from .graph import Graph
from ..analysis.distance import Distance, crop
from ..preparation.reader import resize_to_N
import numpy as np
from tqdm import tqdm
from scipy.cluster import hierarchy
# from joblib import Parallel, delayed

weighted_distance = lambda local_dists={}, weights={}: np.divide(
    np.sum(np.array([weights[k] * local_dists[k] for k in local_dists.keys()])),
    np.array([weights[k] for k in local_dists.keys()]).sum()
)

def compute_dist(pop,
                 population_graph_indices,
                 graph_population_indices,
                 pairs,
                 distances,
                 stokes_to_include,
                 metric,
                 i, j,
                 freq_ids_to_include,
                 matched_idx,
                 min_snr,
                 penalty,
                 step_pattern,
                 window_type,
                 window_args,
                 open_begin,
                 open_end,
                 distance_only,
                 cropped):
    dist = 0
    freq_n = 0
    for freq in pop[i].observations.keys() if freq_ids_to_include is None else freq_ids_to_include:
        this_found, that_found = True, True
        try:
            this = pop[i].observations[freq]
        except KeyError:
            this_found = False

        try:
            that = pop[j].observations[freq]
        except KeyError:
            that_found = False

        if this_found and that_found:
            this_snr_found, that_snr_found = this.snr > min_snr, that.snr > min_snr

            if this_snr_found:
                if i not in population_graph_indices.keys():
                    population_graph_indices[i] = matched_idx
                    graph_population_indices[matched_idx] = i
                    matched_idx += 1

            if that_snr_found:
                if j not in population_graph_indices.keys():
                    population_graph_indices[j] = matched_idx
                    graph_population_indices[matched_idx] = j
                    matched_idx += 1

            local_dists = 0
            stokes_n = 0

            if this_snr_found and that_snr_found:
                args = {'a_phase': this.phase,
                        'b_phase': that.phase,
                        'a_centroid': this.centroid,
                        'a_fwhm': this.fwhm,
                        'b_centroid': that.centroid,
                        'b_fwhm': that.fwhm,
                        'penalty': penalty,
                        'step_pattern': step_pattern,
                        'window_type': window_type,
                        'window_args': window_args,
                        'open_begin': open_begin,
                        'open_end': open_end,
                        'distance_only': distance_only,
                        'cropped': cropped}

                if 'I' in stokes_to_include:
                    dist += Distance(metric).get_distance(this.stokes_I, that.stokes_I, **args)
                    stokes_n += 1
                if this.stokes_Q is not None and that.stokes_Q is not None:
                    if 'Q' in stokes_to_include:
                        local_dists += Distance(metric).get_distance(this.stokes_Q, that.stokes_Q, **args)
                        stokes_n += 1
                    if 'U' in stokes_to_include:
                        local_dists += Distance(metric).get_distance(this.stokes_U, that.stokes_U, **args)
                        stokes_n += 1
                    if 'L' in stokes_to_include:
                        local_dists += Distance(metric).get_distance(this.stokes_L, that.stokes_L, **args)
                        stokes_n += 1
                    if 'V' in stokes_to_include:
                        local_dists += Distance(metric).get_distance(this.stokes_V, that.stokes_V, **args)
                        stokes_n += 1
                    if 'mean_linear_polarization_degree' in stokes_to_include:
                        local_dists += 0.5 * (this.mean_linear_polarization_degree - \
                                            that.mean_linear_polarization_degree)

                        stokes_n += 1
                    if 'model' in stokes_to_include:
                        if this.model is None:
                            this.set_model()
                        if that.model is None:
                            that.set_model()
                        local_dists += 1.5 * Distance(metric).get_distance(this.model, that.model, **args)
                        stokes_n += 1

                    if stokes_n > 0:
                        local_dists *= 1/stokes_n

                if stokes_n > 0:
                    dist += local_dists
                    freq_n += 1

        if 'distance' in stokes_to_include:
            dist += np.abs(pop[i].distance-pop[j].distance)
            freq_n += 1

        if freq_n > 0:
            dist /= freq_n

            pairs += 1
            distances.append((population_graph_indices[i], population_graph_indices[j], dist))



def complete_undirected_weighted_graph(population,
                        metric='kullback-leibler',
                        min_snr=0,
                        stokes_to_include=['Stokes_I'],
                        weights={'Stokes_I': 1},
                        freq_ids_to_include=None,
                        penalty=None,
                        step_pattern='symmetric2',
                        window_type=None,
                        window_args={},
                        open_begin=False,
                        open_end=False,
                        distance_only=True,
                        cropped=False,
                        verbose=False):
    """Make weighted graph from 2D array.

    Parameters
    ----------
    im:
    distance:

    Returns
    -------
    distances: list
        List of distance measures given as list of weighted path [u,v,w], given
        vertices u and v, with weight w.

    """
    if verbose:
        print('Make weighted graph')

    distances = []

    pop = population.as_array()
    population_graph_indices = {}
    graph_population_indices = {}

    pairs = 0

    matched_idx = 0
    for i in tqdm(range(len(pop)-1)):

        # test with only one band
        # Parallel(n_jobs=len(freq_ids_to_include))(
        #     delayed(compute_dist)(
        #         pop,
        #         population_graph_indices,
        #         graph_population_indices,
        #         pairs,
        #         distances,
        #         stokes_to_include,
        #         metric,
        #         i, j,
        #         freq_ids_to_include,
        #         matched_idx,
        #         min_snr,
        #         penalty,
        #         step_pattern,
        #         window_type,
        #         window_args,
        #         open_begin,
        #         open_end,
        #         distance_only,
        #         cropped
        # ) for j in range(i + 1, len(pop)))

        for j in range(i + 1, len(pop)):
            dist = 0
            freq_n = 0
            for freq in pop[i].observations.keys() if freq_ids_to_include is None else freq_ids_to_include:
                this_found, that_found = True, True
                try:
                    this = pop[i].observations[freq]
                except KeyError:
                    this_found = False

                try:
                    that = pop[j].observations[freq]
                except KeyError:
                    that_found = False

                if this_found and that_found:
                    this_snr_found, that_snr_found = this.snr > min_snr, that.snr > min_snr

                    if this_snr_found:
                        if i not in population_graph_indices.keys():
                            population_graph_indices[i] = matched_idx
                            graph_population_indices[matched_idx] = i
                            matched_idx += 1

                    if that_snr_found:
                        if j not in population_graph_indices.keys():
                            population_graph_indices[j] = matched_idx
                            graph_population_indices[matched_idx] = j
                            matched_idx += 1

                    local_dists = {}
                    stokes_n = 0

                    if this_snr_found and that_snr_found:
                        args = {'a_phase': this.phase,
                                'b_phase': that.phase,
                                'a_centroid': this.centroid,
                                'a_fwhm': this.fwhm,
                                'b_centroid': that.centroid,
                                'b_fwhm': that.fwhm,
                                'penalty': penalty,
                                'step_pattern': step_pattern,
                                'window_type': window_type,
                                'window_args': window_args,
                                'open_begin': open_begin,
                                'open_end': open_end,
                                'distance_only': distance_only,
                                'cropped': cropped}

                        if 'I' in stokes_to_include:
                            if 'I' not in weights:
                                weights['I'] = 1
                            dist += Distance(metric).get_distance(this.stokes_I, that.stokes_I, **args)
                            stokes_n += weights['I']
                        if this.stokes_Q is not None and that.stokes_Q is not None:
                            if 'Q' in stokes_to_include:
                                if 'Q' not in weights:
                                    weights['Q'] = 1
                                local_dists['Stokes_Q'] = Distance(metric).get_distance(this.stokes_Q, that.stokes_Q, **args)
                                stokes_n += weights['Q']
                            if 'U' in stokes_to_include:
                                if 'U' not in weights:
                                    weights['U'] = 1
                                local_dists['Stokes_U'] = Distance(metric).get_distance(this.stokes_U, that.stokes_U, **args)
                                stokes_n += weights['U']
                            if 'L' in stokes_to_include:
                                if 'L' not in weights:
                                    weights['L'] = 1
                                local_dists['Stokes_L'] = Distance(metric).get_distance(this.stokes_L, that.stokes_L, **args)
                                stokes_n += weights['L']
                            if 'V' in stokes_to_include:
                                if 'V' not in weights:
                                    weights['V'] = 1
                                local_dists['Stokes_V'] = Distance(metric).get_distance(this.stokes_V, that.stokes_V, **args)
                                stokes_n += weights['V']
                            if 'mean_linear_polarization_degree' in stokes_to_include:
                                if 'mean_linear_polarization_degree' not in weights:
                                    weights['mean_linear_polarization_degree'] = 1
                                local_dists['mean_linear_polarization_degree'] = Distance('L2').get_distance(this.mean_linear_polarization_degree,
                                                                                                            that.mean_linear_polarization_degree,
                                                                                                            **args)
                                stokes_n += weights['mean_linear_polarization_degree']
                            if 'mean_circular_polarization_degree' in stokes_to_include:
                                if 'mean_circular_polarization_degree' not in weights:
                                    weights['mean_circular_polarization_degree'] = 1
                                local_dists['mean_circular_polarization_degree'] = Distance('L2').get_distance(this.mean_circular_polarization_degree,
                                                                                                              that.mean_circular_polarization_degree,
                                                                                                              **args)
                                stokes_n += weights['mean_circular_polarization_degree']

                        if 'model' in stokes_to_include:
                            if this.model is None:
                                this.set_model()
                            if that.model is None:
                                that.set_model()

                            if 'model' not in weights:
                                weights['model'] = 1

                            try:
                                local_dists['model'] += Distance(metric).get_distance(this.model, that.model, **args)
                            except KeyError:
                                local_dists['model'] = Distance(metric).get_distance(this.model, that.model, **args)
                            stokes_n += weights['model']

                            # if stokes_n > 0:
                            #     local_dists *= 1/stokes_n

                        if stokes_n > 0:
                            # dist += local_dists
                            freq_n += 1

            if 'distance' in stokes_to_include:
                dist += np.abs(pop[i].distance-pop[j].distance)
                freq_n += 1

            if freq_n > 0:
                dist /= freq_n

                pairs += 1

                if stokes_n > 0:
                    for k in local_dists.keys():
                        local_dists[k] *= 1/stokes_n

                distances.append((population_graph_indices[i],
                                  population_graph_indices[j],
                                  weighted_distance(local_dists, weights),
                                  local_dists,
                                  weights,))

    print ('complete graph: |e| =', pairs)

    return distances, population_graph_indices, graph_population_indices

def distances_as_dataframe(distances):
    from pandas import DataFrame
    return DataFrame(distances, columns=['u', 'v', 'w'])

def spanning_tree(distances, type='minimum', verbose=False):
    """Build graph and compute its (minimum) spanning tree.

    Parameters
    ----------
    distances: list
        Array of n*(u, v, w)
    verbose:bool
        Print function information.

    Returns
    -------
    graph: Graph

    """
    if verbose:
        print('Compute minimum spanning tree')

    assert type in ['minimum', 'maximum'], print ("type must be one of: ['minumum', 'maximum'].")
    if type == 'maximum':
        reverse = True
    else:
        reverse = False

    graph = Graph()
    for u, v, w, _, _ in distances:
        graph.add_edge(u, v, w)

    graph.spanning_tree(reverse=reverse)

    return graph

def longest_path(graph, population, graph_population_indices, verbose=False):
    if verbose:
        print("Compute longest path in minimum spanning tree")

    sequence_indices = graph.get_longest_path()
    sequence_pulsars = [population[graph_population_indices[i]] for i in sequence_indices]

    return sequence_pulsars, sequence_indices, graph

def cluster_mst(graph, population, graph_population_indices,
                stokes_to_include='I', frequency_range_id=2,
                common_shape=1024, scalor=5, k=4,
                method='KMedoids',
                state_prefix='',
                verbose=False):
    assert method in ['KMedoids', 'HierarchicalTree', 'dendrogram'], \
        'Clustering method must be one of ["KMedoids", "HierarchicalTree", "dendrogram"]'

    def get_profile(this, stokes_to_include):
        if 'I' in stokes_to_include:
            return this.stokes_I
        if this.stokes_Q is not None:
            if 'Q' in stokes_to_include:
                return this.stokes_Q
            if 'U' in stokes_to_include:
                return this.stokes_U
            if 'L' in stokes_to_include:
                return this.stokes_L
            if 'V' in stokes_to_include:
                return this.stokes_V
            if 'linear_polarization_degree' in stokes_to_include:
                return this.linear_polarization_degree

    if verbose:
        print ("Clustering minimum spanning tree into %d clusters" % k)

    vertices_indices = np.sort(
        np.unique(
            np.append(
                np.unique(np.asarray(graph.mst).T[0]),
                np.unique(np.asarray(graph.mst).T[1])
            )
        )
    ).astype(int)

    # Build time series matrix
    mst_profiles = np.zeros((np.max(vertices_indices)+1, common_shape))
    vertices_labels = []
    for i in vertices_indices:
        profile = population[graph_population_indices[i]].observations[frequency_range_id]
        vertices_labels.append(
            "%s (%s, %s)" % (
                population[graph_population_indices[i]].jname,
                population[graph_population_indices[i]].observations[frequency_range_id].epn_reference_code,
                int(population[graph_population_indices[i]].observations[frequency_range_id].frequency)
            )
        )
        mst_profiles[i] = resize_to_N(
            crop(
                profile.get_property(stokes_to_include),
                profile.centroid,
                profile.fwhm,
                common_shape=common_shape,
                scalor=scalor
            ),
            N=common_shape
        )

    if method == 'KMedoids':
        model = clustering.KMedoids(dtw.distance_matrix_fast, {}, k=k)
    elif method == 'HierarchicalTree':
        model = clustering.HierarchicalTree(dists_fun=dtw.distance_matrix_fast,
                                            dists_options={})
    elif method == 'dendrogram':
        model = hierarchy.linkage(
            mst_profiles,
            metric=dtw.distance_fast,
            optimal_ordering=True,
            # method='weighted'
        )

    if method in ['KMedoids', 'HierarchicalTree']:
        cluster_idx = model.fit(mst_profiles)

        if method == 'HierarchicalTree':
            model.plot(
                "images/cluster_ht_%s_%s.pdf" % (state_prefix, stokes_to_include),
                # ts_height=200,
                # show_tr_label=True,
                # show_ts_label=True
                cmap='gnuplot'
            )

    return model, vertices_indices, vertices_labels
