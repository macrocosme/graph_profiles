#!/usr/bin/env python3

import argparse
import pickle
import os
import copy
import time
import numpy as np

from .preparation import epn
from .topology import topology
from .preparation.pulsar import Population

from .utils.io import load, save, set_state_name, check_underscore

def load_states(state_prefix):
    epn_metadata = load('epn_metadata', state_prefix)
    population = load('population', state_prefix)
    distances = load('distances', state_prefix)
    mst = load('mst', state_prefix)
    if mst is not None:
        mst = mst.as_dataframe()
    sequence_indices = load('sequence_indices', state_prefix)
    sequence_population = load('sequence_population', state_prefix)
    elongation_dict = load('elongation_dict')
    population_graph_indices = load('population_graph_indices', state_prefix)
    graph_population_indices = load('graph_population_indices' , state_prefix)
    models_clustering = load('models_clustering' , state_prefix)
    models_vertices_indices = load('models_vertices_indices' , state_prefix)
    models_vertices_labels = load('models_vertices_labels' , state_prefix)

    return epn_metadata, population, distances, mst, \
            sequence_indices, sequence_population, elongation_dict, \
            population_graph_indices, graph_population_indices, \
            models_clustering, models_vertices_indices, models_vertices_labels

def check_stokes_states_exist(metric, stokes_to_include, freq_ids_to_include, reference, folder='states/'):
    print ('checking if stokes states exist')
    for letter in stokes_to_include:
        prefix = set_state_name('', [metric, letter, freq_ids_to_include, reference])
        if not os.path.exists(
            folder + check_underscore(prefix) + 'distances.pickle'
        ):
            return False
    return True

def average_stokes_distances(metric, stokes_to_include, average_stokes_distances, reference, folder='states/'):
    _distances = []
    for letter in stokes_to_include:
        print ('loading', set_state_name('', [metric, letter, average_stokes_distances, reference]))
        stokes_dist = load('distances', set_state_name('', [metric, letter, average_stokes_distances, reference]), folder=folder)
        _distances.append(stokes_dist)
        # slow IO access a little; otherwise it crashes...
        time.sleep(2)

    distances = []

    n = 1/len(_distances)
    for i in range(len(_distances[0])):
        u = _distances[0][i][0]
        v = _distances[0][i][1]
        w = 0
        for j in range(len(_distances)):
            w += _distances[j][i][2]
        w *= n

        distances.append((u, v, w))
    return distances

def get_epn_sequence(reference=None,
                     exclude_references=None,
                     stokes=None, # filter metadata with this code
                     stokes_to_include='I',
                     freq_ids_to_include=None,
                     min_snr=0,
                     state_store=True,
                     state_prefix='',
                     state_reload=False,
                     skip_steps=[],
                     shift=True,
                     normalize=True,
                     remove_baseline=False,
                     resize=False,
                     common_shape=1024,
                     cropped=False,
                     metric='DTW',
                     penalty=None,
                     clustering_method='KMedoids',
                     input_type='json',
                     k_clusters=[5],
                     verbose=True):

    assert clustering_method in ['KMedoids', 'HierarchicalTree', 'dendrogram'], \
        'clustering_method must be one of ["KMedoids", "HierarchicalTree", "dendrogram"]'

    # Reload a previous variable(s) state(s) from pickle file(s)
    if state_reload or len(skip_steps) > 0:
        print ('reload states')
        epn_metadata, population, distances, mst, \
        sequence_indices, sequence_population, elongation_dict, \
        population_graph_indices, graph_population_indices, \
        models_clustering, models_vertices_indices, models_vertices_labels = load_states(state_prefix)
    else:
        elongation_dict = None

    # If not reloading previous state(s), run function(s)
    if (not state_reload or state_reload is None) and ('load_epn_metadata' not in skip_steps):
        try:
            del epn_metadata
        except:
            pass
        epn_metadata = epn.load_epn_metadata(reference=reference,
                                             exclude_references=exclude_references,
                                             stokes=stokes,
                                             input_type=input_type,
                                             verbose=verbose)
        if state_store:
            save('epn_metadata', epn_metadata, state_prefix=state_prefix)

    if (not state_reload or population is None) and ('load_epn_data' not in skip_steps):
        try:
            del population
        except:
            pass
        population, epn_metadata  = epn.load_epn_data(epn_metadata,
                                                      shift=shift,
                                                      normalize=normalize,
                                                      remove_baseline=remove_baseline,
                                                      resize=resize,
                                                      common_shape=common_shape,
                                                      verbose=verbose)
        if state_store:
            save('epn_metadata', epn_metadata, state_prefix=state_prefix)
            save('population', population, protocol=pickle.DEFAULT_PROTOCOL, state_prefix=state_prefix)

    if (not state_reload or distances is None) and ('complete_undirected_weighted_graph' not in skip_steps):
        try:
            del distances
        except:
            pass
        if len(stokes_to_include) > 1 and check_stokes_states_exist(metric,
                                                                    stokes_to_include,
                                                                    freq_ids_to_include,
                                                                    reference):
            #compute mean from 'distance' state files
            print ('average_stokes_distances', stokes_to_include)
            distances = average_stokes_distances(metric, stokes_to_include, freq_ids_to_include, reference)
        else:
            #compute all distances
            print ('complete_undirected_weighted_graph', stokes_to_include)
            distances, population_graph_indices, graph_population_indices = topology.complete_undirected_weighted_graph(population,
                                                                    metric=metric,
                                                                    stokes_to_include=stokes_to_include,
                                                                    freq_ids_to_include=freq_ids_to_include,
                                                                    min_snr=min_snr,
                                                                    penalty=penalty,
                                                                    cropped=cropped,
                                                                    verbose=verbose)
        if state_store:
            save('distances', distances, state_prefix=state_prefix)
            save('population_graph_indices', population_graph_indices, state_prefix=state_prefix)
            save('graph_population_indices', graph_population_indices, state_prefix=state_prefix)

    if (not state_reload or mst is None) and ('spanning_tree' not in skip_steps):
        try:
            del mst
        except:
            pass
        mst = topology.spanning_tree(distances, type='minimum', verbose=verbose)
        if state_store:
            save('mst', mst, state_prefix=state_prefix)

    if (not state_reload or sequence_population is None or sequence_indices is None) and ('longest_path' not in skip_steps):
        try:
            del sequence_population, sequence_indices
        except:
            pass
        sequence_population, sequence_indices, mst = topology.longest_path(mst,
                                                                      population.as_array(),
                                                                      graph_population_indices,
                                                                      verbose=verbose)

        elongation, normalized_elongation  =  mst.elongation()

        if elongation_dict is None:
            elongation_dict = {}
        elongation_dict[state_prefix] = {'elongation' : elongation,
                                         'normalized_elongation' : normalized_elongation,
                                         'length': mst.length(),
                                         'half_width': mst.estimate_half_width(),
                                         'longest_path' : len(sequence_population),
                                         'N' : mst.V}

        if state_store:
            save('sequence_population', sequence_population, protocol=pickle.DEFAULT_PROTOCOL, state_prefix=state_prefix)
            save('sequence_indices', sequence_indices, state_prefix=state_prefix)
            # save('mst', mst, state_prefix=state_prefix)
            save('elongation_dict', elongation_dict)
            save('farness', mst.farness, state_prefix=state_prefix)

    if (not state_reload or sequence_population is None or sequence_indices is None) and ('clustering' not in skip_steps):
        models_clustering = {}
        models_vertices_indices = {}
        models_vertices_labels = {}

        for freq in freq_ids_to_include:
            if clustering_method == 'KMedoids':
                for k_cluster in k_clusters:
                    models_clustering[k_cluster], models_vertices_indices[k_cluster], models_vertices_labels[k_cluster] = topology.cluster_mst(mst,
                                     population.as_array(),
                                     graph_population_indices,
                                     stokes_to_include=stokes_to_include,
                                     frequency_range_id=freq,
                                     k=k_cluster,
                                     method=clustering_method,
                                     state_prefix=state_prefix,
                                     verbose=verbose)
            elif clustering_method in ['HierarchicalTree', 'dendrogram']:
                models_clustering[clustering_method], models_vertices_indices[clustering_method], models_vertices_labels[clustering_method] = topology.cluster_mst(mst,
                                 population.as_array(),
                                 graph_population_indices,
                                 stokes_to_include=stokes_to_include,
                                 frequency_range_id=freq,
                                 method=clustering_method,
                                 state_prefix=state_prefix,
                                 verbose=verbose)
            if state_store:
                save('models_clustering', models_clustering, protocol=pickle.DEFAULT_PROTOCOL, state_prefix=state_prefix)
                save('models_vertices_indices', models_vertices_indices, state_prefix=state_prefix)
                save('models_vertices_labels', models_vertices_labels, state_prefix=state_prefix)

    return population, sequence_population, sequence_indices, mst, elongation_dict, models_clustering, models_vertices_indices, graph_population_indices

def sequence_clusters(population,
                      models_clustering,
                      graph_population_indices,
                      models_vertices_indices,
                      metric='DTW',
                      stokes_to_include='I',
                      freq_ids_to_include=None,
                      min_snr=0,
                      k_clusters=[5],
                      state_prefix='',
                      state_store=False,
                      verbose=False):
    if verbose:
        print ('Sequencing clusters')

    clusters_populations = {}
    distances = {}
    mst = {}
    population_graph_indices, _graph_population_indices = {}, {}
    sequence_population, sequence_indices = {}, {}
    elongation, normalized_elongation = {}, {}
    elongation_dict = {}

    for k_cluster in k_clusters:
        clusters_populations[k_cluster] = {}
        distances[k_cluster] = {}
        mst[k_cluster] = {}
        population_graph_indices[k_cluster], _graph_population_indices[k_cluster] = {}, {}
        sequence_population[k_cluster], sequence_indices[k_cluster] = {}, {}
        elongation[k_cluster], normalized_elongation[k_cluster] = {}, {}
        k_cluster_state_prefix = set_state_name(state_prefix, ['kcluster', str(k_cluster)])

        # Create a population for each cluster stored in models_clustering for the k_cluster case
        for i, k in enumerate(sorted(models_clustering[k_cluster].cluster_idx,
                                     key=lambda k: len(models_clustering[k_cluster].cluster_idx[k]),
                                     reverse=True)):
            for j in models_clustering[k_cluster].cluster_idx[k]:
                if k not in clusters_populations[k_cluster].keys():
                    clusters_populations[k_cluster][k] = Population()

                pulsar = population.as_array()[graph_population_indices[j]]
                clusters_populations[k_cluster][k].add_pulsar(pulsar)

        # Now for each cluster, run sequencing if there are enough pulsars in the population to do so
        for i, k in enumerate(sorted(models_clustering[k_cluster].cluster_idx,
                                     key=lambda k: len(models_clustering[k_cluster].cluster_idx[k]),
                                     reverse=True)):

            if clusters_populations[k_cluster][k].as_array().size > 2:

                distances[k_cluster][k], population_graph_indices[k_cluster][k], _graph_population_indices[k_cluster][k] = topology.complete_undirected_weighted_graph(clusters_populations[k_cluster][k],
                                                            metric=metric,
                                                            stokes_to_include=stokes_to_include,
                                                            freq_ids_to_include=freq_ids_to_include,
                                                            min_snr=min_snr,
                                                            verbose=False)
                mst[k_cluster][k] = topology.spanning_tree(distances[k_cluster][k], type='minimum', verbose=False)

                sequence_population[k_cluster][k], sequence_indices[k_cluster][k], \
                mst[k_cluster][k] = topology.longest_path(mst[k_cluster][k],
                                                       clusters_populations[k_cluster][k].as_array(),
                                                       _graph_population_indices[k_cluster][k],
                                                       verbose=False)

                elongation[k_cluster][k], normalized_elongation[k_cluster][k]  =  mst[k_cluster][k].elongation()

                cluster_state_prefix = set_state_name(k_cluster_state_prefix, ['cluster_id', str(k)])

                # Save elongation info for cluster graph
                elongation_dict[cluster_state_prefix] = {'elongation' : elongation[k_cluster][k],
                                                         'normalized_elongation' : normalized_elongation[k_cluster][k],
                                                         'length': mst[k_cluster][k].length(),
                                                         'half_width': mst[k_cluster][k].estimate_half_width(),
                                                         'longest_path' : len(sequence_population[k_cluster][k]),
                                                         'N' : mst[k_cluster][k].V}

                if state_store:
                    print ('Saving cluster sequence %s states.' % (cluster_state_prefix))
                    save('population', clusters_populations[k_cluster][k], protocol=pickle.DEFAULT_PROTOCOL, state_prefix=cluster_state_prefix)
                    save('distances', distances[k_cluster][k], state_prefix=cluster_state_prefix)
                    save('population_graph_indices', population_graph_indices[k_cluster][k], state_prefix=cluster_state_prefix)
                    save('graph_population_indices', _graph_population_indices[k_cluster][k], state_prefix=cluster_state_prefix)
                    save('mst', mst[k_cluster][k], state_prefix=cluster_state_prefix)
                    save('sequence_population', sequence_population[k_cluster][k], protocol=pickle.DEFAULT_PROTOCOL, state_prefix=cluster_state_prefix)
                    save('sequence_indices', sequence_indices[k_cluster][k], state_prefix=cluster_state_prefix)
                    save('elongation_dict', elongation_dict)

    # return population, distances, population_graph_indices, graph_population_indices, mst, sequence_population, sequence_indices, elongation_dict, farness

# TODO: check all necessary options for latest version
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference',
                        help="Use data from a specific reference only (ref. code from EPN). (Default: None)",
                        type=str,
                        default=None)
    parser.add_argument('--state_store',
                    help="Store state variables to pickle file (ignored if state_reload is True). (Default: True)",
                    type=bool,
                    default=True)
    parser.add_argument('--state_reload',
                        help="Reload previous state variables from pickle file.  (Default: False)",
                        type=bool,
                        default=False)
    parser.add_argument('--shift',
                    help="Shift profile centroid to center (relative to Stokes I's centroid). (Default: True)",
                    type=bool,
                    default=True)
    parser.add_argument('--normalize',
                        help="Normalize data to [0, 1] range. Stokes QUVL are normalized relative to I. (Default: True)",
                        type=bool,
                        default=True)
    parser.add_argument('--remove_baseline',
                        help="Remove baseline. (Default: False)",
                        type=bool,
                        default=False)
    parser.add_argument('--verbose',
                        help="Print processing steps information (Default: True)",
                        type=bool,
                        default=True)

    args = parser.parse_args()

    sequence_population, sequence_indices = get_epn_sequence(
                     reference=args.reference,
                     state_store=args.state_store,
                     state_reload=args.state_reload,
                     shift=args.shift,
                     normalize=args.normalize,
                     remove_baseline=args.remove_baseline,
                     verbose=args.verbose,
                     )
