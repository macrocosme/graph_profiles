from __future__ import division, print_function

import numpy as np

import argparse
import os

from .graph import Graph
from .data_cleaning import (
    read_metadata_file,
    pool_data,
    resize_and_roll_arrays,
    remove_empty_rows,
    fwhm,
    cut_10,
    normalize_zero_one
)
from .distance import Distance
from .plotting import plot_results

from dtaidistance import dtw

import networkx as nx
import cv2

from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib import rc
import matplotlib.image as mpimg

import scipy.signal
import scipy.stats

rc('font', size=12)
rc('axes', titlesize=14)
rc('axes', labelsize=14)


def compute_fwhm(im, freqs=[], spline=False, save=False, verbose=False):
    if verbose:
        print ('Compute FWHM')

    im_fwhm = [fwhm(i, return_dist=True) for i in im]
    if save:
        l, h = freqs
        folder = '%d-%d' % (l,h)
        spline_folder = 'spline_interp' if spline else 'no_interp'
        os.system('mkdir npy/%s' % folder)
        os.system('mkdir npy/%s/%s' % (folder, spline_folder))
        spline_folder = 'spline_interp' if spline else 'no_interp'
        np.save('npy/%s/%s/epn_new_image_fwhm.npy' % (folder, spline_folder), im_fwhm)
    return  im_fwhm

def compute_npeaks(im, freqs=[], spline=False, save=False, verbose=False):
    if verbose:
        print ('Compute N Peaks')

    im_peaks = [len(scipy.signal.find_peaks(i)[0]) for i in im]
    if save:
        l, h = freqs
        folder = '%d-%d' % (l,h)
        spline_folder = 'spline_interp' if spline else 'no_interp'
        os.system('mkdir npy/%s' % folder)
        os.system('mkdir npy/%s/%s' % (folder, spline_folder))
        spline_folder = 'spline_interp' if spline else 'no_interp'
        np.save('npy/%s/%s/epn_new_image_peaks.npy' % (folder, spline_folder), im_peaks)
    return  im_peaks

def make_weighted_graph(im, metric='kullback-leibler', combine=False, im_fwhm=None, im_npeaks=None, metrics=None, transform=None, verbose=False):
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

    # assert metric in METRICS, "metric should be one of %s" % (str(METRICS))

    for i in tqdm(range(len(im)-1)):
        for j in range(i + 1, len(im)):
            if not combine:
                if metric is 'width':
                    dist = np.abs(im_fwhm[i]-im_fwhm[j])
                elif metric is 'zscore':
                    dist = scipy.stats.zscore([im[i], im[j]], axis=1)
                elif metric is 'npeaks':
                    dist = np.abs(im_npeaks[i]-im_npeaks[j])
                elif metric is 'DTW':
                    dist = dtw.distance_fast(im[i].astype(np.double),
                                             im[j].astype(np.double),
                                             window=np.max(
                                                       [im_fwhm[i],
                                                       im_fwhm[j]]
                                                   ),
                                                   psi=2
                                            )
                    # dist2 = dtw.distance_fast(im[i].astype(np.double), im[j][::-1].astype(np.double))
                    # dist = np.min([dist1, dist2])
                else:
                    dist = Distance(metric).get_distance(im[i], im[j], transform=transform)
            else:
                # dist = Distance(metric).get_distance(im[i], im[j], transform=transform)
                dist = np.abs(im_fwhm[i]-im_fwhm[j])
                dist *= Distance(metric).get_distance(im[i], im[j], transform=transform)
                # dist = -999
                # for metric in metrics:
                #     if dist == -999:
                #         dist = Distance(metric).get_distance(im[i], im[j], transform=transform)
                #     else:
                #         dist *= Distance(metric).get_distance(im[i], im[j], transform=transform)
                #     # dist = dist/len(metrics)
            distances.append((i, j, dist))

    return distances

def compute_minimum_spanning_tree(distances, verbose=False):
    """Build graph and compute its minimum spanning tree.

    Parameters
    ----------
    im: numpy.array
        2D array representing profiles stacked vertically (each row is a profile)
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

    graph = Graph()
    for u, v, w in distances:
        graph.add_edge(u, v, w)

    graph.kruskal_minimum_spanning_tree()

    return graph

def compute_maximum_spanning_tree(distances, verbose=False):
    """Build graph and compute its minimum spanning tree.

    Parameters
    ----------
    im: numpy.array
        2D array representing profiles stacked vertically (each row is a profile)
    distances: list
        Array of n*(u, v, w)
    verbose:bool
        Print function information.

    Returns
    -------
    graph: Graph

    """
    if verbose:
        print('Compute maximum spanning tree')

    graph = Graph()
    for u, v, w in distances:
        graph.add_edge(u, v, w)

    graph.kruskal_minimum_spanning_tree(maximum=True)

    return graph

def compute_minimum_spanning_tree_networkx(im, distances, verbose=False):
    """Build graph and compute its minimum spanning tree.

    Parameters
    ----------
    im: numpy.array
        2D array representing profiles stacked vertically
        (each row is a profile)
    distances: list
        Array of n*(u, v, w)
    verbose:bool
        Print function information.

    Returns
    -------
    mst : NetworkX Graph
       A minimum spanning tree or forest.

    """
    if verbose:
        print('Compute minimum spanning tree')

    graph = nx.Graph()
    graph.add_weighted_edges_from(distances)

    mst = nx.algorithms.tree.minimum_spanning_tree(graph, algorithm='kruskal')

    return mst

def get_filename(nickname='storm', freqs=[], spline=False):
    if nickname == 'gerlumph':
        o_image_full = mpimg.imread('input_images/lc.jp2').copy()
        image = o_image_full.copy()

    if nickname == 'epn_pulse_profiles_unsorted':
        if len(freqs) == 0:
            o_image_full, o_image_full_znorm, df, df_indices = np.load('npy/1374/epn_profiles.npy', allow_pickle=True)
        else:
            l, h = freqs
            spline_folder = 'spline_interp' if spline else 'no_interp'
            print ('reading', l, h)
            o_image_full, o_image_full_znorm, df, df_indices = np.load('npy/%d-%d/%s/epn_profiles.npy' % (l, h, spline_folder), allow_pickle=True)
        image = o_image_full.copy()
        image_znorm = o_image_full_znorm.copy()
        return o_image_full, image, image_znorm, df, df_indices

    if nickname == 'storm':
        o_image_full = cv2.imread("input_images/electricblackpool.jpg")[:, :, 2]
        image = cv2.imread("electricblackpool.jpg")[:, :, 2]

    if nickname == 'circle':
        o_image_full = cv2.imread("input_images/circle.png")[:, :, 2]
        image = cv2.imread("circle.png")[:, :, 2]

    if nickname == 'butterfly':
        o_image_full = cv2.imread("input_images/butterfly.jpg")[:, :, 2]
        image = cv2.imread("butterfly.jpg")[:, :, 2]

    if nickname == 'noise':
        o_image_full = np.random.rand(1000, 1000)
        image = o_image_full.copy()

    if nickname == 'shackles':
        o_image_full = cv2.imread("input_images/shackles.jpg")[:, :, 2]
        image = cv2.imread("shackles.jpg")[:, :, 2]

    if nickname == 'dv':
        o_image_full = cv2.imread("input_images/dv.png")[:, :, 2]
        image = cv2.imread("input_images/dv.png")[:, :, 2]

    return o_image_full, image

def append_image(new_image, visited, original_image, vertex_id):
    new_image.append(original_image[vertex_id])
    visited.append(vertex_id)

    return new_image, visited

def visit_leafs(new_image, visited, original_image, df_mst, root):
    subset = df_mst.loc[(df_mst['u'] == root) | (df_mst['v'] == root)]
    leafs = []
    weights = []
    for hit_uvw in subset.sort_values(by=['w']).values:
        if (np.max(subset.sort_values(by=['w']).groupby('w')['w'].nunique().values) > 1):
            print(subset.sort_values(by=['w']).groupby('w')['w'].nunique())
        hit = hit_uvw[1] if root == hit_uvw[0] else hit_uvw[0]
        if hit not in visited:
            leafs.append(hit)
            weights.append(hit_uvw[2])

    for leaf in leafs:
        new_image, visited = append_image(new_image, visited, original_image, int(leaf))
        new_image, visited, weights = visit_leafs(new_image, visited, original_image, df_mst, int(leaf))

    return new_image, visited, weights

def construct_image(graph, original_image, verbose=False):
    if verbose:
        print("Construct image")

    root, leaf = graph.define_path_starting_vertices()

    # Find neighbours of root
    new_image = []
    visited = []
    df_mst = graph.as_dataframe()

    new_image, visited = append_image(new_image, visited, original_image, root)
    new_image, visited, weights = visit_leafs(new_image, visited, original_image, df_mst, root)

    return new_image, visited, weights

def construct_longest_path_image(graph, original_image, stretch=False, stretch_factor=5, verbose=False):
    if verbose:
        print("Construct image")

    root, leaf = graph.define_path_starting_vertices()

    new_image = []
    sequence = graph.find_longest_path(root)
    for i in sequence:
        if stretch:
            for j in range(stretch_factor):
                new_image.append(original_image[i])
        else:
            new_image.append(original_image[i])

    return new_image, sequence

def test(verbose=True):
    df = read_metadata_file(verbose=verbose)
    image = pool_data(df, verbose=verbose)
    del df
    image = remove_empty_rows(image)
    image = resize_and_roll_arrays(image)
    distances = make_weighted_graph(image, metric='euclidean', verbose=verbose)
    graph_shuffled = compute_minimum_spanning_tree(distances, verbose=verbose)
    new_image = construct_image(graph_shuffled, image)
    plot_results(image, new_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path',
                        help="Base path to input file",
                        type=str,
                        default='www.epta.eu.org/epndb/psrfits',
                        required=False)
    parser.add_argument('--input_filename',
                        help="Name of the input file",
                        type=str,
                        default='file_log.txt',
                        required=False)
    parser.add_argument('--metric',
                        help="Name of the input file",
                        type=str,
                        default='cross-correlation',
                        required=False)
    parser.add_argument('--verbose',
                        help="Verbose mode",
                        type=bool,
                        default=False,
                        required=False)

    args = parser.parse_args()

    df = read_metadata_file(base_path=args.base_path, input_filename=args.input_filename, verbose=args.verbose)
    im = pool_data(df, verbose=args.verbose)
    graph = compute_minimum_spanning_tree(args.metric, verbose=args.verbose)
    # df_mst = graph_mst_to_dataframe(graph)
    # sequence = construct_sequence(im, df_mst, verbose=args.verbose)
