"""Distance class"""

from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance, entropy
# from pywavelets import dwt
# from dtaidistance import dtw
from .stats import fwhm, profile_as_distribution as distribution
from ..preparation.reader import resize_to_N
from numpy import histogram, fft, correlate
import numpy as np
from dtw import dtw

def check_neg(v):
    if v < 0:
        return 0, np.abs(v)
    else:
        return v, 0

def check_bound(s, e):
    if s < e:
        return s, e
    elif s > e:
        return e, s
    else:
        return s, e+1

def check_min_max(s, e, shape):
    if s < 0:
        s = 0
    if e > shape-1:
        e = shape-1
    return s, e

def crop(a, a_centroid, a_fwhm, common_shape=1024, scalor=5):
    a_s, a_e = int(a_centroid - (scalor * (a_fwhm // 2))), \
               int(a_centroid + (scalor * (a_fwhm // 2)))

    a_s, as_pad = check_neg(a_s)
    a_e, ae_pad = check_neg(a_e)

    a_s, a_e = check_bound(a_s, a_e)

    # Compare shape
    a_cropped = np.concatenate((
        np.zeros(as_pad),
        a[a_s : a_e],
        np.zeros(ae_pad)

    ))

    return a_cropped

class Distance:
    scipy_prebuilt_metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean',
                      'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching', 'minkowski',
                      'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
                      'wminkowski', 'yule']
    metrics = ['kl_div', 'cross-correlation', 'entropy', 'kullback-leibler', 'wasserstein', 'L2', 'DTW', 'width', 'shape'] + scipy_prebuilt_metrics

    def __init__(self, metric='cross-correlation'):
        assert metric in self.metrics, "metric should be one of %s" % (str(self.metrics))

        self.metric = metric

    def kl_divergence(self, p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    def get_distance(self, a, b,
                           a_centroid=None, a_fwhm=None,
                           b_centroid=None, b_fwhm=None,
                           common_shape=1024,
                           transform=None,
                           **kwargs):
        # Pre-process?
        if transform != None:
            if transform == 'fft':
                a = fft.fft(a)
                b = fft.fft(b)

        # Compute distance between sets
        if self.metric in self.scipy_prebuilt_metrics:
            return self.scipy_distance(a, b, self.metric)[0]

        if self.metric == 'cross-correlation':
            return self.cross_correlation(a, b)[0]

        if self.metric == 'wasserstein':
            return self.wasserstein_distance(a, b)

        if self.metric == 'kullback-leibler':
            return self.entropy(a, b)

        if self.metric == 'L2':
            return np.linalg.norm(a-b)

        if self.metric == 'shape':
            return self.shape(a, b,
                              a_centroid, a_fwhm,
                              b_centroid, b_fwhm,
                              common_shape,
                              cropped=kwargs['cropped'] if 'cropped' in kwargs.keys() else False)

        if self.metric == 'kl_div':
            a = np.asarray(a)
            if np.min(a) < 0:
                a = a - np.min(a)
                a = a + 0.000001

            b = np.asarray(b)
            if np.min(b) < 0:
                b = b - np.min(b)
                b = b + 0.000001

            a = a/a.sum()
            b = b/b.sum()

            kl = np.min([
                self.kl_divergence(a, b),
                self.kl_divergence(b, a),
                self.kl_divergence(a, b[::-1]),
                self.kl_divergence(b, a[::-1]),
            ])
            return kl

        if self.metric == 'width':
            return np.abs(fwhm(a)-fwhm(b))

        if self.metric == 'DTW':
            cropped = False
            if 'cropped' in kwargs.keys():
                if kwargs['cropped']:
                    a_cropped = crop(a,
                                     a_centroid,
                                     a_fwhm,
                                     common_shape=1024,
                                     scalor=5)

                    b_cropped = crop(b,
                                     b_centroid,
                                     b_fwhm,
                                     common_shape=1024,
                                     scalor=5)
                    cropped = True

            # return dtw.distance_fast(a_cropped.astype(np.double) if cropped else a.astype(np.double),
            #                          b_cropped.astype(np.double) if cropped else b.astype(np.double),
            #                          # window=int(0.75*a.shape[0]),
            #                          penalty=kwargs['penalty'] if 'penalty' in kwargs.keys() else None,
            #                          # psi=2
            #                          )
            if kwargs['step_pattern'] != 'asymmetric':
                d = dtw(a_cropped if cropped else a,
                        b_cropped if cropped else b,
                        keep_internals=False,
                        distance_only=kwargs['distance_only'],
                        step_pattern=kwargs['step_pattern'],
                        window_type=kwargs['window_type'],
                        window_args=kwargs['window_args'],
                        open_begin=kwargs['open_begin'],
                        open_end=kwargs['open_end']).distance
            else:
                d = np.max([dtw(a_cropped if cropped else a,
                                b_cropped if cropped else b,
                                keep_internals=False,
                                distance_only=kwargs['distance_only'],
                                step_pattern=kwargs['step_pattern'],
                                window_type=kwargs['window_type'],
                                window_args=kwargs['window_args'],
                                open_begin=kwargs['open_begin'],
                                open_end=kwargs['open_end']).distance,
                            dtw(b_cropped if cropped else b,
                                a_cropped if cropped else a,
                                keep_internals=False,
                                distance_only=kwargs['distance_only'],
                                step_pattern=kwargs['step_pattern'],
                                window_type=kwargs['window_type'],
                                window_args=kwargs['window_args'],
                                open_begin=kwargs['open_begin'],
                                open_end=kwargs['open_end']).distance])
            return d


    def print_metrics(self):
        print (str(self.scipy_prebuilt_metrics))

    def print_metric(self):
        print(self.metric)

    def scipy_distance(self, a, b, metric):
        return pdist([a, b], metric=metric)

    def cross_correlation(self, a, b):
        return correlate(a, b)

    def wasserstein_distance(self, a, b):
        return wasserstein_distance(a, b)

    def entropy(self, a, b):
        return entropy(histogram(a, density=True)[0], histogram(b, density=True)[0])

    def shape(self, a, b,
                    a_centroid=None, a_fwhm=None,
                    b_centroid=None, b_fwhm=None,
                    common_shape=1024,
                    scalor=5,
                    cropped=False):

        if cropped:
            a_cropped = crop(a,
                             a_centroid,
                             a_fwhm,
                             common_shape=common_shape,
                             scalor=scalor)

            b_cropped = crop(b,
                             b_centroid,
                             b_fwhm,
                             common_shape=common_shape,
                             scalor=scalor)

        dist = np.linalg.norm(
            resize_to_N(
                a_cropped if cropped else a, common_shape
            ) - resize_to_N(
                b_cropped if cropped else b, common_shape
            )
        )

        return dist
