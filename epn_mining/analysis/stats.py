import copy

import numpy as np
from sklearn import mixture
from scipy.stats import norm
from joblib import parallel_backend
from scipy.signal import find_peaks

rmse = lambda data, model: np.sqrt(np.sum((data-model)**2)/data.shape[0])

def median_of_medians(profile, n_chunks=8):
    step = profile.size//n_chunks
    medians = np.asarray([np.median(profile[i:i+step]) for i in range(0, profile.size, step)])
    mom = np.median(medians)
    return mom

def median_of_stdevs(profile, n_chunks=8):
    step = profile.size//n_chunks
    stds = [np.std(profile[i:i+step]) for i in range(0, profile.size, step)]
    mostd = np.median(stds)
    return mostd

snr = lambda profile, mom, mostd: (profile.max()-mom)/mostd

def robust_statistics(profile:np.array, n_chunks:int=8):
    """Compute robust statistics (mom, mostd, snr)

    mom: median of medians (mu)
    mostd: median of standard deviations (signa)
    snr: signal to noise ratio

    Parameters
    ----------
        profile: numpy.array
        n_chunks: int

    Returns
    -------
        mom: float
        mostd: float
        snr: float
    """
    mom = median_of_medians(profile=profile, n_chunks=n_chunks)
    mostd = median_of_stdevs(profile=profile, n_chunks=n_chunks)
    _snr = snr(profile.max(), mom, mostd)
    return mom, mostd, _snr

def fwhm(profile, return_dist=False):
    points = np.where(profile > np.max(profile)/2.0)[0]
    if return_dist:
        if points.shape[0] > 0:
            return np.max(points)-np.min(points)
        else:
            return 0
    else:
        try:
            return np.min(points), np.max(points)
        except ValueError:
            # print ('ValueError', 'profile', profile, 'points', points)
            return 0, profile.size-1


def centroid(profile, type='classic'):
    assert type in ['classic', 'fwhm'], 'type should be one of: "classic", "fwhm"'

    if type == 'classic':
        _sum, _sum_I = 0, 0
        try:
            for Xi, Ii in enumerate(profile[fwhm(profile)[0]:fwhm(profile)[1]]):
                _sum += Xi * Ii
                _sum_I += Ii

            centroid = fwhm(profile)[0] + (_sum/_sum_I)
        except ZeroDivisionError:
            for Xi, Ii in enumerate(profile):
                _sum += Xi * Ii
                _sum_I += Ii

            centroid = _sum/_sum_I
        return centroid

    if type == 'fwhm':
        lb, ub = fwhm(profile)
        return ub - (ub - lb) / 2

def ten_percent(profile, return_dist=False):
    points = np.where(profile > (0.1 * profile.max()))
    if not return_dist:
        return np.min(points), np.max(points)
    else:
        return np.max(points)-np.min(points)

def region_non_zero(profile, return_dist=False):
    points = np.where(profile > 0.)
    if not return_dist:
        return np.min(points), np.max(points)
    else:
        return np.max(points)-np.min(points)

def compute_noise_statistics(profile, stat='median'):
    a,b = fwhm(profile)
    noisy = np.append(profile[0:a], profile[b:], axis=0)

    return np.mean(noisy) if stat == 'mean' else np.median(noisy)

def compute_statistics(profile, stat='median'):
    a,b = fwhm(profile)
    width = b-a
    start = a-width
    end = b+width
    noisy = np.append(profile[0:start], profile[end:], axis=0)

    central = np.mean(noisy) if stat == 'mean' else np.median(noisy)
    stdev = np.std(noisy)

    snr = (profile.max()-central)/stdev if stdev != 0 else -1

    return central, stdev, snr

# Find pulse regions
def range_overlap(x, y):
        if x.start == x.stop or y.start == y.stop:
            return False
        return x.start <= y.stop and y.start <= x.stop

def merge_peaks(lims):
    merged_peaks = []
    merged_items = []
    for i in range(len(lims)):
        for j in range(i+1, len(lims)):
            if range_overlap(lims[i], lims[j]):
                if range(np.min([lims[i].start, lims[j].start]),
                         np.max([lims[i].stop, lims[j].stop])) not in merged_peaks:
                    merged_peaks.append(range(np.min([lims[i].start, lims[j].start]),
                                              np.max([lims[i].stop, lims[j].stop])))
                if i not in merged_items:
                    merged_items.append(i)
                if j not in merged_items:
                    merged_items.append(j)

    for i in range(len(lims)):
        if i not in merged_items:
            merged_peaks.append(lims[i])

    return merged_peaks

def find_pulse_regions(profile):
    threshold = robust_statistics(profile)[0] + 5 * robust_statistics(profile)[1]
    peaks = find_peaks(profile,
                       height=threshold,
                       width=2)
    peaks = peaks[0], peaks[1]['peak_heights'], peaks[1]['widths']

    mu = robust_statistics(profile)[0]
    lims = [range(np.where(profile[:p] < mu)[0][-1],
                  p+np.where(profile[p:] < mu)[0][0]) for p in peaks[0]]

    if len(lims) > 1:
        merged_peaks = merge_peaks(lims)
    else:
        merged_peaks = lims

    return merged_peaks

# DPGMM
def sample_phase_from_profile(phase, profile, size=10000, threshold=False, random=False):
    if threshold:
        prof = copy.deepcopy(profile)
        prof[np.where(prof < robust_statistics(prof)[0] + 1.5 * robust_statistics(prof)[1])] = 0
    if random:
        samples = np.random.choice(a=phase, size=size, p=prof if threshold else profile)
    else:
        samples = np.array([phase[i] for i, v in enumerate(prof if threshold else profile) for ii in range(np.round(v*size).astype(int))])
    return samples

def normalize_profile(profile):
    s = profile - np.median(profile)
    s = s - s.min()
    s = s / s.sum()
    return s

def convert_x_to_phase(profile):
    phase = (np.linspace(0, profile.size, profile.size) - profile.size//2) / profile.size
    return phase

def profile_as_distribution(profile, phase, size=100000, threshold=False, random=False):
    if random:
        normed_profile = normalize_profile(profile)

    line_distribution = sample_phase_from_profile(phase = phase,
                                                  profile = normed_profile if random else profile,
                                                  size = size,
                                                  threshold=threshold,
                                                  random=random)

    number_points = line_distribution.size
    profile_sampling = np.expand_dims(line_distribution, 1)
    return profile_sampling

def evaluate_DPGMM(profile, phase, n_components_start = 30, alpha=10**4, tol=1e-3, max_iter=1000,
                   weight_concentration_prior_type='dirichlet_process',
                   mean_prior=None, mean_precision_prior=None,
                   phase_distribution_size=1000, draw_random=False):

    """
    Draw (size) phase samples from Profile used as PDF

    For speed:
    random=False : best use size ~ 1000  (in this mode, size is a scaling factor on the intensity of a phase bin)
    random=True : best use size ~ 10,000 (total number of draws)
    """
    profile_sampling = profile_as_distribution(profile, phase, size = phase_distribution_size, random=draw_random)

    """
    Fit data
    """
    with parallel_backend('threading', n_jobs=-1):
        gmm = mixture.BayesianGaussianMixture(
            n_components = n_components_start,
            weight_concentration_prior_type=weight_concentration_prior_type,
            weight_concentration_prior=alpha,
            mean_prior=mean_prior,
            mean_precision_prior=mean_precision_prior,
            tol=tol,
            max_iter=max_iter
        )
        gmm.fit(profile_sampling)

    return gmm

def profile_from_gmm(observation, cut=False, scale=False, fwhm=None, fit_whole=True, window=0.3, interpulse=False):
    gmm = observation.gmm
    phase = observation.phase

    if cut:
        try:
            from .distance import (
                check_neg,
                check_bound,
                check_min_max
            )
        except:
            pass

        start, end = int(observation.centroid - (3 * (observation.fwhm if fwhm is None else fwhm))), \
                     int(observation.centroid + (3 * (observation.fwhm if fwhm is None else fwhm)))
        start, _ = check_neg(start)
        end, _ = check_neg(end)
        start, end = check_bound(start, end)
        start, end = check_min_max(start, end, observation.stokes_I.size)


        trim = np.unique(np.exp(gmm.score_samples(observation.phase.reshape(-1, 1))))
        # trim = np.unique(
        #     gmm.predict(
        #         profile_as_distribution(observation.stokes_I[start:end],
        #                                phase[start:end],
        #                                size = 10000)
        #     )
        # )
        # Trim the edges
        trim = trim[np.where(
            (gmm.means_[trim] >= phase[start]) &
            (gmm.means_[trim] <= phase[end])
        )[0]]
    else:
        _gmm_y = np.exp(gmm.score_samples(observation.phase.reshape(-1, 1)))
        # trim = np.unique(_gmm_y)
        # trim = np.unique(
        #     gmm.predict(
        #         profile_as_distribution(observation.stokes_I,
        #                                phase,
        #                                size = 10000)
        #     )
        # )
    # print (trim.shape)

    gauss_mixt = np.array(
        [norm.pdf(phase, mu, sd) / np.trapz(norm.pdf(phase, mu, sd), phase) * p \
         for mu, sd, p in zip(
            gmm.means_.ravel(),
            np.sqrt(gmm.covariances_.ravel()),
            gmm.weights_.ravel())]
    )
    # gauss_mixt_t = np.sum(gauss_mixt, axis = 0)

    if scale:
        # _gmm_y = (_gmm_y - _gmm_y.min()) / (_gmm_y.max() - _gmm_y.min())
        _gmm_y /= _gmm_y.max()

        if not fit_whole:
        #     _gmm_y *= observation.stokes_I.max()
        # else:
            _gmm_y *= observation.stokes_I[
                # This assumes that the interpulse's peak is the maximal value in the window phi in (+/-)0.3
                np.where(np.abs(observation.phase) < window) # This should be done differently. Currently, like pulsar.Observation.get_model
            ].max()
        # _gmm_y = _gmm_y - _gmm_y.min()

        # for i in range(len(gauss_mixt)):
        #     gauss_mixt[i] = (gauss_mixt[i] - gauss_mixt_t.max()) / (gauss_mixt_t.max() - gauss_mixt_t.min())
        #     gauss_mixt[i] = gauss_mixt[i] - gauss_mixt[i].min()

        # gauss_mixt_t = (gauss_mixt_t - gauss_mixt_t.min()) / (gauss_mixt_t.max() - gauss_mixt_t.min())
        # gauss_mixt_t *= observation.stokes_I.max() if fit_whole else observation.stokes_I[
        #     np.where(np.abs(observation.phase) < 0.3) # This should be done differently. Currently, like pulsar.Observation.get_model
        # ].max()

    # central, std, _ = robust_statistics(observation.stokes_I)
    # gauss_mixt_t[np.where(gauss_mixt_t <= central + 3*std)] = central + 3*std

    return _gmm_y, gauss_mixt
