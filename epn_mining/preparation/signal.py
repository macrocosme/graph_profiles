import numpy as np
from ..analysis.stats import (
    compute_noise_statistics,
    centroid as compute_centroid,
    median_of_medians
)
from scipy import ndimage
import copy

rotate = lambda x, i: np.roll(x, i)

def shift_centroid_to_center(profile, phase, centroid=None):
    if centroid is None:
        prof = np.roll(profile, profile.shape[0]// 2 - int(np.round(compute_centroid(profile))))
        c = profile.shape[0]// 2 - int(np.round(compute_centroid(profile)))
    else:
        centroids = np.where(np.isclose(phase, centroid))
        prof = np.roll(profile, profile.shape[0]// 2) - centroids[len(centroids)//2]

    # This is horrible
    #phase[int(np.round(compute_centroid(prof)))]
    return prof, phase[c] if centroid is None else centroids[len(centroids)//2]

def best_alignment(prof1, prof2):
    khi = lambda x, y: np.nansum(np.sqrt(x**2-y**2))

    a = copy.deepcopy(prof1)
    b = copy.deepcopy(prof2)

    diffs = [khi(a, rotate(b, i)) for i in range(a.shape[0])]

    return np.argmin(diffs)

def shift_max_to_center(profile, phase):
    prof = np.roll(profile, int(profile.shape[0] / 2) - np.where(profile == np.nanmax(profile))[0])
    return prof, phase[np.where(prof == np.nanmax(prof))[0]]

def remove_baseline(profile):
    try:
        prof = profile-np.abs(median_of_medians(profile))
    except:
        prof = profile-np.abs(compute_noise_statistics(profile))

    return prof

def resize_to_N(data, N):
    prof = ndimage.zoom(data, N/data.size)
    return prof
