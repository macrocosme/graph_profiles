from os import listdir
from os.path import isfile, join
from pandas import DataFrame, to_numeric
import scipy.interpolate
import scipy.stats
import numpy as np
import pdat
import copy
from numba import njit
import collections

from matplotlib import pyplot as plt
from matplotlib import rc
rc('font', size=12)
rc('axes', titlesize=14)
rc('axes', labelsize=14)



def normalize_zero_one(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def fwhm(profile, return_dist=False):
    points = np.where(profile > np.max(profile)/2.0)[0]
    if not return_dist:
        return np.min(points), np.max(points)
    else:
        if points.shape[0] > 0:
            return np.max(points)-np.min(points)
        else:
            return 0

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
    width = b-a
    start = a-width
    end = b+width
    noisy = np.append(profile[0:start], profile[end:], axis=0)
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


def shift_centroid_to_center(profile, centroid=None):
    # if (np.where(profile == profile.max())[0][-1] < profile.shape[0]/split) or \
    #    (np.where(profile == profile.max())[0][-1] > profile.shape[0] - profile.shape[0]/split):
    #    profile = np.roll(profile, int(profile.shape[0] / 2) - int(round(profile.max())))
    return np.roll(profile, int(profile.shape[0] / 2) - int(round(centroid(profile) if centroid is None else centroid)))

def simple_snr(profile):
    a = np.asanyarray(profile)
    m = a.max()
    # std = a.std()
    std = compute_noise_statistics(a)
    return 0 if np.abs(std) == 0 else np.abs(m-np.median(a))/np.abs(std)

def resize_to_N(data,N):
    M=data.size
    res=np.empty(N,data.dtype)
    carry=0
    m=0
    for n in range(N):
        sum = carry
        while m*N - n*M < M :
            sum += data[m]
            m += 1
        carry = (m-(n+1)*M/N)*data[m-1]
        sum -= carry
        res[n] = sum*N/M
    return res

def sort_dict(x):
    return collections.OrderedDict(x)

def interpolate_with_spline(profile, common_shape=None, s=1, ext=2):
    if common_shape is None:
        common_shape = profile.shape[0]
    x = np.linspace(0, profile.shape[0], profile.shape[0])
    new_x = np.linspace(0, profile.shape[0], int(common_shape))
    spl = scipy.interpolate.splrep(x, profile, s=s)
    return scipy.interpolate.splev(new_x, spl, ext=ext)


def interpolate_with_kronecker(profile, common_shape=512):
    return np.kron(profile, np.ones(int(common_shape / profile.shape[0])))

def scale_set(profiles, min=-1, max=-1):
    for i, arr in enumerate(profiles):
        try:
            profiles[i] = (arr - arr.min()) / (arr.max() - arr.min())
        except:
            pass

    return profiles


def pool_data(df_profiles, smooth_iterations=1, norm=True, get_stokes=False, verbose=False):
    """Pool profile data into a 2D arrays

    Parameter
    ---------
    df_profiles: pandas.Dataframe

    Returns
    -------
    im: numpy.array
         2D array where each row is a profile
    """
    if verbose:
        print ('Pooling data to data frame')

    # Pool data
    profiles = []
    stokes_L = []
    stokes_V = []
    pa = []
    i = 0
    df_index_np_index = {}
    for index, row in df_profiles.iterrows():
        try:
            profile = pdat.psrfits(row['file location'], verbose=False)
            data = profile[2].read(verbose=False)
            if data['DATA'].shape[1] == 1:
                arr = (data['DATA'][0,0,-1] * data['DAT_SCL'][0]) + data['DAT_OFFS'][0]
            else:
                arr = (data['DATA'][0,0,-1] * data['DAT_SCL'][0, 0]) + data['DAT_OFFS'][0, 0]
                # Unused arrays for now but leaving the code here
                arr_stokes_L = np.sqrt(
                    ((data['DATA'][0,1,-1] * data['DAT_SCL'][0, 1]) + data['DAT_OFFS'][0, 1])**2 +
                    ((data['DATA'][0,2,-1] * data['DAT_SCL'][0, 2]) + data['DAT_OFFS'][0, 2])**2
                )
                arr_stokes_V = (data['DATA'][0,3,-1] * data['DAT_SCL'][0, 3]) + data['DAT_OFFS'][0, 3],
                arr_PA = 0.5*np.arctan(
                    ((data['DATA'][0,2,-1] * data['DAT_SCL'][0, 2]) + data['DAT_OFFS'][0, 2]) /
                    ((data['DATA'][0,1,-1] * data['DAT_SCL'][0, 1]) + data['DAT_OFFS'][0, 1])
                )
            profiles.append(arr)
            stokes_L.append(arr_stokes_L if data['DATA'].shape[1] != 1 else [])
            stokes_V.append(arr_stokes_V if data['DATA'].shape[1] != 1 else [])
            pa.append(arr_PA if data['DATA'].shape[1] != 1 else [])
            profile.close()
            del arr
            try:
                del arr_stokes_L
                del arr_stokes_V
                del arr_PA
            except UnboundLocalError:
                pass

            df_profiles.loc[index, 'image index'] = i
            df_profiles.loc[index, 'snr'] = compute_statistics(profiles[i])[2] if not np.isnan(compute_statistics(profiles[i])[2]) else -1
            df_profiles.loc[index, 'n samples'] = profiles[i].shape[0]
            i+=1

        except OSError:
            with open('file_log', 'a') as f:
                f.write('%s not found.\n' % row['file location'])
            df_profiles.drop(index, inplace=True)
        except TypeError:
            with open('file_log', 'a') as f:
                f.write('%s not found.\n' % row['file location'])
            df_profiles.drop(index, inplace=True)

    profiles = scale_set(np.asarray(profiles)) if norm else np.asarray(profiles)
    stokes_L = scale_set(np.asarray(stokes_L)) if norm else np.asarray(stokes_L)
    stokes_V = scale_set(np.asarray(stokes_V)) if norm else np.asarray(stokes_V)
    pa = np.asarray(pa)

    df_profiles['image index'] = df_profiles['image index'].astype('int')
    df_profiles['snr'] = df_profiles['snr'].astype('int')
    df_profiles['n samples'] = df_profiles['n samples'].astype('int')

    if get_stokes:
        return profiles, df_profiles, stokes_L, stokes_V, pa
    else:
        return profiles, df_profiles


def resize_and_roll_arrays(profiles,
                           df_profiles,
                           nsigma=5,
                           remove_noise_steps=1,
                           crop_fwhm=False,
                           spline=False,
                           common_shape=512,
                           verbose=False,
                           plot=False,
                           log_filename='log'):
    if verbose:
        print ('Resize arrays to common shape')

    # Scale arrays to some common_shape
    # profiles = np.asarray(profiles)
    print ("profiles shape", profiles.shape)

    # then stretch shorter arrays to common length
    m, n = len(df_profiles), common_shape
    im = np.zeros([m, n])
    df_indices = []

    with open(log_filename, 'w+') as f:
        f.write("i index")
        i = 0
        for index, row in df_profiles.iterrows():

            # try:
            profiles[i] = shift_centroid_to_center(profiles[i])
            central, stdev, snr = compute_statistics(profiles[i])

            if snr > 40:

                pulse_region = np.where(profiles[i] > nsigma*stdev+central)
                im[i] = normalize_zero_one(
                    resize_to_N(
                        profiles[i][np.min(pulse_region):np.max(pulse_region)],
                        common_shape
                    )
                )

                df_profiles.loc[index, 'peak image index'] = i
                # pulse_region = np.where(profiles[i] > nsigma*stdev+central)
                # im[i] = resize_to_N(
                #     profiles[i][np.min(pulse_region):np.max(pulse_region)],
                #     common_shape
                # )

                # Specific threshold selected via visualisation
                # im[i][np.where(im[i] < 0.2)] = 0.2
                # im[i] -= 0.2

                # profiles[i] = resize_to_N(
                #     profiles[i],
                #     common_shape
                # )

                # im[i] = copy.deepcopy(profiles[i])
                # profiles[i] = cut_10(profiles[i])
                # profiles[i] = shift_centroid_to_center(profiles[i])
                # profiles[i] = remove_baseline(profiles[i])
                # if crop_fwhm:
                #     profiles[i] = remove_noise(profiles[i], times=remove_noise_steps, spline=spline)
                #     vals = np.where(profiles[i] > 0)
                #     profiles[i] = profiles[i][np.min(vals):np.max(vals)]
                #     profiles[i] = resize_to_N(profiles[i], common_shape)
                #     im[i] = copy.deepcopy(profiles[i])
                #     df_indices.append(index)
                # else:
                #     if simple_snr(profiles[i]) > 10:
                #         profiles[i] = remove_noise(profiles[i], times=remove_noise_steps, spline=spline)
                #         profiles[i] = shift_centroid_to_center(profiles[i])
                #         im[i] = copy.deepcopy(profiles[i])
                #         df_indices.append(index)
                #     else:
                #         df_profiles.drop(index, inplace=True)
                #         f.write("Dropped %d %d with snr %.2f\n" % (i, index, simple_snr(profiles[i])))
            else:
                df_profiles.drop(index, inplace=True)
                f.write("Dropped %d %d with SNR %f" % (i, index, snr))
            # except Exception as e:
            #     df_profiles.drop(index, inplace=True)
            #     f.write("Dropped %d %d" % (i, index))
            #     print  ('ERROR', i, index, e)

            i += 1

    im = np.asarray(im)
    im = remove_empty_rows(im)

    df_profiles['peak image index'] = df_profiles['peak image index'].astype('int')

    im_znormalized = []
    for ii in im:
        im_znormalized.append(scipy.stats.zscore(copy.deepcopy(ii)))
        # im_znormalized.append(normalize_zero_one(copy.deepcopy(ii)))
    im_znormalized = np.asarray(im_znormalized)

    # print ("Shapes", im.shape, im_znormalized.shape)

    if (plot):
        plt.clf()
        plt.imshow(im)
        plt.show()

    return im, im_znormalized, df_profiles, df_indices
