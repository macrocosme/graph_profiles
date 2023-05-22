import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

import cfod
from cfod import catalog
from cfod.routines import waterfaller

import h5py
import scipy
import wget

from urllib.error import HTTPError

data_catalog = catalog.as_dataframe()

def boxcar_kernel(width):
    width = int(round(width, 0))
    return np.ones(width, dtype="float32") / np.sqrt(width)


def find_burst(ts, min_width=1, max_width=128):
    min_width = int(min_width)
    max_width = int(max_width)
    # do not search widths bigger than timeseries
    widths = list(range(min_width, min(max_width + 1, len(ts)-2)))
    # envelope finding
    snrs = np.empty_like(widths, dtype=float)
    peaks = np.empty_like(widths, dtype=int)
    for i in range(len(widths)):
        convolved = scipy.signal.convolve(ts, boxcar_kernel(widths[i]), mode="same")
        peaks[i] = np.nanargmax(convolved)
        snrs[i] = convolved[peaks[i]]
    best_idx = np.nanargmax(snrs)
    return peaks[best_idx], widths[best_idx], snrs[best_idx]

def bin_freq_channels(data, fbin_factor=4):
    num_chan = data.shape[0]
    if num_chan % fbin_factor != 0:
        raise ValueError("frequency binning factor `fbin_factor` should be even")
    data = np.nanmean(data.reshape((num_chan // fbin_factor, fbin_factor) + data.shape[1:]), axis=1)
    return data

#find the download url and import the data for a burst given its tns_name from the Data table.
def get_data(burst_index_number):
    
    #print('we here')
    
    example_tns = data_catalog["tns_name"][burst_index_number]
    url_base = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/files/vault/AstroDataCitationDOI/CISTI.CANFAR/21.0007/data/waterfalls/data/"
    waterfall_string = '_waterfall.h5'
    url = url_base + example_tns + waterfall_string
    
    #statement implemented so in testing every run does not create duplicates of the same file (saving runtime and storage)
    try:
        scratch_folder = '/scratch/hugo_v/'
        FRB_data_folder = "FRB_data_files/"
        Data_from_source = example_tns + waterfall_string
        #data = h5py.File(Data_from_source, "r")
        data = h5py.File(f"{FRB_data_folder}{Data_from_source}", "r")

        # print("file from folder")

        
    except:
        # print('file not in folder, downloading')
        Data_from_source = wget.download(url, out='FRB_data_files/') #+example_tns+waterfall_string)
        data = h5py.File(Data_from_source, "r")

    return data

def make_curves(data):
    data = data["frb"]
    # eventname = data.attrs["tns_name"].decode()
    wfall = data["wfall"][:]
    model_wfall = data["model_wfall"][:]
    plot_time = data["plot_time"][:]
    # plot_freq = data["plot_freq"][:]
    ts = data["ts"][:]
    # model_ts = data["model_ts"][:]
    spec = data["spec"][:]
    # model_spec = data["model_spec"][:]
    # extent = data["extent"][:]
    # dm = data.attrs["dm"][()]
    # scatterfit = data.attrs["scatterfit"][()]
    # cal_obs_date = data.attrs["calibration_observation_date"].decode()
    # cal_source_name = data.attrs["calibration_source_name"].decode()
    cal_wfall =  data["calibrated_wfall"][:]

    dt = np.median(np.diff(plot_time)) # the delta (time) between time bins 
    # dt in mu s
    # this value is the same for both caliberated and uncalibrated data
    ts_with_RFI = ts

    q1 = np.nanquantile(spec, 0.25)
    q3 = np.nanquantile(spec, 0.75)
    iqr = q3 - q1

    # additional masking of channels with RFI
    rfi_masking_var_factor = 3

    channel_variance = np.nanvar(wfall, axis=1)
    mean_channel_variance = np.nanmean(channel_variance)

    with np.errstate(invalid="ignore"):
        rfi_mask = (channel_variance > \
                    rfi_masking_var_factor * mean_channel_variance) \
                    | (spec[::-1] < q1 - 1.5 * iqr) | (spec[::-1] > q3 + 1.5 * iqr)
    wfall[rfi_mask,...] = np.nan
    model_wfall[rfi_mask,...] = np.nan
    spec[rfi_mask[::-1]] = np.nan

    # -------------- start plotting ------------
    # remake time-series after RFI masking
    ts = np.nansum(wfall, axis=0)
    model_ts = np.nansum(model_wfall, axis=0)


    peak, width, snr = find_burst(ts)
    # print(f"Peak: {peak} at time sample, Width = {width*dt} ms, SNR = {snr}")

    # bin frequency channels such that we have 16,384/16 = 1024 frequency channels 
    #wfall = bin_freq_channels(wfall, 16)
    

    ### time stamps relative to the peak
    peak_idx = np.argmax(ts)
    plot_time -= plot_time[peak_idx]

    # prepare time-series for histogramming
    plot_time -= dt / 2.
    plot_time = np.append(plot_time, plot_time[-1] + dt)

    # also do so for the calibrated data
    cal_wfall[np.isnan(cal_wfall)] = np.nanmedian(cal_wfall)   # replace nans in the data with the data median
    #bin frequency channels such that we have 16,384/16 = 1024 frequency channels 
    cal_wfall = bin_freq_channels(cal_wfall,16) 
    
    cal_ts = np.nanmean(cal_wfall, axis = 0)
    times = np.arange(len(cal_ts))*dt
    peak_idx = np.argmax(cal_ts)
    times -= times[peak_idx]
    times -= dt / 2.
    
    #make calibrated signal less broad
    deff_index_min = -int(times[0] - plot_time[0])
    len(plot_time)
    times_shorter = times[deff_index_min: (deff_index_min +len(plot_time))]
    cal_ts_shorter = cal_ts[deff_index_min: (deff_index_min +len(plot_time))]

    
    return plot_time, np.append(ts, ts[-1]), np.append(model_ts, model_ts[-1]),times_shorter, cal_ts_shorter, snr


def Get_me_FRB_data(burst_index_number_input, folder=None):
    try:
        data= get_data(burst_index_number =burst_index_number_input)
    except (KeyError, HTTPError):
        return None, None, None, None, None, None

    try:
        plot_time, ts_full_list, model_ts_full_list, times_shorter, cal_ts_shorter, snr = make_curves(data)
        return plot_time, ts_full_list, model_ts_full_list, times_shorter, cal_ts_shorter, snr
    except KeyError:
        return None, None, None, None, None, None
