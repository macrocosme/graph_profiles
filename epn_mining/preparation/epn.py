import numpy as np
import copy
import os
import glob
import json
from tqdm import tqdm
from psrqpy import QueryATNF
import pandas as pd
from .pulsar import Pulsar, Population
from .reader import parse_inputdata_file
from .preparation import read_metadata_file
from ..analysis.stats import compute_statistics
from .. import main

LIMS = (
    [0,  200], # in MHz
    [200,400],
    [400,700],
    [700,1000],
    [1000,1500],
    [1500,2000],
    [2000,1199169832000000]
)
LIMS_DICT = {i:v for i, v in enumerate(LIMS)}

ATNF_PARAMS = [
        'JNAME',
        'BNAME',
        'RAJ',
        'DECJ',
        'P0',
        'P1',
        'P1_I',
        'DM',
        'RM',
        'Tau_sc',
        'age',
        'R_Lum',
        'R_Lum14',
        'BSurf',
        'Edot',
        'PMTot',
        'W50',
        'W10',
        'GL',
        'GB',
        'PMTot',
        'VTrans',
        'B_LC',
        'BINARY',
        'BINCOMP',
        'ECC',
        'ASSOC',
        'TYPE',
        'NGlt',
    ]

def listdir(path):
        for f in os.listdir(path):
            if not f.startswith('.'):
                yield f

def listdir_fullpath(path):
    return glob.glob(os.path.join(path, '*'))

def get_freq_range(freq):
    for freq_bin in LIMS_DICT.keys():
        l, h = LIMS_DICT[freq_bin]
        if (freq >= l) and (freq < h):
            return freq_bin
    return None

def load_epn_metadata(base_path = '../www.epta.eu.org/epndb/json', reference=None, exclude_references=None, stokes=None,
                      verbose = True):
    # First collect pulsar list
    jnames = []
    for ref, ref_path in zip(listdir(base_path), listdir_fullpath(base_path)):
        for jname, pulsar_path in zip(listdir(ref_path), listdir_fullpath(ref_path)):
            if 'WARNING' not in jname and jname not in jnames:
                jnames.append(jname)

    # print (jnames)

    # Fetch ATNF Pulsar info (including bname where available)
    if verbose:
        print('Querying ATNF pulsar catalogue')
    psrcat = QueryATNF(
        params=ATNF_PARAMS,
        psrs=jnames,
        cache=False,
    )
    df_psrcat = psrcat.dataframe

    # Build catalogue metadata
    if verbose:
        print('Build EPN/ATNF dataframe')
    pulsars = []
    n_err, m_err = 0, []
    for ref, ref_path in zip(listdir(base_path), listdir_fullpath(base_path)):
        for jname, pulsar_path in zip(listdir(ref_path), listdir_fullpath(ref_path)):
            if 'WARNING' not in jname:
                for filename, json_path in zip(listdir(pulsar_path), listdir_fullpath(pulsar_path)):
                    if '.bk' not in filename:
                        # print (ref, jname, json_path)
                        # Init a dict with ATNF catalogue metadata
                        try:
                            obs = df_psrcat.loc[df_psrcat['JNAME'] == jname].iloc[0].to_dict()
                        except (IndexError, UnboundLocalError):
                            obs = {}

                        # Add EPN meta
                        # reference
                        obs['reference'] = ref

                        # file info
                        obs['file location'] = json_path
                        obs['file name'] = filename

                        # frequency info (I/O -- should probably do all the stokes storing now)
                        data = json.load(open(json_path, 'r'))
                        try:
                            freq = float(data['hdr']['freq'])
                            obs['frequency (MHz)'] = freq
                            obs['frequency range'] = get_freq_range(freq)

                            obs['stokes'] = 'I' #'I' if int(data['hdr']['npol']) == 1 else 'IQUV'

                            pulsars.append(obs)
                        except KeyError:
                            # Skip saving observations with missing info in hdr
                            n_err += 1
                            m_err.append(json_path)
                            pass

    df_start = pd.DataFrame.from_dict(pulsars)
    df_start.rename(columns={x: x.lower() if 'MHz' not in x else x for x in df_start.columns}, inplace=True)

    dtype = {c.lower():df_psrcat[c].dtype.type for c in df_psrcat.columns}
    dtype.update({
        'file location': str,
        'file name': str,
        'frequency (MHz)': float,
        'frequency range': int,
        'reference': str,
        'stokes': str,
    })
    df_start = df_start.astype(dtype=dtype, errors='ignore')

    if reference is not None:
        assert reference in df_start['reference'].values, 'reference must be one of' + str(df_start['reference'].values)
        df_start = df_start.loc[df_start['reference'] == reference]

    if exclude_references is not None:
        assert type(exclude_references) == list, 'exclude_reference must be a list'
        df_start = df_start[~df_start['reference'].isin(exclude_references)]

    if stokes is not None:
        assert stokes in ['I', 'IQUV'], 'stokes must be one of ["I", "IQUV"]'
        df_start = df_start.loc[df_start['stokes'] == stokes]

    df_start[['jname', 'bname']] = df_start[['jname', 'bname']].astype('str')
    return df_start


def load_epn_data(df_profiles,
                  shift=False,
                  normalize=False,
                  remove_baseline=False,
                  resize=False,
                  common_shape=512,
                  verbose=False):
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
        print ('Load EPN data to Population and update EPN metadata dataframe')

    # Pool observations into a Population object, a collection of Pulsar objects
    population = Population()
    for i, jname in enumerate(tqdm(df_profiles['jname'].unique())):
        pulsar = Pulsar(jname=jname,
                        bname=df_profiles.loc[df_profiles['jname'] == jname, 'bname'].values[0])
        for index, row in df_profiles.loc[df_profiles['jname'] == jname].iterrows():
            # try:
            obs = parse_inputdata_file(row['file location'],
                                       frequency=row['frequency (MHz)'],
                                       frequency_range=row['frequency range'],
                                       epn_reference_code=row['reference'],
                                       shift=shift,
                                       normalize=normalize,
                                       remove_baseline=remove_baseline,
                                       resize=resize,
                                       common_shape=common_shape)

            df_profiles.loc[index, 'n samples'] = obs.stokes_I.shape[0]
            _continue = True
            # except OSError:
            #     # print ('OSError for %s' % pulsar.jname )
            #     # with open('file_log', 'a') as f:
            #     #     f.write('%s not found.\n' % row['file location'])
            #     df_profiles.drop(index, inplace=True)
            #     _continue = False
            # except TypeError:
            #     print ('TypeError for %s' % pulsar.jname )
            #     # with open('file_log', 'a') as f:
            #     #     f.write('%s not found.\n' % row['file location'])
            #     df_profiles.drop(index, inplace=True)
            #     _continue = False
            # except UnboundLocalError:
            #     df_profiles.drop(index, inplace=True)
            #     _continue = False

            if _continue:
                # TODO: could be rather SNR/n_bins
                if obs.frequency_range not in pulsar.observations.keys():
                    pulsar.add_observation(copy.deepcopy(obs))
                elif obs.snr > pulsar.observations[obs.frequency_range].snr:
                    pulsar.add_observation(copy.deepcopy(obs))
                # elif obs.original_stokes_size == pulsar.observations[obs.frequency_range].original_stokes_size:
                #     if obs.snr > pulsar.observations[obs.frequency_range].snr:
                #         pulsar.add_observation(copy.deepcopy(obs))
                # elif obs.original_stokes_size > pulsar.observations[obs.frequency_range].original_stokes_size:
                #     pulsar.add_observation(copy.deepcopy(obs))

        population.add_pulsar(copy.deepcopy(pulsar))

    # df_profiles['n samples'] = df_profiles['n samples'].astype('int')

    return population, df_profiles




# Old code
def load_epn_metadata_deprecated(reference=None, exclude_references=None, stokes=None, input_type='psrfits', atnf_params='basic', verbose=False):
    assert input_type in ['psrfits', 'ascii', 'json'], 'input_type should be one of ["psrfits", "ascii", "json"]'

    if verbose:
        print ('Load EPN metadata to dataframe')

    df_start = read_metadata_file(
        base_path = '../www.epta.eu.org/epndb/%s' % (input_type),
        verbose=verbose
    )
    df_start = df_start.astype(
        dtype={
            'jname': str,
            'bname': str,
            'file location': str,
            'file name': str,
            'frequency (MHz)': float,
            'reference': str,
            'stokes': str,
        },
        errors='ignore'
    )

    freq_range = []
    for i, row in df_start.iterrows():
        found = False
        for j in LIMS_DICT.keys():
            l, h = LIMS_DICT[j]
            if (row['frequency (MHz)'] >= l) and (row['frequency (MHz)'] < h):
                found = True
                freq_range.append(j)
        if not found:
            print ('oups', row['frequency (MHz)'])


    df_start['frequency range'] = copy.deepcopy(freq_range)
    del freq_range

    print ('Retrieve related metadata from psrcat')
    df_psrcat = main.load('df_psrcat')
    if df_psrcat is None:
        psrcat = QueryATNF(
            params=ATNF_PARAMS,
            psrs=[row['jname'] for index, row in df_start.iterrows()]
        )
        df_psrcat = psrcat.dataframe
        main.save('df_psrcat', df_psrcat)

    print ('Add psrcat metadata to epn dataframe')
    for index, row in tqdm(df_psrcat.iterrows(), total=df_psrcat.shape[0]):
        for column in row.keys():
            if column not in ['JNAME', 'BNAME']:
                    df_start.loc[(df_start['jname'] == row['JNAME']), column] = row[column]

    # df_start = df_start.sort_values(by=['AGE', 'jname'])

    if reference is not None:
        assert reference in df_start['reference'].values, 'reference must be one of' + str(df_start['reference'].values)
        df_start = df_start.loc[df_start['reference'] == reference]

    if exclude_references is not None:
        assert type(exclude_references) == list, 'exclude_reference must be a list'
        df_start = df_start[~df_start['reference'].isin(exclude_references)]

    if stokes is not None:
        assert stokes in ['I', 'IQUV'], 'stokes must be one of ["I", "IQUV"]'
        df_start = df_start.loc[df_start['stokes'] == stokes]

    # df_start = df_start.loc[df_start['P0'] >= 0.1]

    return df_start
