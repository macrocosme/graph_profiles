import pdat
import numpy as np
from .pulsar import Observation
from .signal import (
    shift_centroid_to_center,
    remove_baseline as _remove_baseline,
    resize_to_N
)
import json

def _normalize_stokes(obs):
    if obs.stokes_Q is not None:
        obs.stokes_Q = (
            obs.stokes_Q - obs.stokes_I.min()
        ) / (
            obs.stokes_I.max() - obs.stokes_I.min()
        )

    if obs.stokes_U is not None:
        obs.stokes_U = (
            obs.stokes_U - obs.stokes_I.min()
        ) / (
            obs.stokes_I.max() - obs.stokes_I.min()
        )

    if obs.stokes_V is not None:
        obs.stokes_V = (
            obs.stokes_V - obs.stokes_I.min()
        ) / (
            obs.stokes_I.max() - obs.stokes_I.min()
        )

    if obs.stokes_L is not None:
        obs.stokes_L = (
            obs.stokes_L - obs.stokes_I.min()
        ) / (
            obs.stokes_I.max() - obs.stokes_I.min()
        )

    obs.stokes_I_znormed = (obs.stokes_I - obs.stokes_I.mean())/obs.stokes_I.std()

    obs.stokes_I = (
        obs.stokes_I - obs.stokes_I.min()
    ) / (
        obs.stokes_I.max() - obs.stokes_I.min()
    )

    return obs

def load_psrfits_data(file_location):
    profile = pdat.psrfits(file_location, verbose=False)
    data = profile[2].read(verbose=False)

    stokes_I, stokes_Q, stokes_U, stokes_V = None, None, None, None

    if data['DATA'].shape[1] == 1:
        stokes_I = (data['DATA'][0,0,-1] * data['DAT_SCL'][0]) + data['DAT_OFFS'][0]
    else:
        stokes_I = (data['DATA'][0,0,-1] * data['DAT_SCL'][0, 0]) + data['DAT_OFFS'][0, 0]
        stokes_Q = (data['DATA'][0,1,-1] * data['DAT_SCL'][0, 1]) + data['DAT_OFFS'][0, 1]
        stokes_U = (data['DATA'][0,2,-1] * data['DAT_SCL'][0, 2]) + data['DAT_OFFS'][0, 2]
        stokes_V = (data['DATA'][0,3,-1] * data['DAT_SCL'][0, 3]) + data['DAT_OFFS'][0, 3]

    profile.close()

    return stokes_I, stokes_Q, stokes_U, stokes_V

def load_ascii_data(file_location):
    stokes_I, stokes_Q, stokes_U, stokes_V = None, None, None, None

    profile = []
    with open(file_location, 'r') as f:
        for line in f:
            profile.append([float(v) if ('.' in v) or ('e' in v) else int(v) for v in line.split()])
    profile = np.asarray(profile)

    if profile.shape[-1] == 4:
        stokes_I = profile.T[3]
    else:
        stokes_I = profile.T[3]
        stokes_Q = profile.T[4]
        stokes_U = profile.T[5]
        stokes_V = profile.T[6]

    return stokes_I, stokes_Q, stokes_U, stokes_V

def load_json_data(file_location):
    stokes_I, stokes_Q, stokes_U, stokes_V = None, None, None, None
    position_angle, position_angle_phase = None, None
    position_angle_yerr_low, position_angle_yerr_high = None, None

    with open(file_location) as f:
        data = json.load(f)
    phase = np.asarray(data['series']['I']).T[0]
    stokes_I = np.asarray(data['series']['I']).T[1]
    if 'Q' in data['series'].keys():
        stokes_Q = np.asarray(data['series']['Q']).T[1]
        stokes_U = np.asarray(data['series']['U']).T[1]
        stokes_V = np.asarray(data['series']['V']).T[1]

    # Should eventually use PA and PAE
    try:
        position_angle_phase = np.asarray(data['series']['PA']).T[0]
        position_angle = np.asarray(data['series']['PA']).T[1]
        position_angle_yerr_low = position_angle - np.asarray(data['series']['PAE']).T[1]
        position_angle_yerr_high = np.asarray(data['series']['PAE']).T[2] - position_angle

        # position_angle = np.empty(phase.size)
        # position_angle_yerr_low = np.zeros(phase.size)
        # position_angle_yerr_high = np.zeros(phase.size)
        # position_angle[:] = None
        # position_angle_yerr_low[:] = None
        # position_angle_yerr_high[:] = None
        #
        # for i, p in enumerate(position_angle_phase):
        #     cond = np.where(phase == p)[0]
        #     position_angle[cond] = _position_angle[i]
        #     position_angle_yerr_low[cond] = _position_angle_yerr_low[i]
        #     position_angle_yerr_high[cond] = _position_angle_yerr_high[i]
    except (IndexError, KeyError):
        pass

    return phase, stokes_I, stokes_Q, stokes_U, stokes_V, position_angle, position_angle_phase, position_angle_yerr_low, position_angle_yerr_high


def parse_inputdata_file(file_location,
                         frequency=None,
                         frequency_range=None,
                         epn_reference_code=None,
                         shift=True,
                         normalize=True,
                         remove_baseline=False,
                         resize=False,
                         common_shape=1024):
    """Read input data file and fetch stokes and position angle arrays

    Parameter
    ---------
    file_location: str
    frequency: str

    Returns
    -------
    observation: Observation


    """
    obs = Observation(frequency=frequency, frequency_range=frequency_range)

    if file_location.split('.')[-1] == 'fits':
        stokes_I, stokes_Q, stokes_U, stokes_V = load_psrfits_data(file_location)
        position_angle_yerr_low = None
    elif file_location.split('.')[-1] == 'txt':
        stokes_I, stokes_Q, stokes_U, stokes_V = load_ascii_data(file_location)
        position_angle_yerr_low = None
    elif file_location.split('.')[-1] == 'json':
        obs.phase, obs.stokes_I, stokes_Q, stokes_U, stokes_V, \
        position_angle, position_angle_phase, \
        position_angle_yerr_low, position_angle_yerr_high = load_json_data(file_location)

    obs.original_stokes_size = obs.stokes_I.size

    if stokes_Q is None:
        if shift:
            try:
                obs.stokes_I, _ = shift_centroid_to_center(obs.stokes_I, obs.phase)
            except IndexError:
                print (obs.stokes_I.size, obs.phase.size)

        if remove_baseline:
            obs.stokes_I = _remove_baseline(obs.stokes_I)

        if resize:
            obs.stokes_I = resize_to_N(obs.stokes_I, common_shape)
    else:
        obs.stokes_Q = stokes_Q
        obs.stokes_U = stokes_U
        obs.stokes_V = stokes_V
        obs.stokes_L = np.sqrt(obs.stokes_Q**2 + obs.stokes_U**2)
        if position_angle_yerr_low is None:
            obs.position_angle = 0.5*np.arctan(obs.stokes_U / obs.stokes_Q)
            obs.position_angle_phase = obs.phase
            obs.position_angle_yerr_low = np.zeros(obs.phase.size)
            obs.position_angle_yerr_high = np.zeros(obs.phase.size)
        else:
            obs.position_angle = position_angle
            obs.position_angle_phase = position_angle_phase
            obs.position_angle_yerr_low = position_angle_yerr_low
            obs.position_angle_yerr_high = position_angle_yerr_high
        obs.epn_reference_code = epn_reference_code
        obs.file_location = file_location

        if shift:
            # centroid = compute_centroid(obs.stokes_I)
            obs.stokes_I, centroid = shift_centroid_to_center(obs.stokes_I, obs.phase)
            obs.stokes_Q, _ = shift_centroid_to_center(obs.stokes_Q, obs.phase, centroid)
            obs.stokes_U, _ = shift_centroid_to_center(obs.stokes_U, obs.phase, centroid)
            obs.stokes_V, _ = shift_centroid_to_center(obs.stokes_V, obs.phase, centroid)
            obs.stokes_L, _ = shift_centroid_to_center(obs.stokes_L, obs.phase, centroid)
            obs.position_angle, _ = shift_centroid_to_center(obs.position_angle, obs.phase, centroid)
            obs.position_angle_yerr_low, _ = shift_centroid_to_center(obs.position_angle_yerr_low, obs.phase, centroid)
            obs.position_angle_yerr_high, _ = shift_centroid_to_center(obs.position_angle_yerr_high, obs.phase, centroid)

        if resize:
            obs.stokes_I = resize_to_N(obs.stokes_I, common_shape)
            obs.stokes_Q = resize_to_N(obs.stokes_Q, common_shape)
            obs.stokes_U = resize_to_N(obs.stokes_U, common_shape)
            obs.stokes_V = resize_to_N(obs.stokes_V, common_shape)
            obs.stokes_L = resize_to_N(obs.stokes_L, common_shape)
            # obs.position_angle = resize_to_N(obs.position_angle, common_shape)

        if remove_baseline:
            obs.stokes_I = _remove_baseline(obs.stokes_I)
            obs.stokes_Q = _remove_baseline(obs.stokes_Q)
            obs.stokes_U = _remove_baseline(obs.stokes_U)
            obs.stokes_V = _remove_baseline(obs.stokes_V)
            obs.stokes_L = _remove_baseline(obs.stokes_L)

    if normalize:
        obs = _normalize_stokes(obs)

    obs.set_linear_polarization_degree()
    obs.set_circular_polarization_degree()

    return obs
