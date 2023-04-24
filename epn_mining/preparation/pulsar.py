import numpy as np
from ..analysis.stats import (
    fwhm,
    centroid,
    compute_statistics,
    # evaluate_DPGMM,
    profile_from_gmm,
    convert_x_to_phase,
    robust_statistics
)
from .signal import remove_baseline
import copy

class Population:
    def __init__(self):
        self.pulsars = {}

    def add_pulsar(self, pulsar):
        self._pulsars[pulsar.jname] = pulsar

    def remove_pulsar(self, jname):
        try:
            del self._pulsars[jname]
        except KeyError:
            pass

    @property
    def pulsars(self):
        return self._pulsars

    @pulsars.setter
    def pulsars(self, pulsars):
        self._pulsars = pulsars

    def as_array(self, sort_on=None):
        pulsars = np.asarray([self.pulsars[jname] for jname in self.pulsars.keys()])
        if sort_on is not None:
            assert sort_on in ['jname', 'period', 'period_derivative','spindown_energy', 'distance', 'relative_width'], "sort_on must be one of ['jname', 'period', 'period_derivative','spindown_energy', 'distance', 'relative_width']"
            if sort_on == 'jname':
                idx = np.argsort(np.asarray([p.jname for p in pulsars]))
            if sort_on == 'period':
                idx = np.argsort(np.asarray([p.period for p in pulsars]))
            if sort_on == 'period_derivative':
                idx = np.argsort(np.asarray([p.period_derivative for p in pulsars]))
            if sort_on == 'spindown_energy':
                idx = np.argsort(np.asarray([p.spindown_energy for p in pulsars]))
            if sort_on == 'relative_width':
                # Period is in seconds, w10 is in ms
                idx = np.argsort(np.asarray([p.w10/(1000*p.period) for p in pulsars]))
            if sort_on == 'distance':
                try:
                    idx = np.argsort(np.asarray([p.distance for p in pulsars]))
                except AttributeError:
                    return pulsars
            return pulsars[idx]
        else:
            return pulsars

    def scale_set(self):
        """Scale pulsars profiles to [0, 1] range.

        Stokes Q, U, V, and L are normalized relative to stokes I.
        """
        for jname in self.pulsars.keys():
            for i, freq in enumerate(self.pulsars[jname].observations.keys()):
                obs = self.pulsars[jname].observations[freq]

class Pulsar:
    def __init__(self, jname, bname=None, index=None, period=None, period_derivative=None, spindown_energy=None,
                       bsurf=None, w10=None, raj=None, decj=None, gl=None, gb=None, distances=None, dtw_paths=None,
                       atnf_query=None, morphological_class=None, morphological_code=None):
        self.jname = jname
        self.bname = bname
        self.index = index
        self.period = period
        self.period_derivative = period_derivative
        self.spindown_energy = spindown_energy
        self.bsurf = bsurf
        self.w10 = w10
        self.raj = raj
        self.decj = decj
        self.gl = gl
        self.gb = gb
        self.distances = distances
        self.dtw_paths = dtw_paths
        self.atnf_query = atnf_query
        self.observations = {}

        # Esoteric section
        self.morphological_class = morphological_class
        self.morphological_code = morphological_code

    def add_observation(self, observation):
        self._observations[observation.frequency_range] = observation

    @property
    def observations(self):
        return self._observations

    @observations.setter
    def observations(self, observations):
        self._observations = observations

    @property
    def jname(self):
        return self._jname

    #jname.setter
    def jname(self, jname):
        self._jname = jname

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        self._index = index

    @property
    def period(self):
        """ P """
        return self._period

    @period.setter
    def period(self, period):
        self._period = period

    @property
    def period_derivative(self):
        """ P DOT """
        return self._period_derivative

    @period_derivative.setter
    def period_derivative(self, period_derivative):
        self._period_derivative = period_derivative

    @property
    def spindown_energy(self):
        """ E DOT """
        return self._spindown_energy

    @spindown_energy.setter
    def spindown_energy(self, spindown_energy):
        self._spindown_energy = spindown_energy

    @property
    def bsurf(self):
        return self._bsurf

    @bsurf.setter
    def bsurf(self, bsurf):
        self._bsurf = bsurf

    @property
    def w10(self):
        return self._w10

    @w10.setter
    def w10(self, w10):
        self._w10 = w10

    # I'm adding this field to compute the widths myself using L-band
    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width):
        self._width = width

    @property
    def raj(self):
        return self._raj

    @raj.setter
    def raj(self, raj):
        self._raj = raj

    @property
    def decj(self):
        return self._decj

    @decj.setter
    def decj(self, decj):
        self._decj = decj

    @property
    def gl(self):
        """ E DOT """
        return self._gl

    @gl.setter
    def gl(self, gl):
        self._gl = gl

    @property
    def gb(self):
        """ E DOT """
        return self._gb

    @gb.setter
    def gb(self, gb):
        self._gb = gb

    @property
    def distances(self):
        """ E DOT """
        return self._distances

    @distances.setter
    def distances(self, distances):
        self._distances = distances
        self.distance = self.distance

    @property
    def distances_L(self):
        """ E DOT """
        return self._distances_L

    @distances_L.setter
    def distances_L(self, distances_L):
        self._distances_L = distances_L
        self.distance = self.distance

    @property
    def distances_V(self):
        """ E DOT """
        return self._distances_V

    @distances_V.setter
    def distances_V(self, distances_V):
        self._distances_V = distances_V
        self.distance = self.distance

    @property
    def Ns(self):
        """ E DOT """
        return self._Ns

    @Ns.setter
    def Ns(self, Ns):
        self._Ns = Ns

    @property
    def distance(self):
        """ E DOT """
        try:
            # ds = self.distances
            # self.distance = np.sqrt(np.sum([(ds[i] - ds[i-1])**2 for i in range(ds.size-1)]))
            # self.distance = self.distances.mean() * (self.distances.mean() / self.distances.std())
            # print (self.jname, self.distances)
            # self.distance = (self.distances.max() - self.distances.mean())/ self.distances.std()
            # n = (self.w10 / (1000 * self.period)) * 1024
            self.distance = (self.distances.std() + self.distances_L.std()) / 2
        except NameError:
            self.distance = None
        except AttributeError:
            self.distance = None
        return self._distance

    @distance.setter
    def distance(self, distance):
        self._distance = distance

    @property
    def dtw_paths(self):
        """ E DOT """
        return self._dtw_paths

    @dtw_paths.setter
    def dtw_paths(self, dtw_paths):
        self._dtw_paths = dtw_paths

class Observation:
    def __init__(self,
                 frequency=None,
                 frequency_range=None,
                 phase=None,
                 stokes_I=None,
                 stokes_Q=None,
                 stokes_U=None,
                 stokes_V=None,
                 stokes_L=None,  # total linear polarization L
                 position_angle=None,
                 position_angle_phase = None,
                 position_angle_yerr_low = None,
                 position_angle_yerr_high = None,
                 model=None,
                 model_components=None,
                 gmm=None,
                 snr=None,
                 epn_reference_code=None,
                 file_location=None,
                 original_stokes_size=None
                ):

        self.frequency = frequency
        self.frequency_range = frequency_range
        self.phase = phase
        self.stokes_I = stokes_I
        self.stokes_Q = stokes_Q
        self.stokes_U = stokes_U
        self.stokes_V = stokes_V
        self.stokes_L = stokes_L
        self.position_angle = position_angle
        self.position_angle_yerr_low = position_angle_yerr_low
        self.position_angle_yerr_high = position_angle_yerr_low
        self.model = model
        self.model_components = model_components
        self.gmm = gmm
        self.snr = snr
        self.epn_reference_code = epn_reference_code
        self.file_location = file_location
        self.original_stokes_size = original_stokes_size

    @property
    def epn_reference_code(self):
        return self._epn_reference_code

    @epn_reference_code.setter
    def epn_reference_code(self, epn_reference_code):
        self._epn_reference_code = epn_reference_code

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, frequency):
        self._frequency = frequency

    @property
    def frequency_range(self):
        return self._frequency_range

    @frequency_range.setter
    def frequency_range(self, frequency_range):
        self._frequency_range = frequency_range

    @property
    def frequency_range_label(self):
        frequency_range_to_label = {
            0: u'[0-200)', # in MHz
            1: u'[200-400)',
            2: u'[400-700)',
            3: u'[700-1000)',
            4: u'[1000-1500)',
            5: u'[1500-2000)',
            6: u'[2000,)'
        }
        return frequency_range_to_label[self._frequency_range]

    @property
    def fwhm(self):
        return self._fwhm

    @fwhm.setter
    def fwhm(self, fwhm):
        self._fwhm = fwhm

    @property
    def centroid(self):
        return self._centroid

    @centroid.setter
    def centroid(self, centroid):
        self._centroid = centroid

    @property
    def snr(self):
        return self._snr

    @snr.setter
    def snr(self, snr):
        self._snr = snr

    @property
    def morphological_class(self):
        return self._morphological_class

    @morphological_class.setter
    def morphological_class(self, morphological_class):
        self._morphological_class = morphological_class

    @property
    def morphological_class_predict(self):
        return self._morphological_class_predict

    @morphological_class_predict.setter
    def morphological_class_predict(self, morphological_class_predict):
        self._morphological_class_predict = morphological_class_predict

    @property
    def morphological_code(self):
        return self._morphological_code

    @morphological_code.setter
    def morphological_code(self, morphological_code):
        self._morphological_code = morphological_code

    @property
    def central(self):
        return self._central

    @central.setter
    def central(self, central):
        self._central = central

    @property
    def sigma_noise(self):
        return self._sigma_noise

    @sigma_noise.setter
    def sigma_noise(self, sigma_noise):
        self._sigma_noise = sigma_noise

    @property
    def original_stokes_size(self):
        return self._original_stokes_size

    @original_stokes_size.setter
    def original_stokes_size(self, original_stokes_size):
        self._original_stokes_size = original_stokes_size

    @property
    def phase(self):
        if hasattr(self, '_phase'):
            if self._phase is None:
                self.set_phase(prop='stokes_I')
        else:
            self.set_phase(prop='stokes_I')

        return self._phase

    @phase.setter
    def phase(self, phase):
        self._phase = phase

    @property
    def stokes_I(self):
        return self._stokes_I

    @stokes_I.setter
    def stokes_I(self, stokes_I):
        self._stokes_I = stokes_I
        if stokes_I is not None:
            self.set_phase('stokes_I')
            self.set_fwhm('stokes_I')
            self.set_centroid('stokes_I')
            self.set_snr('stokes_I')
            # self.central, self.sigma_noise, _ = compute_statistics(self.stokes_I)
            self.central, self.sigma_noise, _ = robust_statistics(self.stokes_I)

    @property
    def stokes_I_znormed(self):
        return self._stokes_I_znormed

    @stokes_I_znormed.setter
    def stokes_I_znormed(self, stokes_I_znormed):
        self._stokes_I_znormed = stokes_I_znormed

    @property
    def stokes_Q(self):
        return self._stokes_Q

    @stokes_Q.setter
    def stokes_Q(self, stokes_Q):
        self._stokes_Q = stokes_Q

    @property
    def stokes_U(self):
        return self._stokes_U

    @stokes_U.setter
    def stokes_U(self, stokes_U):
        self._stokes_U = stokes_U

    @property
    def stokes_V(self):
        return self._stokes_V

    @stokes_V.setter
    def stokes_V(self, stokes_V):
        self._stokes_V = stokes_V

    @property
    def stokes_L(self):
        return self._stokes_L

    @stokes_L.setter
    def stokes_L(self, stokes_L):
        # if stokes_L is not None and self._stokes_L is None:
        #     # From "Handbook of pulsar astronomy", Section 7.4.3.1, p. 187.
        #     cond = np.where((stokes_L / self.sigma_noise) >= 1.57)
        #     stokes_L[cond] = self.sigma_noise * ((stokes_L[cond] / self.sigma_noise)**2 - 1)**0.5
        #     stokes_L[np.where((stokes_L / self.sigma_noise) < 1.57)] = 0
        #     self._stokes_L = stokes_L
        # else:
        self._stokes_L = stokes_L

    def set_linear_polarization_degree(self):
        if self.stokes_L is not None:
            norm = lambda arr: arr - robust_statistics(arr)[0]
            self.stokes_L_indices = np.where(((self.stokes_I - self.central) / self.sigma_noise) > 3)
            linear_polarization_degree = np.zeros(self.stokes_I.size)
            try:
                linear_polarization_degree[self.stokes_L_indices] = np.divide(norm(self.stokes_L[self.stokes_L_indices]),
                                                                              self.stokes_I[self.stokes_L_indices])
                l = linear_polarization_degree[self.stokes_L_indices]
                self.linear_polarization_degree = linear_polarization_degree
                self.mean_linear_polarization_degree = np.mean(l[np.where(np.abs(l) < np.inf)[0]])
            except ValueError:
                self.linear_polarization_degree = None
        else:
            self.linear_polarization_degree = None

    @property
    def linear_polarization_degree(self):
        return self._linear_polarization_degree

    @linear_polarization_degree.setter
    def linear_polarization_degree(self, linear_polarization_degree):
        self._linear_polarization_degree = linear_polarization_degree

    @property
    def mean_linear_polarization_degree(self):
        # norm = lambda arr: arr - robust_statistics(arr)[0]
        # linear_polarization_degree = np.zeros(self.stokes_I.size)
        # linear_polarization_degree[self.stokes_L_indices] = np.divide(norm(self.stokes_L[self.stokes_L_indices]),
        #                                                               self.stokes_I[self.stokes_L_indices])
        # l = linear_polarization_degree[self.stokes_L_indices]
        # self.mean_linear_polarization_degree = np.mean(l[np.where(np.abs(l) < np.inf)[0]])
        return self._mean_linear_polarization_degree

    @mean_linear_polarization_degree.setter
    def mean_linear_polarization_degree(self, mean_linear_polarization_degree):
        self._mean_linear_polarization_degree = mean_linear_polarization_degree

    def set_circular_polarization_degree(self):
        if self.stokes_V is not None:
            norm = lambda arr: arr - robust_statistics(arr)[0]
            indices = np.where(((self.stokes_I - self.central) / self.sigma_noise) > 3)
            try:
                degree = np.divide(norm(self.stokes_V[indices]), self.stokes_I[indices])
                self.circular_polarization_degree = degree
                self.mean_circular_polarization_degree = np.mean(degree[np.where(np.abs(degree) < np.inf)[0]])
            except ValueError:
                self.circular_polarization_degree = None
        else:
            self.circular_polarization_degree = None

    @property
    def circular_polarization_degree(self):
        return self._circular_polarization_degree

    @circular_polarization_degree.setter
    def circular_polarization_degree(self, circular_polarization_degree):
        self._circular_polarization_degree = circular_polarization_degree

    @property
    def mean_circular_polarization_degree(self, average=True):
        # norm = lambda arr: arr - robust_statistics(arr)[0]
        # indices = np.where(((self.stokes_I - self.central) / self.sigma_noise) > 3)
        # degree = np.divide(norm(self.stokes_V[indices]), self.stokes_I[indices])
        # self.mean_circular_polarization_degree = np.mean(degree[np.where(np.abs(degree) < np.inf)[0]])

        return self._mean_circular_polarization_degree

    @mean_circular_polarization_degree.setter
    def mean_circular_polarization_degree(self, mean_circular_polarization_degree):
        self._mean_circular_polarization_degree = mean_circular_polarization_degree

    @property
    def position_angle(self):
        return self._position_angle

    @position_angle.setter
    def position_angle(self, position_angle):
        self._position_angle = position_angle

    @property
    def position_angle_yerr_low(self):
        return self._position_angle_yerr_low

    @position_angle_yerr_low.setter
    def position_angle_yerr_low(self, position_angle_yerr_low):
        self._position_angle_yerr_low = position_angle_yerr_low

    @property
    def position_angle_yerr_high(self):
        return self._position_angle_yerr_high

    @position_angle_yerr_high.setter
    def position_angle_yerr_high(self, position_angle_yerr_high):
        self._position_angle_yerr_high = position_angle_yerr_high

    # @property
    # def position_angle(self):
    #     return self._position_angle
    #
    # @position_angle.setter
    # def position_angle(self, position_angle):
    #     self._position_angle = np.empty(self.stokes_I.size)
    #     self._position_angle[:] = np.NaN
    #
    #     # Would be nice but currently doesn't automatically update when array is resized...
    #     # if 'valid_indices' in self.__dict__.keys():
    #     #     try:
    #     #         self._position_angle[self.valid_indices] = position_angle[self.valid_indices]
    #     #     except IndexError:
    #     #         print (self.valid_indices)
    #     #         exit(1)
    #     # else:
    #     self.valid_indices = np.where( ((self.stokes_I - self.central) / self.sigma_noise) > 3 )
    #     self._position_angle[self.valid_indices] = position_angle[self.valid_indices]



    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def model_components(self):
        return self._model_components

    def scale_model_components(self):
        for i in range(len(self.model_components)):
            self.model_components[i] = (self.model_components[i] - self.model.max()) / (self.model.max() - self.model.min())
            self.model_components[i] = self.model_components[i] - self.model_components[i].min()

    @model_components.setter
    def model_components(self, model_components):
        self._model_components = model_components

    @property
    def gmm(self):
        return self._gmm

    @gmm.setter
    def gmm(self, gmm):
        self._gmm = gmm

    def get_property(self, prop='stokes_I', phase_fraction:float=None):
        if 'freq' in prop:
            X = self.frequency

        if 'I' in prop:
            X = self.stokes_I

        if 'Q' in prop:
            X = self.stokes_Q

        if 'U' in prop:
            X = self.stokes_U

        if 'V' in prop:
            X = self.stokes_V

        if 'L' in prop:
            X = self.stokes_L

        if 'angle' in prop:
            X = self.position_angle

        if 'model' in prop:
            X = self.model

        if 'phase' in prop:
            X = self.phase

        return X if phase_fraction is None else X[np.where(np.abs(self.phase) < phase_fraction)]

    def get_centroid(self, prop='stokes_I'):
        return centroid(
            self.get_property(prop)
        )

    def set_centroid(self, prop='stokes_I'):
        self.centroid = self.get_centroid(prop)

    def get_fwhm(self, prop='stokes_I'):
        return fwhm(
            self.get_property(prop),
            return_dist=True
        )

    def set_fwhm(self, prop='stokes_I'):
        self.fwhm = self.get_fwhm(prop)

    def get_phase(self, prop='stokes_I'):
        return convert_x_to_phase(
            self.get_property(prop)
        )

    def set_phase(self, prop='stokes_I'):
        self.phase = self.get_phase(prop)

    def get_snr(self, prop='stokes_I'):
        return robust_statistics(
            self.get_property(prop)
        )[2]

    def set_snr(self, prop='stokes_I'):
        self.snr = self.get_snr(prop)

    # def get_model(self, prop='stokes_I',
    #                     n_components=30,
    #                     alpha=10**4,
    #                     tol=1e-3,
    #                     max_iter=1000,
    #                     mean_prior=None,
    #                     mean_precision_prior=None,
    #                     threshold=False,
    #                     fit_whole=True,
    #                     window=0.3,
    #                     phase_distribution_size=1000,
    #                     draw_random=False,
    #                     roll=False,
    #                     window_roll=None,
    #                     cut=False,
    #                     scale=False,
    #                     fwhm=None,
    #               ):
    #     profile = copy.deepcopy(self.get_property(prop)) if threshold else self.get_property(prop)
    #     if threshold:
    #         profile[np.where(np.abs(profile) <= self.central + 3 * self.sigma_noise)] = self.central + 3 * self.sigma_noise
    #         profile = (profile - profile.min()) / (profile.max() - profile.min())
    #
    #     if not fit_whole:
    #         cond = np.where(np.abs(self.phase) < window)
    #
    #     if roll:
    #         n_roll = profile.size//2 if window_roll is None else window_roll
    #         profile = np.roll(profile, n_roll)
    #
    #     # Will need to handle two-case situations when with_interpulse is true
    #     self.gmm = evaluate_DPGMM(
    #         profile if fit_whole else profile[cond],
    #         self.get_property('phase') if fit_whole else self.phase[cond],
    #         n_components_start = n_components,
    #         alpha = alpha,
    #         tol=tol,
    #         max_iter=max_iter,
    #         mean_prior=mean_prior,
    #         mean_precision_prior=mean_precision_prior,
    #         phase_distribution_size=phase_distribution_size,
    #         draw_random=draw_random
    #     )
    #
    #     if roll == 0:
    #         self.model, self.model_components = profile_from_gmm(
    #             self,
    #             cut=cut,
    #             scale=scale,
    #             fwhm=fwhm,
    #             fit_whole=fit_whole,
    #             window=window
    #         )
    #     else:
    #         self.stokes_I = profile  # assumes prop is stokes I
    #         model, model_components = profile_from_gmm(
    #             self,
    #             cut=cut,
    #             scale=scale,
    #             fwhm=fwhm,
    #             fit_whole=fit_whole,
    #             window=window,
    #             interpulse=True
    #         )
    #
    #         model = np.roll(model, -n_roll)
    #         self.model = np.mean([self.model, model], axis=0)
    #         for i, c in enumerate(model_components):
    #             model_components[i] = np.roll(c, -n_roll)
    #         self.model_components = np.append(self.model_components, model_components, axis=0)
    #
    #         self.stokes_I = np.roll(self.stokes_I, -n_roll)
    #
    #     return self.gmm
    #
    # # The experimental changes to deal with interpulses made this function deprecated potentially...
    # def set_model(self, prop='stokes_I',
    #                     threshold=False,
    #                     alpha=10**4,
    #                     tol=1e-3,
    #                     max_iter=1000,
    #                     n_components=30,
    #                     cut=False,
    #                     scale=False,
    #                     override=False,
    #                     mean_prior=None,
    #                     mean_precision_prior=None,
    #                     fwhm=None,
    #                     fit_whole=True,
    #                     window=0.3,
    #                     window_roll=None,
    #                     phase_distribution_size=1000,
    #                     draw_random=False,
    #                     has_interpulse=False
    #               ):
    #
    #     for roll in range(2 if has_interpulse else 1):
    #         if self.gmm is None or override:
    #             self.get_model(prop,
    #                            n_components=n_components,
    #                            alpha=alpha,
    #                            tol=tol,
    #                            max_iter=max_iter,
    #                            mean_prior=mean_prior,
    #                            mean_precision_prior=mean_precision_prior,
    #                            threshold=threshold,
    #                            fit_whole=fit_whole,
    #                            window=window,
    #                            phase_distribution_size=phase_distribution_size,
    #                            draw_random=draw_random,
    #                            cut=cut,
    #                            scale=scale,
    #                            fwhm=fwhm,
    #                            roll=roll,
    #                            window_roll=window_roll)

    def remove_baseline(self, prop='stokes_I'):
        return remove_baseline(
            self.get_property(prop)
        )






class Component:
    def __init__(self, amplitude=None, fwhm=None, mean=None):
        self.amplitude = amplitude
        self.fwhm = fwhm
        self.mean = mean

    @property
    def amplitude(self):
        return self._amplitude

    @amplitude.setter
    def amplitude(self, amplitude):
        self._amplitude = amplitude

    @property
    def fwhm(self):
        return self._fwhm

    @fwhm.setter
    def fwhm(self, fwhm):
        self._fwhm = fwhm

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = mean

class Model:
    def __init__(self, components:[]=None, phase:np.array=None, rms=None):
        self.components = [] if components is None else components
        self.phase = phase
        self.rms = rms

    def gaussian(self, amp, fwhm, mean):
        return lambda x: amp * np.exp(-4. * np.log(2) * (x-mean)**2 / fwhm**2)

    def add_component(self, amplitude, fwhm, mean):
        self.components.append(Component(amplitude, fwhm, mean))

    @property
    def model(self):
        model = self.noise
        for c in self.components:
            # print (c.amplitude, c.fwhm, c.mean)
            model = model + self.gaussian(c.amplitude, c.fwhm, c.mean)(self.phase)
        return model

    @property
    def noise(self):
        return np.random.randn(self.phase.size) * self.rms

    @property
    def components(self):
        return self._components

    @components.setter
    def components(self, components):
        self._components = components

    @property
    def amplitudes(self):
        return [c.amplitude for c in self.components]

    @property
    def fwhms(self):
        return [c.fwhm for c in self.components]

    @property
    def means(self):
        return [c.mean for c in self.components]


