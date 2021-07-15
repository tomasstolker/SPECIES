"""
Module with reading functionalities for atmospheric model spectra.
"""

import os
import math
import warnings
import configparser

from typing import Optional, Dict, Tuple, List

import h5py
import spectres
import numpy as np

from typeguard import typechecked
from scipy.interpolate import interp1d, interp2d, RegularGridInterpolator

from species.analysis import photometry
from species.core import box, constants
from species.data import database
from species.read import read_calibration, read_filter, read_planck
from species.util import dust_util, mcmc_util, read_util


class ReadModel:
    """
    Class for reading a model spectrum from the database.
    """

    @typechecked
    def __init__(self,
                 model: str,
                 wavel_range: Optional[Tuple[float, float]] = None,
                 filter_name: Optional[str] = None):
        """
        Parameters
        ----------
        model : str
            Name of the atmospheric model.
        wavel_range : tuple(float, float), None
            Wavelength range (um). Full spectrum is selected if set to ``None``. Not used if
            ``filter_name`` is not ``None``.
        filter_name : str, None
            Filter name that is used for the wavelength range. The ``wavel_range`` is used if set
            to ``None``.

        Returns
        -------
        NoneType
            None
        """

        self.model = model

        self.spectrum_interp = None
        self.wl_points = None
        self.wl_index = None

        self.filter_name = filter_name
        self.wavel_range = wavel_range

        if self.filter_name is not None:
            transmission = read_filter.ReadFilter(self.filter_name)

            self.wavel_range = transmission.wavelength_range()
            self.mean_wavelength = transmission.mean_wavelength()

        else:
            self.mean_wavelength = None

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

        self.extra_param = ['radius', 'distance', 'mass', 'luminosity', 'lognorm_radius',
                            'lognorm_sigma', 'lognorm_ext', 'ism_ext', 'ism_red', 'powerlaw_max',
                            'powerlaw_exp', 'powerlaw_ext', 'disk_teff', 'disk_radius',
                            'veil_a', 'veil_b', 'veil_ref']

    @typechecked
    def open_database(self) -> h5py._hl.files.File:
        """
        Internal function for opening the `species` database.

        Returns
        -------
        h5py._hl.files.File
            The HDF5 database.
        """

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        database_path = config['species']['database']

        h5_file = h5py.File(database_path, 'r')

        try:
            h5_file[f'models/{self.model}']

        except KeyError:
            raise ValueError(f'The \'{self.model}\' model spectra are not present in the '
                             f'database.')

        return h5_file

    @typechecked
    def wavelength_points(self,
                          hdf5_file: h5py._hl.files.File) -> Tuple[np.ndarray, np.ndarray]:
        """
        Internal function for extracting the wavelength points and indices that are used.

        Parameters
        ----------
        hdf5_file : h5py._hl.files.File
            The HDF5 database.

        Returns
        -------
        np.ndarray
            Wavelength points (um).
        np.ndarray
            Array with the size of the original wavelength grid. The booleans indicate if a
            wavelength point was used.
        """

        wl_points = np.asarray(hdf5_file[f'models/{self.model}/wavelength'])

        if self.wavel_range is None:
            wl_index = np.ones(wl_points.shape[0], dtype=bool)

        else:
            wl_index = (wl_points > self.wavel_range[0]) & (wl_points < self.wavel_range[1])
            index = np.where(wl_index)[0]

            if index[0]-1 >= 0:
                wl_index[index[0] - 1] = True

            if index[-1]+1 < wl_index.size:
                wl_index[index[-1] + 1] = True

        return wl_points[wl_index], wl_index

    @typechecked
    def interpolate_model(self) -> None:
        """
        Internal function for linearly interpolating the full grid of model spectra.

        Returns
        -------
        NoneType
            None
        """

        h5_file = self.open_database()

        points = []
        for item in self.get_points().values():
            points.append(item)

        if self.wl_points is None:
            self.wl_points, self.wl_index = self.wavelength_points(h5_file)

        flux = np.asarray(h5_file[f'models/{self.model}/flux'])
        flux = flux[..., self.wl_index]

        self.spectrum_interp = RegularGridInterpolator(points,
                                                       flux,
                                                       method='linear',
                                                       bounds_error=False,
                                                       fill_value=np.nan)

    @typechecked
    def interpolate_grid(self,
                         wavel_resample: Optional[np.ndarray] = None,
                         smooth: bool = False,
                         spec_res: Optional[float] = None) -> None:
        """
        Internal function for linearly interpolating the grid of model spectra for a given
        filter or wavelength sampling.

        wavel_resample : np.ndarray, None
            Wavelength points for the resampling of the spectrum. The ``filter_name`` is used
            if set to ``None``.
        smooth : bool
            Smooth the spectrum with a Gaussian line spread function. Only recommended in case the
            input wavelength sampling has a uniform spectral resolution.
        spec_res : float
            Spectral resolution that is used for the Gaussian filter when ``smooth=True``.

        Returns
        -------
        NoneType
            None
        """

        self.interpolate_model()

        if smooth and wavel_resample is None:
            raise ValueError('Smoothing is only required if the spectra are resampled to a new '
                             'wavelength grid, therefore requiring the \'wavel_resample\' '
                             'argument.')

        points = []
        for item in self.get_points().values():
            points.append(list(item))

        param = self.get_parameters()
        n_param = len(param)

        dim_size = []
        for item in points:
            dim_size.append(len(item))

        if self.filter_name is not None:
            dim_size.append(1)
        else:
            dim_size.append(wavel_resample.size)

        flux_new = np.zeros(dim_size)

        if n_param == 1:
            model_param = {}

            for i, item_0 in enumerate(points[0]):
                model_param[param[0]] = item_0

                if self.filter_name is not None:
                    flux_new[i] = self.get_flux(model_param)[0]

                else:
                    flux_new[i, :] = self.get_model(model_param,
                                                    spec_res=spec_res,
                                                    wavel_resample=wavel_resample,
                                                    smooth=smooth).flux

        elif n_param == 2:
            model_param = {}

            for i, item_0 in enumerate(points[0]):
                for j, item_1 in enumerate(points[1]):
                    model_param[param[0]] = item_0
                    model_param[param[1]] = item_1

                    if self.filter_name is not None:
                        flux_new[i, j] = self.get_flux(model_param)[0]

                    else:
                        flux_new[i, j, :] = self.get_model(model_param,
                                                           spec_res=spec_res,
                                                           wavel_resample=wavel_resample,
                                                           smooth=smooth).flux

        elif n_param == 3:
            model_param = {}

            for i, item_0 in enumerate(points[0]):
                for j, item_1 in enumerate(points[1]):
                    for k, item_2 in enumerate(points[2]):
                        model_param[param[0]] = item_0
                        model_param[param[1]] = item_1
                        model_param[param[2]] = item_2

                        if self.filter_name is not None:
                            flux_new[i, j, k] = self.get_flux(model_param)[0]

                        else:
                            flux_new[i, j, k, :] = self.get_model(model_param,
                                                                  spec_res=spec_res,
                                                                  wavel_resample=wavel_resample,
                                                                  smooth=smooth).flux

        elif n_param == 4:
            model_param = {}

            for i, item_0 in enumerate(points[0]):
                for j, item_1 in enumerate(points[1]):
                    for k, item_2 in enumerate(points[2]):
                        for m, item_3 in enumerate(points[3]):
                            model_param[param[0]] = item_0
                            model_param[param[1]] = item_1
                            model_param[param[2]] = item_2
                            model_param[param[3]] = item_3

                            if self.filter_name is not None:
                                flux_new[i, j, k, m] = self.get_flux(model_param)[0]

                            else:
                                flux_new[i, j, k, m, :] = self.get_model(
                                    model_param, spec_res=spec_res, wavel_resample=wavel_resample,
                                    smooth=smooth).flux

        elif n_param == 5:
            model_param = {}

            for i, item_0 in enumerate(points[0]):
                for j, item_1 in enumerate(points[1]):
                    for k, item_2 in enumerate(points[2]):
                        for m, item_3 in enumerate(points[3]):
                            for n, item_4 in enumerate(points[4]):
                                model_param[param[0]] = item_0
                                model_param[param[1]] = item_1
                                model_param[param[2]] = item_2
                                model_param[param[3]] = item_3
                                model_param[param[4]] = item_4

                                if self.filter_name is not None:
                                    flux_new[i, j, k, m, n] = self.get_flux(model_param)[0]

                                else:
                                    flux_new[i, j, k, m, n, :] = self.get_model(
                                        model_param, spec_res=spec_res,
                                        wavel_resample=wavel_resample, smooth=smooth).flux

        if self.filter_name is not None:
            transmission = read_filter.ReadFilter(self.filter_name)
            self.wl_points = [transmission.mean_wavelength()]

        else:
            self.wl_points = wavel_resample

        self.spectrum_interp = RegularGridInterpolator(points,
                                                       flux_new,
                                                       method='linear',
                                                       bounds_error=False,
                                                       fill_value=np.nan)

    @staticmethod
    @typechecked
    def apply_ext_lnorm_distr(wavelength: np.ndarray,
                              flux: np.ndarray,
                              radius_interp: float,
                              sigma_interp: float,
                              v_band_ext: float) -> np.ndarray:
        """
        Internal function for applying extinction by dust to a spectrum.

        wavelength : np.ndarray
            Wavelengths (um) of the spectrum.
        flux : np.ndarray
            Fluxes (W m-2 um-1) of the spectrum.
        radius_interp : float
            Logarithm of the mean geometric radius (um) of the log-normal size distribution.
        sigma_interp : float
            Geometric standard deviation (dimensionless) of the log-normal size distribution.
        v_band_ext : float
            The extinction (mag) in the V band.

        Returns
        -------
        np.ndarray
            Fluxes (W m-2 um-1) with the extinction applied.
        """

        database_path = dust_util.check_dust_database()

        with h5py.File(database_path, 'r') as h5_file:
            dust_cross = np.asarray(h5_file['dust/lognorm/mgsio3/crystalline/cross_section'])
            dust_wavel = np.asarray(h5_file['dust/lognorm/mgsio3/crystalline/wavelength'])
            dust_radius = np.asarray(h5_file['dust/lognorm/mgsio3/crystalline/radius_g'])
            dust_sigma = np.asarray(h5_file['dust/lognorm/mgsio3/crystalline/sigma_g'])

        if wavelength[0] < dust_wavel[0]:
            raise ValueError(f'The shortest wavelength ({wavelength[0]:.2e} um) for which the '
                             f'spectrum will be calculated is smaller than the shortest '
                             f'wavelength ({dust_wavel[0]:.2e} um) of the grid with dust cross '
                             f'sections.')

        if wavelength[-1] > dust_wavel[-1]:
            raise ValueError(f'The longest wavelength ({wavelength[-1]:.2e} um) for which the '
                             f'spectrum  will be calculated is larger than the longest wavelength '
                             f'({dust_wavel[-1]:.2e} um) of the grid with dust cross sections.')

        dust_interp = RegularGridInterpolator((dust_wavel, dust_radius, dust_sigma),
                                              dust_cross,
                                              method='linear',
                                              bounds_error=True)

        read_filt = read_filter.ReadFilter('Generic/Bessell.V')
        filt_trans = read_filt.get_filter()

        cross_phot = np.zeros((dust_radius.shape[0], dust_sigma.shape[0]))

        for i in range(dust_radius.shape[0]):
            for j in range(dust_sigma.shape[0]):
                cross_interp = interp1d(dust_wavel,
                                        dust_cross[:, i, j],
                                        kind='linear',
                                        bounds_error=True)

                cross_tmp = cross_interp(filt_trans[:, 0])

                integral1 = np.trapz(filt_trans[:, 1]*cross_tmp, filt_trans[:, 0])
                integral2 = np.trapz(filt_trans[:, 1], filt_trans[:, 0])

                # Filter-weighted average of the extinction cross section
                cross_phot[i, j] = integral1/integral2

        cross_interp = interp2d(dust_sigma,
                                dust_radius,
                                cross_phot,
                                kind='linear',
                                bounds_error=True)

        cross_v_band = cross_interp(sigma_interp, 10.**radius_interp)[0]

        radius_full = np.full(wavelength.shape[0], 10.**radius_interp)
        sigma_full = np.full(wavelength.shape[0], sigma_interp)

        cross_new = dust_interp(np.column_stack((wavelength, radius_full, sigma_full)))

        n_grains = v_band_ext / cross_v_band / 2.5 / np.log10(np.exp(1.))

        return flux * np.exp(-cross_new*n_grains)

    @staticmethod
    @typechecked
    def apply_ext_plaw_distr(wavelength: np.ndarray,
                             flux: np.ndarray,
                             r_max_interp: float,
                             exp_interp: float,
                             v_band_ext: float) -> np.ndarray:
        """
        Internal function for applying extinction by dust to a spectrum.

        wavelength : np.ndarray
            Wavelengths (um) of the spectrum.
        flux : np.ndarray
            Fluxes (W m-2 um-1) of the spectrum.
        r_max_interp : float
            Maximum radius (um) of the power-law size distribution.
        exp_interp : float
            Exponent of the power-law size distribution.
        v_band_ext : float
            The extinction (mag) in the V band.

        Returns
        -------
        np.ndarray
            Fluxes (W m-2 um-1) with the extinction applied.
        """

        database_path = dust_util.check_dust_database()

        with h5py.File(database_path, 'r') as h5_file:
            dust_cross = np.asarray(h5_file['dust/powerlaw/mgsio3/crystalline/cross_section'])
            dust_wavel = np.asarray(h5_file['dust/powerlaw/mgsio3/crystalline/wavelength'])
            dust_r_max = np.asarray(h5_file['dust/powerlaw/mgsio3/crystalline/radius_max'])
            dust_exp = np.asarray(h5_file['dust/powerlaw/mgsio3/crystalline/exponent'])

        dust_interp = RegularGridInterpolator((dust_wavel, dust_r_max, dust_exp),
                                              dust_cross,
                                              method='linear',
                                              bounds_error=True)

        if wavelength[0] < dust_wavel[0]:
            raise ValueError(f'The shortest wavelength ({wavelength[0]:.2e} um) for which the '
                             f'spectrum will be calculated is smaller than the shortest '
                             f'wavelength ({dust_wavel[0]:.2e} um) of the grid with dust cross '
                             f'sections.')

        if wavelength[-1] > dust_wavel[-1]:
            raise ValueError(f'The longest wavelength ({wavelength[-1]:.2e} um) for which the '
                             f'spectrum  will be calculated is larger than the longest wavelength '
                             f'({dust_wavel[-1]:.2e} um) of the grid with dust cross sections.')

        read_filt = read_filter.ReadFilter('Generic/Bessell.V')
        filt_trans = read_filt.get_filter()

        cross_phot = np.zeros((dust_r_max.shape[0], dust_exp.shape[0]))

        for i in range(dust_r_max.shape[0]):
            for j in range(dust_exp.shape[0]):
                cross_interp = interp1d(dust_wavel,
                                        dust_cross[:, i, j],
                                        kind='linear',
                                        bounds_error=True)

                cross_tmp = cross_interp(filt_trans[:, 0])

                integral1 = np.trapz(filt_trans[:, 1]*cross_tmp, filt_trans[:, 0])
                integral2 = np.trapz(filt_trans[:, 1], filt_trans[:, 0])

                # Filter-weighted average of the extinction cross section
                cross_phot[i, j] = integral1/integral2

        cross_interp = interp2d(dust_exp,
                                dust_r_max,
                                cross_phot,
                                kind='linear',
                                bounds_error=True)

        cross_v_band = cross_interp(exp_interp, 10.**r_max_interp)[0]

        r_max_full = np.full(wavelength.shape[0], 10.**r_max_interp)
        exp_full = np.full(wavelength.shape[0], exp_interp)

        cross_new = dust_interp(np.column_stack((wavelength, r_max_full, exp_full)))

        n_grains = v_band_ext / cross_v_band / 2.5 / np.log10(np.exp(1.))

        return flux * np.exp(-cross_new*n_grains)

    @staticmethod
    @typechecked
    def apply_ext_ism(wavelengths: np.ndarray,
                      flux: np.ndarray,
                      v_band_ext: float,
                      v_band_red: float) -> np.ndarray:
        """
        Internal function for applying ISM extinction to a spectrum.

        wavelengths : np.ndarray
            Wavelengths (um) of the spectrum.
        flux : np.ndarray
            Fluxes (W m-2 um-1) of the spectrum.
        v_band_ext : float
            Extinction (mag) in the V band.
        v_band_red : float
            Reddening in the V band.

        Returns
        -------
        np.ndarray
            Fluxes (W m-2 um-1) with the extinction applied.
        """

        ext_mag = dust_util.ism_extinction(v_band_ext, v_band_red, wavelengths)

        return flux * 10.**(-0.4*ext_mag)

    @staticmethod
    @typechecked
    def apply_ext_powerlaw(wavelengths: np.ndarray,
                           flux: np.ndarray,
                           v_band_ext: float,
                           powerl_index: float) -> np.ndarray:
        """
        Internal function for applying ISM extinction to a spectrum.

        wavelengths : np.ndarray
            Wavelengths (um) of the spectrum.
        flux : np.ndarray
            Fluxes (W m-2 um-1) of the spectrum.
        v_band_ext : float
            Extinction (mag) in the V band.
        powerl_index : float
            Power-law index

        Returns
        -------
        np.ndarray
            Fluxes (W m-2 um-1) with the extinction applied.
        """

        ext_mag = dust_util.powerl_extinction(v_band_ext, powerl_index, wavelengths)

        return flux * 10.**(-0.4*ext_mag)

    @typechecked
    def get_model(self,
                  model_param: Dict[str, float],
                  spec_res: Optional[float] = None,
                  wavel_resample: Optional[np.ndarray] = None,
                  magnitude: bool = False,
                  smooth: bool = False) -> box.ModelBox:
        """
        Function for extracting a model spectrum by linearly interpolating the model grid.

        Parameters
        ----------
        model_param : dict
            Dictionary with the model parameters and values. The values should be within the
            boundaries of the grid. The grid boundaries of the spectra in the database can be
            obtained with :func:`~species.read.read_model.ReadModel.get_bounds()`.
        spec_res : float, None
            Spectral resolution that is used for smoothing the spectrum with a Gaussian kernel
            when ``smooth=True``. The wavelengths will be resampled to the argument of
            ``spec_res`` if ``smooth=False``.
        wavel_resample : np.ndarray, None
            Wavelength points (um) to which the spectrum is resampled. In that case, ``spec_res``
            can still be used for smoothing the spectrum with a Gaussian kernel. The original
            wavelength points are used if the argument is set to ``None``.
        magnitude : bool
            Normalize the spectrum with a flux calibrated spectrum of Vega and return the magnitude
            instead of flux density.
        smooth : bool
            If ``True``, the spectrum is smoothed with a Gaussian kernel to the spectral resolution
            of ``spec_res``. This requires either a uniform spectral resolution of the input
            spectra (fast) or a uniform wavelength spacing of the input spectra (slow).

        Returns
        -------
        species.core.box.ModelBox
            Box with the model spectrum.
        """

        # Get grid boundaries

        grid_bounds = self.get_bounds()

        # Check if all parameters are present and within the grid boundaries

        for key in self.get_parameters():
            if key not in model_param.keys():
                raise ValueError(f'The \'{key}\' parameter is required by \'{self.model}\'. '
                                 f'The mandatory parameters are {self.get_parameters()}.')

            if model_param[key] < grid_bounds[key][0]:
                raise ValueError(f'The input value of \'{key}\' is smaller than the lower '
                                 f'boundary of the model grid ({model_param[key]} < '
                                 f'{grid_bounds[key][0]}).')

            if model_param[key] > grid_bounds[key][1]:
                raise ValueError(f'The input value of \'{key}\' is larger than the upper '
                                 f'boundary of the model grid ({model_param[key]} > '
                                 f'{grid_bounds[key][1]}).')

        # Print a warning if redundant parameters are included in the dictionary

        for key in model_param.keys():
            if key not in self.get_parameters() and key not in self.extra_param:
                warnings.warn(f'The \'{key}\' parameter is not required by \'{self.model}\' so '
                              f'the parameter will be ignored. The mandatory parameters are '
                              f'{self.get_parameters()}.')

        # Interpolate the model grid

        if self.spectrum_interp is None:
            self.interpolate_model()

        # Set the wavelength range

        if self.wavel_range is None:
            wl_points = self.get_wavelengths()
            self.wavel_range = (wl_points[0], wl_points[-1])

        # Create a list with the parameter values

        parameters = []

        if 'teff' in model_param:
            parameters.append(model_param['teff'])

        if 'logg' in model_param:
            parameters.append(model_param['logg'])

        if 'feh' in model_param:
            parameters.append(model_param['feh'])

        if 'co' in model_param:
            parameters.append(model_param['co'])

        if 'fsed' in model_param:
            parameters.append(model_param['fsed'])

        # Interpolate the spectrum from the grid

        flux = self.spectrum_interp(parameters)[0]

        # Add the radius to the parameter dictionary if the mass if given

        if 'mass' in model_param and 'radius' not in model_param:
            mass = 1e3 * model_param['mass'] * constants.M_JUP  # (g)
            radius = math.sqrt(1e3 * constants.GRAVITY * mass / (10.**model_param['logg']))  # (cm)
            model_param['radius'] = 1e-2 * radius / constants.R_JUP  # (Rjup)

        # Apply (radius/distance)^2 scaling

        if 'radius' in model_param and 'distance' in model_param:
            scaling = (model_param['radius']*constants.R_JUP)**2 / \
                      (model_param['distance']*constants.PARSEC)**2

            flux *= scaling

        # Add blackbody disk component to the spectrum

        if 'disk_teff' in model_param and 'disk_radius' in model_param:
            disk_param = {'teff': model_param['disk_teff'],
                          'radius': model_param['disk_radius'],
                          'distance': model_param['distance']}

            readplanck = read_planck.ReadPlanck((0.9*self.wavel_range[0], 1.1*self.wavel_range[-1]))

            if spec_res is None:
                planck_box = readplanck.get_spectrum(disk_param, 1000., smooth=False)

            else:
                planck_box = readplanck.get_spectrum(disk_param, spec_res, smooth=False)

            flux += spectres.spectres(self.wl_points, planck_box.wavelength, planck_box.flux)

        # Create ModelBox with the spectrum

        model_box = box.create_box(boxtype='model',
                                   model=self.model,
                                   wavelength=self.wl_points,
                                   flux=flux,
                                   parameters=model_param,
                                   quantity='flux')

        # Apply veiling

        if 'veil_a' in model_param and 'veil_b' in model_param and 'veil_ref' in model_param:
            lambda_ref = 0.5  # (um)

            veil_flux = model_param['veil_ref'] + \
                model_param['veil_b']*(model_box.wavelength - lambda_ref)

            model_box.flux = model_param['veil_a']*model_box.flux + veil_flux

        # Apply extinction

        if 'lognorm_radius' in model_param and 'lognorm_sigma' in model_param and \
                'lognorm_ext' in model_param:

            model_box.flux = self.apply_ext_norm_distr(model_box.wavelength,
                                                       model_box.flux,
                                                       model_param['lognorm_radius'],
                                                       model_param['lognorm_sigma'],
                                                       model_param['lognorm_ext'])

        if 'powerlaw_max' in model_param and 'powerlaw_exp' in model_param and \
                'powerlaw_ext' in model_param:

            model_box.flux = self.apply_ext_plaw_distr(model_box.wavelength,
                                                       model_box.flux,
                                                       model_param['powerlaw_max'],
                                                       model_param['powerlaw_exp'],
                                                       model_param['powerlaw_ext'])

        if 'ism_ext' in model_param:
            ism_reddening = model_param.get('ism_red', 3.1)

            model_box.flux = self.apply_ext_ism(model_box.wavelength,
                                                model_box.flux,
                                                model_param['ism_ext'],
                                                ism_reddening)

        if 'powerl_ext' in model_param and 'powerl_exp' in model_param:

            model_box.flux = self.apply_ext_powerlaw(model_box.wavelength,
                                                     model_box.flux,
                                                     model_param['powerl_ext'],
                                                     model_param['powerl_exp'])

        # Smooth the spectrum

        if smooth and spec_res is not None:
            model_box.flux = read_util.smooth_spectrum(
                model_box.wavelength, model_box.flux, spec_res)

        elif smooth and spec_res is None:
            warnings.warn('Smoothing of a spectrum (smooth=True) is only possible when setting '
                          'the argument of \'spec_res\'.')

        # Resample the spectrum

        if wavel_resample is not None:
            model_box.flux = spectres.spectres(wavel_resample,
                                               model_box.wavelength,
                                               model_box.flux,
                                               spec_errs=None,
                                               fill=np.nan,
                                               verbose=True)

            model_box.wavelength = wavel_resample

        elif spec_res is not None and not smooth:
            index = np.where(np.isnan(model_box.flux))[0]

            if index.size > 0:
                raise ValueError('Flux values should not contains NaNs. Please make sure that '
                                 'the parameter values and the wavelength range are within '
                                 'the grid boundaries as stored in the database.')

            wavel_resample = read_util.create_wavelengths(
                (self.wl_points[0], self.wl_points[-1]), spec_res)

            indices = np.where((wavel_resample > self.wl_points[0]) &
                               (wavel_resample < self.wl_points[-2]))[0]

            wavel_resample = wavel_resample[indices]

            model_box.flux = spectres.spectres(wavel_resample,
                                               model_box.wavelength,
                                               model_box.flux,
                                               spec_errs=None,
                                               fill=np.nan,
                                               verbose=True)

            model_box.wavelength = wavel_resample

        # Convert flux to magnitude

        if magnitude:
            with h5py.File(self.database, 'r') as h5_file:
                try:
                    h5_file['spectra/calibration/vega']

                except KeyError:
                    h5_file.close()
                    species_db = database.Database()
                    species_db.add_spectra('vega')
                    h5_file = h5py.File(self.database, 'r')

            readcalib = read_calibration.ReadCalibration('vega', filter_name=None)
            calibbox = readcalib.get_spectrum()

            flux_vega, _ = spectres.spectres(model_box.wavelength,
                                             calibbox.wavelength,
                                             calibbox.flux,
                                             spec_errs=calibbox.error,
                                             fill=np.nan,
                                             verbose=True)

            model_box.flux = -2.5*np.log10(model_box.flux/flux_vega)
            model_box.quantity = 'magnitude'

        # Check if the contains NaNs

        if np.isnan(np.sum(model_box.flux)):
            warnings.warn(f'The resampled spectrum contains {np.sum(np.isnan(model_box.flux))} '
                          f'NaNs, probably because the original wavelength range does not fully '
                          f'encompass the new wavelength range. The happened with the '
                          f'following parameters: {model_param}.')

        # Add the luminosity to the parameter dictionary

        if 'radius' in model_box.parameters:
            model_box.parameters['luminosity'] = 4. * np.pi * (
                model_box.parameters['radius'] * constants.R_JUP)**2 * constants.SIGMA_SB * \
                model_box.parameters['teff']**4. / constants.L_SUN  # (Lsun)

        # Add the blackbody disk component to the luminosity

        if 'disk_teff' in model_box.parameters and 'disk_radius' in model_box.parameters:
            model_box.parameters['luminosity'] += 4. * np.pi * (
                model_box.parameters['disk_radius'] * constants.R_JUP)**2 * constants.SIGMA_SB * \
                model_box.parameters['disk_teff']**4. / constants.L_SUN  # (Lsun)

        # Add the planet mass to the parameter dictionary

        if 'radius' in model_param and 'logg' in model_param:
            model_param['mass'] = read_util.get_mass(model_param['logg'], model_param['radius'])

        return model_box

    @typechecked
    def get_data(self,
                 model_param: Dict[str, float],
                 spec_res: Optional[float] = None,
                 wavel_resample: Optional[np.ndarray] = None) -> box.ModelBox:
        """
        Function for selecting a model spectrum (without interpolation) for a set of parameter
        values that coincide with the grid points. The stored grid points can be inspected with
        :func:`~species.read.read_model.ReadModel.get_points`.

        Parameters
        ----------
        model_param : dict
            Model parameters and values. Only discrete values from the original grid are possible.
            Else, the nearest grid values are selected.
        spec_res : float, None
            Spectral resolution that is used for smoothing the spectrum with a Gaussian kernel. No
            smoothing is applied to the spectrum if the argument is set to ``None``.
        wavel_resample : np.ndarray, None
            Wavelength points (um) to which the spectrum will be resampled. In that case,
            ``spec_res`` can still be used for smoothing the spectrum with a Gaussian kernel. The
            original wavelength points are used if the argument is set to ``None``.

        Returns
        -------
        species.core.box.ModelBox
            Box with the model spectrum.
        """

        # Check if all parameters are present

        for key in self.get_parameters():
            if key not in model_param.keys():
                raise ValueError(f'The \'{key}\' parameter is required by \'{self.model}\'. '
                                 f'The mandatory parameters are {self.get_parameters()}.')

        # Print a warning if redundant parameters are included in the dictionary

        for key in model_param.keys():
            if key not in self.get_parameters() and key not in self.extra_param:
                warnings.warn(f'The \'{key}\' parameter is not required by \'{self.model}\' so '
                              f'the parameter will be ignored. The mandatory parameters are '
                              f'{self.get_parameters()}.')

        # Open de HDF5 database

        h5_file = self.open_database()

        # Set the wavelength range

        if self.wavel_range is None:
            wl_points = self.get_wavelengths()
            self.wavel_range = (wl_points[0], wl_points[-1])

        # Create lists with the parameter names and values

        param_key = []
        param_val = []

        if 'teff' in model_param:
            param_key.append('teff')
            param_val.append(model_param['teff'])

        if 'logg' in model_param:
            param_key.append('logg')
            param_val.append(model_param['logg'])

        if 'feh' in model_param:
            param_key.append('feh')
            param_val.append(model_param['feh'])

        if 'co' in model_param:
            param_key.append('co')
            param_val.append(model_param['co'])

        if 'fsed' in model_param:
            param_key.append('fsed')
            param_val.append(model_param['fsed'])

        # Read the grid of fluxes from the database

        flux = np.asarray(h5_file[f'models/{self.model}/flux'])

        # Find the indices of the grid points for which the spectrum will be extracted

        indices = []

        for item in param_key:
            data = np.asarray(h5_file[f'models/{self.model}/{item}'])
            data_index = np.argwhere(np.round(data, 4) == np.round(model_param[item], 4))[0]

            if len(data_index) == 0:
                raise ValueError('The parameter {item}={model_val[i]} is not found.')

            indices.append(data_index[0])

        # Read the wavelength grid and the indices that will be used for the wavelength range

        wl_points, wl_index = self.wavelength_points(h5_file)

        # Append the wavelength indices to the list of grid indices

        indices.append(wl_index)

        # Extract the spectrum at the requested grid point and wavelength range

        flux = flux[tuple(indices)]

        # Close the HDF5 database

        h5_file.close()

        # Apply (radius/distance)^2 scaling

        if 'radius' in model_param and 'distance' in model_param:
            scaling = (model_param['radius']*constants.R_JUP)**2 / \
                      (model_param['distance']*constants.PARSEC)**2

            flux *= scaling

        # Add blackbody disk component to the spectrum

        if 'disk_teff' in model_param and 'disk_radius' in model_param:
            disk_param = {'teff': model_param['disk_teff'],
                          'radius': model_param['disk_radius'],
                          'distance': model_param['distance']}

            readplanck = read_planck.ReadPlanck((0.9*self.wavel_range[0], 1.1*self.wavel_range[-1]))

            if spec_res is None:
                planck_box = readplanck.get_spectrum(disk_param, 1000., smooth=False)

            else:
                planck_box = readplanck.get_spectrum(disk_param, spec_res, smooth=False)

            flux += spectres.spectres(wl_points, planck_box.wavelength, planck_box.flux)

        # Create ModelBox with the spectrum

        model_box = box.create_box(boxtype='model',
                                   model=self.model,
                                   wavelength=wl_points,
                                   flux=flux,
                                   parameters=model_param,
                                   quantity='flux')

        # Apply extinction

        if 'lognorm_radius' in model_param and 'lognorm_sigma' in model_param and \
                'lognorm_ext' in model_param:

            model_box.flux = self.apply_ext_lnorm_distr(model_box.wavelength,
                                                        model_box.flux,
                                                        model_param['lognorm_radius'],
                                                        model_param['lognorm_sigma'],
                                                        model_param['lognorm_ext'])

        if 'powerlaw_max' in model_param and 'powerlaw_exp' in model_param and \
                'powerlaw_ext' in model_param:

            model_box.flux = self.apply_ext_plaw_distr(model_box.wavelength,
                                                       model_box.flux,
                                                       model_param['powerlaw_max'],
                                                       model_param['powerlaw_exp'],
                                                       model_param['powerlaw_ext'])

        if 'ism_ext' in model_param:
            ism_reddening = model_param.get('ism_red', 3.1)

            model_box.flux = self.apply_ext_ism(model_box.wavelength,
                                                model_box.flux,
                                                model_param['ism_ext'],
                                                ism_reddening)

        if 'powerl_ext' in model_param and 'powerl_exp' in model_param:

            model_box.flux = self.apply_ext_powerlaw(model_box.wavelength,
                                                     model_box.flux,
                                                     model_param['powerl_ext'],
                                                     model_param['powerl_exp'])

        # Smooth the spectrum

        if spec_res is not None:
            model_box.flux = read_util.smooth_spectrum(
                model_box.wavelength, model_box.flux, spec_res)

        # Resample the spectrum

        if wavel_resample is not None:
            model_box.flux = spectres.spectres(wavel_resample,
                                               model_box.wavelength,
                                               model_box.flux,
                                               spec_errs=None,
                                               fill=np.nan,
                                               verbose=True)

            model_box.wavelength = wavel_resample

        # Add the luminosity to the parameter dictionary

        if 'radius' in model_box.parameters:
            model_box.parameters['luminosity'] = 4. * np.pi * (
                model_box.parameters['radius'] * constants.R_JUP)**2 * constants.SIGMA_SB * \
                model_box.parameters['teff']**4. / constants.L_SUN  # (Lsun)

        # Add the blackbody disk component to the luminosity

        if 'disk_teff' in model_box.parameters and 'disk_radius' in model_box.parameters:
            model_box.parameters['luminosity'] += 4. * np.pi * (
                model_box.parameters['disk_radius'] * constants.R_JUP)**2 * constants.SIGMA_SB * \
                model_box.parameters['disk_teff']**4. / constants.L_SUN  # (Lsun)

        return model_box

    @typechecked
    def get_flux(self,
                 model_param: Dict[str, float],
                 synphot=None) -> Tuple[float, None]:
        """
        Function for calculating the average flux density for the ``filter_name``.

        Parameters
        ----------
        model_param : dict
            Model parameters and values.
        synphot : species.analysis.photometry.SyntheticPhotometry, None
            Synthetic photometry object. The object is created if set to ``None``.

        Returns
        -------
        float
            Average flux (W m-2 um-1).
        float, None
            Uncertainty (W m-2 um-1), which is set to ``None``.
        """

        for key in self.get_parameters():
            if key not in model_param.keys():
                raise ValueError(f'The \'{key}\' parameter is required by \'{self.model}\'. '
                                 f'The mandatory parameters are {self.get_parameters()}.')

        if self.spectrum_interp is None:
            self.interpolate_model()

        spectrum = self.get_model(model_param)

        if synphot is None:
            synphot = photometry.SyntheticPhotometry(self.filter_name)

        return synphot.spectrum_to_flux(spectrum.wavelength, spectrum.flux)

    @typechecked
    def get_magnitude(self,
                      model_param: Dict[str, float]) -> Tuple[Optional[float], Optional[float]]:
        """
        Function for calculating the apparent and absolute magnitudes for the ``filter_name``.

        Parameters
        ----------
        model_param : dict
            Dictionary with the model parameters. A ``radius`` (Rjup) and ``distance`` (pc)
            are required for the apparent magnitude (i.e. to scale the flux from the planet to the
            observer). Only a ``radius`` is required for the absolute magnitude.

        Returns
        -------
        float
            Apparent magnitude. A ``None`` is returned if the dictionary of ``model_param`` does
            not contain a ``radius`` and ``distance``.
        float, None
            Absolute magnitude. A ``None`` is returned if the dictionary of ``model_param`` does
            not contain a ``radius``.
        """

        for key in self.get_parameters():
            if key not in model_param.keys():
                raise ValueError(f'The \'{key}\' parameter is required by \'{self.model}\'. '
                                 f'The mandatory parameters are {self.get_parameters()}.')

        if self.spectrum_interp is None:
            self.interpolate_model()

        try:
            spectrum = self.get_model(model_param)

        except ValueError:
            warnings.warn(f'The set of model parameters {model_param} is outside the grid range '
                          f'{self.get_bounds()} so returning a NaN.')

            return np.nan, np.nan

        if spectrum.wavelength.size == 0:
            app_mag = (np.nan, None)
            abs_mag = (np.nan, None)

        else:
            synphot = photometry.SyntheticPhotometry(self.filter_name)

            if 'radius' in model_param and 'distance' in model_param:
                app_mag, abs_mag = synphot.spectrum_to_magnitude(
                    spectrum.wavelength, spectrum.flux, distance=(model_param['distance'], None))

            else:
                app_mag = (None, None)
                abs_mag = (None, None)

                if 'radius' in model_param:
                    distance = 10.  # (pc)

                    spectrum.flux *= (model_param['radius']*constants.R_JUP)**2
                    spectrum.flux /= (distance*constants.PARSEC)**2

                    _, abs_mag = synphot.spectrum_to_magnitude(
                        spectrum.wavelength, spectrum.flux, distance=(distance, None))

        return app_mag[0], abs_mag[0]

    @typechecked
    def log_likelihood(self,
                       model_param: Dict[str, float],
                       object_box: box.ObjectBox) -> Tuple[float, Dict[str, float]]:
        """
        Function for calculating the log-likelihood for a given set of parameters and selected
        data of an ``ObjectBox``. The ``weight`` parameter is currently not implemented (see
        ``weights`` parameter of ``FitModel``). Please open an issue on Github if this parameter
        is required.

        Parameters
        ----------
        model_param : dict
            Dictionary with the model parameters and values.
        object_box : species.core.box.ObjectBox
            The :class:`~species.core.box.ObjectBox` with the photometric fluxes and/or spectra
            for which the log-likelihood will be calculated. Any scaling parameters in
            ``model_param`` will be applied so the :func:`~species.util.read_util.update_spectra`
            should not have been used on the ``object_box`` (i.e. the original spectra are
            required).

        Returns
        -------
        float
            Total (natural) log-likelihood.
        dict
            Dictionary with the log-likelihood for the individual components.
        """

        if self.spectrum_interp is None:
            self.interpolate_model()

        spectrum = self.get_model(model_param)

        print('\rCalculating the log-likelihood:')

        lnlike_dict = {}

        if object_box.flux is not None:
            for filter_name, phot_flux in object_box.flux.items():
                synphot = photometry.SyntheticPhotometry(filter_name)

                model_flux, _ = synphot.spectrum_to_flux(spectrum.wavelength, spectrum.flux)

                lnlike_dict[filter_name] = mcmc_util.lnlike_phot(
                    phot_flux[0], phot_flux[1]**2, model_flux, weight=None)

                print(f'   - {filter_name} = {lnlike_dict[filter_name]:.2f}')

        if object_box.spectrum is not None:
            for spec_name, spec_data in object_box.spectrum.items():
                if f'scaling_{spec_name}' in model_param:
                    # Write to spec_flux such that the ObjectBox fluxes are not overwritten
                    spec_flux = model_param[f'scaling_{spec_name}'] * spec_data[0][:, 1]

                else:
                    spec_flux = spec_data[0][:, 1]

                if spec_data[1] is not None:
                    cov_matrix = spec_data[1]

                elif f'log_corr_len_{spec_name}' in model_param and \
                        f'corr_amp_{spec_name}' in model_param:

                    cov_matrix = mcmc_util.sqexp_kernel(spec_data[0][:, 0],
                                                        spec_data[0][:, 2],
                                                        model_param[f'corr_amp_{spec_name}'],
                                                        model_param[f'log_corr_len_{spec_name}'])

                elif f'log_glob_len_{spec_name}' in model_param and \
                        f'log_glob_amp_{spec_name}' in model_param:

                    log_glob_amp = model_param[f'log_glob_amp_{spec_name}']
                    log_glob_len = model_param[f'log_glob_len_{spec_name}']

                    cov_matrix = mcmc_util.matern32_kernel(spec_data[0][:, 0],
                                                           spec_data[0][:, 2]**2,
                                                           log_glob_amp,
                                                           log_glob_len)

                else:
                    cov_matrix = None

                if spec_data[2] is not None:
                    inv_cov_matrix = spec_data[2]

                else:
                    inv_cov_matrix = None

                model_box = self.get_model(model_param,
                                           spec_res=spec_data[3],
                                           wavel_resample=spec_data[0][:, 0],
                                           smooth=True)

                lnlike_dict[spec_name] = mcmc_util.lnlike_spec(
                    spec_flux, spec_data[0][:, 2]**2, model_box.flux, weight=None,
                    cov_matrix=cov_matrix, inv_cov_matrix=inv_cov_matrix)

                print(f'   - {spec_name} = {lnlike_dict[spec_name]:.2f}')

        if len(lnlike_dict) == 0:
            raise ValueError('The dictionary with log-likelihood values is empty. Please '
                             'make sure that the object_box contains photometric and/or '
                             'spectral data.')

        print(f'Total log-likelihood: {sum(lnlike_dict.values()):.2f}')

        return sum(lnlike_dict.values()), lnlike_dict

    @typechecked
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Function for extracting the grid boundaries.

        Returns
        -------
        dict
            Boundaries of parameter grid.
        """

        h5_file = self.open_database()

        parameters = self.get_parameters()

        bounds = {}

        for item in parameters:
            data = h5_file[f'models/{self.model}/{item}']
            bounds[item] = (data[0], data[-1])

        h5_file.close()

        return bounds

    @typechecked
    def get_wavelengths(self) -> np.ndarray:
        """
        Function for extracting the wavelength points.

        Returns
        -------
        np.ndarray
            Wavelength points (um).
        """

        with self.open_database() as h5_file:
            wavelength = np.asarray(h5_file[f'models/{self.model}/wavelength'])

        return wavelength

    @typechecked
    def get_points(self) -> Dict[str, np.ndarray]:
        """
        Function for extracting the grid points.

        Returns
        -------
        dict
            Parameter points of the model grid.
        """

        points = {}

        h5_file = self.open_database()

        parameters = self.get_parameters()

        points = {}

        for item in parameters:
            data = h5_file[f'models/{self.model}/{item}']
            points[item] = np.asarray(data)

        h5_file.close()

        return points

    @typechecked
    def get_parameters(self) -> List[str]:
        """
        Function for extracting the parameter names.

        Returns
        -------
        list(str)
            Model parameters.
        """

        h5_file = self.open_database()

        dset = h5_file[f'models/{self.model}']

        if 'n_param' in dset.attrs:
            n_param = dset.attrs['n_param']

        elif 'nparam' in dset.attrs:
            n_param = dset.attrs['nparam']

        param = []
        for i in range(n_param):
            param.append(dset.attrs[f'parameter{i}'])

        h5_file.close()

        return param

    @typechecked
    def get_spec_res(self) -> float:
        """
        Function for extracting the spectral resolution as stored in the database.

        Returns
        -------
        float
            Spectral resolution.
        """

        wavel_points = self.get_wavelengths()

        wavel_mean = (wavel_points[1:]+wavel_points[:-1])/2.

        # R = lambda / delta_lambda / 2, because twice as many points as R are
        # required to resolve two features that are lambda / R apart
        spec_res = wavel_mean / np.diff(wavel_points) / 2.

        return np.mean(spec_res)
