"""
Module with functionalities for calculating synthetic photometry.
"""

import os
import math
import warnings
import configparser

from typing import Optional, Union, Tuple, List

import h5py
import numpy as np

from typeguard import typechecked

from species.data import database
from species.read import read_filter, read_calibration
from species.util import phot_util


class SyntheticPhotometry:
    """
    Class for calculating synthetic photometry from a spectrum and also for conversion between
    magnitudes and fluxes. Note that depending on the detector type (energy- or photon-counting)
    the integral for the filter-weighted flux contains an additional wavelength factor.
    """

    @typechecked
    def __init__(self,
                 filter_name: str) -> None:
        """
        Parameters
        ----------
        filter_name : str
            Filter name as listed in the database. Filters from the SVO Filter Profile Service are
            automatically downloaded and added to the database.

        Returns
        -------
        NoneType
            None
        """

        self.filter_name = filter_name
        self.filter_interp = None
        self.wavel_range = None

        self.vega_mag = 0.03  # (mag)

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

        read_filt = read_filter.ReadFilter(self.filter_name)
        self.det_type = read_filt.detector_type()

    @typechecked
    def zero_point(self) -> np.float64:
        """
        Internal function for calculating the zero point of the provided ``filter_name``.

        Returns
        -------
        float
            Zero-point flux (W m-2 um-1).
        """

        if self.wavel_range is None:
            transmission = read_filter.ReadFilter(self.filter_name)
            self.wavel_range = transmission.wavelength_range()

        h5_file = h5py.File(self.database, 'r')

        try:
            h5_file['spectra/calibration/vega']

        except KeyError:
            h5_file.close()
            species_db = database.Database()
            species_db.add_spectra('vega')
            h5_file = h5py.File(self.database, 'r')

        readcalib = read_calibration.ReadCalibration('vega', None)
        calibbox = readcalib.get_spectrum()

        wavelength = calibbox.wavelength
        flux = calibbox.flux

        wavelength_crop = wavelength[(wavelength > self.wavel_range[0]) &
                                     (wavelength < self.wavel_range[1])]

        flux_crop = flux[(wavelength > self.wavel_range[0]) &
                         (wavelength < self.wavel_range[1])]

        h5_file.close()

        return self.spectrum_to_flux(wavelength_crop, flux_crop)[0]

    @typechecked
    def spectrum_to_flux(self,
                         wavelength: np.ndarray,
                         flux: np.ndarray,
                         error: Optional[np.ndarray] = None,
                         threshold: Optional[float] = 0.05) -> Tuple[
                             Union[np.float32, np.float64],
                             Union[Optional[np.float32], Optional[np.float64]]]:
        """
        Function for calculating the average flux from a spectrum and a filter profile. The error
        is propagated by sampling 200 random values from the error distributions.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength points (um).
        flux : np.ndarray
            Flux (W m-2 um-1).
        error : np.ndarray, None
            Uncertainty (W m-2 um-1). Not used if set to ``None``.
        threshold : float, None
            Transmission threshold (value between 0 and 1). If the minimum transmission value is
            larger than the threshold, a NaN is returned. This will happen if the input spectrum
            does not cover the full wavelength range of the filter profile. Not used if set to
            ``None``.

        Returns
        -------
        float
            Average flux (W m-2 um-1).
        float, None
            Uncertainty (W m-2 um-1).
        """

        if error is not None:
            # The error calculation requires the original spectrum because spectrum_to_flux is used
            wavel_error = wavelength.copy()
            flux_error = flux.copy()

        if self.filter_interp is None:
            transmission = read_filter.ReadFilter(self.filter_name)
            self.filter_interp = transmission.interpolate_filter()

            if self.wavel_range is None:
                self.wavel_range = transmission.wavelength_range()

        if wavelength.size == 0:
            raise ValueError(f'Calculation of the mean flux for {self.filter_name} is not '
                             f'possible because the wavelength array is empty.')

        indices = np.where((self.wavel_range[0] <= wavelength) &
                           (wavelength <= self.wavel_range[1]))[0]

        if indices.size < 2:
            syn_flux = np.nan

            warnings.warn('Calculating a synthetic flux requires more than one wavelength '
                          'point. Photometry is set to NaN.')

        else:
            if threshold is None and (wavelength[0] > self.wavel_range[0] or
                                      wavelength[-1] < self.wavel_range[1]):

                warnings.warn(f'The filter profile of {self.filter_name} '
                              f'({self.wavel_range[0]:.4f}-{self.wavel_range[1]:.4f}) extends '
                              f'beyond the wavelength range of the spectrum ({wavelength[0]:.4f} '
                              f'-{wavelength[-1]:.4f}). The flux is set to NaN. Setting the '
                              f'\'threshold\' parameter will loosen the wavelength constraints.')

                syn_flux = np.nan

            else:
                wavelength = wavelength[indices]
                flux = flux[indices]

                transmission = self.filter_interp(wavelength)

                if threshold is not None and \
                        (transmission[0] > threshold or transmission[-1] > threshold) and \
                        (wavelength[0] < self.wavel_range[0] or wavelength[-1] >
                         self.wavel_range[-1]):

                    warnings.warn(f'The filter profile of {self.filter_name} '
                                  f'({self.wavel_range[0]:.4f}-{self.wavel_range[1]:.4f}) '
                                  f'extends beyond the wavelength range of the spectrum '
                                  f'({wavelength[0]:.4f}-{wavelength[-1]:.4f}). The flux '
                                  f'is set to NaN. Increasing the \'threshold\' parameter '
                                  f'({threshold}) will loosen the wavelength constraint.')

                    syn_flux = np.nan

                else:
                    indices = np.isnan(transmission)
                    indices = np.logical_not(indices)

                    if self.det_type == 'energy':
                        # Energy counting detector
                        integrand1 = transmission[indices]*flux[indices]
                        integrand2 = transmission[indices]

                    elif self.det_type == 'photon':
                        # Photon counting detector
                        integrand1 = wavelength[indices]*transmission[indices]*flux[indices]
                        integrand2 = wavelength[indices]*transmission[indices]

                    integral1 = np.trapz(integrand1, wavelength[indices])
                    integral2 = np.trapz(integrand2, wavelength[indices])

                    syn_flux = integral1/integral2

        if error is not None and not np.any(np.isnan(error)):
            phot_random = np.zeros(200)

            for i in range(200):
                # Use the original spectrum size (i.e. wavel_error and flux_error)
                spec_random = flux_error + np.random.normal(loc=0.,
                                                            scale=1.,
                                                            size=wavel_error.shape[0])*error

                phot_random[i] = self.spectrum_to_flux(wavel_error,
                                                       spec_random,
                                                       error=None,
                                                       threshold=threshold)[0]

            error_flux = np.std(phot_random)

        elif error is not None and np.any(np.isnan(error)):
            warnings.warn('Spectum contains NaN so can not calculate the error.')
            error_flux = None

        else:
            error_flux = None

        return syn_flux, error_flux

    @typechecked
    def spectrum_to_magnitude(self,
                              wavelength: np.ndarray,
                              flux: np.ndarray,
                              error: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                              distance: Optional[Tuple[float, Optional[float]]] = None,
                              threshold: Optional[float] = 0.05) -> Tuple[
                                  Tuple[float, Optional[float]],
                                  Optional[Tuple[Optional[float], Optional[float]]]]:
        """
        Function for calculating the apparent and absolute magnitude from a spectrum and a
        filter profile. The error is propagated by sampling 200 random values from the error
        distributions.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength points (um).
        flux : np.ndarray
            Flux (W m-2 um-1).
        error : np.ndarray, list(np.ndarray), None
            Uncertainty (W m-2 um-1).
        distance : tuple(float, float), None
            Distance and uncertainty (pc). No absolute magnitude is calculated if set to ``None``.
            No error on the absolute magnitude is calculated if the uncertainty is set to ``None``.
        threshold : float, None
            Transmission threshold (value between 0 and 1). If the minimum transmission value is
            larger than the threshold, a NaN is returned. This will happen if the input spectrum
            does not cover the full wavelength range of the filter profile. Not used if set to
            ``None``.

        Returns
        -------
        tuple(float, float)
            Apparent magnitude and uncertainty.
        tuple(float, float)
            Absolute magnitude and uncertainty.
        """

        zp_flux = self.zero_point()

        syn_flux = self.spectrum_to_flux(wavelength,
                                         flux,
                                         error=error,
                                         threshold=threshold)

        app_mag = self.vega_mag - 2.5*math.log10(syn_flux[0]/zp_flux)

        if error is not None and not np.any(np.isnan(error)):
            mag_random = np.zeros(200)

            for i in range(200):
                spec_random = flux + np.random.normal(loc=0.,
                                                      scale=1.,
                                                      size=wavelength.shape[0])*error

                flux_random = self.spectrum_to_flux(wavelength,
                                                    spec_random,
                                                    error=None,
                                                    threshold=threshold)

                mag_random[i] = self.vega_mag - 2.5*np.log10(flux_random[0]/zp_flux)

            error_app_mag = np.std(mag_random)

        elif error is not None and np.any(np.isnan(error)):
            warnings.warn('Spectum contains NaN so can not calculate the error.')
            error_app_mag = None

        else:
            error_app_mag = None

        if distance is None:
            abs_mag = None
            error_abs_mag = None

        else:
            abs_mag = app_mag - 5.*np.log10(distance[0]) + 5.

            if error_app_mag is not None and distance[1] is not None:
                error_dist = distance[1] * (5./(distance[0]*math.log(10.)))
                error_abs_mag = math.sqrt(error_app_mag**2 + error_dist**2)

            else:
                error_abs_mag = None

        return (app_mag, error_app_mag), (abs_mag, error_abs_mag)

    @typechecked
    def magnitude_to_flux(self,
                          magnitude: float,
                          error: Optional[float] = None,
                          zp_flux: Optional[float] = None) -> Tuple[np.float64, np.float64]:
        """
        Function for converting a magnitude to a flux.

        Parameters
        ----------
        magnitude : float
            Magnitude.
        error : float, None
            Error on the magnitude. Not used if set to ``None``.
        zp_flux : float, None
            Zero-point flux (W m-2 um-1). The value is calculated if set to ``None``.

        Returns
        -------
        float
            Flux (W m-2 um-1).
        float
            Error (W m-2 um-1).
        """

        if zp_flux is None:
            zp_flux = self.zero_point()

        flux = 10.**(-0.4*(magnitude-self.vega_mag))*zp_flux

        if error is None:
            error_flux = None

        else:
            error_upper = flux * (10.**(0.4*error) - 1.)
            error_lower = flux * (1. - 10.**(-0.4*error))
            error_flux = (error_lower+error_upper)/2.

        return flux, error_flux

    @typechecked
    def flux_to_magnitude(self,
                          flux: float,
                          error: Optional[Union[float, np.ndarray]] = None,
                          distance: Optional[Union[Tuple[float, Optional[float]],
                                                   Tuple[np.ndarray, Optional[np.ndarray]]]] = None
                          ) -> Tuple[Union[Tuple[float, Optional[float]],
                                           Tuple[np.ndarray, Optional[np.ndarray]]],
                                     Union[Tuple[float, Optional[float]],
                                           Tuple[np.ndarray, Optional[np.ndarray]]]]:
        """
        Function for converting a flux into a magnitude.

        Parameters
        ----------
        flux : float, np.ndarray
            Flux (W m-2 um-1).
        error : float, np.ndarray, None
            Uncertainty (W m-2 um-1). Not used if set to None.
        distance : tuple(float, float), tuple(np.ndarray, np.ndarray)
            Distance and uncertainty (pc). The returned absolute magnitude is set to None in case
            ``distance`` is set to None. The error is not propagated into the error on the absolute
            magnitude in case the distance uncertainty is set to None, for example
            ``distance=(20., None)``

        Returns
        -------
        tuple(float, float), tuple(np.ndarray, np.ndarray)
            Apparent magnitude and uncertainty.
        tuple(float, float), tuple(np.ndarray, np.ndarray)
            Absolute magnitude and uncertainty.
        """

        zp_flux = self.zero_point()

        app_mag = self.vega_mag - 2.5*np.log10(flux/zp_flux)

        if error is None:
            error_app_mag = None
            error_abs_mag = None

        else:
            error_app_lower = app_mag - (self.vega_mag - 2.5*np.log10((flux+error)/zp_flux))
            error_app_upper = (self.vega_mag - 2.5*np.log10((flux-error)/zp_flux)) - app_mag
            error_app_mag = (error_app_lower+error_app_upper)/2.

        if distance is None:
            abs_mag = None
            error_abs_mag = None

        else:
            abs_mag, error_abs_mag = phot_util.apparent_to_absolute(
                (app_mag, error_app_mag), distance)

        return (app_mag, error_app_mag), (abs_mag, error_abs_mag)
