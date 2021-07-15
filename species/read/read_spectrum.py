"""
Module with reading functionalities for spectral libraries.
"""

import configparser
import os

from typing import List

import h5py
import numpy as np

from typeguard import typechecked

from species.analysis import photometry
from species.core import box
from species.data import database
from species.read import read_filter


class ReadSpectrum:
    """
    Class for reading spectral library data from the database.
    """

    @typechecked
    def __init__(self,
                 spec_library: str,
                 filter_name: str = None) -> None:
        """
        Parameters
        ----------
        spec_library : str
            Name of the spectral library ('irtf', 'spex') or other type of spectrum ('vega').
        filter_name : str, None
            Filter name for the wavelength range. Full spectra are read if set to ``None``.

        Returns
        -------
        NoneType
            None
        """

        self.spec_library = spec_library
        self.filter_name = filter_name

        if filter_name is None:
            self.wavel_range = None

        else:
            transmission = read_filter.ReadFilter(filter_name)
            self.wavel_range = transmission.wavelength_range()

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

    @typechecked
    def get_spectrum(self,
                     sptypes: List[str] = None,
                     exclude_nan: bool = True) -> box.SpectrumBox:
        """
        Function for selecting spectra from the database.

        Parameters
        ----------
        sptypes : list(str), None
            Spectral types to select from a library. The spectral types should be indicated with
            two characters (e.g. 'M5', 'L2', 'T3'). All spectra are selected if set to ``None``.
            For each object in the ``spec_library``, the requested ``sptypes`` are first compared
            with the optical spectral type and, if not available, secondly the near-infrared
            spectral type.
        exclude_nan : bool
            Exclude wavelength points for which the flux is NaN.

        Returns
        -------
        species.core.box.SpectrumBox
            Box with the spectra.
        """

        h5_file = h5py.File(self.database, 'r')

        try:
            h5_file[f'spectra/{self.spec_library}']

        except KeyError:
            h5_file.close()
            species_db = database.Database()
            species_db.add_spectra(self.spec_library, sptypes)
            h5_file = h5py.File(self.database, 'r')

        list_wavelength = []
        list_flux = []
        list_error = []
        list_name = []
        list_simbad = []
        list_sptype = []
        list_distance = []

        for item in h5_file[f'spectra/{self.spec_library}']:
            dset = h5_file[f'spectra/{self.spec_library}/{item}']

            wavelength = dset[0, :]  # (um)
            flux = dset[1, :]  # (W m-2 um-1)
            error = dset[2, :]  # (W m-2 um-1)

            if exclude_nan:
                indices = np.isnan(flux)
                indices = np.logical_not(indices)
                indices = np.where(indices)[0]

                wavelength = wavelength[indices]
                flux = flux[indices]
                error = error[indices]

            if self.wavel_range is None:
                wl_index = np.arange(0, len(wavelength), 1)

            else:
                wl_index = (flux > 0.) & (wavelength > self.wavel_range[0]) & \
                           (wavelength < self.wavel_range[1])

            count = np.count_nonzero(wl_index)

            if count > 0:
                index = np.where(wl_index)[0]

                if index[0] > 0:
                    wl_index[index[0] - 1] = True

                if index[-1] < len(wl_index)-1:
                    wl_index[index[-1] + 1] = True

                list_wavelength.append(wavelength[wl_index])
                list_flux.append(flux[wl_index])
                list_error.append(error[wl_index])

                attrs = dset.attrs

                if 'name' in attrs:
                    if isinstance(dset.attrs['name'], str):
                        list_name.append(dset.attrs['name'])
                    else:
                        list_name.append(dset.attrs['name'].decode('utf-8'))
                else:
                    list_name.append('')

                if 'simbad' in attrs:
                    if isinstance(dset.attrs['simbad'], str):
                        list_simbad.append(dset.attrs['simbad'])
                    else:
                        list_simbad.append(dset.attrs['simbad'].decode('utf-8'))
                else:
                    list_simbad.append('')

                if 'sptype' in attrs:
                    if isinstance(dset.attrs['sptype'], str):
                        list_sptype.append(dset.attrs['sptype'])
                    else:
                        list_sptype.append(dset.attrs['sptype'].decode('utf-8'))
                else:
                    list_sptype.append('None')

                if 'distance' in attrs:
                    list_distance.append((dset.attrs['distance'], dset.attrs['distance_error']))
                else:
                    list_distance.append((np.nan, np.nan))

            else:
                list_wavelength.append(np.array([]))
                list_flux.append(np.array([]))
                list_error.append(np.array([]))
                list_name.append('')
                list_simbad.append('')
                list_sptype.append('None')
                list_distance.append((np.nan, np.nan))

        specbox = box.SpectrumBox()
        specbox.spec_library = self.spec_library

        if sptypes is not None:
            indices = []

            specbox.wavelength = []
            specbox.flux = []
            specbox.error = []
            specbox.name = []
            specbox.simbad = []
            specbox.sptype = []
            specbox.distance = []

            for item in sptypes:

                for i, spec_item in enumerate(list_sptype):
                    if item == spec_item[:2]:
                        specbox.wavelength.append(list_wavelength[i])
                        specbox.flux.append(list_flux[i])
                        specbox.error.append(list_error[i])
                        specbox.name.append(list_name[i])
                        specbox.simbad.append(list_simbad[i])
                        specbox.sptype.append(list_sptype[i])
                        specbox.distance.append(list_distance[i])

        else:
            specbox.wavelength = list_wavelength
            specbox.flux = list_flux
            specbox.error = list_error
            specbox.name = list_name
            specbox.simbad = list_simbad
            specbox.sptype = list_sptype
            specbox.distance = list_distance

        return specbox

    @typechecked
    def get_flux(self,
                 sptypes: List[str] = None) -> box.PhotometryBox:
        """
        Function for calculating the average flux density for the ``filter_name``.

        Parameters
        ----------
        sptypes : list(str), None
            Spectral types to select from a library. The spectral types should be indicated with
            two characters (e.g. 'M5', 'L2', 'T3'). All spectra are selected if set to ``None``.

        Returns
        -------
        species.core.box.PhotometryBox
            Box with the synthetic photometry.
        """

        specbox = self.get_spectrum(sptypes=sptypes,
                                    exclude_nan=True)

        n_spectra = len(specbox.wavelength)

        filter_profile = read_filter.ReadFilter(filter_name=self.filter_name)
        mean_wavel = filter_profile.mean_wavelength()

        wavelengths = np.full(n_spectra, mean_wavel)
        filters = np.full(n_spectra, self.filter_name)

        synphot = photometry.SyntheticPhotometry(filter_name=self.filter_name)

        phot_flux = []

        for i in range(n_spectra):
            flux = synphot.spectrum_to_flux(wavelength=specbox.wavelength[i],
                                            flux=specbox.flux[i],
                                            error=specbox.error[i])

            phot_flux.append(flux)

        phot_flux = np.asarray(phot_flux)

        return box.create_box(boxtype='photometry',
                              name=specbox.name,
                              sptype=specbox.sptype,
                              wavelength=wavelengths,
                              flux=phot_flux,
                              app_mag=None,
                              abs_mag=None,
                              filter_name=filters)

    @typechecked
    def get_magnitude(self,
                      sptypes: List[str] = None) -> box.PhotometryBox:
        """
        Function for calculating the apparent magnitude for the ``filter_name``.

        Parameters
        ----------
        sptypes : list(str)
            Spectral types to select from a library. The spectral types should be indicated with
            two characters (e.g. 'M5', 'L2', 'T3'). All spectra are selected if set to ``None``.

        Returns
        -------
        species.core.box.PhotometryBox
            Box with the synthetic photometry.
        """

        specbox = self.get_spectrum(sptypes=sptypes,
                                    exclude_nan=True)

        n_spectra = len(specbox.wavelength)

        filter_profile = read_filter.ReadFilter(filter_name=self.filter_name)
        mean_wavel = filter_profile.mean_wavelength()

        wavelengths = np.full(n_spectra, mean_wavel)
        filters = np.full(n_spectra, self.filter_name)

        synphot = photometry.SyntheticPhotometry(filter_name=self.filter_name)

        app_mag = []
        abs_mag = []

        for i in range(n_spectra):

            if np.isnan(specbox.distance[i][0]):
                app_tmp = (np.nan, np.nan)
                abs_tmp = (np.nan, np.nan)

            else:
                app_tmp, abs_tmp = synphot.spectrum_to_magnitude(
                    specbox.wavelength[i], specbox.flux[i], error=specbox.error[i],
                    distance=(float(specbox.distance[i][0]), float(specbox.distance[i][1])))

            app_mag.append(app_tmp)
            abs_mag.append(abs_tmp)

        return box.create_box(boxtype='photometry',
                              name=specbox.name,
                              sptype=specbox.sptype,
                              wavelength=wavelengths,
                              flux=None,
                              app_mag=np.asarray(app_mag),
                              abs_mag=np.asarray(abs_mag),
                              filter_name=filters)
