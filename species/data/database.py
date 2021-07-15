"""
Module with functionalities for reading and writing of data.
"""

import configparser
import os
import json
import warnings

from typing import Dict, List, Optional, Tuple, Union

import emcee
import h5py
import numpy as np
import tqdm

from astropy.io import fits
from scipy import linalg
from typeguard import typechecked

from species.analysis import photometry
from species.core import box, constants
from species.data import ames_cond, ames_dusty, atmo, blackbody, btcond, btcond_feh, btnextgen, \
                         btsettl, btsettl_cifist, companions, drift_phoenix, dust, exo_rem, \
                         filters, irtf, isochrones, leggett, petitcode, spex, vega, vlm_plx, \
                         kesseli2017, morley2012, bonnefoy2014, allers2013
from species.read import read_calibration, read_filter, read_model, read_object, read_planck
from species.util import data_util, dust_util, read_util


class Database:
    """
    Class with reading and writing functionalities for the HDF5 database.
    """

    @typechecked
    def __init__(self) -> None:
        """
        Returns
        -------
        NoneType
            None
        """

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']
        self.input_path = config['species']['data_folder']

    @typechecked
    def list_content(self) -> None:
        """
        Returns
        -------
        NoneType
            None
        """

        print('Database content:')

        @typechecked
        def descend(h5_object: Union[h5py._hl.files.File,
                                     h5py._hl.group.Group,
                                     h5py._hl.dataset.Dataset],
                    seperator: str = '') -> None:
            """
            Function for descending into an HDF5 dataset and printing its content.

            Parameters
            ----------
            h5_object : h5py._hl.files.File, h5py._hl.group.Group, h5py._hl.dataset.Dataset
                The ``h5py`` object.
            separator : str
                Separator that is used between items.

            Returns
            -------
            NoneType
                None
            """

            if isinstance(h5_object, (h5py._hl.files.File, h5py._hl.group.Group)):
                for key in h5_object.keys():
                    print(seperator+'- '+key+': '+str(h5_object[key]))
                    descend(h5_object[key], seperator=seperator+'\t')

            elif isinstance(h5_object, h5py._hl.dataset.Dataset):
                for key in h5_object.attrs.keys():
                    print(seperator+'- '+key+': '+str(h5_object.attrs[key]))

        with h5py.File(self.database, 'r') as hdf_file:
            descend(hdf_file)

    @staticmethod
    @typechecked
    def list_companions() -> None:
        """
        Returns
        -------
        NoneType
            None
        """

        spec_data = companions.get_spec_data()

        for planet_name, planet_dict in companions.get_data().items():
            distance = planet_dict['distance']
            app_mag = planet_dict['app_mag']

            print(f'Object name = {planet_name}')
            print(f'Distance (pc) = {distance[0]} +/- {distance[1]}')

            for mag_name, mag_dict in app_mag.items():
                print(f'{mag_name} (mag) = {mag_dict[0]} +/- {mag_dict[1]}')

            if planet_name in spec_data:
                for key, value in spec_data[planet_name].items():
                    print(f'{key} spectrum from {value[3]}')

            print()

    @typechecked
    def delete_data(self,
                    dataset: str) -> None:
        """
        Function for deleting a dataset from the HDF5 database.

        Parameters
        ----------
        dataset : str
            Dataset path in the HDF5 database.

        Returns
        -------
        NoneType
            None
        """

        with h5py.File(self.database, 'a') as hdf_file:
            if dataset in hdf_file:
                print(f'Deleting data: {dataset}...', end='', flush=True)
                del hdf_file[dataset]
                print(' [DONE]')

            else:
                warnings.warn(f'The dataset {dataset} is not found in {self.database}.')

    @typechecked
    def add_companion(self,
                      name: Union[Optional[str], Optional[List[str]]] = None) -> None:
        """
        Function for adding the magnitudes and spectra of directly imaged planets and brown dwarfs
        from :class:`~species.data.companions.get_data` and
        :class:`~species.data.companions.get_comp_spec` to the database.

        Parameters
        ----------
        name : str, list(str), None
            Name or list with names of the directly imaged planets and brown dwarfs (e.g.
            ``'HR 8799 b'`` or ``['HR 8799 b', '51 Eri b', 'PZ Tel B']``). All the available
            companion data are added if set to ``None``.

        Returns
        -------
        NoneType
            None
        """

        if isinstance(name, str):
            name = list((name, ))

        data = companions.get_data()

        if name is None:
            name = data.keys()

        for item in name:
            spec_dict = companions.companion_spectra(self.input_path, item)

            self.add_object(object_name=item,
                            distance=data[item]['distance'],
                            app_mag=data[item]['app_mag'],
                            spectrum=spec_dict)

    @typechecked
    def add_dust(self) -> None:
        """
        Function for adding optical constants of MgSiO3 and Fe, and MgSiO3 cross sections for
        a log-normal and power-law size distribution to the database. The optical constants have
        been compiled by Mollière et al. (2019) for petitRADTRANS from the following sources:

        - MgSiO3, crystalline
            - Scott & Duley (1996), ApJS, 105, 401
            - Jäger et al. (1998), A&A, 339, 904

        - MgSiO3, amorphous
            - Jäger et al. (2003), A&A, 408, 193

        - Fe, crystalline
            - Henning & Stognienko (1996), A&A, 311, 291

        - Fe, amorphous
            - Pollack et al. (1994), ApJ, 421, 615

        Returns
        -------
        NoneType
            None
        """

        h5_file = h5py.File(self.database, 'a')

        if 'dust' in h5_file:
            del h5_file['dust']

        h5_file.create_group('dust')

        dust.add_optical_constants(self.input_path, h5_file)
        dust.add_cross_sections(self.input_path, h5_file)

        h5_file.close()

    @typechecked
    def add_filter(self,
                   filter_name: str,
                   filename: Optional[str] = None,
                   detector_type: str = 'photon') -> None:
        """
        Function for adding a filter profile to the database, either from the SVO Filter profile
        Service or from an input file. Additional filters that are automatically added are
        Magellan/VisAO.rp, Magellan/VisAO.ip, Magellan/VisAO.zp, Magellan/VisAO.Ys, ALMA/band6,
        and ALMA/band7.

        Parameters
        ----------
        filter_name : str
            Filter name from the SVO Filter Profile Service (e.g., 'Paranal/NACO.Lp') or a
            user-defined name if a ``filename`` is specified.
        filename : str
            Filename of the filter profile. The first column should contain the wavelength
            (um) and the second column the fractional transmission. The profile is downloaded from
            the SVO Filter Profile Service if the argument of ``filename`` is set to ``None``.
        detector_type : str
            The detector type ('photon' or 'energy'). The argument is only used if a ``filename``
            is provided. Otherwise, for filters that are fetched from the SVO website, the detector
            type is read from the SVO data. The detector type determines if a wavelength factor
            is included in the integral for the synthetic photometry.

        Returns
        -------
        NoneType
            None
        """

        print(f'Adding filter: {filter_name}...', end='', flush=True)

        filter_split = filter_name.split('/')

        h5_file = h5py.File(self.database, 'a')

        if f'filters/{filter_name}' in h5_file:
            del h5_file[f'filters/{filter_name}']

        if 'filters' not in h5_file:
            h5_file.create_group('filters')

        if f'filters/{filter_split[0]}' not in h5_file:
            h5_file.create_group(f'filters/{filter_split[0]}')

        if filename is not None:
            data = np.loadtxt(filename)
            wavelength = data[:, 0]
            transmission = data[:, 1]

        else:
            wavelength, transmission, detector_type = filters.download_filter(filter_name)

        if wavelength is not None and transmission is not None:
            dset = h5_file.create_dataset(f'filters/{filter_name}',
                                          data=np.column_stack((wavelength, transmission)))

            dset.attrs['det_type'] = str(detector_type)

        h5_file.close()

        print(' [DONE]')

    @typechecked
    def add_isochrones(self,
                       filename: str,
                       tag: str,
                       model: str = 'baraffe') -> None:
        """
        Function for adding isochrone data to the database.

        Parameters
        ----------
        filename : str
            Filename with the isochrone data.
        tag : str
            Database tag name where the isochrone that will be stored.
        model : str
            Evolutionary model ('baraffe' or 'marleau'). For 'baraffe' models, the isochrone data
            can be downloaded from https://phoenix.ens-lyon.fr/Grids/. For 'marleau' models, the
            data can be requested from Gabriel Marleau.

        Returns
        -------
        NoneType
            None
        """

        h5_file = h5py.File(self.database, 'a')

        if 'isochrones' not in h5_file:
            h5_file.create_group('isochrones')

        if 'isochrones/'+tag in h5_file:
            del h5_file[f'isochrones/{tag}']

        if model[0:7] == 'baraffe':
            isochrones.add_baraffe(h5_file, tag, filename)

        elif model[0:7] == 'marleau':
            isochrones.add_marleau(h5_file, tag, filename)

        h5_file.close()

    @typechecked
    def add_model(self,
                  model: str,
                  wavel_range: Optional[Tuple[float, float]] = None,
                  spec_res: Optional[float] = None,
                  teff_range: Optional[Tuple[float, float]] = None,
                  data_folder: Optional[str] = None) -> None:
        """
        Method for adding a grid of model spectra to the database. All spectra have been resampled
        to a lower, constant spectral resolution (typically :math:`R = 5000`).

        Parameters
        ----------
        model : str
            Model name ('ames-cond', 'ames-dusty', 'atmo', 'bt-settl', 'bt-settl-cifist',
            'bt-nextgen', 'drift-phoenix', 'petitcode-cool-clear', 'petitcode-cool-cloudy',
            'petitcode-hot-clear', 'petitcode-hot-cloudy', 'exo-rem', 'blackbody', bt-cond',
            'bt-cond-feh, 'morley-2012').
        wavel_range : tuple(float, float), None
            Wavelength range (um) for adding a subset of the spectra. The full wavelength range
            is used if the argument is set to ``None``.
        spec_res : float, None
            Spectral resolution to which the spectra will be resampled. This parameter is optional
            since the spectra have already been resampled to a lower, constant resolution
            (typically :math:`R = 5000`). The argument is only used if ``wavel_range`` is not
            ``None``.
        teff_range : tuple(float, float), None
            Effective temperature range (K) for adding a subset of the model grid. The full
            parameter grid will be added if the argument is set to ``None``.
        data_folder : str, None
            DEPRECATED: Folder where the input data is located. This parameter is no longer in use
            since all model spectra are publicly available. The parameter will be removed in a
            future release.

        Returns
        -------
        NoneType
            None
        """

        if data_folder is not None:
            warnings.warn('The \'data_folder\' parameter has been deprecated since '
                          'all supported model spectra are publicly available. The'
                          'parameter will therefore be ignored and will cause an error '
                          'in a future release.')

        # proprietary = ['petitcode-hot-clear', 'petitcode-hot-cloudy']

        # if model in proprietary and data_folder is None:
        #     raise ValueError(f'The {model} model is not publicly available and needs to '
        #                      f'be imported by setting the \'data_folder\' parameter.')

        # if model in ['bt-nextgen'] and wavel_range is None:
        #     raise ValueError(f'The \'wavel_range\' should be set for the \'{model}\' models to '
        #                      f'resample the original spectra on a fixed wavelength grid.')

        # if model in ['bt-nextgen'] and spec_res is None:
        #     raise ValueError(f'The \'spec_res\' should be set for the \'{model}\' models to '
        #                      f'resample the original spectra on a fixed wavelength grid.')

        # if model == 'bt-nextgen' and teff_range is None:
        #     warnings.warn('The temperature range is not restricted with the \'teff_range\' '
        #                   'parameter. Therefore, adding the BT-Settl or BT-NextGen spectra '
        #                   'will be very slow.')

        h5_file = h5py.File(self.database, 'a')

        if 'models' not in h5_file:
            h5_file.create_group('models')

        if model == 'ames-cond':
            ames_cond.add_ames_cond(self.input_path,
                                    h5_file,
                                    wavel_range,
                                    teff_range,
                                    spec_res)

            data_util.add_missing(model, ['teff', 'logg'], h5_file)

        elif model == 'ames-dusty':
            ames_dusty.add_ames_dusty(self.input_path,
                                      h5_file,
                                      wavel_range,
                                      teff_range,
                                      spec_res)

            data_util.add_missing(model, ['teff', 'logg'], h5_file)

        elif model == 'atmo':
            atmo.add_atmo(self.input_path,
                          h5_file,
                          wavel_range,
                          teff_range,
                          spec_res)

            data_util.add_missing(model, ['teff', 'logg'], h5_file)

        elif model == 'blackbody':
            blackbody.add_blackbody(self.input_path,
                                    h5_file,
                                    wavel_range,
                                    teff_range,
                                    spec_res)

            data_util.add_missing(model, ['teff'], h5_file)

        elif model == 'bt-cond':
            btcond.add_btcond(self.input_path,
                              h5_file,
                              wavel_range,
                              teff_range,
                              spec_res)

            data_util.add_missing(model, ['teff', 'logg'], h5_file)

        elif model == 'bt-cond-feh':
            btcond_feh.add_btcond_feh(self.input_path,
                                      h5_file,
                                      wavel_range,
                                      teff_range,
                                      spec_res)

            data_util.add_missing(model, ['teff', 'logg', 'feh'], h5_file)

        elif model == 'bt-settl':
            btsettl.add_btsettl(self.input_path,
                                h5_file,
                                wavel_range,
                                teff_range,
                                spec_res)

            data_util.add_missing(model, ['teff', 'logg'], h5_file)

        elif model == 'bt-settl-cifist':
            btsettl_cifist.add_btsettl(self.input_path,
                                       h5_file,
                                       wavel_range,
                                       teff_range,
                                       spec_res)

            data_util.add_missing(model, ['teff', 'logg'], h5_file)

        elif model == 'bt-nextgen':
            btnextgen.add_btnextgen(self.input_path,
                                    h5_file,
                                    wavel_range,
                                    teff_range,
                                    spec_res)

            data_util.add_missing(model, ['teff', 'logg', 'feh'], h5_file)

        elif model == 'drift-phoenix':
            drift_phoenix.add_drift_phoenix(self.input_path,
                                            h5_file,
                                            wavel_range,
                                            teff_range,
                                            spec_res)

            data_util.add_missing(model, ['teff', 'logg', 'feh'], h5_file)

        elif model == 'morley-2012':
            morley2012.add_morley2012(self.input_path,
                                      h5_file,
                                      wavel_range,
                                      teff_range,
                                      spec_res)

            data_util.add_missing(model, ['teff', 'logg', 'fsed'], h5_file)

        elif model == 'petitcode-cool-clear':
            petitcode.add_petitcode_cool_clear(self.input_path,
                                               h5_file,
                                               wavel_range,
                                               teff_range,
                                               spec_res)

            data_util.add_missing(model, ['teff', 'logg', 'feh'], h5_file)

        elif model == 'petitcode-cool-cloudy':
            petitcode.add_petitcode_cool_cloudy(self.input_path,
                                                h5_file,
                                                wavel_range,
                                                teff_range,
                                                spec_res)

            data_util.add_missing(model, ['teff', 'logg', 'feh', 'fsed'], h5_file)

        elif model == 'petitcode-hot-clear':
            petitcode.add_petitcode_hot_clear(self.input_path,
                                              h5_file,
                                              wavel_range,
                                              teff_range,
                                              spec_res)

            data_util.add_missing(model, ['teff', 'logg', 'feh', 'co'], h5_file)

        elif model == 'petitcode-hot-cloudy':
            petitcode.add_petitcode_hot_cloudy(self.input_path,
                                               h5_file,
                                               wavel_range,
                                               teff_range,
                                               spec_res)

            data_util.add_missing(model, ['teff', 'logg', 'feh', 'co', 'fsed'], h5_file)

        elif model == 'exo-rem':
            exo_rem.add_exo_rem(self.input_path,
                                h5_file,
                                wavel_range,
                                teff_range,
                                spec_res)

            data_util.add_missing(model, ['teff', 'logg', 'feh', 'co'], h5_file)

        else:
            raise ValueError(f'The {model} atmospheric model is not available. Please choose from '
                             f'\'ames-cond\', \'ames-dusty\', \'atmo\', \'bt-settl\', '
                             f'\'bt-nextgen\', \'drift-phoexnix\', \'petitcode-cool-clear\', '
                             f'\'petitcode-cool-cloudy\', \'petitcode-hot-clear\', '
                             f'\'petitcode-hot-cloudy\', \'exo-rem\', \'bt-settl-cifist\', '
                             f'\'bt-cond\', \'bt-cond-feh\', \'blackbody\'.')

        h5_file.close()

    @typechecked
    def add_object(self,
                   object_name: str,
                   distance: Optional[Tuple[float, float]] = None,
                   app_mag: Optional[Dict[str,
                                          Union[Tuple[float, float],
                                                List[Tuple[float, float]]]]] = None,
                   flux_density: Optional[Dict[str, Tuple[float, float]]] = None,
                   spectrum: Optional[Dict[str,
                                           Tuple[str,
                                                 Optional[str],
                                                 Optional[float]]]] = None,
                   deredden: Union[Dict[str, float], float] = None) -> None:
        """
        Function for adding the photometric and/or spectroscopic data of an object to the database.

        Parameters
        ----------
        object_name: str
            Object name that will be used as label in the database.
        distance : tuple(float, float), None
            Distance and uncertainty (pc). Not stored if set to None.
        app_mag : dict, None
            Dictionary with the filter names, apparent magnitudes, and uncertainties. For example,
            ``{'Paranal/NACO.Lp': (15., 0.2), 'Paranal/NACO.Mp': (13., 0.3)}``. For the use of
            duplicate filter names, the magnitudes have to be provided in a list, for example
            ``{'Paranal/NACO.Lp': [(15., 0.2), (14.5, 0.5)], 'Paranal/NACO.Mp': (13., 0.3)}``.
            No photometric data is stored if set to ``None``.
        flux_density : dict, None
            Dictionary with filter names, flux densities (W m-2 um-1), and uncertainties
            (W m-1 um-1). For example, ``{'Paranal/NACO.Lp': (1e-15, 1e-16)}``. Currently,
            the use of duplicate filters is not implemented. The use of ``app_mag`` is preferred
            over ``flux_density`` because with ``flux_density`` only fluxes are stored while with
            ``app_mag`` both magnitudes and fluxes. However, ``flux_density`` can be used in case
            the magnitudes and/or filter profiles are not available. In that case, the fluxes can
            still be selected with ``inc_phot`` in :class:`~species.analysis.fit_model.FitModel`.
            The argument of ``flux_density`` is ignored if set to ``None``.
        spectrum : dict, None
            Dictionary with the spectrum, optional covariance matrix, and spectral resolution for
            each instrument. The input data can either have a FITS or ASCII format. The spectra
            should have 3 columns with wavelength (um), flux (W m-2 um-1), and uncertainty
            (W m-2 um-1). The covariance matrix should be 2D with the same number of wavelength
            points as the spectrum. For example, ``{'SPHERE': ('spectrum.dat', 'covariance.fits',
            50.)}``. No covariance data is stored if set to None, for example, ``{'SPHERE':
            ('spectrum.dat', None, 50.)}``. The ``spectrum`` parameter is ignored if set to None.
            For GRAVITY data, the same FITS file can be provided as spectrum and covariance matrix.
        deredden : dict, float, None
            Dictionary with ``spectrum`` and ``app_mag`` names that will de dereddened with the
            provided A_V. For example, ``deredden={'SPHERE': 1.5, 'Keck/NIRC2.J': 1.5}`` will
            deredden the provided spectrum named 'SPHERE' and the Keck/NIRC2 J-band photometry with
            a visual extinction of 1.5. For photometric fluxes, the filter-averaged extinction is
            used for the dereddening.

        Returns
        -------
        NoneType
            None
        """

        h5_file = h5py.File(self.database, 'a')

        if deredden is None:
            deredden = {}

        if app_mag is not None:
            if 'spectra/calibration/vega' not in h5_file:
                self.add_spectra('vega')

            for item in app_mag:
                if f'filters/{item}' not in h5_file:
                    self.add_filter(item)

        if flux_density is not None:
            if 'spectra/calibration/vega' not in h5_file:
                self.add_spectra('vega')

            for item in flux_density:
                if f'filters/{item}' not in h5_file:
                    self.add_filter(item)

        print(f'Adding object: {object_name}')

        if 'objects' not in h5_file:
            h5_file.create_group('objects')

        if f'objects/{object_name}' not in h5_file:
            h5_file.create_group(f'objects/{object_name}')

        if distance is not None:
            print(f'   - Distance (pc) = {distance[0]:.2f} +/- {distance[1]:.2f}')

            if f'objects/{object_name}/distance' in h5_file:
                del h5_file[f'objects/{object_name}/distance']

            h5_file.create_dataset(f'objects/{object_name}/distance',
                                   data=distance)  # (pc)

        flux = {}
        error = {}
        dered_phot = {}

        if app_mag is not None:
            for mag_item in app_mag:
                if isinstance(deredden, float) or mag_item in deredden:
                    read_filt = read_filter.ReadFilter(mag_item)
                    filter_profile = read_filt.get_filter()

                    if isinstance(deredden, float):
                        ext_mag = dust_util.ism_extinction(deredden, 3.1, filter_profile[:, 0])

                    else:
                        ext_mag = dust_util.ism_extinction(deredden[mag_item], 3.1,
                                                           filter_profile[:, 0])

                    synphot = photometry.SyntheticPhotometry(mag_item)

                    dered_phot[mag_item], _ = synphot.spectrum_to_flux(filter_profile[:, 0],
                                                                       10.**(0.4*ext_mag))

                else:
                    dered_phot[mag_item] = 1.

                if isinstance(app_mag[mag_item], tuple):

                    try:
                        synphot = photometry.SyntheticPhotometry(mag_item)

                        flux[mag_item], error[mag_item] = synphot.magnitude_to_flux(
                            app_mag[mag_item][0], app_mag[mag_item][1])

                        flux[mag_item] *= dered_phot[mag_item]

                    except KeyError:
                        warnings.warn(f'Filter \'{mag_item}\' is not available on the SVO Filter '
                                      f'Profile Service so a flux calibration can not be done. '
                                      f'Please add the filter manually with the \'add_filter\' '
                                      f'function. For now, only the \'{mag_item}\' magnitude of '
                                      f'\'{object_name}\' is stored.')

                        # Write NaNs if the filter is not available
                        flux[mag_item], error[mag_item] = np.nan, np.nan

                elif isinstance(app_mag[mag_item], list):
                    flux_list = []
                    error_list = []

                    for i, dupl_item in enumerate(app_mag[mag_item]):

                        try:
                            synphot = photometry.SyntheticPhotometry(mag_item)

                            flux_dupl, error_dupl = synphot.magnitude_to_flux(
                                dupl_item[0], dupl_item[1])

                            flux_dupl *= dered_phot[mag_item]

                        except KeyError:
                            warnings.warn(f'Filter \'{mag_item}\' is not available on the SVO '
                                          f'Filter Profile Service so a flux calibration can not '
                                          f'be done. Please add the filter manually with the '
                                          f'\'add_filter\' function. For now, only the '
                                          f'\'{mag_item}\' magnitude of \'{object_name}\' is '
                                          f'stored.')

                            # Write NaNs if the filter is not available
                            flux_dupl, error_dupl = np.nan, np.nan

                        flux_list.append(flux_dupl)
                        error_list.append(error_dupl)

                    flux[mag_item] = flux_list
                    error[mag_item] = error_list

                else:
                    raise ValueError('The values in the dictionary with magnitudes should be '
                                     'tuples or a list with tuples (in case duplicate filter '
                                     'names are required).')

            for mag_item in app_mag:
                if f'objects/{object_name}/{mag_item}' in h5_file:
                    del h5_file[f'objects/{object_name}/{mag_item}']

                if isinstance(app_mag[mag_item], tuple):
                    n_phot = 1

                    app_mag[mag_item] = (app_mag[mag_item][0] - 2.5*np.log10(dered_phot[mag_item]),
                                         app_mag[mag_item][1])

                    print(f'   - {mag_item}:')

                    print(f'      - Apparent magnitude = {app_mag[mag_item][0]:.2f} +/- '
                          f'{app_mag[mag_item][1]:.2f}')

                    print(f'      - Flux (W m-2 um-1) = {flux[mag_item]:.2e} +/- '
                          f'{error[mag_item]:.2e}')

                    if isinstance(deredden, float):
                        print(f'      - Dereddening A_V: {deredden}')

                    elif mag_item in deredden:
                        print(f'      - Dereddening A_V: {deredden[mag_item]}')

                    data = np.asarray([app_mag[mag_item][0],
                                       app_mag[mag_item][1],
                                       flux[mag_item],
                                       error[mag_item]])

                elif isinstance(app_mag[mag_item], list):
                    n_phot = len(app_mag[mag_item])
                    print(f'   - {mag_item} ({n_phot} values):')

                    mag_list = []
                    mag_err_list = []

                    for i, dupl_item in enumerate(app_mag[mag_item]):
                        dered_mag = app_mag[mag_item][i][0] - 2.5*np.log10(dered_phot[mag_item])
                        app_mag_item = (dered_mag, app_mag[mag_item][i][1])

                        print(f'      - Apparent magnitude = {app_mag_item[0]:.2f} +/- '
                              f'{app_mag_item[1]:.2f}')

                        print(f'      - Flux (W m-2 um-1) = {flux[mag_item][i]:.2e} +/- '
                              f'{error[mag_item][i]:.2e}')

                        mag_list.append(app_mag_item[0])
                        mag_err_list.append(app_mag_item[1])

                        if isinstance(deredden, float):
                            print(f'      - Dereddening A_V: {deredden}')

                        elif mag_item in deredden:
                            print(f'      - Dereddening A_V: {deredden[mag_item]}')

                    data = np.asarray([mag_list,
                                       mag_err_list,
                                       flux[mag_item],
                                       error[mag_item]])

                # (mag), (mag), (W m-2 um-1), (W m-2 um-1)
                dset = h5_file.create_dataset(f'objects/{object_name}/{mag_item}',
                                              data=data)

                dset.attrs['n_phot'] = n_phot

        if flux_density is not None:
            for flux_item in flux_density:
                if isinstance(deredden, float) or flux_item in deredden:
                    warnings.warn(f'The deredden parameter is not supported by flux_density. '
                                  f'Please use app_mag instead and/or open an issue on Github. '
                                  f'Ignoring the dereddening of {flux_item}.')

                if f'objects/{object_name}/{flux_item}' in h5_file:
                    del h5_file[f'objects/{object_name}/{flux_item}']

                if isinstance(flux_density[flux_item], tuple):
                    print(f'   - {flux_item}:')

                    print(f'      - Flux (W m-2 um-1) = {flux_density[flux_item][0]:.2e} +/- '
                          f'{flux_density[flux_item][1]:.2e}')

                    data = np.asarray([np.nan,
                                       np.nan,
                                       flux_density[flux_item][0],
                                       flux_density[flux_item][1]])

                    # None, None, (W m-2 um-1), (W m-2 um-1)
                    dset = h5_file.create_dataset(f'objects/{object_name}/{flux_item}',
                                                  data=data)

                    dset.attrs['n_phot'] = 1

        if spectrum is not None:
            read_spec = {}
            read_cov = {}

            if f'objects/{object_name}/spectrum' in h5_file:
                del h5_file[f'objects/{object_name}/spectrum']

            # Read spectra

            for key, value in spectrum.items():
                if value[0].endswith('.fits') or value[0].endswith('.fit'):
                    with fits.open(value[0]) as hdulist:
                        if 'INSTRU' in hdulist[0].header and \
                                hdulist[0].header['INSTRU'] == 'GRAVITY':
                            # Read data from a FITS file with the GRAVITY format
                            print('   - GRAVITY spectrum:')

                            gravity_object = hdulist[0].header['OBJECT']
                            print(f'      - Object: {gravity_object}')

                            wavelength = hdulist[1].data['WAVELENGTH']  # (um)
                            flux = hdulist[1].data['FLUX']  # (W m-2 um-1)
                            covariance = hdulist[1].data['COVARIANCE']  # (W m-2 um-1)^2
                            error = np.sqrt(np.diag(covariance))  # (W m-2 um-1)

                            read_spec[key] = np.column_stack([wavelength, flux, error])

                        else:
                            # Otherwise try to read a 2D dataset with 3 columns
                            print('   - Spectrum:')

                            for i, hdu_item in enumerate(hdulist):
                                data = np.asarray(hdu_item.data)

                                if data.ndim == 2 and 3 in data.shape and key not in read_spec:
                                    read_spec[key] = data

                            if key not in read_spec:
                                raise ValueError(f'The spectrum data from {value[0]} can not be '
                                                 f'read. The data format should be 2D with 3 '
                                                 f'columns.')

                else:
                    try:
                        data = np.loadtxt(value[0])
                    except UnicodeDecodeError:
                        raise ValueError(f'The spectrum data from {value[0]} can not be read. '
                                         f'Please provide a FITS or ASCII file.')

                    if data.ndim != 2 or 3 not in data.shape:
                        raise ValueError(f'The spectrum data from {value[0]} can not be read. The '
                                         f'data format should be 2D with 3 columns.')

                    print('   - Spectrum:')
                    read_spec[key] = data

                if isinstance(deredden, float):
                    ext_mag = dust_util.ism_extinction(deredden, 3.1, read_spec[key][:, 0])
                    read_spec[key][:, 1] *= 10.**(0.4*ext_mag)

                elif key in deredden:
                    ext_mag = dust_util.ism_extinction(deredden[key], 3.1, read_spec[key][:, 0])
                    read_spec[key][:, 1] *= 10.**(0.4*ext_mag)

                wavelength = read_spec[key][:, 0]
                flux = read_spec[key][:, 1]
                error = read_spec[key][:, 2]

                print(f'      - Database tag: {key}')
                print(f'      - Filename: {value[0]}')
                print(f'      - Data shape: {read_spec[key].shape}')
                print(f'      - Wavelength range (um): {wavelength[0]:.2f} - {wavelength[-1]:.2f}')
                print(f'      - Mean flux (W m-2 um-1): {np.nanmean(flux):.2e}')
                print(f'      - Mean error (W m-2 um-1): {np.nanmean(error):.2e}')

                if isinstance(deredden, float):
                    print(f'      - Dereddening A_V: {deredden}')

                elif key in deredden:
                    print(f'      - Dereddening A_V: {deredden[key]}')

            # Read covariance matrix

            for key, value in spectrum.items():
                if value[1] is None:
                    read_cov[key] = None

                elif value[1].endswith('.fits') or value[1].endswith('.fit'):
                    with fits.open(value[1]) as hdulist:
                        if 'INSTRU' in hdulist[0].header and \
                                hdulist[0].header['INSTRU'] == 'GRAVITY':
                            # Read data from a FITS file with the GRAVITY format
                            print('   - GRAVITY covariance matrix:')

                            gravity_object = hdulist[0].header['OBJECT']
                            print(f'      - Object: {gravity_object}')

                            read_cov[key] = hdulist[1].data['COVARIANCE']  # (W m-2 um-1)^2

                        else:
                            # Otherwise try to read a square, 2D dataset
                            print('   - Covariance matrix:')

                            for i, hdu_item in enumerate(hdulist):
                                data = np.asarray(hdu_item.data)

                                corr_warn = f'The matrix from {value[1]} contains ' \
                                            f'ones along the diagonal. Converting this ' \
                                            f'correlation matrix into a covariance matrix.'

                                if data.ndim == 2 and data.shape[0] == data.shape[1]:
                                    if key not in read_cov:
                                        if data.shape[0] == read_spec[key].shape[0]:
                                            if np.all(np.diag(data) == 1.):
                                                warnings.warn(corr_warn)

                                                read_cov[key] = data_util.correlation_to_covariance(
                                                    data, read_spec[key][:, 2])

                                            else:
                                                read_cov[key] = data

                            if key not in read_cov:
                                raise ValueError(f'The covariance matrix from {value[1]} can not '
                                                 f'be read. The data format should be 2D with the '
                                                 f'same number of wavelength points as the '
                                                 f'spectrum.')

                else:
                    try:
                        data = np.loadtxt(value[1])
                    except UnicodeDecodeError:
                        raise ValueError(f'The covariance matrix from {value[1]} can not be read. '
                                         f'Please provide a FITS or ASCII file.')

                    if data.ndim != 2 or 3 not in data.shape:
                        raise ValueError(f'The covariance matrix from {value[1]} can not be read. '
                                         f'The data format should be 2D with the same number of '
                                         f'wavelength points as the spectrum.')

                    print('   - Covariance matrix:')

                    if np.all(np.diag(data) == 1.):
                        warnings.warn(f'The matrix from {value[1]} contains ones on '
                                      f'the diagonal. Converting this correlation matrix into a '
                                      f'covariance matrix.')

                        read_cov[key] = data_util.correlation_to_covariance(
                            data, read_spec[key][:, 2])

                    else:
                        read_cov[key] = data

                if read_cov[key] is not None:
                    sign, log_det_cov = np.linalg.slogdet(read_cov[key])

                    if sign != 1.:
                        raise ValueError(f'Unexpected sign ({sign}) from np.linalg.slogdet.')

                    print(f'      - Database tag: {key}')
                    print(f'      - Filename: {value[1]}')
                    print(f'      - Data shape: {read_cov[key].shape}')
                    print(f'      - Log-determinant: {log_det_cov:.2f}')

            print('   - Spectral resolution:')

            for key, value in spectrum.items():

                h5_file.create_dataset(f'objects/{object_name}/spectrum/{key}/spectrum',
                                       data=read_spec[key])

                if read_cov[key] is not None:
                    h5_file.create_dataset(f'objects/{object_name}/spectrum/{key}/covariance',
                                           data=read_cov[key])

                    _, log_det_cov = np.linalg.slogdet(read_cov[key])

                    dset = h5_file[f'objects/{object_name}/spectrum/{key}/covariance']
                    dset.attrs['log_det_cov'] = log_det_cov

                    h5_file.create_dataset(f'objects/{object_name}/spectrum/{key}/inv_covariance',
                                           data=linalg.inv(read_cov[key]))

                    if not np.allclose(read_spec[key][:, 2], np.sqrt(np.diag(read_cov[key]))):
                        raise ValueError(f'The uncertainties from the spectrum file of {key} are '
                                         f'different from the uncorrelated uncertainties from the '
                                         f'covariance file.')

                dset = h5_file[f'objects/{object_name}/spectrum/{key}']

                if value[2] is None:
                    print(f'      - {key}: None')
                    dset.attrs['specres'] = 0.

                else:
                    print(f'      - {key}: {value[2]:.1f}')
                    dset.attrs['specres'] = value[2]

        h5_file.close()

    @typechecked
    def add_photometry(self,
                       phot_library: str) -> None:
        """
        Parameters
        ----------
        phot_library : str
            Photometric library ('vlm-plx' or 'leggett').

        Returns
        -------
        NoneType
            None
        """

        h5_file = h5py.File(self.database, 'a')

        if 'photometry' not in h5_file:
            h5_file.create_group('photometry')

        if 'photometry/'+phot_library in h5_file:
            del h5_file['photometry/'+phot_library]

        if phot_library[0:7] == 'vlm-plx':
            vlm_plx.add_vlm_plx(self.input_path, h5_file)

        elif phot_library[0:7] == 'leggett':
            leggett.add_leggett(self.input_path, h5_file)

        h5_file.close()

    @typechecked
    def add_calibration(self,
                        tag: str,
                        filename: Optional[str] = None,
                        data: Optional[np.ndarray] = None,
                        units: Optional[Dict[str, str]] = None,
                        scaling: Optional[Tuple[float, float]] = None) -> None:
        """
        Function for adding a calibration spectrum to the database.

        Parameters
        ----------
        tag : str
            Tag name in the database.
        filename : str, None
            Filename with the calibration spectrum. The first column should contain the wavelength
            (um), the second column the flux density (W m-2 um-1), and the third column
            the error (W m-2 um-1). The ``data`` argument is used if set to ``None``.
        data : np.ndarray, None
            Spectrum stored as 3D array with shape ``(n_wavelength, 3)``. The first column should
            contain the wavelength (um), the second column the flux density (W m-2 um-1),
            and the third column the error (W m-2 um-1). The ``filename`` argument is used if set
            to ``None``.
        units : dict, None
            Dictionary with the wavelength and flux units, e.g. ``{'wavelength': 'angstrom',
            'flux': 'w m-2'}``. The default units (um and W m-2 um-1) are used if set to ``None``.
        scaling : tuple(float, float), None
            Scaling for the wavelength and flux as ``(scaling_wavelength, scaling_flux)``. Not used
            if set to ``None``.

        Returns
        -------
        NoneType
            None
        """

        if filename is None and data is None:
            raise ValueError('Either the \'filename\' or \'data\' argument should be provided.')

        if scaling is None:
            scaling = (1., 1.)

        h5_file = h5py.File(self.database, 'a')

        if 'spectra/calibration' not in h5_file:
            h5_file.create_group('spectra/calibration')

        if 'spectra/calibration/'+tag in h5_file:
            del h5_file['spectra/calibration/'+tag]

        if filename is not None:
            data = np.loadtxt(filename)

        if units is None:
            wavelength = scaling[0]*data[:, 0]  # (um)
            flux = scaling[1]*data[:, 1]  # (W m-2 um-1)

        else:
            if units['wavelength'] == 'um':
                wavelength = scaling[0]*data[:, 0]  # (um)

            elif units['wavelength'] == 'angstrom':
                wavelength = scaling[0]*data[:, 0]*1e-4  # (um)

            if units['flux'] == 'w m-2 um-1':
                flux = scaling[1]*data[:, 1]  # (W m-2 um-1)

            elif units['flux'] == 'w m-2':
                if units['wavelength'] == 'um':
                    flux = scaling[1]*data[:, 1]/wavelength  # (W m-2 um-1)

        if data.shape[1] == 3:
            if units is None:
                error = scaling[1]*data[:, 2]  # (W m-2 um-1)

            else:
                if units['flux'] == 'w m-2 um-1':
                    error = scaling[1]*data[:, 2]  # (W m-2 um-1)

                elif units['flux'] == 'w m-2':
                    if units['wavelength'] == 'um':
                        error = scaling[1]*data[:, 2]/wavelength  # (W m-2 um-1)

        else:
            error = np.repeat(0., wavelength.size)

        print(f'Adding calibration spectrum: {tag}...', end='', flush=True)

        h5_file.create_dataset(f'spectra/calibration/{tag}',
                               data=np.vstack((wavelength, flux, error)))

        h5_file.close()

        print(' [DONE]')

    @typechecked
    def add_spectrum(self,
                     spec_library: str,
                     sptypes: Optional[List[str]] = None) -> None:
        """
        DEPRECATION: This method is deprecated and will be removed in a future release. Please use
        the :meth:`~species.data.database.Database.add_spectra` method instead.

        Parameters
        ----------
        spec_library : str
            Spectral library ('irtf', 'spex', 'kesseli+2017', 'bonnefoy+2014', 'allers+2013').
        sptypes : list(str)
            Spectral types ('F', 'G', 'K', 'M', 'L', 'T'). Currently only implemented for 'irtf'.

        Returns
        -------
        NoneType
            None
        """

        warnings.warn('This method is deprecated and will be removed in a future release. Please '
                      'use the add_spectra method instead.')

        h5_file = h5py.File(self.database, 'a')

        if 'spectra' not in h5_file:
            h5_file.create_group('spectra')

        if 'spectra/'+spec_library in h5_file:
            del h5_file['spectra/'+spec_library]

        if spec_library[0:5] == 'vega':
            vega.add_vega(self.input_path, h5_file)

        elif spec_library[0:5] == 'irtf':
            irtf.add_irtf(self.input_path, h5_file, sptypes)

        elif spec_library[0:5] == 'spex':
            spex.add_spex(self.input_path, h5_file)

        elif spec_library[0:12] == 'kesseli+2017':
            kesseli2017.add_kesseli2017(self.input_path, h5_file)

        elif spec_library[0:13] == 'bonnefoy+2014':
            bonnefoy2014.add_bonnefoy2014(self.input_path, h5_file)

        elif spec_library[0:11] == 'allers+2013':
            allers2013.add_allers2013(self.input_path, h5_file)

        h5_file.close()

    @typechecked
    def add_spectra(self,
                    spec_library: str,
                    sptypes: Optional[List[str]] = None) -> None:
        """
        Parameters
        ----------
        spec_library : str
            Spectral library ('irtf', 'spex', 'kesseli+2017', 'bonnefoy+2014', 'allers+2013').
        sptypes : list(str)
            Spectral types ('F', 'G', 'K', 'M', 'L', 'T'). Currently only implemented for 'irtf'.

        Returns
        -------
        NoneType
            None
        """

        h5_file = h5py.File(self.database, 'a')

        if 'spectra' not in h5_file:
            h5_file.create_group('spectra')

        if 'spectra/'+spec_library in h5_file:
            del h5_file['spectra/'+spec_library]

        if spec_library[0:5] == 'vega':
            vega.add_vega(self.input_path, h5_file)

        elif spec_library[0:5] == 'irtf':
            irtf.add_irtf(self.input_path, h5_file, sptypes)

        elif spec_library[0:5] == 'spex':
            spex.add_spex(self.input_path, h5_file)

        elif spec_library[0:12] == 'kesseli+2017':
            kesseli2017.add_kesseli2017(self.input_path, h5_file)

        elif spec_library[0:13] == 'bonnefoy+2014':
            bonnefoy2014.add_bonnefoy2014(self.input_path, h5_file)

        elif spec_library[0:11] == 'allers+2013':
            allers2013.add_allers2013(self.input_path, h5_file)

        h5_file.close()

    @typechecked
    def add_samples(self,
                    sampler: str,
                    samples: np.ndarray,
                    ln_prob: np.ndarray,
                    ln_evidence: Optional[Tuple[float, float]],
                    mean_accept: Optional[float],
                    spectrum: Tuple[str, str],
                    tag: str,
                    modelpar: List[str],
                    distance: Optional[float],
                    spec_labels: Optional[List[str]]):
        """
        Parameters
        ----------
        sampler : str
            Sampler ('emcee', 'multinest', or 'ultranest').
        samples : np.ndarray
            Samples of the posterior.
        ln_prob : np.ndarray
            Log posterior for each sample.
        ln_evidence : tuple(float, float)
            Log evidence and uncertainty. Set to ``None`` when ``sampler`` is 'emcee'.
        mean_accept : float, None
            Mean acceptance fraction. Set to ``None`` when ``sampler`` is 'multinest' or
            'ultranest'.
        spectrum : tuple(str, str)
            Tuple with the spectrum type ('model' or 'calibration') and spectrum name (e.g.
            'drift-phoenix').
        tag : str
            Database tag.
        modelpar : list(str)
            List with the model parameter names.
        distance : float, None
            Distance to the object (pc). Not used if set to ``None``.
        spec_labels : list(str), None
            List with the spectrum labels that are used for fitting an additional scaling
            parameter. Not used if set to ``None``.

        Returns
        -------
        NoneType
            None
        """

        if spec_labels is None:
            spec_labels = []

        h5_file = h5py.File(self.database, 'a')

        if 'results' not in h5_file:
            h5_file.create_group('results')

        if 'results/fit' not in h5_file:
            h5_file.create_group('results/fit')

        if f'results/fit/{tag}' in h5_file:
            del h5_file[f'results/fit/{tag}']

        dset = h5_file.create_dataset(f'results/fit/{tag}/samples', data=samples)
        h5_file.create_dataset(f'results/fit/{tag}/ln_prob', data=ln_prob)

        dset.attrs['type'] = str(spectrum[0])
        dset.attrs['spectrum'] = str(spectrum[1])
        dset.attrs['n_param'] = int(len(modelpar))
        dset.attrs['sampler'] = str(sampler)

        if mean_accept is not None:
            dset.attrs['mean_accept'] = float(mean_accept)
            print(f'Mean acceptance fraction: {mean_accept:.3f}')

        if distance is not None:
            dset.attrs['distance'] = float(distance)

        if ln_evidence is not None:
            dset.attrs['ln_evidence'] = ln_evidence

        count_scaling = 0

        for i, item in enumerate(modelpar):
            dset.attrs[f'parameter{i}'] = str(item)

            if item in spec_labels:
                dset.attrs[f'scaling{count_scaling}'] = str(item)
                count_scaling += 1

        dset.attrs['n_scaling'] = int(count_scaling)

        try:
            int_auto = emcee.autocorr.integrated_time(samples)
            print(f'Integrated autocorrelation time = {int_auto}')

        except emcee.autocorr.AutocorrError:
            int_auto = None

            print('The chain is shorter than 50 times the integrated autocorrelation time. '
                  '[WARNING]')

        if int_auto is not None:
            for i, item in enumerate(int_auto):
                dset.attrs[f'autocorrelation{i}'] = float(item)

        h5_file.close()

    @typechecked
    def get_probable_sample(self,
                            tag: str,
                            burnin: Optional[int] = None) -> Dict[str, float]:
        """
        Function for extracting the sample parameters with the highest posterior probability.

        Parameters
        ----------
        tag : str
            Database tag with the posterior results.
        burnin : int, None
            Number of burnin steps. No burnin is removed if set to ``None``.

        Returns
        -------
        dict
            Parameters and values for the sample with the maximum posterior probability.
        """

        if burnin is None:
            burnin = 0

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file[f'results/fit/{tag}/samples']

        samples = np.asarray(dset)
        ln_prob = np.asarray(h5_file[f'results/fit/{tag}/ln_prob'])

        if 'n_param' in dset.attrs:
            n_param = dset.attrs['n_param']
        elif 'nparam' in dset.attrs:
            n_param = dset.attrs['nparam']

        if samples.ndim == 3:
            if burnin > samples.shape[1]:
                raise ValueError(f'The \'burnin\' value is larger than the number of steps '
                                 f'({samples.shape[1]}) that are made by the walkers.')

            samples = samples[:, burnin:, :]
            ln_prob = ln_prob[:, burnin:]

            samples = np.reshape(samples, (-1, n_param))
            ln_prob = np.reshape(ln_prob, -1)

        index_max = np.unravel_index(ln_prob.argmax(), ln_prob.shape)

        # max_prob = ln_prob[index_max]
        max_sample = samples[index_max]

        prob_sample = {}

        for i in range(n_param):
            par_key = dset.attrs[f'parameter{i}']
            par_value = max_sample[i]

            prob_sample[par_key] = par_value

        if dset.attrs.__contains__('distance'):
            prob_sample['distance'] = dset.attrs['distance']

        h5_file.close()

        return prob_sample

    @typechecked
    def get_median_sample(self,
                          tag: str,
                          burnin: Optional[int] = None) -> Dict[str, float]:
        """
        Function for extracting the median parameter values from the posterior samples.

        Parameters
        ----------
        tag : str
            Database tag with the posterior results.
        burnin : int, None
            Number of burnin steps. No burnin is removed if set to ``None``.

        Returns
        -------
        dict
            Parameters and values for the sample with the maximum posterior probability.
        """

        if burnin is None:
            burnin = 0

        with h5py.File(self.database, 'r') as h5_file:
            dset = h5_file[f'results/fit/{tag}/samples']

            if 'n_param' in dset.attrs:
                n_param = dset.attrs['n_param']
            elif 'nparam' in dset.attrs:
                n_param = dset.attrs['nparam']

            samples = np.asarray(dset)

            # samples = samples[samples[:, 2] > 100., ]

            if samples.ndim == 3:
                if burnin > samples.shape[1]:
                    raise ValueError(f'The \'burnin\' value is larger than the number of steps '
                                     f'({samples.shape[1]}) that are made by the walkers.')

                if burnin is not None:
                    samples = samples[:, burnin:, :]

                samples = np.reshape(samples, (-1, n_param))

            median_sample = {}

            for i in range(n_param):
                par_key = dset.attrs[f'parameter{i}']
                par_value = np.percentile(samples[:, i], 50.)
                median_sample[par_key] = par_value

            if dset.attrs.__contains__('distance'):
                median_sample['distance'] = dset.attrs['distance']

        return median_sample

    @typechecked
    def get_compare_sample(self,
                           tag: str,
                           spec_fix: Optional[str] = None) -> Dict[str, float]:
        """
        Function for extracting the sample parameters with the highest posterior probability.

        Parameters
        ----------
        tag : str
            Database tag where the results from
            :meth:`~species.analysis.compare_spectra.CompareSpectra.compare_model` are stored.
        spec_fix : str, None
            After comparing multiple spectra with a model grid, one of the flux scalings need to
            be used to calculate the planet radius (i.e. scaling = (radius/distance)^2). The name
            of this spectrum is specified as argument of ``spec_fix``. For the other spectra,
            the same radius will be used and an additional scaling parameter will be included in
            the returned dictionary. When passing the returned dictionary to
            :func:`~species.util.read_util.update_spectra`, the spectra can be updated with the
            derived scaling corrections. The argument can be set to ``None`` if a single spectrum
            was used for the comparison.

        Returns
        -------
        dict
            Dictionary with the best-fit parameters.
        """

        with h5py.File(self.database, 'a') as h5_file:
            dset = h5_file[f'results/comparison/{tag}/goodness_of_fit']

            n_param = dset.attrs['n_param']
            n_spec_name = dset.attrs['n_spec_name']

            model_param = {}

            for i in range(n_param):
                model_param[dset.attrs[f'parameter{i}']] = dset.attrs[f'best_param{i}']

            model_param['distance'] = dset.attrs['distance']

            if n_spec_name == 1:
                spec_name = dset.attrs['spec_name0']
                model_param['radius'] = dset.attrs[f'radius_{spec_name}']

            else:
                if spec_fix is None:
                    raise ValueError('The argument of \'spec_fix\' should be set when the results '
                                     'from CompareSpectra.compare_model have been obtained by '
                                     'combining multiple spectra (i.e. the argument of '
                                     '\'spec_name\' in CompareSpectra).')

                model_param['radius'] = dset.attrs[f'radius_{spec_fix}']

                for i in range(n_spec_name):
                    spec_name = dset.attrs[f'spec_name{i}']

                    if spec_name == spec_fix:
                        continue

                    # Factor for scaling the observed spectrum to the model spectrum
                    model_param[f'scaling_{spec_name}'] = (
                        dset.attrs[f'radius_{spec_fix}'] / dset.attrs[f'radius_{spec_name}'])**2

        return model_param

    @typechecked
    def get_mcmc_spectra(self,
                         tag: str,
                         random: int,
                         burnin: Optional[int] = None,
                         wavel_range: Optional[Union[Tuple[float, float], str]] = None,
                         spec_res: Optional[float] = None,
                         wavel_resample: Optional[np.ndarray] = None) -> Union[
                             List[box.ModelBox], List[box.SpectrumBox]]:
        """
        Function for drawing random spectra from the sampled posterior distributions.

        Parameters
        ----------
        tag : str
            Database tag with the posterior samples.
        random : int
            Number of random samples.
        burnin : int, None
            Number of burnin steps. No burnin is removed if set to ``None``. Not required when
            using nested sampling.
        wavel_range : tuple(float, float), str, None
            Wavelength range (um) or filter name. Full spectrum is used if set to ``None``.
        spec_res : float
            Spectral resolution that is used for the smoothing with a Gaussian kernel. No smoothing
            is applied if the argument set to ``None``.
        wavel_resample : np.ndarray, None
            Wavelength points (um) to which the model spectrum will be resampled. The resampling is
            applied after the optional smoothing to the resolution of ``spec_res``.

        Returns
        -------
        list(species.core.box.ModelBox)
            List with ``ModelBox`` objects.
        """

        if burnin is None:
            burnin = 0

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file[f'results/fit/{tag}/samples']

        spectrum_type = dset.attrs['type']
        spectrum_name = dset.attrs['spectrum']

        if 'n_param' in dset.attrs:
            n_param = dset.attrs['n_param']
        elif 'nparam' in dset.attrs:
            n_param = dset.attrs['nparam']

        if 'n_scaling' in dset.attrs:
            n_scaling = dset.attrs['n_scaling']
        elif 'nscaling' in dset.attrs:
            n_scaling = dset.attrs['nscaling']
        else:
            n_scaling = 0

        if 'n_error' in dset.attrs:
            n_error = dset.attrs['n_error']
        else:
            n_error = 0

        ignore_param = []

        for i in range(n_scaling):
            ignore_param.append(dset.attrs[f'scaling{i}'])

        for i in range(n_error):
            ignore_param.append(dset.attrs[f'error{i}'])

        for i in range(n_param):
            if dset.attrs[f'parameter{i}'][:13] == 'log_corr_len_':
                ignore_param.append(dset.attrs[f'parameter{i}'])

            elif dset.attrs[f'parameter{i}'][:9] == 'corr_amp_':
                ignore_param.append(dset.attrs[f'parameter{i}'])

        if spec_res is not None and spectrum_type == 'calibration':
            warnings.warn('Smoothing of the spectral resolution is not implemented for calibration '
                          'spectra.')

        if dset.attrs.__contains__('distance'):
            distance = dset.attrs['distance']
        else:
            distance = None

        samples = np.asarray(dset)

        # samples = samples[samples[:, 2] > 100., ]

        if samples.ndim == 2:
            ran_index = np.random.randint(samples.shape[0], size=random)
            samples = samples[ran_index, ]

        elif samples.ndim == 3:
            if burnin > samples.shape[1]:
                raise ValueError(f'The \'burnin\' value is larger than the number of steps '
                                 f'({samples.shape[1]}) that are made by the walkers.')

            samples = samples[:, burnin:, :]

            ran_walker = np.random.randint(samples.shape[0], size=random)
            ran_step = np.random.randint(samples.shape[1], size=random)
            samples = samples[ran_walker, ran_step, :]

        param = []
        for i in range(n_param):
            param.append(dset.attrs[f'parameter{i}'])

        if spectrum_type == 'model':
            if spectrum_name == 'planck':
                readmodel = read_planck.ReadPlanck(wavel_range)

            elif spectrum_name == 'powerlaw':
                pass

            else:
                readmodel = read_model.ReadModel(spectrum_name, wavel_range=wavel_range)

        elif spectrum_type == 'calibration':
            readcalib = read_calibration.ReadCalibration(spectrum_name, filter_name=None)

        boxes = []

        for i in tqdm.tqdm(range(samples.shape[0]), desc='Getting MCMC spectra'):
            model_param = {}
            for j in range(samples.shape[1]):
                if param[j] not in ignore_param:
                    model_param[param[j]] = samples[i, j]

            if distance:
                model_param['distance'] = distance

            if spectrum_type == 'model':
                if spectrum_name == 'planck':
                    specbox = readmodel.get_spectrum(model_param,
                                                     spec_res,
                                                     smooth=True,
                                                     wavel_resample=wavel_resample)

                elif spectrum_name == 'powerlaw':
                    if wavel_resample is not None:
                        warnings.warn('The \'wavel_resample\' parameter is not support by the '
                                      '\'powerlaw\' model so the argument will be ignored.')

                    specbox = read_util.powerlaw_spectrum(wavel_range, model_param)

                else:
                    specbox = readmodel.get_model(model_param,
                                                  spec_res=spec_res,
                                                  wavel_resample=wavel_resample,
                                                  smooth=True)

            elif spectrum_type == 'calibration':
                specbox = readcalib.get_spectrum(model_param)

            boxes.append(specbox)

        h5_file.close()

        return boxes

    @typechecked
    def get_mcmc_photometry(self,
                            tag: str,
                            filter_name: str,
                            burnin: Optional[int] = None,
                            phot_type: str = 'magnitude') -> np.ndarray:
        """
        Function for calculating synthetic magnitudes or fluxes from the posterior samples.

        Parameters
        ----------
        tag : str
            Database tag with the posterior samples.
        filter_name : str
            Filter name for which the synthetic photometry will be computed.
        burnin : int, None
            Number of burnin steps. No burnin is removed if set to ``None``. Not required when
            using nested sampling.
        phot_type : str
            Photometry type ('magnitude' or 'flux').

        Returns
        -------
        np.ndarray
            Synthetic magnitudes or fluxes (W m-2 um-1).
        """

        if phot_type not in ['magnitude', 'flux']:
            raise ValueError('The argument of \'phot_type\' is not recognized and should be '
                             'set to \'magnitude\' or \'flux\'.')

        if burnin is None:
            burnin = 0

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file[f'results/fit/{tag}/samples']

        if 'n_param' in dset.attrs:
            n_param = dset.attrs['n_param']
        elif 'nparam' in dset.attrs:
            n_param = dset.attrs['nparam']

        spectrum_type = dset.attrs['type']
        spectrum_name = dset.attrs['spectrum']

        if dset.attrs.__contains__('distance'):
            distance = dset.attrs['distance']
        else:
            distance = None

        samples = np.asarray(dset)

        if samples.ndim == 3:
            if burnin > samples.shape[1]:
                raise ValueError(f'The \'burnin\' value is larger than the number of steps '
                                 f'({samples.shape[1]}) that are made by the walkers.')

            samples = samples[:, burnin:, :]
            samples = samples.reshape((samples.shape[0]*samples.shape[1], n_param))

        param = []
        for i in range(n_param):
            param.append(dset.attrs[f'parameter{i}'])

        h5_file.close()

        if spectrum_type == 'model':
            if spectrum_name == 'powerlaw':
                synphot = photometry.SyntheticPhotometry(filter_name)
                synphot.zero_point()  # Set the wavel_range attribute

            else:
                readmodel = read_model.ReadModel(spectrum_name, filter_name=filter_name)

        elif spectrum_type == 'calibration':
            readcalib = read_calibration.ReadCalibration(spectrum_name, filter_name=filter_name)

        mcmc_phot = np.zeros((samples.shape[0]))

        for i in tqdm.tqdm(range(samples.shape[0]), desc='Getting MCMC photometry'):
            model_param = {}

            for j in range(n_param):
                model_param[param[j]] = samples[i, j]

            if distance is not None:
                model_param['distance'] = distance

            if spectrum_type == 'model':
                if spectrum_name == 'powerlaw':
                    pl_box = read_util.powerlaw_spectrum(synphot.wavel_range, model_param)

                    if phot_type == 'magnitude':
                        app_mag, _ = synphot.spectrum_to_magnitude(pl_box.wavelength, pl_box.flux)
                        mcmc_phot[i] = app_mag[0]

                    elif phot_type == 'flux':
                        mcmc_phot[i], _ = synphot.spectrum_to_flux(pl_box.wavelength, pl_box.flux)

                else:
                    if phot_type == 'magnitude':
                        mcmc_phot[i], _ = readmodel.get_magnitude(model_param)

                    elif phot_type == 'flux':
                        mcmc_phot[i], _ = readmodel.get_flux(model_param)

            elif spectrum_type == 'calibration':
                if phot_type == 'magnitude':
                    app_mag, _ = readcalib.get_magnitude(model_param=model_param, distance=None)
                    mcmc_phot[i] = app_mag[0]

                elif phot_type == 'flux':
                    mcmc_phot[i], _ = readcalib.get_flux(model_param=model_param)

        return mcmc_phot

    @typechecked
    def get_object(self,
                   object_name: str,
                   inc_phot: Union[bool, List[str]] = True,
                   inc_spec: Union[bool, List[str]] = True) -> box.ObjectBox:
        """
        Function for extracting the photometric and/or spectroscopic data of an object from the
        database. The spectroscopic data contains optionally the covariance matrix and its inverse.

        Parameters
        ----------
        object_name : str
            Object name in the database.
        inc_phot : bool, list(str)
            Include photometric data. If a boolean, either all (``True``) or none (``False``) of
            the data are selected. If a list, a subset of filter names (as stored in the database)
            can be provided.
        inc_spec : bool, list(str)
            Include spectroscopic data. If a boolean, either all (``True``) or none (``False``) of
            the data are selected. If a list, a subset of spectrum names (as stored in the database
            with :func:`~species.data.database.Database.add_object`) can be provided.

        Returns
        -------
        species.core.box.ObjectBox
            Box with the object's data.
        """

        print(f'Getting object: {object_name}...', end='', flush=True)

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file[f'objects/{object_name}']

        distance = np.asarray(dset['distance'])

        if inc_phot:

            magnitude = {}
            flux = {}
            mean_wavel = {}

            for observatory in dset.keys():
                if observatory not in ['distance', 'spectrum']:
                    for filter_name in dset[observatory]:
                        name = f'{observatory}/{filter_name}'

                        if isinstance(inc_phot, bool) or name in inc_phot:
                            magnitude[name] = dset[name][0:2]
                            flux[name] = dset[name][2:4]

                            filter_trans = read_filter.ReadFilter(name)
                            mean_wavel[name] = filter_trans.mean_wavelength()

            phot_filters = list(magnitude.keys())

        else:

            magnitude = None
            flux = None
            phot_filters = None
            mean_wavel = None

        if inc_spec and f'objects/{object_name}/spectrum' in h5_file:
            spectrum = {}

            for item in h5_file[f'objects/{object_name}/spectrum']:
                data_group = f'objects/{object_name}/spectrum/{item}'

                if isinstance(inc_spec, bool) or item in inc_spec:

                    if f'{data_group}/covariance' not in h5_file:
                        spectrum[item] = (np.asarray(h5_file[f'{data_group}/spectrum']),
                                          None,
                                          None,
                                          h5_file[f'{data_group}'].attrs['specres'])

                    else:
                        spectrum[item] = (np.asarray(h5_file[f'{data_group}/spectrum']),
                                          np.asarray(h5_file[f'{data_group}/covariance']),
                                          np.asarray(h5_file[f'{data_group}/inv_covariance']),
                                          h5_file[f'{data_group}'].attrs['specres'])

        else:
            spectrum = None

        h5_file.close()

        print(' [DONE]')

        return box.create_box('object',
                              name=object_name,
                              filters=phot_filters,
                              mean_wavel=mean_wavel,
                              magnitude=magnitude,
                              flux=flux,
                              distance=distance,
                              spectrum=spectrum)

    @typechecked
    def get_samples(self,
                    tag: str,
                    burnin: Optional[int] = None,
                    random: Optional[int] = None,
                    json_file: Optional[str] = None) -> box.SamplesBox:
        """
        Parameters
        ----------
        tag: str
            Database tag with the samples.
        burnin : int, None
            Number of burnin samples to exclude. All samples are selected if set to ``None``.
            The parameter is only required for samples obtained with ``emcee`` and is therefore
            not used for samples obtained with ``MultiNest`` or ``UltraNest``.
        random : int, None
            Number of random samples to select. All samples (with the burnin excluded) are
            selected if set to ``None``.
        json_file : str, None
            JSON file to store the posterior samples. The data will not be written if the argument
            is set to ``None``.

        Returns
        -------
        species.core.box.SamplesBox
            Box with the posterior samples.
        """

        if burnin is None:
            burnin = 0

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file[f'results/fit/{tag}/samples']
        ln_prob = np.asarray(h5_file[f'results/fit/{tag}/ln_prob'])

        spectrum = dset.attrs['spectrum']

        if 'n_param' in dset.attrs:
            n_param = dset.attrs['n_param']
        elif 'nparam' in dset.attrs:
            n_param = dset.attrs['nparam']

        if 'ln_evidence' in dset.attrs:
            # Use if condition for backward compatibility
            ln_evidence = dset.attrs['ln_evidence']
        else:
            ln_evidence = None

        samples = np.asarray(dset)

        if samples.ndim == 3:
            if burnin > samples.shape[1]:
                raise ValueError(f'The \'burnin\' value is larger than the number of steps '
                                 f'({samples.shape[1]}) that are made by the walkers.')

            samples = samples[:, burnin:, :]

            if random:
                ran_walker = np.random.randint(samples.shape[0], size=random)
                ran_step = np.random.randint(samples.shape[1], size=random)
                samples = samples[ran_walker, ran_step, :]

        param = []
        for i in range(n_param):
            param.append(dset.attrs[f'parameter{i}'])

        h5_file.close()

        if samples.ndim == 3:
            prob_sample = self.get_probable_sample(tag, burnin)
        else:
            prob_sample = None

        median_sample = self.get_median_sample(tag, burnin)

        if json_file is not None:
            samples_dict = {}

            for i, item in enumerate(param):
                samples_dict[item] = list(samples[:, i])

            with open(json_file, 'w') as out_file:
                json.dump(samples_dict, out_file, indent=4)

        return box.create_box('samples',
                              spectrum=spectrum,
                              parameters=param,
                              samples=samples,
                              ln_prob=ln_prob,
                              ln_evidence=ln_evidence,
                              prob_sample=prob_sample,
                              median_sample=median_sample)

    @typechecked
    def add_empirical(self,
                      tag: str,
                      names: List[str],
                      sptypes: List[str],
                      goodness_of_fit: List[float],
                      flux_scaling: List[np.ndarray],
                      av_ext: List[float],
                      rad_vel: List[float],
                      object_name: str,
                      spec_name: List[str],
                      spec_library: str) -> None:
        """
        Parameters
        ----------
        tag : str
            Database tag where the results will be stored.
        names : list(str)
            Array with the names of the empirical spectra.
        sptypes : list(str)
            Array with the spectral types of ``names``.
        goodness_of_fit : list(float)
            Array with the goodness-of-fit values.
        flux_scaling : list(np.ndarray)
            List with arrays with the best-fit scaling values to match the library spectra with
            the data. The size of each array is equal to the number of spectra that are provided
            as argument of ``spec_name``.
        av_ext : list(float)
            Array with the visual extinction :math:`A_V`.
        rad_vel : list(float)
            Array with the radial velocities (km s-1).
        object_name : str
            Object name as stored in the database with
            :func:`~species.data.database.Database.add_object` or
            :func:`~species.data.database.Database.add_companion`.
        spec_name : list(str)
            List with spectrum names that are stored at the object data of ``object_name``.
        spec_library : str
            Name of the spectral library that was used for the empirical comparison.

        Returns
        -------
        NoneType
            None
        """

        with h5py.File(self.database, 'a') as h5_file:

            if 'results' not in h5_file:
                h5_file.create_group('results')

            if 'results/empirical' not in h5_file:
                h5_file.create_group('results/empirical')

            if f'results/empirical/{tag}' in h5_file:
                del h5_file[f'results/empirical/{tag}']

            dtype = h5py.special_dtype(vlen=str)

            dset = h5_file.create_dataset(f'results/empirical/{tag}/names',
                                          (np.size(names), ), dtype=dtype)

            dset[...] = names

            dset.attrs['object_name'] = str(object_name)
            dset.attrs['spec_library'] = str(spec_library)
            dset.attrs['n_spec_name'] = len(spec_name)

            for i, item in enumerate(spec_name):
                dset.attrs[f'spec_name{i}'] = item

            dset = h5_file.create_dataset(f'results/empirical/{tag}/sptypes',
                                          (np.size(sptypes), ), dtype=dtype)

            dset[...] = sptypes

            h5_file.create_dataset(f'results/empirical/{tag}/goodness_of_fit', data=goodness_of_fit)
            h5_file.create_dataset(f'results/empirical/{tag}/flux_scaling', data=flux_scaling)
            h5_file.create_dataset(f'results/empirical/{tag}/av_ext', data=av_ext)
            h5_file.create_dataset(f'results/empirical/{tag}/rad_vel', data=rad_vel)

    @typechecked
    def add_comparison(self,
                       tag: str,
                       goodness_of_fit: np.ndarray,
                       flux_scaling: np.ndarray,
                       model_param: List[str],
                       coord_points: List[np.ndarray],
                       object_name: str,
                       spec_name: List[str],
                       model: str) -> None:
        """
        Parameters
        ----------
        tag : str
            Database tag where the results will be stored.
        goodness_of_fit : np.ndarray
            Array with the goodness-of-fit values.
        flux_scaling : np.ndarray
            Array with the best-fit scaling values to match the model spectra with the data.
        model_param : list(str)
            List with the names of the model parameters.
        coord_points : list(np.ndarray)
            List with 1D arrays of the model grid points, in the same order as ``model_param``.
        object_name : str
            Object name as stored in the database with
            :func:`~species.data.database.Database.add_object` or
            :func:`~species.data.database.Database.add_companion`.
        spec_name : list(str)
            List with spectrum names that are stored at the object data of ``object_name``.
        model : str
            Atmospheric model grid that is used for the comparison.

        Returns
        -------
        NoneType
            None
        """

        read_obj = read_object.ReadObject(object_name)
        distance = read_obj.get_distance()[0]  # (pc)

        with h5py.File(self.database, 'a') as h5_file:

            if 'results' not in h5_file:
                h5_file.create_group('results')

            if 'results/comparison' not in h5_file:
                h5_file.create_group('results/comparison')

            if f'results/comparison/{tag}' in h5_file:
                del h5_file[f'results/comparison/{tag}']

            dset = h5_file.create_dataset(f'results/comparison/{tag}/goodness_of_fit',
                                          data=goodness_of_fit)

            dset.attrs['object_name'] = str(object_name)
            dset.attrs['model'] = str(model)
            dset.attrs['n_param'] = len(model_param)
            dset.attrs['n_spec_name'] = len(spec_name)
            dset.attrs['distance'] = distance

            for i, item in enumerate(model_param):
                dset.attrs[f'parameter{i}'] = item

            for i, item in enumerate(spec_name):
                dset.attrs[f'spec_name{i}'] = item

            h5_file.create_dataset(f'results/comparison/{tag}/flux_scaling', data=flux_scaling)

            for i, item in enumerate(coord_points):
                h5_file.create_dataset(f'results/comparison/{tag}/coord_points{i}', data=item)

            # Sum the goodness-of-fit of the different spectra
            goodness_sum = np.sum(goodness_of_fit, axis=-1)

            # Indices of the best-fit model
            best_index = np.unravel_index(goodness_sum.argmin(), goodness_sum.shape)
            dset.attrs['best_fit'] = goodness_sum[best_index]

            print('Best-fit parameters:')
            print(f'   - Goodness-of-fit = {goodness_sum[best_index]:.2e}')

            for i, item in enumerate(model_param):
                best_param = coord_points[i][best_index[i]]
                dset.attrs[f'best_param{i}'] = best_param
                print(f'   - {item} = {best_param}')

            for i, item in enumerate(spec_name):
                scaling = flux_scaling[best_index[0], best_index[1], best_index[2], i]

                radius = np.sqrt(scaling * (distance*constants.PARSEC)**2)  # (m)
                radius /= constants.R_JUP  # (Rjup)

                dset.attrs[f'radius_{item}'] = radius
                print(f'   - {item} radius (Rjup) = {radius:.2f}')

                dset.attrs[f'scaling_{item}'] = scaling
                print(f'   - {item} scaling = {scaling:.2e}')
