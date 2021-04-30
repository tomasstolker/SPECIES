"""
Module with functionalities for empirical spectral analysis.
"""

import configparser
import os

from typing import List, Optional, Tuple, Union

import h5py
import numpy as np

from scipy.interpolate import interp1d
from typeguard import typechecked

from species.core import constants
from species.data import database
from species.read import read_object
from species.util import dust_util, read_util


class CompareSpectra:
    """
    Class for comparing a spectrum of an object with the spectra of a library.
    """

    @typechecked
    def __init__(self,
                 object_name: str,
                 spec_name: str,
                 spec_library: str) -> None:
        """
        Parameters
        ----------
        object_name : str
            Object name as stored in the database with
            :func:`~species.data.database.Database.add_object` or
            :func:`~species.data.database.Database.add_companion`.
        spec_name : str
            Name of the spectrum that is stored at the object data of ``object_name``.
        spec_library : str
            Name of the spectral library ('irtf', 'spex', or 'kesseli+2017).

        Returns
        -------
        NoneType
            None
        """

        self.object_name = object_name
        self.spec_name = spec_name
        self.spec_library = spec_library

        self.object = read_object.ReadObject(object_name)

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

    @typechecked
    def spectral_type(self,
                      tag: str,
                      wavel_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
                      sptypes: Optional[List[str]] = None,
                      av_ext: Optional[Union[List[float], np.array]] = None,
                      rad_vel: Optional[Union[List[float], np.array]] = None) -> None:
        """
        Method for finding the best fitting empirical spectra from a selected library by
        calculating for each spectrum the goodness-of-fit statistic from Cushing et al. (2008).

        Parameters
        ----------
        tag : str
            Database tag where for each spectrum from the spectral library the best-fit parameters
            will be stored. So when testing a range of values for ``av_ext`` and ``rad_vel``, only
            the parameters that minimize the goodness-of-fit statistic will be stored.
        wavel_range : tuple(float, float), None
            Wavelength range (um) that is used for the empirical comparison.
        sptypes : list(str), None
            List with spectral types to compare with. The list should only contains types, for
            example ``sptypes=['M', 'L']``. All available spectral types in the ``spec_library``
            are compared with if set to ``None``.
        av_ext : list(float), np.array, None
            List of A_V extinctions for which the goodness-of-fit statistic is tested. The
            extinction is calculated with the empirical relation from Cardelli et al. (1989).
        rad_vel : list(float), np.array, None
            List of radial velocities (km s-1) for which the goodness-of-fit statistic is tested.

        Returns
        -------
        NoneType
            None
        """

        w_i = 1.

        if av_ext is None:
            av_ext = [0.]

        if rad_vel is None:
            rad_vel = [0.]

        h5_file = h5py.File(self.database, 'r')

        try:
            h5_file[f'spectra/{self.spec_library}']

        except KeyError:
            h5_file.close()
            species_db = database.Database()
            species_db.add_spectrum(self.spec_library)
            h5_file = h5py.File(self.database, 'r')

        # Read object spectrum
        obj_spec = self.object.get_spectrum()[self.spec_name][0]

        # Read inverted covariance matrix
        obj_inv_cov = self.object.get_spectrum()[self.spec_name][2]

        # Read spectral resolution
        obj_res = self.object.get_spectrum()[self.spec_name][3]

        name_list = []
        spt_list = []
        gk_list = []
        ck_list = []
        av_list = []
        rv_list = []

        print_message = ''

        for i, item in enumerate(h5_file[f'spectra/{self.spec_library}']):
            # Read spectrum spectral type from library
            dset = h5_file[f'spectra/{self.spec_library}/{item}']

            if isinstance(dset.attrs['sptype'], str):
                item_sptype = dset.attrs['sptype']
            else:
                # Use decode for backward compatibility
                item_sptype = dset.attrs['sptype'].decode('utf-8')

            if item_sptype == 'None':
                continue

            if sptypes is None or item_sptype[0] in sptypes:
                # Convert HDF5 dataset into numpy array
                spectrum = np.asarray(dset)

                if wavel_range is not None:
                    # Select subset of the spectrum

                    if wavel_range[0] is None:
                        indices = np.where((spectrum[:, 0] < wavel_range[1]))[0]

                    elif wavel_range[1] is None:
                        indices = np.where((spectrum[:, 0] > wavel_range[0]))[0]

                    else:
                        indices = np.where((spectrum[:, 0] > wavel_range[0]) &
                                           (spectrum[:, 0] < wavel_range[1]))[0]

                    if len(indices) == 0:
                        raise ValueError('The selected wavelength range does not cover any '
                                         'wavelength points of the input spectrum. Please '
                                         'use a broader range as argument of \'wavel_range\'.')

                    spectrum = spectrum[indices, ]

                empty_message = len(print_message)*' '
                print(f'\r{empty_message}', end='')

                print_message = f'Processing spectra... {item}'
                print(f'\r{print_message}', end='')

                for av_item in av_ext:
                    for rv_item in rad_vel:
                        # Dust extinction
                        ism_ext = dust_util.ism_extinction(av_item, 3.1, spectrum[:, 0])
                        flux_scaling = 10.**(-0.4*ism_ext)

                        # Shift wavelengths by RV
                        wavel_shifted = spectrum[:, 0] + spectrum[:, 0]*1e3*rv_item/constants.LIGHT

                        # Smooth spectrum
                        flux_smooth = read_util.smooth_spectrum(wavel_shifted,
                                                                spectrum[:, 1]*flux_scaling,
                                                                spec_res=obj_res,
                                                                force_smooth=True)

                        # Interpolate library spectrum to object wavelengths
                        interp_spec = interp1d(spectrum[:, 0],
                                               flux_smooth,
                                               kind='linear',
                                               fill_value='extrapolate')

                        indices = np.where((obj_spec[:, 0] > np.amin(spectrum[:, 0])) &
                                           (obj_spec[:, 0] < np.amax(spectrum[:, 0])))[0]

                        flux_resample = interp_spec(obj_spec[indices, 0])

                        c_numer = w_i*obj_spec[indices, 1]*flux_resample/obj_spec[indices, 2]**2
                        c_denom = w_i*flux_resample**2/obj_spec[indices, 2]**2

                        c_k = np.sum(c_numer) / np.sum(c_denom)

                        chi_sq = (obj_spec[indices, 1] - c_k*flux_resample) / obj_spec[indices, 2]
                        g_k = np.sum(w_i * chi_sq**2)

                        # obj_inv_cov_crop = obj_inv_cov[indices, :]
                        # obj_inv_cov_crop = obj_inv_cov_crop[:, indices]
                        #
                        # g_k = np.dot(obj_spec[indices, 1]-c_k*flux_resample,
                        #     np.dot(obj_inv_cov_crop, obj_spec[indices, 1]-c_k*flux_resample))

                        name_list.append(item)
                        spt_list.append(item_sptype)
                        gk_list.append(g_k)
                        ck_list.append(c_k)
                        av_list.append(av_item)
                        rv_list.append(rv_item)

        empty_message = len(print_message)*' '
        print(f'\r{empty_message}', end='')

        print('\rProcessing spectra... [DONE]')

        h5_file.close()

        name_list = np.asarray(name_list)
        spt_list = np.asarray(spt_list)
        gk_list = np.asarray(gk_list)
        ck_list = np.asarray(ck_list)
        av_list = np.asarray(av_list)
        rv_list = np.asarray(rv_list)

        sort_index = np.argsort(gk_list)

        name_list = name_list[sort_index]
        spt_list = spt_list[sort_index]
        gk_list = gk_list[sort_index]
        ck_list = ck_list[sort_index]
        av_list = av_list[sort_index]
        rv_list = rv_list[sort_index]

        name_select = []
        spt_select = []
        gk_select = []
        ck_select = []
        av_select = []
        rv_select = []

        for i, item in enumerate(name_list):
            if item not in name_select:
                name_select.append(item)
                spt_select.append(spt_list[i])
                gk_select.append(gk_list[i])
                ck_select.append(ck_list[i])
                av_select.append(av_list[i])
                rv_select.append(rv_list[i])

        print('Best-fitting spectra:')

        if len(gk_select) < 10:
            for i in range(len(gk_select)):
                print(f'   {i+1:2d}. G = {gk_select[i]:.2e} -> {name_select[i]}, {spt_select[i]}, '
                      f'A_V = {av_select[i]:.2f}, RV = {rv_select[i]:.0f} km/s')

        else:
            for i in range(10):
                print(f'   {i+1:2d}. G = {gk_select[i]:.2e} -> {name_select[i]}, {spt_select[i]}, '
                      f'A_V = {av_select[i]:.2f}, RV = {rv_select[i]:.0f} km/s')

        species_db = database.Database()

        species_db.add_empirical(tag=tag,
                                 names=name_select,
                                 sptypes=spt_select,
                                 goodness_of_fit=gk_select,
                                 flux_scaling=ck_select,
                                 av_ext=av_select,
                                 rad_vel=rv_select,
                                 object_name=self.object_name,
                                 spec_name=self.spec_name,
                                 spec_library=self.spec_library)
