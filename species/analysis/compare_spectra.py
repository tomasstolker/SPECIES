"""
Module with functionalities for comparing a spectrum with a library of empirical or model spectra.
"""

import configparser
import os
import warnings

from typing import List, Optional, Tuple, Union

import h5py
import numpy as np

import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from typeguard import typechecked

from species.core import constants
from species.data import database
from species.read import read_model, read_object
from species.util import dust_util, read_util


class CompareSpectra:
    """
    Class for comparing a spectrum of an object with a library of empirical or model spectra.
    """

    @typechecked
    def __init__(self,
                 object_name: str,
                 spec_name: Union[str, List[str]],
                 spec_library: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        object_name : str
            Object name as stored in the database with
            :func:`~species.data.database.Database.add_object` or
            :func:`~species.data.database.Database.add_companion`.
        spec_name : str, list(str)
            Name of the spectrum or list with spectrum names that are stored at the object data
            of ``object_name``. The argument can be either a string or a list of strings.
        spec_library : str, None
            DEPRECATED: Name of the spectral library ('irtf', 'spex', or 'kesseli+2017).

        Returns
        -------
        NoneType
            None
        """

        self.object_name = object_name
        self.spec_name = spec_name

        if isinstance(self.spec_name, str):
            self.spec_name = [self.spec_name]

        if spec_library is not None:
            warnings.warn('The \'spec_library\' parameter is no longer used by the constructor '
                          'of CompareSpectra and will be removed in a future release.',
                          DeprecationWarning)

        self.object = read_object.ReadObject(object_name)

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

    @typechecked
    def spectral_type(self,
                      tag: str,
                      spec_library,
                      wavel_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
                      sptypes: Optional[List[str]] = None,
                      av_ext: Optional[Union[List[float], np.array]] = None,
                      rad_vel: Optional[Union[List[float], np.array]] = None) -> None:
        """
        Method for finding the best fitting empirical spectra from a selected library by
        evaluating the goodness-of-fit statistic from Cushing et al. (2008).

        Parameters
        ----------
        tag : str
            Database tag where for each spectrum from the spectral library the best-fit parameters
            will be stored. So when testing a range of values for ``av_ext`` and ``rad_vel``, only
            the parameters that minimize the goodness-of-fit statistic will be stored.
        spec_library : str
            Name of the spectral library ('irtf', 'spex', 'kesseli+2017', 'bonnefoy+2014').
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
            h5_file[f'spectra/{spec_library}']

        except KeyError:
            h5_file.close()
            species_db = database.Database()
            species_db.add_spectra(spec_library)
            h5_file = h5py.File(self.database, 'r')

        # Read object spectra and resolution

        obj_spec = []
        obj_res = []

        for item in self.spec_name:
            obj_spec.append(self.object.get_spectrum()[item][0])
            obj_res.append(self.object.get_spectrum()[item][3])

        # Read inverted covariance matrix
        # obj_inv_cov = self.object.get_spectrum()[self.spec_name][2]

        # Create empty lists for results

        name_list = []
        spt_list = []
        gk_list = []
        ck_list = []
        av_list = []
        rv_list = []

        print_message = ''

        # Start looping over library spectra

        for i, item in enumerate(h5_file[f'spectra/{spec_library}']):
            # Read spectrum spectral type from library
            dset = h5_file[f'spectra/{spec_library}/{item}']

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

                # Loop over all values of A_V and RV that will be tested

                for av_item in av_ext:
                    for rv_item in rad_vel:
                        for j, spec_item in enumerate(obj_spec):
                            # Dust extinction
                            ism_ext = dust_util.ism_extinction(av_item, 3.1, spectrum[:, 0])
                            flux_scaling = 10.**(-0.4*ism_ext)

                            # Shift wavelengths by RV
                            wavel_shifted = spectrum[:, 0] + \
                                spectrum[:, 0]*1e3*rv_item/constants.LIGHT

                            # Smooth spectrum
                            flux_smooth = read_util.smooth_spectrum(wavel_shifted,
                                                                    spectrum[:, 1]*flux_scaling,
                                                                    spec_res=obj_res[j],
                                                                    force_smooth=True)

                            # Interpolate library spectrum to object wavelengths
                            interp_spec = interp1d(spectrum[:, 0],
                                                   flux_smooth,
                                                   kind='linear',
                                                   fill_value='extrapolate')

                            indices = np.where((spec_item[:, 0] > np.amin(spectrum[:, 0])) &
                                               (spec_item[:, 0] < np.amax(spectrum[:, 0])))[0]

                            flux_resample = interp_spec(spec_item[indices, 0])

                            c_numer = w_i*spec_item[indices, 1]*flux_resample / \
                                spec_item[indices, 2]**2

                            c_denom = w_i*flux_resample**2/spec_item[indices, 2]**2

                            if j == 0:
                                g_k = 0.
                                c_k_spec = []

                            c_k = np.sum(c_numer) / np.sum(c_denom)
                            c_k_spec.append(c_k)

                            chi_sq = (spec_item[indices, 1] - c_k*flux_resample) / \
                                spec_item[indices, 2]

                            g_k += np.sum(w_i * chi_sq**2)

                            # obj_inv_cov_crop = obj_inv_cov[indices, :]
                            # obj_inv_cov_crop = obj_inv_cov_crop[:, indices]

                            # g_k = np.dot(spec_item[indices, 1]-c_k*flux_resample,
                            #     np.dot(obj_inv_cov_crop,
                            #            spec_item[indices, 1]-c_k*flux_resample))

                        # Append to the lists of results

                        name_list.append(item)
                        spt_list.append(item_sptype)
                        gk_list.append(g_k)
                        ck_list.append(c_k_spec)
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
            for i, gk_item in enumerate(gk_select):
                print(f'   {i+1:2d}. G = {gk_item:.2e} -> {name_select[i]}, {spt_select[i]}, '
                      f'A_V = {av_select[i]:.2f}, RV = {rv_select[i]:.0f} km/s,\n'
                      f'                      scalings = {ck_select[i]}')

        else:
            for i in range(10):
                print(f'   {i+1:2d}. G = {gk_select[i]:.2e} -> {name_select[i]}, {spt_select[i]}, '
                      f'A_V = {av_select[i]:.2f}, RV = {rv_select[i]:.0f} km/s,\n'
                      f'                      scalings = {ck_select[i]}')

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
                                 spec_library=spec_library)

    @typechecked
    def compare_model(self,
                      tag: str,
                      model: str,
                      av_points: Optional[Union[List[float], np.array]] = None,
                      fix_logg: Optional[float] = None,
                      fix_spec: Optional[List[str]] = None) -> None:
        """
        Method for finding the best fitting spectrum from a grid of atmospheric model spectra by
        evaluating the goodness-of-fit statistic from Cushing et al. (2008). Currently, this method
        only supports model grids with only :math:`T_\\mathrm{eff}` and :math:`\\log(g)` as free
        parameters (e.g. BT-Settl). Please create an issue on Github if support for models with
        more than two parameters is required.

        Parameters
        ----------
        tag : str
            Database tag where for each spectrum from the spectral library the best-fit parameters
            will be stored. So when testing a range of values for ``av_ext`` and ``rad_vel``, only
            the parameters that minimize the goodness-of-fit statistic will be stored.
        model : str
            Name of the atmospheric model grid with synthetic spectra.
        av_points : list(float), np.array, None
            List of :math:`A_V` extinction values for which the goodness-of-fit statistic will be
            tested. The extinction is calculated with the relation from Cardelli et al. (1989).
        fix_logg : float, None
            Fix the value of :math:`\log(g)`, for example if estimated from gravity-sensitive
            spectral features. Typically, :math:`\log(g)` can not be accurately determined when
            comparing the spectra over a broad wavelength range.
        fix_spec : list(str), None
            List with names of spectra for which the relative scaling will be fixed.

        Returns
        -------
        NoneType
            None
        """

        w_i = {}

        for spec_item in self.spec_name:
            obj_wavel = self.object.get_spectrum()[spec_item][0][:, 0]

            diff = (np.diff(obj_wavel)[1:] + np.diff(obj_wavel)[:-1]) / 2.
            diff = np.insert(diff, 0, diff[0])
            diff = np.append(diff, diff[-1])

            # w_i[spec_item] = diff
            w_i[spec_item] = np.ones(obj_wavel.shape[0])

        if av_points is None:
            av_points = np.array([0.])

        elif isinstance(av_points, list):
            av_points = np.array(av_points)

        readmodel = read_model.ReadModel(model)

        model_param = readmodel.get_parameters()
        grid_points = readmodel.get_points()

        coord_points = []
        for key, value in grid_points.items():
            if key == 'logg' and fix_logg is not None:
                if fix_logg in value:
                    coord_points.append(np.array([fix_logg]))

                else:
                    raise ValueError(f'The argument of \'fix_logg\' ({fix_logg}) is not found '
                                     f'in the parameter grid of the model spectra. The following '
                                     f'values of log(g) are available: {value}')

            else:
                coord_points.append(value)

        if av_points is not None:
            model_param.append('ism_ext')
            coord_points.append(av_points)

        grid_shape = []
        for item in coord_points:
            grid_shape.append(len(item))

        grid_shape.append(len(self.spec_name))

        fit_stat = np.zeros(grid_shape)
        flux_scaling = np.zeros(grid_shape)

        count = 1

        if len(coord_points) == 2:
            n_iter = len(coord_points[0])*len(coord_points[1])

            for i, item_i in enumerate(coord_points[0]):
                for j, item_j in enumerate(coord_points[1]):
                    model_spec = {}

                    for k, spec_item in enumerate(self.spec_name):
                        print(f'Processing model spectrum {count}/{n_iter}...', end='')

                        obj_spec = self.object.get_spectrum()[spec_item][0]
                        obj_res = self.object.get_spectrum()[spec_item][3]

                        param_dict = {model_param[0]: item_i,
                                      model_param[1]: item_j}

                        wavel_range = (0.9*obj_spec[0, 0], 1.1*obj_spec[-1, 0])
                        readmodel = read_model.ReadModel(model, wavel_range=wavel_range)

                        model_box = readmodel.get_data(param_dict,
                                                       spec_res=obj_res,
                                                       wavel_resample=obj_spec[:, 0])

                        model_spec[spec_item] = model_box.flux

                    for k, spec_item in enumerate(self.spec_name):
                        obj_spec = self.object.get_spectrum()[spec_item][0]

                        c_numer = w_i[spec_item] * obj_spec[:, 1] * model_spec[spec_item] / obj_spec[:, 2]**2
                        c_denom = w_i[spec_item] * model_spec[spec_item]**2 / obj_spec[:, 2]**2

                        flux_scaling[i, j, k] = np.sum(c_numer) / np.sum(c_denom)

                        residual = obj_spec[:, 1] - flux_scaling[i, j, k]*model_spec[spec_item]
                        fit_stat[i, j, k] = np.sum(w_i[spec_item] * (residual/obj_spec[:, 2])**2)

                    count += 1

        if len(coord_points) == 3:
            n_iter = len(coord_points[0])*len(coord_points[1])*len(coord_points[2])

            for i, item_i in enumerate(coord_points[0]):
                for j, item_j in enumerate(coord_points[1]):
                    for k, item_k in enumerate(coord_points[2]):
                        model_spec = {}

                        for m, spec_item in enumerate(self.spec_name):
                            print(f'\rProcessing model spectrum {count}/{n_iter}...', end='')

                            obj_spec = self.object.get_spectrum()[spec_item][0]
                            obj_res = self.object.get_spectrum()[spec_item][3]

                            param_dict = {model_param[0]: item_i,
                                          model_param[1]: item_j,
                                          model_param[2]: item_k}

                            wavel_range = (0.9*obj_spec[0, 0], 1.1*obj_spec[-1, 0])
                            readmodel = read_model.ReadModel(model, wavel_range=wavel_range)

                            model_box = readmodel.get_data(param_dict,
                                                           spec_res=obj_res,
                                                           wavel_resample=obj_spec[:, 0])

                            model_spec[spec_item] = model_box.flux

                        if fix_spec is not None:
                            for spec_item in fix_spec:
                                if spec_item == fix_spec[0]:
                                    obj_add = self.object.get_spectrum()[spec_item][0]

                                    # if spec_item == 'MUSE':
                                    #     obj_add[:, 2] *= 3.

                                    model_add = model_spec[spec_item]
                                    w_i_add = w_i[spec_item]

                                else:
                                    spec_tmp = self.object.get_spectrum()[spec_item][0]
                                    obj_add = np.vstack((obj_add, spec_tmp))
                                    model_add = np.append(model_add, model_spec[spec_item])
                                    w_i_add = np.append(w_i_add, w_i[spec_item])

                            c_numer = w_i_add * obj_add[:, 1] * model_add / obj_add[:, 2]**2
                            c_denom = w_i_add * model_add**2 / obj_add[:, 2]**2

                            scaling_fix = np.sum(c_numer) / np.sum(c_denom)

                            residual = obj_add[:, 1] - scaling_fix*model_add
                            stat_fix = np.sum(w_i_add * (residual/obj_add[:, 2])**2)

                            # def func(x, scaling):
                            #     return scaling*model_add
                            #
                            # popt, pcov = curve_fit(func, xdata=obj_add[:, 0], ydata=obj_add[:, 1])
                            # scaling_fix = popt[0]

                            # if item_i == 2600. and item_k == 1.:
                            #     plt.plot(obj_add[:, 0], obj_add[:, 1])
                            #     plt.plot(obj_add[:, 0], scaling_fix*model_add)
                            #     plt.show()
                            #     exit()

                        for m, spec_item in enumerate(self.spec_name):
                            if fix_spec is not None and spec_item == fix_spec[0]:
                                flux_scaling[i, j, k, m] = scaling_fix
                                fit_stat[i, j, j, m] = stat_fix

                            elif fix_spec is not None and spec_item in fix_spec:
                                flux_scaling[i, j, k, m] = scaling_fix
                                fit_stat[i, j, j, m] = 0.

                            else:
                                obj_spec = self.object.get_spectrum()[spec_item][0]

                                c_numer = w_i[spec_item] * obj_spec[:, 1] * model_spec[spec_item] / obj_spec[:, 2]**2
                                c_denom = w_i[spec_item] * model_spec[spec_item]**2 / obj_spec[:, 2]**2

                                flux_scaling[i, j, k, m] = np.sum(c_numer) / np.sum(c_denom)

                                residual = obj_spec[:, 1] - flux_scaling[i, j, k, m]*model_spec[spec_item]
                                fit_stat[i, j, k, m] = np.sum(w_i[spec_item] * (residual/obj_spec[:, 2])**2)

                        count += 1

        print(' [DONE]')

        species_db = database.Database()

        species_db.add_comparison(tag=tag,
                                  goodness_of_fit=fit_stat,
                                  flux_scaling=flux_scaling,
                                  model_param=model_param,
                                  coord_points=coord_points,
                                  object_name=self.object_name,
                                  spec_name=self.spec_name,
                                  model=model)
