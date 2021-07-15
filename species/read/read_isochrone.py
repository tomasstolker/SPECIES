"""
Module with reading functionalities for isochrones.
"""

import configparser
import os
import warnings

from typing import Optional, Tuple

import h5py
import numpy as np

from scipy.interpolate import griddata
from typeguard import typechecked

from species.core import box
from species.read import read_model
from species.util import read_util


class ReadIsochrone:
    """
    Class for reading isochrone data from the database.
    """

    @typechecked
    def __init__(self,
                 tag: str) -> None:
        """
        Parameters
        ----------
        tag : str
            Database tag of the isochrone data.

        Returns
        -------
        NoneType
            None
        """

        self.tag = tag

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

    @typechecked
    def get_isochrone(self,
                      age: float,
                      masses: np.ndarray,
                      filters_color: Optional[Tuple[str, str]] = None,
                      filter_mag: Optional[str] = None) -> box.IsochroneBox:
        """
        Function for selecting an isochrone.

        Parameters
        ----------
        age : float
            Age (Myr) at which the isochrone data is interpolated.
        masses : np.ndarray
            Masses (Mjup) at which the isochrone data is interpolated.
        filters_color : tuple(str, str), None
            Filter names for the color as listed in the file with the isochrone data. Not selected
            if set to ``None`` or if only evolutionary tracks are available.
        filter_mag : str, None
            Filter name for the absolute magnitude as listed in the file with the isochrone data.
            Not selected if set to ``None`` or if only evolutionary tracks are available.

        Returns
        -------
        species.core.box.IsochroneBox
            Box with the isochrone.
        """

        age_points = np.full(masses.shape[0], age)  # (Myr)

        color = None
        mag_abs = None

        index_teff = 2
        index_logg = 4

        # Read isochrone data

        with h5py.File(self.database, 'r') as h5_file:
            model = h5_file[f'isochrones/{self.tag}/evolution'].attrs['model']
            evolution = np.asarray(h5_file[f'isochrones/{self.tag}/evolution'])

            if model == 'baraffe':
                filters = list(h5_file[f'isochrones/{self.tag}/filters'])
                magnitudes = np.asarray(h5_file[f'isochrones/{self.tag}/magnitudes'])

                # Convert the h5py list of filters from bytes to strings
                for i, item in enumerate(filters):
                    if isinstance(item, bytes):
                        filters[i] = item.decode('utf-8')

        if model == 'baraffe':
            if filters_color is not None:
                index_color_1 = filters.index(filters_color[0])
                index_color_2 = filters.index(filters_color[1])

            if filter_mag is not None:
                index_mag = filters.index(filter_mag)

            if filters_color is not None:
                mag_color_1 = griddata(points=evolution[:, 0:2],
                                       values=magnitudes[:, index_color_1],
                                       xi=np.stack((age_points, masses), axis=1),
                                       method='linear',
                                       fill_value='nan',
                                       rescale=False)

                mag_color_2 = griddata(points=evolution[:, 0:2],
                                       values=magnitudes[:, index_color_2],
                                       xi=np.stack((age_points, masses), axis=1),
                                       method='linear',
                                       fill_value='nan',
                                       rescale=False)

                color = mag_color_1-mag_color_2

            if filter_mag is not None:
                mag_abs = griddata(points=evolution[:, 0:2],
                                   values=magnitudes[:, index_mag],
                                   xi=np.stack((age_points, masses), axis=1),
                                   method='linear',
                                   fill_value='nan',
                                   rescale=False)

        teff = griddata(points=evolution[:, 0:2],
                        values=evolution[:, index_teff],
                        xi=np.stack((age_points, masses), axis=1),
                        method='linear',
                        fill_value='nan',
                        rescale=False)

        logg = griddata(points=evolution[:, 0:2],
                        values=evolution[:, index_logg],
                        xi=np.stack((age_points, masses), axis=1),
                        method='linear',
                        fill_value='nan',
                        rescale=False)

        return box.create_box(boxtype='isochrone',
                              model=self.tag,
                              filters_color=filters_color,
                              filter_mag=filter_mag,
                              color=color,
                              magnitude=mag_abs,
                              teff=teff,
                              logg=logg,
                              masses=masses)

    @typechecked
    def get_color_magnitude(self,
                            age: float,
                            masses: np.ndarray,
                            model: str,
                            filters_color: Tuple[str, str],
                            filter_mag: str,
                            adapt_logg: bool = False) -> box.ColorMagBox:
        """
        Function for calculating color-magnitude combinations from a selected isochrone.

        Parameters
        ----------
        age : float
            Age (Myr) at which the isochrone data is interpolated.
        masses : np.ndarray
            Masses (Mjup) at which the isochrone data is interpolated.
        model : str
            Atmospheric model used to compute the synthetic photometry.
        filters_color : tuple(str, str)
            Filter names for the color as listed in the file with the isochrone data. The filter
            names should be provided in the format of the SVO Filter Profile Service.
        filter_mag : str
            Filter name for the absolute magnitude as listed in the file with the isochrone data.
            The value should be equal to one of the ``filters_color`` values.
        adapt_logg : bool
            Adapt :math:`\\log(g)` to the upper or lower boundary of the atmospheric model grid
            whenever the :math:`\\log(g)` that has been calculated from the isochrone mass and
            radius lies outside the available range of the synthetic spectra. Typically
            :math:`\\log(g)` has only a minor impact on the broadband magnitudes and colors.

        Returns
        -------
        species.core.box.ColorMagBox
            Box with the color-magnitude data.
        """

        isochrone = self.get_isochrone(age=age,
                                       masses=masses,
                                       filters_color=None,
                                       filter_mag=None)

        model1 = read_model.ReadModel(model=model, filter_name=filters_color[0])
        model2 = read_model.ReadModel(model=model, filter_name=filters_color[1])

        param_bounds = model1.get_bounds()

        if model1.get_parameters() != ['teff', 'logg']:
            raise ValueError('Creating synthetic colors and magnitudes from isochrones is '
                             'currently only implemented for models with only Teff and log(g) '
                             'as free parameters. Please create an issue on the Github page if '
                             'additional functionalities are required.')

        mag1 = np.zeros(isochrone.masses.shape[0])
        mag2 = np.zeros(isochrone.masses.shape[0])
        radius = np.zeros(isochrone.masses.shape[0])

        for i, mass_item in enumerate(isochrone.masses):
            model_param = {'teff': isochrone.teff[i],
                           'logg': isochrone.logg[i],
                           'mass': mass_item,
                           'distance': 10.}

            radius[i] = read_util.get_radius(model_param['logg'], model_param['mass'])  # (Rjup)

            if np.isnan(isochrone.teff[i]):
                mag1[i] = np.nan
                mag2[i] = np.nan

                warnings.warn(f'The value of Teff is NaN for the following isochrone sample: '
                              f'{model_param}. Setting the magnitudes to NaN.')

            else:
                for item_bounds in param_bounds:
                    if model_param[item_bounds] <= param_bounds[item_bounds][0]:
                        if adapt_logg and item_bounds == 'logg':
                            warnings.warn(f'The log(g) is {model_param[item_bounds]} but the '
                                          f'lower boundary of the model grid is '
                                          f'{param_bounds[item_bounds][0]}. Adapting '
                                          f'log(g) to {param_bounds[item_bounds][0]} since '
                                          f'adapt_logg=True.')

                            model_param['logg'] = param_bounds['logg'][0]

                        else:
                            mag1[i] = np.nan
                            mag2[i] = np.nan

                            warnings.warn(f'The value of {item_bounds} is '
                                          f'{model_param[item_bounds]}, which is below the lower '
                                          f'bound of the model grid '
                                          f'({param_bounds[item_bounds][0]}). Setting the '
                                          f'magnitudes to NaN for the following isochrone sample: '
                                          f'{model_param}.')

                    elif model_param[item_bounds] >= param_bounds[item_bounds][1]:
                        if adapt_logg and item_bounds == 'logg':
                            warnings.warn(f'The log(g) is {model_param[item_bounds]} but the '
                                          f'upper boundary of the model grid is '
                                          f'{param_bounds[item_bounds][1]}. Adapting '
                                          f'log(g) to {param_bounds[item_bounds][1]} since '
                                          f'adapt_logg=True.')

                            model_param['logg'] = param_bounds['logg'][1]

                        else:
                            mag1[i] = np.nan
                            mag2[i] = np.nan

                            warnings.warn(f'The value of {item_bounds} is '
                                          f'{model_param[item_bounds]}, which is above the upper '
                                          f'bound of the model grid '
                                          f'({param_bounds[item_bounds][1]}). Setting the '
                                          f'magnitudes to NaN for the following isochrone sample: '
                                          f'{model_param}.')

                if not np.isnan(mag1[i]):
                    mag1[i], _ = model1.get_magnitude(model_param)
                    mag2[i], _ = model2.get_magnitude(model_param)

        if filter_mag == filters_color[0]:
            abs_mag = mag1

        elif filter_mag == filters_color[1]:
            abs_mag = mag2

        else:
            raise ValueError('The argument of filter_mag should be equal to one of the two filter '
                             'values of filters_color.')

        return box.create_box(boxtype='colormag',
                              library=model,
                              object_type='model',
                              filters_color=filters_color,
                              filter_mag=filter_mag,
                              color=mag1-mag2,
                              magnitude=abs_mag,
                              names=None,
                              sptype=masses,
                              mass=masses,
                              radius=radius)

    @typechecked
    def get_color_color(self,
                        age: float,
                        masses: np.ndarray,
                        model: str,
                        filters_colors: Tuple[Tuple[str, str],
                                              Tuple[str, str]]) -> box.ColorColorBox:
        """
        Function for calculating color-magnitude combinations from a selected isochrone.

        Parameters
        ----------
        age : float
            Age (Myr) at which the isochrone data is interpolated.
        masses : np.ndarray
            Masses (Mjup) at which the isochrone data is interpolated.
        model : str
            Atmospheric model used to compute the synthetic photometry.
        filters_colors : tuple(tuple(str, str), tuple(str, str))
            Filter names for the colors as listed in the file with the isochrone data. The filter
            names should be provided in the format of the SVO Filter Profile Service.

        Returns
        -------
        species.core.box.ColorColorBox
            Box with the color-color data.
        """

        isochrone = self.get_isochrone(age=age,
                                       masses=masses,
                                       filters_color=None,
                                       filter_mag=None)

        model1 = read_model.ReadModel(model=model, filter_name=filters_colors[0][0])
        model2 = read_model.ReadModel(model=model, filter_name=filters_colors[0][1])
        model3 = read_model.ReadModel(model=model, filter_name=filters_colors[1][0])
        model4 = read_model.ReadModel(model=model, filter_name=filters_colors[1][1])

        if model1.get_parameters() != ['teff', 'logg']:
            raise ValueError('Creating synthetic colors and magnitudes from isochrones is '
                             'currently only implemented for models with only Teff and log(g) '
                             'as free parameters. Please contact Tomas Stolker if additional '
                             'functionalities are required.')

        mag1 = np.zeros(isochrone.masses.shape[0])
        mag2 = np.zeros(isochrone.masses.shape[0])
        mag3 = np.zeros(isochrone.masses.shape[0])
        mag4 = np.zeros(isochrone.masses.shape[0])
        radius = np.zeros(isochrone.masses.shape[0])

        for i, mass_item in enumerate(isochrone.masses):
            model_param = {'teff': isochrone.teff[i],
                           'logg': isochrone.logg[i],
                           'mass': mass_item,
                           'distance': 10.}

            radius[i] = read_util.get_radius(model_param['logg'], model_param['mass'])  # (Rjup)

            if np.isnan(isochrone.teff[i]):
                mag1[i] = np.nan
                mag2[i] = np.nan
                mag3[i] = np.nan
                mag4[i] = np.nan

                warnings.warn(f'The value of Teff is NaN for the following isochrone sample: '
                              f'{model_param}. Setting the magnitudes to NaN.')

            else:
                for item_bounds in model1.get_bounds():
                    if model_param[item_bounds] <= model1.get_bounds()[item_bounds][0]:
                        mag1[i] = np.nan
                        mag2[i] = np.nan
                        mag3[i] = np.nan
                        mag4[i] = np.nan

                        warnings.warn(f'The value of {item_bounds} is {model_param[item_bounds]}, '
                                      f'which is below the lower bound of the model grid '
                                      f'({model1.get_bounds()[item_bounds][0]}). Setting the '
                                      f'magnitudes to NaN for the following isochrone sample: '
                                      f'{model_param}.')

                    elif model_param[item_bounds] >= model1.get_bounds()[item_bounds][1]:
                        mag1[i] = np.nan
                        mag2[i] = np.nan
                        mag3[i] = np.nan
                        mag4[i] = np.nan

                        warnings.warn(f'The value of {item_bounds} is {model_param[item_bounds]}, '
                                      f'which is above the upper bound of the model grid '
                                      f'({model1.get_bounds()[item_bounds][1]}). Setting the '
                                      f'magnitudes to NaN for the following isochrone sample: '
                                      f'{model_param}.')

                if not np.isnan(mag1[i]) and not np.isnan(mag2[i]) and\
                        not np.isnan(mag3[i]) and not np.isnan(mag4[i]):
                    mag1[i], _ = model1.get_magnitude(model_param)
                    mag2[i], _ = model2.get_magnitude(model_param)
                    mag3[i], _ = model3.get_magnitude(model_param)
                    mag4[i], _ = model4.get_magnitude(model_param)

        return box.create_box(boxtype='colorcolor',
                              library=model,
                              object_type='model',
                              filters=filters_colors,
                              color1=mag1-mag2,
                              color2=mag3-mag4,
                              names=None,
                              sptype=masses,
                              mass=masses,
                              radius=radius)
