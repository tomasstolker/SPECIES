"""
Module with a function for plotting spectra.
"""

import os
import math
import warnings
import itertools

from typing import Optional, Union, Tuple, List

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from typeguard import typechecked
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, ScalarFormatter

from species.core import box, constants
from species.read import read_filter
from species.util import plot_util


@typechecked
def plot_spectrum(boxes: list,
                  filters: Optional[List[str]] = None,
                  residuals: Optional[box.ResidualsBox] = None,
                  plot_kwargs: Optional[List[Optional[dict]]] = None,
                  xlim: Optional[Tuple[float, float]] = None,
                  ylim: Optional[Tuple[float, float]] = None,
                  ylim_res: Optional[Tuple[float, float]] = None,
                  scale: Optional[Tuple[str, str]] = None,
                  title: Optional[str] = None,
                  offset: Optional[Tuple[float, float]] = None,
                  legend: Optional[Union[str, dict, Tuple[float, float],
                                   List[Optional[Union[dict, str, Tuple[float, float]]]]]] = None,
                  figsize: Optional[Tuple[float, float]] = (10., 5.),
                  object_type: str = 'planet',
                  quantity: str = 'flux density',
                  output: str = 'spectrum.pdf'):
    """
    Parameters
    ----------
    boxes : list(species.core.box, )
        Boxes with data.
    filters : list(str, ), None
        Filter IDs for which the transmission profile is plotted. Not plotted if set to None.
    residuals : species.core.box.ResidualsBox, None
        Box with residuals of a fit. Not plotted if set to None.
    plot_kwargs : list(dict, ), None
        List with dictionaries of keyword arguments for each box. For example, if the ``boxes``
        are a ``ModelBox`` and ``ObjectBox``:

        .. code-block:: python

            plot_kwargs=[{'ls': '-', 'lw': 1., 'color': 'black'},
                         {'spectrum_1': {'marker': 'o', 'ms': 3., 'color': 'tab:brown', 'ls': 'none'},
                          'spectrum_2': {'marker': 'o', 'ms': 3., 'color': 'tab:blue', 'ls': 'none'},
                          'Paranal/SPHERE.IRDIS_D_H23_3': {'marker': 's', 'ms': 4., 'color': 'tab:cyan', 'ls': 'none'},
                          'Paranal/SPHERE.IRDIS_D_K12_1': [{'marker': 's', 'ms': 4., 'color': 'tab:orange', 'ls': 'none'},
                                                           {'marker': 's', 'ms': 4., 'color': 'tab:red', 'ls': 'none'}],
                          'Paranal/NACO.Lp': {'marker': 's', 'ms': 4., 'color': 'tab:green', 'ls': 'none'},
                          'Paranal/NACO.Mp': {'marker': 's', 'ms': 4., 'color': 'tab:green', 'ls': 'none'}}]

        For an ``ObjectBox``, the dictionary contains items for the different spectrum and filter
        names stored with :func:`~species.data.database.Database.add_object`. In case both
        and ``ObjectBox`` and a ``SynphotBox`` are provided, then the latter can be set to ``None``
        in order to use the same (but open) symbols as the data from the ``ObjectBox``. Note that
        if a filter name is duplicated in an ``ObjectBox`` (Paranal/SPHERE.IRDIS_D_K12_1 in the
        example) then a list with two dictionaries should be provided. Colors are automatically
        chosen if ``plot_kwargs`` is set to ``None``.
    xlim : tuple(float, float)
        Limits of the wavelength axis.
    ylim : tuple(float, float)
        Limits of the flux axis.
    ylim_res : tuple(float, float), None
        Limits of the residuals axis. Automatically chosen (based on the minimum and maximum
        residual value) if set to None.
    scale : tuple(str, str), None
        Scale of the x and y axes ('linear' or 'log'). The scale is set to ``('linear', 'linear')``
        if set to ``None``.
    title : str
        Title.
    offset : tuple(float, float)
        Offset for the label of the x- and y-axis.
    legend : str, tuple, dict, list(dict, dict), None
        Location of the legend (str or tuple(float, float)) or a dictionary with the ``**kwargs``
        of ``matplotlib.pyplot.legend``, for example ``{'loc': 'upper left', 'fontsize: 12.}``.
        Alternatively, a list with two values can be provided to separate the model and data
        handles in two legends. Each of these two elements can be set to ``None``. For example,
        ``[None, {'loc': 'upper left', 'fontsize: 12.}]``, if only the data points should be
        included in a legend.
    figsize : tuple(float, float)
        Figure size.
    object_type : str
        Object type ('planet' or 'star'). With 'planet', the radius and mass are expressed in
        Jupiter units. With 'star', the radius and mass are expressed in solar units.
    quantity: str
        The quantity of the y-axis ('flux density', 'flux', or 'magnitude').
    output : str
        Output filename.

    Returns
    -------
    NoneType
        None
    """

    mpl.rcParams['font.serif'] = ['Bitstream Vera Serif']
    mpl.rcParams['font.family'] = 'serif'

    plt.rc('axes', edgecolor='black', linewidth=2.2)
    plt.rcParams['axes.axisbelow'] = False

    if plot_kwargs is None:
        plot_kwargs = []

    elif plot_kwargs is not None and len(boxes) != len(plot_kwargs):
        raise ValueError(f'The number of \'boxes\' ({len(boxes)}) should be equal to the '
                         f'number of items in \'plot_kwargs\' ({len(plot_kwargs)}).')

    if residuals is not None and filters is not None:
        plt.figure(1, figsize=figsize)
        gridsp = mpl.gridspec.GridSpec(3, 1, height_ratios=[1, 3, 1])
        gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(gridsp[1, 0])
        ax2 = plt.subplot(gridsp[0, 0])
        ax3 = plt.subplot(gridsp[2, 0])

    elif residuals is not None:
        plt.figure(1, figsize=figsize)
        gridsp = mpl.gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(gridsp[0, 0])
        ax2 = None
        ax3 = plt.subplot(gridsp[1, 0])

    elif filters is not None:
        plt.figure(1, figsize=figsize)
        gridsp = mpl.gridspec.GridSpec(2, 1, height_ratios=[1, 4])
        gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(gridsp[1, 0])
        ax2 = plt.subplot(gridsp[0, 0])
        ax3 = None

    else:
        plt.figure(1, figsize=figsize)
        gridsp = mpl.gridspec.GridSpec(1, 1)
        gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(gridsp[0, 0])
        ax2 = None
        ax3 = None

    if residuals is not None:
        labelbottom = False
    else:
        labelbottom = True

    if scale is None:
        scale = ('linear', 'linear')

    ax1.set_xscale(scale[0])
    ax1.set_yscale(scale[1])

    if filters is not None:
        ax2.set_xscale(scale[0])

    if residuals is not None:
        ax3.set_xscale(scale[0])

    ax1.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                    direction='in', width=1, length=5, labelsize=12, top=True,
                    bottom=True, left=True, right=True, labelbottom=labelbottom)

    ax1.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                    direction='in', width=1, length=3, labelsize=12, top=True,
                    bottom=True, left=True, right=True, labelbottom=labelbottom)

    if filters is not None:
        ax2.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                        direction='in', width=1, length=5, labelsize=12, top=True,
                        bottom=True, left=True, right=True, labelbottom=False)

        ax2.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                        direction='in', width=1, length=3, labelsize=12, top=True,
                        bottom=True, left=True, right=True, labelbottom=False)

    if residuals is not None:
        ax3.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                        direction='in', width=1, length=5, labelsize=12, top=True,
                        bottom=True, left=True, right=True)

        ax3.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                        direction='in', width=1, length=3, labelsize=12, top=True,
                        bottom=True, left=True, right=True)

    if scale[0] == 'linear':
        ax1.xaxis.set_minor_locator(AutoMinorLocator(5))

    if scale[1] == 'linear':
        ax1.yaxis.set_minor_locator(AutoMinorLocator(5))

    # ax1.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    # ax3.set_yticks([-2., 0., 2.])

    if filters is not None:
        if scale[0] == 'linear':
            ax2.xaxis.set_minor_locator(AutoMinorLocator(5))

    if residuals is not None:
        if scale[0] == 'linear':
            ax3.xaxis.set_minor_locator(AutoMinorLocator(5))

    if residuals is not None and filters is not None:
        ax1.set_xlabel('')
        ax2.set_xlabel('')
        ax3.set_xlabel('Wavelength (µm)', fontsize=13)

    elif residuals is not None:
        ax1.set_xlabel('')
        ax3.set_xlabel('Wavelength (µm)', fontsize=11)

    elif filters is not None:
        ax1.set_xlabel('Wavelength (µm)', fontsize=13)
        ax2.set_xlabel('')

    else:
        ax1.set_xlabel('Wavelength (µm)', fontsize=13)

    if filters is not None:
        ax2.set_ylabel(r'T$_\lambda$', fontsize=13)

    if residuals is not None:
        if quantity == 'flux density':
            ax3.set_ylabel(r'$\Delta$$\mathregular{F}_\lambda$ ($\sigma$)', fontsize=11)

        elif quantity == 'flux':
            ax3.set_ylabel(r'$\Delta$$\mathregular{F}_\lambda$ ($\sigma$)', fontsize=11)

    if xlim is None:
        ax1.set_xlim(0.6, 6.)
    else:
        ax1.set_xlim(xlim[0], xlim[1])

    if quantity == 'magnitude':
        scaling = 1.
        ax1.set_ylabel('Flux contrast (mag)', fontsize=13)

        if ylim:
            ax1.set_ylim(ylim[0], ylim[1])

    else:
        if ylim:
            ax1.set_ylim(ylim[0], ylim[1])

            ylim = ax1.get_ylim()

            exponent = math.floor(math.log10(ylim[1]))
            scaling = 10.**exponent

            if quantity == 'flux density':
                ylabel = r'$\mathregular{F}_\lambda$ (10$^{'+str(exponent)+r'}$ W m$^{-2}$ µm$^{-1}$)'

            elif quantity == 'flux':
                ylabel = r'$\lambda$$\mathregular{F}_\lambda$ (10$^{'+str(exponent)+r'}$ W m$^{-2}$)'

            ax1.set_ylabel(ylabel, fontsize=11)
            ax1.set_ylim(ylim[0]/scaling, ylim[1]/scaling)

            if ylim[0] < 0.:
                ax1.axhline(0.0, ls='--', lw=0.7, color='gray', dashes=(2, 4), zorder=0.5)

        else:
            if quantity == 'flux density':
                ax1.set_ylabel(r'$\mathregular{F}_\lambda$ (W m$^{-2}$ µm$^{-1}$)', fontsize=11)

            elif quantity == 'flux':
                ax1.set_ylabel(r'$\lambda$$\mathregular{F}_\lambda$ (W m$^{-2}$)', fontsize=11)

            scaling = 1.

    xlim = ax1.get_xlim()

    if filters is not None:
        ax2.set_xlim(xlim[0], xlim[1])
        ax2.set_ylim(0., 1.)

    if residuals is not None:
        ax3.set_xlim(xlim[0], xlim[1])

    if offset is not None and residuals is not None and filters is not None:
        ax3.get_xaxis().set_label_coords(0.5, offset[0])

        ax1.get_yaxis().set_label_coords(offset[1], 0.5)
        ax2.get_yaxis().set_label_coords(offset[1], 0.5)
        ax3.get_yaxis().set_label_coords(offset[1], 0.5)

    elif offset is not None and filters is not None:
        ax1.get_xaxis().set_label_coords(0.5, offset[0])

        ax1.get_yaxis().set_label_coords(offset[1], 0.5)
        ax2.get_yaxis().set_label_coords(offset[1], 0.5)

    elif offset is not None and residuals is not None:
        ax3.get_xaxis().set_label_coords(0.5, offset[0])

        ax1.get_yaxis().set_label_coords(offset[1], 0.5)
        ax3.get_yaxis().set_label_coords(offset[1], 0.5)

    elif offset is not None:
        ax1.get_xaxis().set_label_coords(0.5, offset[0])
        ax1.get_yaxis().set_label_coords(offset[1], 0.5)

    else:
        ax1.get_xaxis().set_label_coords(0.5, -0.12)
        ax1.get_yaxis().set_label_coords(-0.1, 0.5)

    for j, boxitem in enumerate(boxes):
        flux_scaling = 1.

        if j < len(boxes):
            plot_kwargs.append(None)

        if isinstance(boxitem, (box.SpectrumBox, box.ModelBox)):
            wavelength = boxitem.wavelength
            flux = boxitem.flux

            if isinstance(wavelength[0], (np.float32, np.float64)):
                data = np.array(flux, dtype=np.float64)
                masked = np.ma.array(data, mask=np.isnan(data))

                if isinstance(boxitem, box.ModelBox):
                    param = boxitem.parameters

                    par_key, par_unit, par_label = plot_util.quantity_unit(
                        param=list(param.keys()), object_type=object_type)

                    label = ''
                    newline = False

                    for i, item in enumerate(par_key):
                        if item[:4] == 'teff':
                            value = f'{param[item]:.0f}'

                        elif item in ['logg', 'feh', 'fsed', 'lognorm_ext',
                                      'powerlaw_ext', 'ism_ext']:
                            value = f'{param[item]:.1f}'

                        elif item in ['co']:
                            value = f'{param[item]:.2f}'

                        elif item[:6] == 'radius':
                            if object_type == 'planet':
                                value = f'{param[item]:.1f}'

                                # if item == 'radius_1':
                                #     value = f'{param[item]:.0f}'
                                # else:
                                #     value = f'{param[item]:.1f}'

                            elif object_type == 'star':
                                value = f'{param[item]*constants.R_JUP/constants.R_SUN:.1f}'

                        # elif item == 'mass':
                        #     if object_type == 'planet':
                        #         value = f'{param[item]:.0f}'
                        #
                        #     elif object_type == 'star':
                        #         value = f'{param[item]*constants.M_JUP/constants.M_SUN:.1f}'

                        elif item == 'luminosity':
                            value = f'{np.log10(param[item]):.2f}'

                        else:
                            continue

                        # if len(label) > 80 and newline == False:
                        #     label += '\n'
                        #     newline = True

                        if par_unit[i] is None:
                            label += f'{par_label[i]} = {value}'
                        else:
                            label += f'{par_label[i]} = {value} {par_unit[i]}'

                        if i < len(par_key)-1:
                            label += ', '

                else:
                    label = None

                if plot_kwargs[j]:
                    kwargs_copy = plot_kwargs[j].copy()

                    if 'label' in kwargs_copy:
                        if kwargs_copy['label'] is None:
                            label = None
                        else:
                            label = kwargs_copy['label']

                        del kwargs_copy['label']

                    if quantity == 'flux':
                        flux_scaling = wavelength

                    ax1.plot(wavelength, flux_scaling*masked/scaling, zorder=2, label=label, **kwargs_copy)

                else:
                    if quantity == 'flux':
                        flux_scaling = wavelength

                    ax1.plot(wavelength, flux_scaling*masked/scaling, lw=0.5, label=label, zorder=2)

            elif isinstance(wavelength[0], (np.ndarray)):
                for i, item in enumerate(wavelength):
                    data = np.array(flux[i], dtype=np.float64)
                    masked = np.ma.array(data, mask=np.isnan(data))

                    if isinstance(boxitem.name[i], bytes):
                        label = boxitem.name[i].decode('utf-8')
                    else:
                        label = boxitem.name[i]

                    if quantity == 'flux':
                        flux_scaling = item

                    ax1.plot(item, flux_scaling*masked/scaling, lw=0.5, label=label)

        elif isinstance(boxitem, list):
            for i, item in enumerate(boxitem):
                wavelength = item.wavelength
                flux = item.flux

                data = np.array(flux, dtype=np.float64)
                masked = np.ma.array(data, mask=np.isnan(data))

                if quantity == 'flux':
                    flux_scaling = wavelength

                if plot_kwargs[j]:
                    ax1.plot(wavelength, flux_scaling*masked/scaling, zorder=1, **plot_kwargs[j])
                else:
                    ax1.plot(wavelength, flux_scaling*masked/scaling, color='gray', lw=0.2, alpha=0.5, zorder=1)

        elif isinstance(boxitem, box.PhotometryBox):
            label_check = []

            for i, item in enumerate(boxitem.wavelength):
                transmission = read_filter.ReadFilter(boxitem.filter_name[i])
                fwhm = transmission.filter_fwhm()

                if quantity == 'flux':
                    flux_scaling = item

                if plot_kwargs[j]:
                    if 'label' in plot_kwargs[j] and plot_kwargs[j]['label'] not in label_check:
                        label_check.append(plot_kwargs[j]['label'])

                    elif 'label' in plot_kwargs[j] and plot_kwargs[j]['label'] in label_check:
                        del plot_kwargs[j]['label']

                    if boxitem.flux[i][1] is None:
                        ax1.errorbar(item, flux_scaling*boxitem.flux[i][0]/scaling, xerr=fwhm/2.,
                                     yerr=None, zorder=3, **plot_kwargs[j])

                    else:
                        ax1.errorbar(item, flux_scaling*boxitem.flux[i][0]/scaling, xerr=fwhm/2.,
                                     yerr=flux_scaling*boxitem.flux[i][1]/scaling, zorder=3, **plot_kwargs[j])

                else:
                    if boxitem.flux[i][1] is None:
                        ax1.errorbar(item, flux_scaling*boxitem.flux[i][0]/scaling, xerr=fwhm/2.,
                                     yerr=None, marker='s', ms=6, color='black', zorder=3)

                    else:
                        ax1.errorbar(item, flux_scaling*boxitem.flux[i][0]/scaling, xerr=fwhm/2.,
                                     yerr=flux_scaling*boxitem.flux[i][1]/scaling, marker='s', ms=6, color='black',
                                     zorder=3)

        elif isinstance(boxitem, box.ObjectBox):
            if boxitem.spectrum is not None:
                spec_list = []
                wavel_list = []

                for item in boxitem.spectrum:
                    spec_list.append(item)
                    wavel_list.append(boxitem.spectrum[item][0][0, 0])

                sort_index = np.argsort(wavel_list)
                spec_sort = []

                for i in range(sort_index.size):
                    spec_sort.append(spec_list[sort_index[i]])

                for key in spec_sort:
                    masked = np.ma.array(boxitem.spectrum[key][0],
                                         mask=np.isnan(boxitem.spectrum[key][0]))

                    if quantity == 'flux':
                        flux_scaling = masked[:, 0]

                    if not plot_kwargs[j] or key not in plot_kwargs[j]:
                        plot_obj = ax1.errorbar(masked[:, 0], flux_scaling*masked[:, 1]/scaling,
                                                yerr=flux_scaling*masked[:, 2]/scaling, ms=2, marker='s',
                                                zorder=2.5, ls='none')

                        if plot_kwargs[j] is None:
                            plot_kwargs[j] = {}

                        plot_kwargs[j][key] = {'marker': 's', 'ms': 2., 'ls': 'none',
                                               'color': plot_obj[0].get_color()}

                    elif 'marker' not in plot_kwargs[j][key]:
                        # Plot the spectrum as a line without error bars
                        # (e.g. when the spectrum has a high spectral resolution)
                        plot_obj = ax1.plot(masked[:, 0], flux_scaling*masked[:, 1]/scaling,
                                            **plot_kwargs[j][key])

                    else:
                        ax1.errorbar(masked[:, 0], flux_scaling*masked[:, 1]/scaling, yerr=flux_scaling*masked[:, 2]/scaling,
                                     zorder=2.5, **plot_kwargs[j][key])

            if boxitem.flux is not None:
                filter_list = []
                wavel_list = []

                for item in boxitem.flux:
                    read_filt = read_filter.ReadFilter(item)
                    filter_list.append(item)
                    wavel_list.append(read_filt.mean_wavelength())

                sort_index = np.argsort(wavel_list)
                filter_sort = []

                for i in range(sort_index.size):
                    filter_sort.append(filter_list[sort_index[i]])

                for item in filter_sort:
                    transmission = read_filter.ReadFilter(item)
                    wavelength = transmission.mean_wavelength()
                    fwhm = transmission.filter_fwhm()

                    if not plot_kwargs[j] or item not in plot_kwargs[j]:
                        if not plot_kwargs[j]:
                            plot_kwargs[j] = {}

                        if quantity == 'flux':
                            flux_scaling = wavelength

                        scale_tmp = flux_scaling / scaling

                        if isinstance(boxitem.flux[item][0], np.ndarray):
                            for i in range(boxitem.flux[item].shape[1]):

                                plot_obj = ax1.errorbar(wavelength,
                                                        scale_tmp*boxitem.flux[item][0, i],
                                                        xerr=fwhm/2.,
                                                        yerr=scale_tmp*boxitem.flux[item][1, i],
                                                        marker='s', ms=5, zorder=3, color='black')

                        else:

                            plot_obj = ax1.errorbar(wavelength,
                                                    scale_tmp*boxitem.flux[item][0],
                                                    xerr=fwhm/2.,
                                                    yerr=scale_tmp*boxitem.flux[item][1],
                                                    marker='s', ms=5, zorder=3, color='black')

                        plot_kwargs[j][item] = {'marker': 's', 'ms': 5., 'color': plot_obj[0].get_color()}

                    else:
                        if quantity == 'flux':
                            flux_scaling = wavelength

                        if isinstance(boxitem.flux[item][0], np.ndarray):
                            if not isinstance(plot_kwargs[j][item], list):
                                raise ValueError(f'A list with {boxitem.flux[item].shape[1]} '
                                                 f'dictionaries are required because the filter '
                                                 f'{item} has {boxitem.flux[item].shape[1]} '
                                                 f'values.')

                            for i in range(boxitem.flux[item].shape[1]):
                                ax1.errorbar(wavelength, flux_scaling*boxitem.flux[item][0, i]/scaling, xerr=fwhm/2.,
                                             yerr=flux_scaling*boxitem.flux[item][1, i]/scaling, zorder=3, **plot_kwargs[j][item][i])

                        else:
                            if boxitem.flux[item][1] == 0.:
                                ax1.errorbar(wavelength, flux_scaling*boxitem.flux[item][0]/scaling,
                                             xerr=fwhm/2., yerr=0.5*flux_scaling*boxitem.flux[item][0]/scaling,
                                             uplims=True, capsize=2., capthick=0., zorder=3, **plot_kwargs[j][item])

                            else:
                                ax1.errorbar(wavelength, flux_scaling*boxitem.flux[item][0]/scaling,
                                             xerr=fwhm/2., yerr=flux_scaling*boxitem.flux[item][1]/scaling,
                                             zorder=3, **plot_kwargs[j][item])

        elif isinstance(boxitem, box.SynphotBox):
            for i, find_item in enumerate(boxes):
                if isinstance(find_item, box.ObjectBox):
                    obj_index = i
                    break

            for item in boxitem.flux:
                transmission = read_filter.ReadFilter(item)
                wavelength = transmission.mean_wavelength()
                fwhm = transmission.filter_fwhm()

                if quantity == 'flux':
                    flux_scaling = wavelength

                if not plot_kwargs[obj_index] or item not in plot_kwargs[obj_index]:
                    ax1.errorbar(wavelength, flux_scaling*boxitem.flux[item]/scaling, xerr=fwhm/2., yerr=None,
                                 alpha=0.7, marker='s', ms=5, zorder=4, mfc='white')

                else:
                    if isinstance(plot_kwargs[obj_index][item], list):
                        # In case of multiple photometry values for the same filter, use the
                        # plot_kwargs of the first data point

                        kwargs_copy = plot_kwargs[obj_index][item][0].copy()

                        if 'label' in kwargs_copy:
                            del kwargs_copy['label']

                        ax1.errorbar(wavelength, flux_scaling*boxitem.flux[item]/scaling, xerr=fwhm/2., yerr=None,
                                     zorder=4, mfc='white', **kwargs_copy)

                    else:
                        kwargs_copy = plot_kwargs[obj_index][item].copy()

                        if 'label' in kwargs_copy:
                            del kwargs_copy['label']

                        if 'mfc' in kwargs_copy:
                            del kwargs_copy['mfc']

                        ax1.errorbar(wavelength, flux_scaling*boxitem.flux[item]/scaling, xerr=fwhm/2., yerr=None,
                                     zorder=4, mfc='white', **kwargs_copy)

    if filters is not None:
        for i, item in enumerate(filters):
            transmission = read_filter.ReadFilter(item)
            data = transmission.get_filter()

            ax2.plot(data[:, 0], data[:, 1], '-', lw=0.7, color='black', zorder=1)

    if residuals is not None:
        for i, find_item in enumerate(boxes):
            if isinstance(find_item, box.ObjectBox):
                obj_index = i
                break

        res_max = 0.

        if residuals.photometry is not None:
            for item in residuals.photometry:
                if not plot_kwargs[obj_index] or item not in plot_kwargs[obj_index]:
                    ax3.plot(residuals.photometry[item][0], residuals.photometry[item][1], marker='s',
                             ms=5, linestyle='none', zorder=2)

                else:
                    if residuals.photometry[item].ndim == 1:
                        ax3.errorbar(residuals.photometry[item][0], residuals.photometry[item][1],
                                     zorder=2, **plot_kwargs[obj_index][item])

                    elif residuals.photometry[item].ndim == 2:
                        for i in range(residuals.photometry[item].shape[1]):
                            if isinstance(plot_kwargs[obj_index][item], list):
                                ax3.errorbar(residuals.photometry[item][0, i],
                                             residuals.photometry[item][1, i], zorder=2,
                                             **plot_kwargs[obj_index][item][i])

                            else:
                                ax3.errorbar(residuals.photometry[item][0, i],
                                             residuals.photometry[item][1, i], zorder=2,
                                             **plot_kwargs[obj_index][item])

                res_max = np.nanmax(np.abs(residuals.photometry[item][1]))

        if residuals.spectrum is not None:
            for key, value in residuals.spectrum.items():
                if not plot_kwargs[obj_index] or key not in plot_kwargs[obj_index]:
                    ax3.errorbar(value[:, 0], value[:, 1], marker='o', ms=2, ls='none', zorder=1)

                else:
                    ax3.errorbar(value[:, 0], value[:, 1], zorder=1, **plot_kwargs[obj_index][key])

                max_tmp = np.nanmax(np.abs(value[:, 1]))

                if max_tmp > res_max:
                    res_max = max_tmp

        res_lim = math.ceil(1.1*res_max)

        if res_lim > 10.:
            res_lim = 5.

        ax3.axhline(0., ls='--', lw=0.7, color='gray', dashes=(2, 4), zorder=0.5)
        # ax3.axhline(-2.5, ls=':', lw=0.7, color='gray', dashes=(1, 4), zorder=0.5)
        # ax3.axhline(2.5, ls=':', lw=0.7, color='gray', dashes=(1, 4), zorder=0.5)

        if ylim_res is None:
            ax3.set_ylim(-res_lim, res_lim)

        else:
            ax3.set_ylim(ylim_res[0], ylim_res[1])

    if filters is not None:
        ax2.set_ylim(0., 1.1)

    print(f'Plotting spectrum: {output}...', end='', flush=True)

    if title is not None:
        if filters:
            ax2.set_title(title, y=1.02, fontsize=13)
        else:
            ax1.set_title(title, y=1.02, fontsize=13)

    handles, labels = ax1.get_legend_handles_labels()

    if handles and legend is not None:
        if isinstance(legend, list):
            model_handles = []
            data_handles = []

            model_labels = []
            data_labels = []

            for i, item in enumerate(handles):
                if isinstance(item, mpl.lines.Line2D):
                    model_handles.append(item)
                    model_labels.append(labels[i])

                elif isinstance(item, mpl.container.ErrorbarContainer):
                    data_handles.append(item)
                    data_labels.append(labels[i])

                else:
                    warnings.warn(f'The object type {item} is not implemented for the legend.')

            if legend[0] is not None:
                if isinstance(legend[0], (str, tuple)):
                    leg_1 = ax1.legend(model_handles, model_labels, loc=legend[0], fontsize=10., frameon=False)
                else:
                    leg_1 = ax1.legend(model_handles, model_labels, **legend[0])

            else:
                leg_1 = None

            if legend[1] is not None:
                if isinstance(legend[1], (str, tuple)):
                    leg_2 = ax1.legend(data_handles, data_labels, loc=legend[1], fontsize=8, frameon=False)
                else:
                    leg_2 = ax1.legend(data_handles, data_labels, **legend[1])

            if leg_1 is not None:
                ax1.add_artist(leg_1)

        elif isinstance(legend, (str, tuple)):
            ax1.legend(loc=legend, fontsize=8, frameon=False)

        else:
            ax1.legend(**legend)

    if scale[0] == 'log':
        ax1.xaxis.set_major_formatter(ScalarFormatter())

        if ax2 is not None:
            ax2.xaxis.set_major_formatter(ScalarFormatter())

        if ax3 is not None:
            ax3.xaxis.set_major_formatter(ScalarFormatter())

    # filters = ['Paranal/SPHERE.ZIMPOL_N_Ha',
    #            'MUSE/Hbeta',
    #            'ALMA/855']
    #
    # filters = ['Paranal/SPHERE.IRDIS_B_Y',
    #            'MKO/NSFCam.J',
    #            'Paranal/SPHERE.IRDIS_D_H23_2',
    #            'Paranal/SPHERE.IRDIS_D_H23_3',
    #            'Paranal/SPHERE.IRDIS_D_K12_1',
    #            'Paranal/SPHERE.IRDIS_D_K12_2',
    #            'Paranal/NACO.Lp',
    #            'Paranal/NACO.NB405',
    #            'Paranal/NACO.Mp']
    #
    # for i, item in enumerate(filters):
    #     readfilter = read_filter.ReadFilter(item)
    #     filter_wavelength = readfilter.mean_wavelength()
    #     filter_width = readfilter.filter_fwhm()
    #
    #     # if i == 5:
    #     #     ax1.errorbar(filter_wavelength, 1.3e4, xerr=filter_width/2., color='dimgray', elinewidth=2.5, zorder=10)
    #     # else:
    #     #     ax1.errorbar(filter_wavelength, 6e3, xerr=filter_width/2., color='dimgray', elinewidth=2.5, zorder=10)
    #
    #     if i == 0:
    #         ax1.text(filter_wavelength, 1e-2, r'H$\alpha$', ha='center', va='center', fontsize=10, color='black')
    #     elif i == 1:
    #         ax1.text(filter_wavelength, 1e-2, r'H$\beta$', ha='center', va='center', fontsize=10, color='black')
    #     elif i == 2:
    #         ax1.text(filter_wavelength, 1e-2, 'ALMA\nband 7 rms', ha='center', va='center', fontsize=8, color='black')
    #
    #     if i == 0:
    #         ax1.text(filter_wavelength, 1.4, 'Y', ha='center', va='center', fontsize=10, color='black')
    #     elif i == 1:
    #         ax1.text(filter_wavelength, 1.4, 'J', ha='center', va='center', fontsize=10, color='black')
    #     elif i == 2:
    #         ax1.text(filter_wavelength-0.04, 1.4, 'H2', ha='center', va='center', fontsize=10, color='black')
    #     elif i == 3:
    #         ax1.text(filter_wavelength+0.04, 1.4, 'H3', ha='center', va='center', fontsize=10, color='black')
    #     elif i == 4:
    #         ax1.text(filter_wavelength, 1.4, 'K1', ha='center', va='center', fontsize=10, color='black')
    #     elif i == 5:
    #         ax1.text(filter_wavelength, 1.4, 'K2', ha='center', va='center', fontsize=10, color='black')
    #     elif i == 6:
    #         ax1.text(filter_wavelength, 1.4, 'L$\'$', ha='center', va='center', fontsize=10, color='black')
    #     elif i == 7:
    #         ax1.text(filter_wavelength, 1.4, 'NB4.05', ha='center', va='center', fontsize=10, color='black')
    #     elif i == 8:
    #         ax1.text(filter_wavelength, 1.4, 'M$\'}$', ha='center', va='center', fontsize=10, color='black')
    #
    # ax1.text(1.26, 0.58, 'VLT/SPHERE', ha='center', va='center', fontsize=8., color='slateblue', rotation=43.)
    # ax1.text(2.5, 1.28, 'VLT/SINFONI', ha='left', va='center', fontsize=8., color='darkgray')

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.clf()
    plt.close()

    print(' [DONE]')
