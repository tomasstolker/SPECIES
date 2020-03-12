"""
Module with a function for plotting spectra.
"""

import os
import math
import itertools

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from species.core import box, constants
from species.read import read_filter
from species.util import plot_util


mpl.rcParams['font.serif'] = ['Bitstream Vera Serif']
mpl.rcParams['font.family'] = 'serif'

plt.rc('axes', edgecolor='black', linewidth=2.2)
plt.rcParams['axes.axisbelow'] = False


def plot_spectrum(boxes,
                  filters=None,
                  residuals=None,
                  colors=None,
                  xlim=None,
                  ylim=None,
                  ylim_res=None,
                  scale=('linear', 'linear'),
                  title=None,
                  offset=None,
                  legend='upper left',
                  figsize=(7., 5.),
                  object_type='planet',
                  quantity='flux',
                  output='spectrum.pdf'):
    """
    Parameters
    ----------
    boxes : list(species.core.box, )
        Boxes with data.
    filters : list(str, ), None
        Filter IDs for which the transmission profile is plotted. Not plotted if set to None.
    residuals : species.core.box.ResidualsBox, None
        Box with residuals of a fit. Not plotted if set to None.
    colors : list(str, ), None
        Colors to be used for the different boxes. Note that a box with residuals requires a tuple
        with two colors (i.e., for the photometry and spectrum). Automatic colors are used if set
        to None.
    xlim : tuple(float, float)
        Limits of the wavelength axis.
    ylim : tuple(float, float)
        Limits of the flux axis.
    ylim_res : tuple(float, float), None
        Limits of the residuals axis. Automatically chosen (based on the minimum and maximum
        residual value) if set to None.
    scale : tuple(str, str)
        Scale of the axes ('linear' or 'log').
    title : str
        Title.
    offset : tuple(float, float)
        Offset for the label of the x- and y-axis.
    legend : str, None
        Location of the legend.
    figsize : tuple(float, float)
        Figure size.
    object_type : str
        Object type ('planet' or 'star'). With 'planet', the radius and mass are expressed in
        Jupiter units. With 'star', the radius and mass are expressed in solar units.
    quantity: str
        The quantity of the y-axis ('flux' or 'magnitude').
    output : str
        Output filename.

    Returns
    -------
    NoneType
        None
    """

    marker = itertools.cycle(('o', 's', '*', 'p', '<', '>', 'P', 'v', '^'))

    if colors is not None and len(boxes) != len(colors):
        raise ValueError(f'The number of \'boxes\' ({len(boxes)}) should be equal to the '
                         f'number of \'colors\' ({len(colors)}).')

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
        ax3 = plt.subplot(gridsp[1, 0])

    elif filters is not None:
        plt.figure(1, figsize=figsize)
        gridsp = mpl.gridspec.GridSpec(2, 1, height_ratios=[1, 4])
        gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(gridsp[1, 0])
        ax2 = plt.subplot(gridsp[0, 0])

    else:
        plt.figure(1, figsize=figsize)
        gridsp = mpl.gridspec.GridSpec(1, 1)
        gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(gridsp[0, 0])

    if residuals is not None:
        labelbottom = False
    else:
        labelbottom = True

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

    if residuals is not None and filters is not None:
        ax1.set_xlabel('', fontsize=13)
        ax2.set_xlabel('', fontsize=13)
        ax3.set_xlabel(r'Wavelength ($\mu$m)', fontsize=13)

    elif residuals is not None:
        ax1.set_xlabel('', fontsize=13)
        ax3.set_xlabel(r'Wavelength ($\mu$m)', fontsize=13)

    elif filters is not None:
        ax1.set_xlabel(r'Wavelength ($\mu$m)', fontsize=13)
        ax2.set_xlabel('', fontsize=13)

    else:
        ax1.set_xlabel(r'Wavelength ($\mu$m)', fontsize=13)

    if filters is not None:
        ax2.set_ylabel('Transmission', fontsize=13)

    if residuals is not None:
        ax3.set_ylabel(r'Residual ($\sigma$)', fontsize=13)

    if xlim is not None:
        ax1.set_xlim(xlim[0], xlim[1])
    else:
        ax1.set_xlim(0.6, 6.)

    if quantity == 'magnitude':
        scaling = 1.
        ax1.set_ylabel('Flux contrast (mag)', fontsize=13)

        if ylim:
            ax1.set_ylim(ylim[0], ylim[1])

    elif quantity == 'flux':
        if ylim:
            ax1.set_ylim(ylim[0], ylim[1])

            ylim = ax1.get_ylim()

            exponent = math.floor(math.log10(ylim[1]))
            scaling = 10.**exponent

            ylabel = r'Flux (10$^{'+str(exponent)+r'}$ W m$^{-2}$ $\mu$m$^{-1}$)'

            ax1.set_ylabel(ylabel, fontsize=13)
            ax1.set_ylim(ylim[0]/scaling, ylim[1]/scaling)

            if ylim[0] < 0.:
                ax1.axhline(0.0, linestyle='--', color='gray', dashes=(2, 4), zorder=0.5)

        else:
            ax1.set_ylabel(r'Flux (W m$^{-2}$ $\mu$m$^{-1}$)', fontsize=13)
            scaling = 1.

    if filters is not None:
        ax2.set_ylim(0., 1.)

    xlim = ax1.get_xlim()

    if filters is not None:
        ax2.set_xlim(xlim[0], xlim[1])

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

    ax1.set_xscale(scale[0])
    ax1.set_yscale(scale[1])

    if filters is not None:
        ax2.set_xscale(scale[0])

    if residuals is not None:
        ax3.set_xscale(scale[0])

    color_obj_phot = None
    color_obj_spec = None

    for j, boxitem in enumerate(boxes):
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

                        if item == 'teff':
                            value = f'{param[item]:.1f}'

                        elif item in ['logg', 'feh', 'co', 'fsed']:
                            value = f'{param[item]:.2f}'

                        elif item == 'radius':

                            if object_type == 'planet':
                                value = f'{param[item]:.2f}'

                            elif object_type == 'star':
                                value = f'{param[item]*constants.R_JUP/constants.R_SUN:.2f}'

                        elif item == 'mass':
                            if object_type == 'planet':
                                value = f'{param[item]:.2f}'
                            elif object_type == 'star':
                                value = f'{param[item]*constants.M_JUP/constants.M_SUN:.2f}'

                        elif item == 'luminosity':
                            value = f'{param[item]:.1e}'

                        else:
                            continue

                        # if len(label) > 110 and newline == False:
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

                if colors:
                    ax1.plot(wavelength, masked/scaling, color=colors[j], lw=0.5,
                             label=label, zorder=2)
                else:
                    ax1.plot(wavelength, masked/scaling, lw=0.5, label=label, zorder=2)

            elif isinstance(wavelength[0], (np.ndarray)):
                for i, item in enumerate(wavelength):
                    data = np.array(flux[i], dtype=np.float64)
                    masked = np.ma.array(data, mask=np.isnan(data))

                    if isinstance(boxitem.name[i], bytes):
                        label = boxitem.name[i].decode('utf-8')
                    else:
                        label = boxitem.name[i]

                    ax1.plot(item, masked/scaling, lw=0.5, label=label)

        elif isinstance(boxitem, list):
            for i, item in enumerate(boxitem):
                wavelength = item.wavelength
                flux = item.flux

                data = np.array(flux, dtype=np.float64)
                masked = np.ma.array(data, mask=np.isnan(data))

                if colors:
                    ax1.plot(wavelength, masked/scaling, lw=0.2, color=colors[j],
                             alpha=0.5, zorder=1)
                else:
                    ax1.plot(wavelength, masked/scaling, lw=0.2, alpha=0.5, zorder=1)

        elif isinstance(boxitem, box.PhotometryBox):
            marker = next(marker)

            for i, item in enumerate(boxitem.wavelength):
                transmission = read_filter.ReadFilter(boxitem.filter_name[i])
                fwhm = transmission.filter_fwhm()

                if colors:
                    ax1.errorbar(item, boxitem.flux[i][0]/scaling, xerr=fwhm/2.,
                                 yerr=boxitem.flux[i][1], marker=marker, ms=6, color=colors[j],
                                 zorder=3)
                else:
                    ax1.errorbar(item, boxitem.flux[i][0]/scaling, xerr=fwhm/2.,
                                yerr=boxitem.flux[i][1], marker=marker, ms=6, color='black',
                                zorder=3)

        elif isinstance(boxitem, box.ObjectBox):
            if boxitem.flux is not None:
                for item in boxitem.flux:
                    transmission = read_filter.ReadFilter(item)
                    wavelength = transmission.mean_wavelength()
                    fwhm = transmission.filter_fwhm()

                    if colors is None:
                        ax1.errorbar(wavelength, boxitem.flux[item][0]/scaling, xerr=fwhm/2.,
                                     yerr=boxitem.flux[item][1]/scaling, marker='s', ms=5, zorder=3,
                                     markerfacecolor=color_obj_phot)

                    else:
                        color_obj_phot = colors[j][0]

                        ax1.errorbar(wavelength, boxitem.flux[item][0]/scaling, xerr=fwhm/2.,
                                     yerr=boxitem.flux[item][1]/scaling, marker='s', ms=5, zorder=3,
                                     color=color_obj_phot, markerfacecolor=color_obj_phot)

            if boxitem.spectrum is not None:
                for key, value in boxitem.spectrum.items():
                    masked = np.ma.array(boxitem.spectrum[key][0],
                                         mask=np.isnan(boxitem.spectrum[key][0]))

                    if colors is None:
                        ax1.errorbar(masked[:, 0], masked[:, 1]/scaling, yerr=masked[:, 2]/scaling,
                                     ms=2, marker='s', zorder=2.5, ls='none')

                    else:
                        color_obj_spec = colors[j][1]

                        ax1.errorbar(masked[:, 0], masked[:, 1]/scaling, yerr=masked[:, 2]/scaling,
                                     marker='o', ms=2, zorder=2.5, color=color_obj_spec,
                                     markerfacecolor=color_obj_spec, ls='none')

        elif isinstance(boxitem, box.SynphotBox):
            for item in boxitem.flux:
                transmission = read_filter.ReadFilter(item)
                wavelength = transmission.mean_wavelength()
                fwhm = transmission.filter_fwhm()

                ax1.errorbar(wavelength, boxitem.flux[item]/scaling, xerr=fwhm/2., yerr=None,
                             alpha=0.7, marker='s', ms=5, zorder=4, color=colors[j],
                             markerfacecolor='white')

    if filters is not None:
        for i, item in enumerate(filters):
            transmission = read_filter.ReadFilter(item)
            data = transmission.get_filter()

            ax2.plot(data[0, ], data[1, ], '-', lw=0.7, color='black', zorder=1)

    if residuals is not None:
        res_max = 0.

        if residuals.photometry is not None:
            ax3.plot(residuals.photometry[0, ], residuals.photometry[1, ], marker='s',
                     ms=5, linestyle='none', color=color_obj_phot, zorder=2)

            res_max = np.nanmax(np.abs(residuals.photometry[1, ]))

        if residuals.spectrum is not None:
            for key, value in residuals.spectrum.items():
                if colors is None:
                    ax3.plot(value[:, 0], value[:, 1], marker='o', ms=2,
                             linestyle='none', zorder=1)

                else:
                    ax3.plot(value[:, 0], value[:, 1], marker='o',
                             ms=2, linestyle='none', color=color_obj_spec, zorder=1)

                max_tmp = np.nanmax(np.abs(value[:, 1]))

                if max_tmp > res_max:
                    res_max = max_tmp

        res_lim = math.ceil(1.1*res_max)

        if res_lim > 10.:
            res_lim = 5.

        ax3.axhline(0.0, linestyle='--', color='gray', dashes=(2, 4), zorder=0.5)

        if ylim_res is None:
            ax3.set_ylim(-res_lim, res_lim)
        else:
            ax3.set_ylim(ylim_res[0], ylim_res[1])

    if filters is not None:
        ax2.set_ylim(0., 1.1)

    print(f'Plotting spectrum: {output}...', end='', flush=True)

    if title is not None:
        if filters:
            ax2.set_title(title, y=1.02, fontsize=15)
        else:
            ax1.set_title(title, y=1.02, fontsize=15)

    handles, _ = ax1.get_legend_handles_labels()

    if handles and legend:
        ax1.legend(loc=legend, fontsize=10, frameon=False)

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.clf()
    plt.close()

    print(' [DONE]')
