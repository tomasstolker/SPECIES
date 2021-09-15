"""
Module with functions for creating color-magnitude and color-color plot.
"""

import math
import os

from typing import List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colorbar import Colorbar
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d
from typeguard import typechecked

from species.core import box
from species.data import companions
from species.read import read_filter, read_object
from species.util import dust_util, plot_util


@typechecked
def plot_color_magnitude(boxes: list,
                         objects: Optional[Union[List[Tuple[str, str, str, str]],
                                                 List[Tuple[str, str, str, str, Optional[dict],
                                                            Optional[dict]]]]] = None,
                         mass_labels: Optional[Union[List[float], List[Tuple[float, str]]]] = None,
                         teff_labels: Optional[Union[List[float], List[Tuple[float, str]]]] = None,
                         companion_labels: bool = False,
                         accretion: bool = False,
                         reddening: Optional[List[Tuple[Tuple[str, str], Tuple[str, float], str,
                                                        float, Tuple[float, float]]]] = None,
                         ism_red: Optional[List[Tuple[Tuple[str, str], str, float,
                                                      Tuple[float, float]]]] = None,
                         field_range: Optional[Tuple[str, str]] = None,
                         label_x: str = 'Color',
                         label_y: str = 'Absolute magnitude',
                         xlim: Optional[Tuple[float, float]] = None,
                         ylim: Optional[Tuple[float, float]] = None,
                         offset: Optional[Tuple[float, float]] = None,
                         legend: Optional[Union[str, dict, Tuple[float, float]]] = 'upper left',
                         output: str = 'color-magnitude.pdf') -> None:
    """
    Function for creating a color-magnitude diagram.

    Parameters
    ----------
    boxes : list(species.core.box.ColorMagBox, species.core.box.IsochroneBox, )
        Boxes with the color-magnitude and isochrone data from photometric libraries, spectral
        libraries, and/or atmospheric models. The synthetic data have to be created with
        :func:`~species.read.read_isochrone.ReadIsochrone.get_color_magnitude`. These boxes
        contain synthetic colors and magnitudes for a given age and a range of masses.
    objects : list(tuple(str, str, str, str), ),
              list(tuple(str, str, str, str, dict, dict), ), None
        Tuple with individual objects. The objects require a tuple with their database tag, the two
        filter names for the color, and the filter name for the absolute magnitude. Optionally, a
        dictionary with keyword arguments can be provided for the object's marker and label,
        respectively. For example, ``{'marker': 'o', 'ms': 10}`` for the marker and
        ``{'ha': 'left', 'va': 'bottom', 'xytext': (5, 5)})`` for the label. Not used if set to
        None.
    mass_labels : list(float, ), list(tuple(float, str), ), None
        Plot labels with masses next to the isochrone data of `models`. The list with masses has
        to be provided in Jupiter mass. Alternatively, a list of tuples can be provided with
        the planet mass and position of the label ('left' or 'right), for example
        ``[(10., 'left'), (20., 'right')]``. No labels are shown if set to None.
    teff_labels : list(float, ), list(tuple(float, str), ), None
        Plot labels with temperatures (K) next to the synthetic Planck photometry. Alternatively,
        a list of tuples can be provided with the planet mass and position of the label ('left' or
        'right), for example ``[(1000., 'left'), (1200., 'right')]``. No labels are shown if set
        to None.
    companion_labels : bool
        Plot labels with the names of the directly imaged companions.
    accretion : bool
        Plot accreting, directly imaged objects with a different symbol than the regular, directly
        imaged objects. The object names from ``objects`` will be compared with the data from
        :func:`~species.data.companions.get_data` to check if a companion is accreting or not.
    reddening : list(tuple(tuple(str, str), tuple(str, float), str, float,
                tuple(float, float))), None
        Include reddening arrows by providing a list with tuples. Each tuple contains the filter
        names for the color, the filter name and value of the magnitude, the mean particle radius
        (um), and the start position (color, mag) of the arrow in the plot, so ``((filter_color_1,
        filter_color_2), (filter_mag, mag_value), composition, radius, (x_pos, y_pos))``. The
        composition can be either ``'Fe'`` or ``'MgSiO3'`` (both with crystalline structure). A
        log-normal size distribution is used with the specified mean radius and the geometric
        standard deviation is fixed to 2. Both ``xlim`` and ``ylim`` need to be set for the correct
        rotation of the reddening label. The parameter is not used if set to ``None``.
    ism_red : list(tuple(tuple(str, str), str, float, tuple(float, float))), None
        List with reddening arrows for ISM extinction. Each item in the list is a tuple that itself
        contain a tuple with the filter names for the color, the filter name of the magnitude, the
        visual extinction, and the start position (color, mag) of the arrow in the plot, so
        ``((filter_color_1, filter_color_2), filter_mag, A_V, (x_pos, y_pos))``. The parameter is
        not used if the argument is set to ``None``.
    field_range : tuple(str, str), None
        Range of the discrete colorbar for the field dwarfs. The tuple should contain the lower
        and upper value ('early M', 'late M', 'early L', 'late L', 'early T', 'late T', 'early Y).
        The full range is used if set to None.
    label_x : str
        Label for the x-axis.
    label_y : str
        Label for the y-axis.
    xlim : tuple(float, float), None
        Limits for the x-axis. Not used if set to None.
    ylim : tuple(float, float), None
        Limits for the y-axis. Not used if set to None.
    offset : tuple(float, float), None
        Offset of the x- and y-axis label.
    legend : str, tuple(float, float), dict, None
        Legend position or keyword arguments. No legend is shown if set to ``None``.
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

    model_color = ('#234398', '#f6a432', 'black')
    model_linestyle = ('-', '--', ':', '-.')

    isochrones = []
    planck = []
    models = []
    empirical = []

    for item in boxes:
        if isinstance(item, box.IsochroneBox):
            isochrones.append(item)

        elif isinstance(item, box.ColorMagBox):
            if item.object_type == 'model':
                models.append(item)

            elif item.library == 'planck':
                planck.append(item)

            else:
                empirical.append(item)

        else:
            raise ValueError(f'Found a {type(item)} while only ColorMagBox and IsochroneBox '
                             f'objects can be provided to \'boxes\'.')

    if empirical:
        plt.figure(1, figsize=(4., 4.8))
        gridsp = mpl.gridspec.GridSpec(3, 1, height_ratios=[0.2, 0.1, 4.5])
        gridsp.update(wspace=0., hspace=0., left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(gridsp[2, 0])
        ax2 = plt.subplot(gridsp[0, 0])

    else:
        plt.figure(1, figsize=(4., 4.5))
        gridsp = mpl.gridspec.GridSpec(1, 1)
        gridsp.update(wspace=0., hspace=0., left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(gridsp[0, 0])

    ax1.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                    direction='in', width=1, length=5, labelsize=12, top=True,
                    bottom=True, left=True, right=True)

    ax1.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                    direction='in', width=1, length=3, labelsize=12, top=True,
                    bottom=True, left=True, right=True)

    ax1.xaxis.set_major_locator(MultipleLocator(1.))
    ax1.yaxis.set_major_locator(MultipleLocator(1.))

    ax1.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.2))

    ax1.set_xlabel(label_x, fontsize=14)
    ax1.set_ylabel(label_y, fontsize=14)

    ax1.invert_yaxis()

    if offset is not None:
        ax1.get_xaxis().set_label_coords(0.5, offset[0])
        ax1.get_yaxis().set_label_coords(offset[1], 0.5)

    else:
        ax1.get_xaxis().set_label_coords(0.5, -0.08)
        ax1.get_yaxis().set_label_coords(-0.12, 0.5)

    if xlim is not None:
        ax1.set_xlim(xlim[0], xlim[1])

    if ylim is not None:
        ax1.set_ylim(ylim[0], ylim[1])

    if models is not None:
        count = 0

        model_dict = {}

        for j, item in enumerate(models):
            if item.library not in model_dict:
                model_dict[item.library] = [count, 0]
                count += 1

            else:
                model_dict[item.library] = [model_dict[item.library][0],
                                            model_dict[item.library][1]+1]

            model_count = model_dict[item.library]

            if model_count[0] == 3:
                raise ValueError('Only three different types of model atmospheres can be added.')

            if model_count[1] == 0:
                label = plot_util.model_name(item.library)

                if item.library == 'zhu2015':
                    ax1.plot(item.color, item.magnitude, marker='x', ms=5,
                             linestyle=model_linestyle[model_count[1]],
                             linewidth=0.6, color='gray', label=label, zorder=0)

                    xlim = ax1.get_xlim()
                    ylim = ax1.get_ylim()

                    for i, teff_item in enumerate(item.sptype):
                        teff_label = rf'{teff_item:.0e} $M_\mathregular{{Jup}}^{2}$ yr$^{{-1}}$'

                        if item.magnitude[i] > ylim[1]:
                            ax1.annotate(teff_label, (item.color[i], item.magnitude[i]),
                                         color='gray', fontsize=8, ha='left', va='center',
                                         xytext=(item.color[i]+0.1, item.magnitude[i]+0.05),
                                         zorder=3)

                else:
                    ax1.plot(item.color, item.magnitude, linestyle=model_linestyle[model_count[1]],
                             lw=1., color=model_color[model_count[0]], label=label, zorder=0)

                    if mass_labels is not None:
                        interp_magnitude = interp1d(item.sptype, item.magnitude)
                        interp_color = interp1d(item.sptype, item.color)

                        for i, mass_item in enumerate(mass_labels):
                            if isinstance(mass_item, tuple):
                                mass_val = mass_item[0]
                                mass_pos = mass_item[1]

                            else:
                                mass_val = mass_item
                                mass_pos = 'right'

                            if j == 0 or (j > 0 and mass_val < 20.):
                                pos_color = interp_color(mass_val)
                                pos_mag = interp_magnitude(mass_val)

                                if mass_pos == 'left':
                                    mass_ha = 'right'
                                    mass_xytext = (pos_color-0.05, pos_mag)

                                else:
                                    mass_ha = 'left'
                                    mass_xytext = (pos_color+0.05, pos_mag)

                                mass_label = str(int(mass_val))+r' M$_\mathregular{J}$'

                                xlim = ax1.get_xlim()
                                ylim = ax1.get_ylim()

                                if xlim[0]+0.2 < pos_color < xlim[1]-0.2 and \
                                        ylim[1]+0.2 < pos_mag < ylim[0]-0.2:

                                    ax1.scatter(pos_color, pos_mag, c=model_color[model_count[0]],
                                                s=15, edgecolor='none', zorder=0)

                                    ax1.annotate(mass_label, (pos_color, pos_mag),
                                                 color=model_color[model_count[0]], fontsize=9,
                                                 xytext=mass_xytext, zorder=3, ha=mass_ha,
                                                 va='center')

            else:
                ax1.plot(item.color, item.magnitude, linestyle=model_linestyle[model_count[1]],
                         linewidth=0.6, color=model_color[model_count[0]], zorder=0)

    if planck is not None:
        planck_count = 0

        for j, item in enumerate(planck):
            if planck_count == 0:
                label = plot_util.model_name(item.library)
            else:
                label = None

            ax1.plot(item.color, item.magnitude, linestyle='--', linewidth=0.8,
                     color='black', label=label, zorder=0)

            if teff_labels is not None and planck_count == 0:
                interp_magnitude = interp1d(item.sptype, item.magnitude)
                interp_color = interp1d(item.sptype, item.color)

                for i, teff_item in enumerate(teff_labels):
                    if isinstance(teff_item, tuple):
                        teff_val = teff_item[0]
                        teff_pos = teff_item[1]

                    else:
                        teff_val = teff_item
                        teff_pos = 'right'

                    if j == 0 or (j > 0 and teff_val < 20.):
                        pos_color = interp_color(teff_val)
                        pos_mag = interp_magnitude(teff_val)

                        if teff_pos == 'left':
                            teff_ha = 'right'
                            teff_xytext = (pos_color-0.05, pos_mag)

                        else:
                            teff_ha = 'left'
                            teff_xytext = (pos_color+0.05, pos_mag)

                        teff_label = f'{int(teff_val)} K'

                        xlim = ax1.get_xlim()
                        ylim = ax1.get_ylim()

                        if xlim[0]+0.2 < pos_color < xlim[1]-0.2 and \
                                ylim[1]+0.2 < pos_mag < ylim[0]-0.2:

                            ax1.scatter(pos_color, pos_mag, c='black', s=15, ec='none', zorder=0)

                            if planck_count == 0:
                                ax1.annotate(teff_label, (pos_color, pos_mag), color='black',
                                             fontsize=9, xytext=teff_xytext, zorder=3, ha=teff_ha,
                                             va='center')

            planck_count += 1

    if empirical:
        cmap = plt.cm.viridis

        bounds, ticks, ticklabels = plot_util.field_bounds_ticks(field_range)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        for item in empirical:
            sptype = item.sptype
            color = item.color
            magnitude = item.magnitude
            names = item.names

            if isinstance(sptype, list):
                sptype = np.array(sptype)

            if item.object_type in ['field', None]:
                indices = np.where(sptype != 'None')[0]

                sptype = sptype[indices]
                color = color[indices]
                magnitude = magnitude[indices]

                spt_disc = plot_util.sptype_substellar(sptype, color.shape)

                _, unique = np.unique(color, return_index=True)

                sptype = sptype[unique]
                color = color[unique]
                magnitude = magnitude[unique]
                spt_disc = spt_disc[unique]

                scat = ax1.scatter(color, magnitude, c=spt_disc, cmap=cmap, norm=norm, s=50,
                                   alpha=0.7, edgecolor='none', zorder=2)

                cb = Colorbar(ax=ax2, mappable=scat, orientation='horizontal',
                              ticklocation='top', format='%.2f')

                cb.ax.tick_params(width=1, length=5, labelsize=10, direction='in', color='black')

                cb.set_ticks(ticks)
                cb.set_ticklabels(ticklabels)

            elif item.object_type == 'young':
                if objects is not None:
                    object_names = []

                    for obj_item in objects:
                        object_names.append(obj_item[0])

                    indices = plot_util.remove_color_duplicates(object_names, names)

                    color = color[indices]
                    magnitude = magnitude[indices]

                ax1.plot(color, magnitude, marker='s', ms=4, linestyle='none', alpha=0.7,
                         color='gray', markeredgecolor='black', label='Young/low-gravity', zorder=2)

                # for item in names[indices]:
                #
                #     if item == '2MASSWJ2244316+204343':
                #         item = '2MASS 2244+20'
                #
                #     kwargs = {'ha': 'left', 'va': 'center', 'fontsize': 8.5,
                #               'xytext': (5., 0.), 'color': 'black'}
                #
                #     ax1.annotate(item, (color, magnitude), zorder=3,
                #                  textcoords='offset points', **kwargs)

    if isochrones:
        for item in isochrones:
            ax1.plot(item.color, item.magnitude, linestyle='-', linewidth=1., color='black')

    if reddening is not None:
        for item in reddening:
            ext_1, ext_2 = dust_util.calc_reddening(item[0],
                                                    item[1],
                                                    composition=item[2],
                                                    structure='crystalline',
                                                    radius_g=item[3])

            delta_x = ext_1 - ext_2
            delta_y = item[1][1]

            x_pos = item[4][0] + delta_x
            y_pos = item[4][1] + delta_y

            ax1.annotate('', (x_pos, y_pos), xytext=(item[4][0], item[4][1]),
                         fontsize=8, arrowprops={'arrowstyle': '->'}, color='black', zorder=3.)

            x_pos_text = item[4][0] + delta_x/2.
            y_pos_text = item[4][1] + delta_y/2.

            vector_len = math.sqrt(delta_x**2+delta_y**2)

            if item[2] == 'MgSiO3':
                dust_species = r'MgSiO$_{3}$'
            elif item[2] == 'Fe':
                dust_species = 'Fe'

            if (item[3]).is_integer():
                red_label = f'{dust_species} ({item[3]:.0f} µm)'
            else:
                red_label = f'{dust_species} ({item[3]:.1f} µm)'

            text = ax1.annotate(red_label, (x_pos_text, y_pos_text),
                                xytext=(7.*delta_y/vector_len, 7.*delta_x/vector_len),
                                textcoords='offset points', fontsize=8., color='black',
                                ha='center', va='center')

            ax1.plot([item[4][0], x_pos], [item[4][1], y_pos], '-', color='white')

            sp1 = ax1.transData.transform_point((item[4][0], item[4][1]))
            sp2 = ax1.transData.transform_point((x_pos, y_pos))

            angle = np.degrees(np.arctan2(sp2[1]-sp1[1], sp2[0]-sp1[0]))
            text.set_rotation(angle)

    if ism_red is not None:
        for item in ism_red:
            # Color filters
            read_filt_0 = read_filter.ReadFilter(item[0][0])
            read_filt_1 = read_filter.ReadFilter(item[0][1])

            # Magnitude filter
            read_filt_2 = read_filter.ReadFilter(item[1])

            mean_wavel = np.array([read_filt_0.mean_wavelength(),
                                   read_filt_1.mean_wavelength(),
                                   read_filt_2.mean_wavelength()])

            ext_mag = dust_util.ism_extinction(item[2], 3.1, mean_wavel)

            delta_x = ext_mag[0] - ext_mag[1]
            delta_y = ext_mag[2]

            x_pos = item[3][0] + delta_x
            y_pos = item[3][1] + delta_y

            ax1.annotate('', (x_pos, y_pos), xytext=(item[3][0], item[3][1]),
                         fontsize=8, arrowprops={'arrowstyle': '->'}, color='black', zorder=3.)

            x_pos_text = item[3][0] + delta_x/2.
            y_pos_text = item[3][1] + delta_y/2.

            vector_len = math.sqrt(delta_x**2+delta_y**2)

            if (item[2]).is_integer():
                red_label = fr'A$_\mathregular{{V}}$ = {item[2]:.0f}'
            else:
                red_label = fr'A$_\mathregular{{V}}$ = {item[2]:.1f}'

            text = ax1.annotate(red_label, (x_pos_text, y_pos_text),
                                xytext=(8.*delta_y/vector_len, 8.*delta_x/vector_len),
                                textcoords='offset points', fontsize=8., color='black',
                                ha='center', va='center')

            ax1.plot([item[3][0], x_pos], [item[3][1], y_pos], '-', color='white')

            sp1 = ax1.transData.transform_point((item[3][0], item[3][1]))
            sp2 = ax1.transData.transform_point((x_pos, y_pos))

            angle = np.degrees(np.arctan2(sp2[1]-sp1[1], sp2[0]-sp1[0]))
            text.set_rotation(angle)

    if objects is not None:
        for i, item in enumerate(objects):
            objdata = read_object.ReadObject(item[0])

            objcolor1 = objdata.get_photometry(item[1])
            objcolor2 = objdata.get_photometry(item[2])

            if objcolor1.ndim == 2:
                print(f'Found {objcolor1.shape[1]} values for filter {item[1]} of {item[0]}')
                print(f'so using the first value:  {objcolor1[0, 0]} +/- {objcolor1[1, 0]} mag')
                objcolor1 = objcolor1[:, 0]

            if objcolor2.ndim == 2:
                print(f'Found {objcolor2.shape[1]} values for filter {item[2]} of {item[0]}')
                print(f'so using the first value:  {objcolor2[0, 0]} +/- {objcolor2[1, 0]} mag')
                objcolor2 = objcolor2[:, 0]

            abs_mag, abs_err = objdata.get_absmag(item[3])

            if isinstance(abs_mag, np.ndarray):
                abs_mag = abs_mag[0]
                abs_err = abs_err[0]

            colorerr = math.sqrt(objcolor1[1]**2+objcolor2[1]**2)
            x_color = objcolor1[0]-objcolor2[0]

            companion_data = companions.get_data()

            if len(item) > 4 and item[4] is not None:
                kwargs = item[4]

            else:
                kwargs = {'marker': '>',
                          'ms': 6.,
                          'color': 'black',
                          'mfc': 'white',
                          'mec': 'black',
                          'label': 'Direct imaging'}

                if accretion and item[0] in companion_data and companion_data[item[0]]['accretion']:
                    kwargs['marker'] = 'X'
                    kwargs['ms'] = 7.
                    kwargs['label'] = 'Accreting'

            ax1.errorbar(x_color, abs_mag, yerr=abs_err, xerr=colorerr, zorder=3, **kwargs)

            if companion_labels:
                if len(item) > 4:
                    kwargs = item[5]

                else:
                    kwargs = {'ha': 'left',
                              'va': 'bottom',
                              'fontsize': 8.5,
                              'xytext': (5., 5.),
                              'color': 'black'}

                ax1.annotate(objdata.object_name, (x_color, abs_mag), zorder=3,
                             textcoords='offset points', **kwargs)

    print(f'Plotting color-magnitude diagram: {output}...', end='', flush=True)

    if legend is not None:
        handles, labels = ax1.get_legend_handles_labels()

        # prevent duplicates
        by_label = dict(zip(labels, handles))

        if handles:
            ax1.legend(by_label.values(), by_label.keys(), loc=legend, fontsize=8.5,
                       frameon=False, numpoints=1)

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.clf()
    plt.close()

    print(' [DONE]')


@typechecked
def plot_color_color(boxes: list,
                     objects: Optional[Union[List[Tuple[str, Tuple[str, str], Tuple[str, str]]],
                                             List[Tuple[str, Tuple[str, str], Tuple[str, str],
                                                        Optional[dict], Optional[dict]]]]] = None,
                     mass_labels: Optional[Union[List[float], List[Tuple[float, str]]]] = None,
                     teff_labels: Optional[Union[List[float], List[Tuple[float, str]]]] = None,
                     companion_labels: bool = False,
                     reddening: Optional[List[Tuple[Tuple[str, str], Tuple[str, str],
                                                    Tuple[str, float], str, float,
                                                    Tuple[float, float]]]] = None,
                     field_range: Optional[Tuple[str, str]] = None,
                     label_x: str = 'Color',
                     label_y: str = 'Color',
                     xlim: Optional[Tuple[float, float]] = None,
                     ylim: Optional[Tuple[float, float]] = None,
                     offset: Optional[Tuple[float, float]] = None,
                     legend: Optional[Union[str, dict, Tuple[float, float]]] = 'upper left',
                     figsize: Optional[Tuple[float, float]] = (4., 4.3),
                     output: str = 'color-color.pdf') -> None:
    """
    Function for creating a color-color diagram.

    Parameters
    ----------
    boxes : list(species.core.box.ColorColorBox, species.core.box.IsochroneBox, )
        Boxes with the color-color and isochrone data from photometric libraries, spectral
        libraries, and/or atmospheric models. The synthetic data have to be created with
        :func:`~species.read.read_isochrone.ReadIsochrone.get_color_color`. These boxes
        contain synthetic colors for a given age and a range of masses.
    objects : tuple(tuple(str, tuple(str, str), tuple(str, str)), ),
              tuple(tuple(str, tuple(str, str), tuple(str, str), dict, dict), ), None
        Tuple with individual objects. The objects require a tuple with their database tag, the two
        filter names for the first color, and the two filter names for the second color.
        Optionally, a dictionary with keyword arguments can be provided for the object's marker and
        label, respectively. For example, ``{'marker': 'o', 'ms': 10}`` for the marker and
        ``{'ha': 'left', 'va': 'bottom', 'xytext': (5, 5)})`` for the label. Not used if set to
        None.
    mass_labels : list(float, ), list(tuple(float, str), ), None
        Plot labels with masses next to the isochrone data of `models`. The list with masses has
        to be provided in Jupiter mass. Alternatively, a list of tuples can be provided with
        the planet mass and position of the label ('left' or 'right), for example
        ``[(10., 'left'), (20., 'right')]``. No labels are shown if set to None.
    teff_labels : list(float, ), list(tuple(float, str), ), None
        Plot labels with temperatures (K) next to the synthetic Planck photometry. Alternatively,
        a list of tuples can be provided with the planet mass and position of the label ('left' or
        'right), for example ``[(1000., 'left'), (1200., 'right')]``. No labels are shown if set
        to None.
    companion_labels : bool
        Plot labels with the names of the directly imaged companions.
    reddening : list(tuple(tuple(str, str), tuple(str, str), tuple(str, float), str, float,
                tuple(float, float)), None
        Include reddening arrows by providing a list with tuples. Each tuple contains the filter
        names for the color, the filter name for the magnitude, the particle radius (um), and the
        start position (color, mag) of the arrow in the plot, so (filter_color_1, filter_color_2,
        filter_mag, composition, radius, (x_pos, y_pos)). The composition can be either 'Fe' or
        'MgSiO3' (both with crystalline structure). The parameter is not used if set to ``None``.
    field_range : tuple(str, str), None
        Range of the discrete colorbar for the field dwarfs. The tuple should contain the lower
        and upper value ('early M', 'late M', 'early L', 'late L', 'early T', 'late T', 'early Y).
        The full range is used if set to None.
    label_x : str
        Label for the x-axis.
    label_y : str
        Label for the y-axis.
    xlim : tuple(float, float)
        Limits for the x-axis.
    ylim : tuple(float, float)
        Limits for the y-axis.
    offset : tuple(float, float), None
        Offset of the x- and y-axis label.
    legend : str, tuple(float, float), dict, None
        Legend position or keyword arguments. No legend is shown if set to ``None``.
    figsize : tuple(float, float)
        Figure size.
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

    model_color = ('#234398', '#f6a432', 'black')
    model_linestyle = ('-', '--', ':', '-.')

    isochrones = []
    planck = []
    models = []
    empirical = []

    for item in boxes:
        if isinstance(item, box.IsochroneBox):
            isochrones.append(item)

        elif isinstance(item, box.ColorColorBox):
            if item.object_type == 'model':
                models.append(item)

            elif item.library == 'planck':
                planck.append(item)

            else:
                empirical.append(item)

        else:
            raise ValueError(f'Found a {type(item)} while only ColorColorBox and IsochroneBox '
                             f'objects can be provided to \'boxes\'.')

    plt.figure(1, figsize=figsize)
    gridsp = mpl.gridspec.GridSpec(3, 1, height_ratios=[0.2, 0.1, 4.])
    gridsp.update(wspace=0., hspace=0., left=0, right=1, bottom=0, top=1)

    ax1 = plt.subplot(gridsp[2, 0])
    ax2 = plt.subplot(gridsp[0, 0])

    ax1.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                    direction='in', width=1, length=5, labelsize=12, top=True,
                    bottom=True, left=True, right=True)

    ax1.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                    direction='in', width=1, length=3, labelsize=12, top=True,
                    bottom=True, left=True, right=True)

    ax1.xaxis.set_major_locator(MultipleLocator(0.5))
    ax1.yaxis.set_major_locator(MultipleLocator(0.5))

    ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax1.set_xlabel(label_x, fontsize=14)
    ax1.set_ylabel(label_y, fontsize=14)

    ax1.invert_yaxis()

    if offset:
        ax1.get_xaxis().set_label_coords(0.5, offset[0])
        ax1.get_yaxis().set_label_coords(offset[1], 0.5)
    else:
        ax1.get_xaxis().set_label_coords(0.5, -0.08)
        ax1.get_yaxis().set_label_coords(-0.12, 0.5)

    if xlim:
        ax1.set_xlim(xlim[0], xlim[1])

    if ylim:
        ax1.set_ylim(ylim[0], ylim[1])

    if models is not None:
        count = 0

        model_dict = {}

        for j, item in enumerate(models):
            if item.library not in model_dict:
                model_dict[item.library] = [count, 0]
                count += 1

            else:
                model_dict[item.library] = [model_dict[item.library][0],
                                            model_dict[item.library][1]+1]

            model_count = model_dict[item.library]

            if model_count[1] == 0:
                label = plot_util.model_name(item.library)

                if item.library == 'zhu2015':
                    ax1.plot(item.color1, item.color2, marker='x', ms=5,
                             linestyle=model_linestyle[model_count[1]],
                             linewidth=0.6, color='gray', label=label, zorder=0)

                    xlim = ax1.get_xlim()
                    ylim = ax1.get_ylim()

                    for i, teff_item in enumerate(item.sptype):
                        teff_label = rf'{teff_item:.0e} $M_\mathregular{{Jup}}^{2}$ yr$^{{-1}}$'

                        if item.color2[i] < ylim[1]:
                            ax1.annotate(teff_label, (item.color1[i], item.color2[i]),
                                         color='gray', fontsize=8, ha='left', va='center',
                                         xytext=(item.color1[i]+0.1, item.color2[i]-0.05),
                                         zorder=3)

                else:
                    ax1.plot(item.color1, item.color2, linestyle=model_linestyle[model_count[1]],
                             lw=1., color=model_color[model_count[0]], label=label, zorder=0)

                    if mass_labels is not None:
                        interp_color1 = interp1d(item.sptype, item.color1)
                        interp_color2 = interp1d(item.sptype, item.color2)

                        for i, mass_item in enumerate(mass_labels):
                            if isinstance(mass_item, tuple):
                                mass_val = mass_item[0]
                                mass_pos = mass_item[1]

                            else:
                                mass_val = mass_item
                                mass_pos = 'right'

                            # if j == 0 or (j > 0 and mass_val < 20.):
                            if j == 0:
                                pos_color1 = interp_color1(mass_val)
                                pos_color2 = interp_color2(mass_val)

                                if mass_pos == 'left':
                                    mass_ha = 'right'
                                    mass_xytext = (pos_color1-0.05, pos_color2)

                                else:
                                    mass_ha = 'left'
                                    mass_xytext = (pos_color1+0.05, pos_color2)

                                mass_label = str(int(mass_val))+r' M$_\mathregular{J}$'

                                xlim = ax1.get_xlim()
                                ylim = ax1.get_ylim()

                                if xlim[0]+0.2 < pos_color1 < xlim[1]-0.2 and \
                                        ylim[0]+0.2 < pos_color2 < ylim[1]-0.2:

                                    ax1.scatter(pos_color1, pos_color2,
                                                c=model_color[model_count[0]],
                                                s=15, edgecolor='none', zorder=0)

                                    ax1.annotate(mass_label, (pos_color1, pos_color2),
                                                 color=model_color[model_count[0]], fontsize=9,
                                                 xytext=mass_xytext, ha=mass_ha, va='center',
                                                 zorder=3)

            else:
                ax1.plot(item.color1, item.color2, linestyle=model_linestyle[model_count[1]],
                         linewidth=0.6, color=model_color[model_count[0]], label=label, zorder=0)

    if planck is not None:
        planck_count = 0

        for j, item in enumerate(planck):

            if planck_count == 0:
                label = plot_util.model_name(item.library)

                ax1.plot(item.color1, item.color2, ls='--', linewidth=0.8, color='black',
                         label=label, zorder=0)

                if teff_labels is not None:
                    interp_color1 = interp1d(item.sptype, item.color1)
                    interp_color2 = interp1d(item.sptype, item.color2)

                    for i, teff_item in enumerate(teff_labels):
                        if isinstance(teff_item, tuple):
                            teff_val = teff_item[0]
                            teff_pos = teff_item[1]

                        else:
                            teff_val = teff_item
                            teff_pos = 'right'

                        if j == 0 or (j > 0 and teff_val < 20.):
                            pos_color1 = interp_color1(teff_val)
                            pos_color2 = interp_color2(teff_val)

                            if teff_pos == 'left':
                                teff_ha = 'right'
                                teff_xytext = (pos_color1-0.05, pos_color2)

                            else:
                                teff_ha = 'left'
                                teff_xytext = (pos_color1+0.05, pos_color2)

                            teff_label = f'{int(teff_val)} K'

                            xlim = ax1.get_xlim()
                            ylim = ax1.get_ylim()

                            if xlim[0]+0.2 < pos_color1 < xlim[1]-0.2 and \
                                    ylim[0]+0.2 < pos_color2 < ylim[1]-0.2:

                                ax1.scatter(pos_color1, pos_color2, c='black', s=15,
                                            edgecolor='none', zorder=0)

                                ax1.annotate(teff_label, (pos_color1, pos_color2),
                                             color='black', fontsize=9,
                                             xytext=teff_xytext, zorder=3, ha=teff_ha, va='center')

            else:
                ax1.plot(item.color1, item.color2, ls='--', lw=0.5, color='black', zorder=0)

            planck_count += 1

    if empirical:
        cmap = plt.cm.viridis

        bounds, ticks, ticklabels = plot_util.field_bounds_ticks(field_range)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        for item in empirical:
            sptype = item.sptype
            names = item.names
            color1 = item.color1
            color2 = item.color2

            if isinstance(sptype, list):
                sptype = np.array(sptype)

            if item.object_type in ['field', None]:
                indices = np.where(sptype != 'None')[0]

                sptype = sptype[indices]
                color1 = color1[indices]
                color2 = color2[indices]

                spt_disc = plot_util.sptype_substellar(sptype, color1.shape)
                _, unique = np.unique(color1, return_index=True)

                sptype = sptype[unique]
                color1 = color1[unique]
                color2 = color2[unique]
                spt_disc = spt_disc[unique]

                scat = ax1.scatter(color1, color2, c=spt_disc, cmap=cmap, norm=norm, s=50,
                                   alpha=0.7, edgecolor='none', zorder=2)

                cb = Colorbar(ax=ax2, mappable=scat, orientation='horizontal',
                              ticklocation='top', format='%.2f')

                cb.ax.tick_params(width=1, length=5, labelsize=10, direction='in', color='black')

                cb.set_ticks(ticks)
                cb.set_ticklabels(ticklabels)

            elif item.object_type == 'young':
                if objects is not None:
                    object_names = []

                    for obj_item in objects:
                        object_names.append(obj_item[0])

                    indices = plot_util.remove_color_duplicates(object_names, names)

                    color1 = color1[indices]
                    color2 = color2[indices]

                ax1.plot(color1, color2, marker='s', ms=4, linestyle='none', alpha=0.7,
                         color='gray', markeredgecolor='black', label='Young/low-gravity', zorder=2)

    if isochrones:
        for item in isochrones:
            ax1.plot(item.colors[0], item.colors[1], linestyle='-', linewidth=1., color='black')

    if reddening is not None:
        for item in reddening:
            ext_1, ext_2 = dust_util.calc_reddening(item[0],
                                                    item[2],
                                                    composition=item[3],
                                                    structure='crystalline',
                                                    radius_g=item[4])

            ext_3, ext_4 = dust_util.calc_reddening(item[1],
                                                    item[2],
                                                    composition=item[3],
                                                    structure='crystalline',
                                                    radius_g=item[4])

            delta_x = ext_1 - ext_2
            delta_y = ext_3 - ext_4

            x_pos = item[5][0] + delta_x
            y_pos = item[5][1] + delta_y

            ax1.annotate('', (x_pos, y_pos), xytext=(item[5][0], item[5][1]),
                         fontsize=8, arrowprops={'arrowstyle': '->'}, color='black', zorder=3.)

            x_pos_text = item[5][0] + delta_x/2.
            y_pos_text = item[5][1] + delta_y/2.

            vector_len = math.sqrt(delta_x**2+delta_y**2)

            if item[3] == 'MgSiO3':
                dust_species = r'MgSiO$_{3}$'

            elif item[3] == 'Fe':
                dust_species = 'Fe'

            if item[4].is_integer():
                red_label = f'{dust_species} ({item[4]:.0f} µm)'

            else:
                red_label = f'{dust_species} ({item[4]:.1f} µm)'

            text = ax1.annotate(red_label, (x_pos_text, y_pos_text),
                                xytext=(-7.*delta_y/vector_len, 7.*delta_x/vector_len),
                                textcoords='offset points', fontsize=8., color='black',
                                ha='center', va='center')

            ax1.plot([item[5][0], x_pos], [item[5][1], y_pos], '-', color='white')

            sp1 = ax1.transData.transform_point((item[5][0], item[5][1]))
            sp2 = ax1.transData.transform_point((x_pos, y_pos))

            angle = np.degrees(np.arctan2(sp2[1]-sp1[1], sp2[0]-sp1[0]))
            text.set_rotation(angle)

    if objects is not None:
        for i, item in enumerate(objects):
            objdata = read_object.ReadObject(item[0])

            objphot1 = objdata.get_photometry(item[1][0])
            objphot2 = objdata.get_photometry(item[1][1])
            objphot3 = objdata.get_photometry(item[2][0])
            objphot4 = objdata.get_photometry(item[2][1])

            if objphot1.ndim == 2:
                print(f'Found {objphot1.shape[1]} values for filter {item[1][0]} of {item[0]}')
                print(f'so using the first value:  {objphot1[0, 0]} +/- {objphot1[1, 0]} mag')
                objphot1 = objphot1[:, 0]

            if objphot2.ndim == 2:
                print(f'Found {objphot2.shape[1]} values for filter {item[1][1]} of {item[0]}')
                print(f'so using the first value:  {objphot2[0, 0]} +/- {objphot2[1, 0]} mag')
                objphot2 = objphot2[:, 0]

            if objphot3.ndim == 2:
                print(f'Found {objphot3.shape[1]} values for filter {item[2][0]} of {item[0]}')
                print(f'so using the first value:  {objphot3[0, 0]} +/- {objphot3[1, 0]} mag')
                objphot3 = objphot3[:, 0]

            if objphot4.ndim == 2:
                print(f'Found {objphot4.shape[1]} values for filter {item[2][1]} of {item[0]}')
                print(f'so using the first value:  {objphot4[0, 0]} +/- {objphot4[1, 0]} mag')
                objphot4 = objphot4[:, 0]

            color1 = objphot1[0] - objphot2[0]
            color2 = objphot3[0] - objphot4[0]

            error1 = math.sqrt(objphot1[1]**2+objphot2[1]**2)
            error2 = math.sqrt(objphot3[1]**2+objphot4[1]**2)

            if len(item) > 3 and item[3] is not None:
                kwargs = item[3]

            else:
                kwargs = {'marker': '>',
                          'ms': 6.,
                          'color': 'black',
                          'mfc': 'white',
                          'mec': 'black',
                          'label': 'Direct imaging'}

            ax1.errorbar(color1, color2, xerr=error1, yerr=error2, zorder=3, **kwargs)

            if companion_labels:
                if len(item) > 3:
                    kwargs = item[4]

                else:
                    kwargs = {'ha': 'left',
                              'va': 'bottom',
                              'fontsize': 8.5,
                              'xytext': (5., 5.),
                              'color': 'black'}

                ax1.annotate(objdata.object_name, (color1, color2), zorder=3,
                             textcoords='offset points', **kwargs)

    print(f'Plotting color-color diagram: {output}...', end='', flush=True)

    handles, labels = ax1.get_legend_handles_labels()

    if legend is not None:
        handles, labels = ax1.get_legend_handles_labels()

        # prevent duplicates
        by_label = dict(zip(labels, handles))

        if handles:
            ax1.legend(by_label.values(), by_label.keys(), loc=legend, fontsize=8.5,
                       frameon=False, numpoints=1)

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.clf()
    plt.close()

    print(' [DONE]')
