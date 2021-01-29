"""
Module with a function for plotting results from the empirical spectral analysis.
"""

import configparser
import os

from typing import Optional, Tuple

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import interp1d
from typeguard import typechecked

from species.core import constants
from species.read import read_object
from species.util import dust_util, read_util


@typechecked
def plot_statistic(tag: str,
                   xlim: Optional[Tuple[float, float]] = None,
                   ylim: Optional[Tuple[float, float]] = None,
                   title: Optional[str] = None,
                   offset: Optional[Tuple[float, float]] = None,
                   figsize: Optional[Tuple[float, float]] = (4., 2.5),
                   output: str = 'statistic.pdf'):
    """
    Function for plotting the goodness-of-fit statistic of the empirical spectral comparison.

    Parameters
    ----------
    tag : str
        Database tag where the results from the empirical comparison with
        :class:`~species.analysis.empirical.CompareSpectra.spectral_type` are stored.
    xlim : tuple(float, float)
        Limits of the spectral type axis in numbers (i.e. 0=M0, 5=M5, 10=L0, etc.).
    ylim : tuple(float, float)
        Limits of the goodness-of-fit axis.
    title : str
        Plot title.
    offset : tuple(float, float)
        Offset for the label of the x- and y-axis.
    figsize : tuple(float, float)
        Figure size.
    output : str
        Output filename.

    Returns
    -------
    NoneType
        None
    """

    print(f'Plotting goodness-of-fit statistic: {output}...', end='')

    config_file = os.path.join(os.getcwd(), 'species_config.ini')

    config = configparser.ConfigParser()
    config.read_file(open(config_file))

    db_path = config['species']['database']

    h5_file = h5py.File(db_path, 'r')

    dset = h5_file[f'results/empirical/{tag}/names']

    names = np.array(dset)
    sptypes = np.array(h5_file[f'results/empirical/{tag}/sptypes'])
    g_fit = np.array(h5_file[f'results/empirical/{tag}/goodness_of_fit'])

    mpl.rcParams['font.serif'] = ['Bitstream Vera Serif']
    mpl.rcParams['font.family'] = 'serif'

    plt.rc('axes', edgecolor='black', linewidth=2.2)
    plt.rcParams['axes.axisbelow'] = False

    plt.figure(1, figsize=figsize)
    gridsp = mpl.gridspec.GridSpec(1, 1)
    gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    ax = plt.subplot(gridsp[0, 0])

    ax.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                   direction='in', width=1, length=5, labelsize=12, top=True,
                   bottom=True, left=True, right=True)

    ax.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                   direction='in', width=1, length=3, labelsize=12, top=True,
                   bottom=True, left=True, right=True)

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.set_xlabel('Spectral type', fontsize=13)
    ax.set_ylabel('G', fontsize=13)

    if offset is not None:
        ax.get_xaxis().set_label_coords(0.5, offset[0])
        ax.get_yaxis().set_label_coords(offset[1], 0.5)

    else:
        ax.get_xaxis().set_label_coords(0.5, -0.1)
        ax.get_yaxis().set_label_coords(-0.1, 0.5)

    if title is not None:
        ax.set_title(title, y=1.02, fontsize=13)

    ax.set_xticks(np.linspace(0., 30., 7, endpoint=True))
    ax.set_xticklabels(['M0', 'M5', 'L0', 'L5', 'T0', 'T5', 'Y0'])

    if xlim is None:
        ax.set_xlim(0., 30.)
    else:
        ax.set_xlim(xlim[0], xlim[1])

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    sptype_num = np.zeros(names.shape[0])

    for i, item in enumerate(sptypes):
        for j in range(10):
            if item == f'M{j}':
                sptype_num[i] = float(j)

            elif item == f'L{j}':
                sptype_num[i] = float(10+j)

            elif item == f'T{j}':
                sptype_num[i] = float(20+j)

    ax.plot(sptype_num, g_fit, 's', ms=3., mew=0.5, color='lightgray', markeredgecolor='darkgray')

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.clf()
    plt.close()

    h5_file.close()

    print(' [DONE]')


@typechecked
def plot_empirical_spectra(tag: str,
                           n_spectra: int,
                           xlim: Optional[Tuple[float, float]] = None,
                           ylim: Optional[Tuple[float, float]] = None,
                           title: Optional[str] = None,
                           offset: Optional[Tuple[float, float]] = None,
                           figsize: Optional[Tuple[float, float]] = (4., 2.5),
                           output: str = 'empirical.pdf'):
    """
    Function for plotting the results from the empirical spectrum comparison.

    Parameters
    ----------
    tag : str
        Database tag where the results from the empirical comparison with
        :class:`~species.analysis.empirical.CompareSpectra.spectral_type` are stored.
    n_spectra : int
        The number of spectra with the lowest goodness-of-fit statistic that will be plotted in
        comparison with the data.
    xlim : tuple(float, float)
        Limits of the spectral type axis.
    ylim : tuple(float, float)
        Limits of the goodness-of-fit axis.
    title : str
        Plot title.
    offset : tuple(float, float)
        Offset for the label of the x- and y-axis.
    figsize : tuple(float, float)
        Figure size.
    output : str
        Output filename.

    Returns
    -------
    NoneType
        None
    """

    print(f'Plotting empirical spectra comparison: {output}...', end='')

    config_file = os.path.join(os.getcwd(), 'species_config.ini')

    config = configparser.ConfigParser()
    config.read_file(open(config_file))

    db_path = config['species']['database']

    h5_file = h5py.File(db_path, 'r')

    dset = h5_file[f'results/empirical/{tag}/names']

    object_name = dset.attrs['object_name']
    spec_name = dset.attrs['spec_name']
    spec_library = dset.attrs['spec_library']

    names = np.array(dset)
    flux_scaling = np.array(h5_file[f'results/empirical/{tag}/flux_scaling'])
    av_ext = np.array(h5_file[f'results/empirical/{tag}/av_ext'])

    rad_vel = np.array(h5_file[f'results/empirical/{tag}/rad_vel'])
    rad_vel *= 1e3  # (m s-1)

    mpl.rcParams['font.serif'] = ['Bitstream Vera Serif']
    mpl.rcParams['font.family'] = 'serif'

    plt.rc('axes', edgecolor='black', linewidth=2.2)
    plt.rcParams['axes.axisbelow'] = False

    plt.figure(1, figsize=figsize)
    gridsp = mpl.gridspec.GridSpec(1, 1)
    gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    ax = plt.subplot(gridsp[0, 0])

    ax.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                   direction='in', width=1, length=5, labelsize=12, top=True,
                   bottom=True, left=True, right=True)

    ax.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                   direction='in', width=1, length=3, labelsize=12, top=True,
                   bottom=True, left=True, right=True)

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.set_xlabel('Wavelength (µm)', fontsize=13)
    ax.set_ylabel(r'$\mathregular{F}_\lambda$ (W m$^{-2}$ µm$^{-1}$)', fontsize=11)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    if offset is not None:
        ax.get_xaxis().set_label_coords(0.5, offset[0])
        ax.get_yaxis().set_label_coords(offset[1], 0.5)

    else:
        ax.get_xaxis().set_label_coords(0.5, -0.1)
        ax.get_yaxis().set_label_coords(-0.1, 0.5)

    if title is not None:
        ax.set_title(title, y=1.02, fontsize=13)

    read_obj = read_object.ReadObject(object_name)

    obj_spec = read_obj.get_spectrum()[spec_name][0]
    obj_res = read_obj.get_spectrum()[spec_name][3]

    for i in range(n_spectra):
        spectrum = np.asarray(h5_file[f'spectra/{spec_library}/{names[i]}'])

        ism_ext = dust_util.ism_extinction(av_ext[i], 3.1, spectrum[:, 0])
        ext_scaling = 10.**(-0.4*ism_ext)

        wavel_shifted = spectrum[:, 0] + spectrum[:, 0] * rad_vel[i] / constants.LIGHT

        flux_smooth = read_util.smooth_spectrum(wavel_shifted,
                                                spectrum[:, 1]*ext_scaling,
                                                spec_res=obj_res,
                                                force_smooth=True)

        interp_spec = interp1d(spectrum[:, 0],
                               flux_smooth,
                               fill_value='extrapolate')

        indices = np.where((obj_spec[:, 0] > np.amin(spectrum[:, 0])) &
                           (obj_spec[:, 0] < np.amax(spectrum[:, 0])))[0]

        flux_resample = interp_spec(obj_spec[indices, 0])

        ax.plot(obj_spec[indices, 0], flux_scaling[i]*flux_resample, color='gray', lw=0.3,
                alpha=0.5, zorder=1)

    ax.plot(obj_spec[:, 0], obj_spec[:, 1], '-', lw=0.6, color='black')

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.clf()
    plt.close()

    h5_file.close()

    print(' [DONE]')
