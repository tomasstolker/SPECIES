"""
Utility functions for plotting data.
"""

from typing import Optional, Tuple, List

import numpy as np

from typeguard import typechecked


@typechecked
def sptype_substellar(sptype: np.ndarray,
                      shape: Tuple[int]) -> np.ndarray:
    """
    Function for mapping the spectral types of substellar objects (M, L, T, and Y) to numbers.

    Parameters
    ----------
    sptype : np.ndarray
        Array with spectral types.
    shape : tuple(int)
        Shape (1D) of the output array

    Returns
    -------
    np.ndarray
        Array with spectral types mapped to numbers.
    """

    spt_disc = np.zeros(shape)

    for i, item in enumerate(sptype):
        if item[0:2] in ['M0', 'M1', 'M2', 'M3', 'M4']:
            spt_disc[i] = 0.5

        elif item[0:2] in ['M5', 'M6', 'M7', 'M8', 'M9']:
            spt_disc[i] = 1.5

        elif item[0:2] in ['L0', 'L1', 'L2', 'L3', 'L4']:
            spt_disc[i] = 2.5

        elif item[0:2] in ['L5', 'L6', 'L7', 'L8', 'L9']:
            spt_disc[i] = 3.5

        elif item[0:2] in ['T0', 'T1', 'T2', 'T3', 'T4']:
            spt_disc[i] = 4.5

        elif item[0:2] in ['T5', 'T6', 'T7', 'T8', 'T9']:
            spt_disc[i] = 5.5

        elif 'Y' in item:
            spt_disc[i] = 6.5

        else:
            spt_disc[i] = np.nan
            continue

    return spt_disc


@typechecked
def sptype_stellar(sptype: np.ndarray,
                   shape: Tuple[int]) -> np.ndarray:
    """
    Function for mapping all spectral types (O through Y) to numbers.

    Parameters
    ----------
    sptype : np.ndarray
        Array with spectral types.
    shape : tuple(int)
        Shape (1D) of the output array

    Returns
    -------
    np.ndarray
        Array with spectral types mapped to numbers.
    """

    spt_disc = np.zeros(shape)

    for i, item in enumerate(sptype):
        if item[0] == 'O':
            spt_disc[i] = 0.5

        elif item[0] == 'B':
            spt_disc[i] = 1.5

        elif item[0] == 'A':
            spt_disc[i] = 2.5

        elif item[0] == 'F':
            spt_disc[i] = 3.5

        elif item[0] == 'G':
            spt_disc[i] = 4.5

        elif item[0] == 'K':
            spt_disc[i] = 5.5

        elif item[0] == 'M':
            spt_disc[i] = 6.5

        elif item[0] == 'L':
            spt_disc[i] = 7.5

        elif item[0] == 'T':
            spt_disc[i] = 8.5

        elif item[0] == 'Y':
            spt_disc[i] = 9.5

        else:
            spt_disc[i] = np.nan
            continue

    return spt_disc


@typechecked
def update_labels(param: List[str]) -> List[str]:
    """
    Function for updating the fitted parameters to labels used in plots.

    Parameters
    ----------
    param : list
        List with parameter names.

    Returns
    -------
    list
        List with parameter labels for plots.
    """

    if 'teff' in param:
        index = param.index('teff')
        param[index] = r'$\mathregular{T}_\mathregular{eff}$ (K)'

    if 'logg' in param:
        index = param.index('logg')
        param[index] = r'$\mathregular{log}\,\mathregular{g}$'

    if 'feh' in param:
        index = param.index('feh')
        param[index] = r'[Fe/H]'

    if 'fsed' in param:
        index = param.index('fsed')
        param[index] = r'$\mathregular{f}_\mathregular{sed}$'

    if 'co' in param:
        index = param.index('co')
        param[index] = r'C/O'

    if 'radius' in param:
        index = param.index('radius')
        param[index] = r'$\mathregular{R}$ ($\mathregular{R_J}$)'

    if 'mass' in param:
        index = param.index('mass')
        param[index] = r'$\mathregular{M}$ ($\mathregular{M_J}$)'

    if 'luminosity' in param:
        index = param.index('luminosity')
        param[index] = r'$\mathregular{log}\,\mathregular{L}/\mathregular{L}_\mathregular{\odot}$'

    if 'luminosity_ratio' in param:
        index = param.index('luminosity_ratio')
        param[index] = r'$\mathregular{log}\,\mathregular{L_1}/\mathregular{L_2}$'

    if 'luminosity_disk_planet' in param:
        index = param.index('luminosity_disk_planet')
        param[index] = r'$\mathregular{L_{disk}}/\mathregular{L_{atm}}$'

    if 'lognorm_radius' in param:
        index = param.index('lognorm_radius')
        param[index] = r'$\mathregular{log}\,\mathregular{r_g}$'

    if 'lognorm_sigma' in param:
        index = param.index('lognorm_sigma')
        param[index] = r'$\mathregular{\sigma_g}$'

    if 'lognorm_ext' in param:
        index = param.index('lognorm_ext')
        param[index] = r'$\mathregular{A_V}$'

    if 'powerlaw_min' in param:
        index = param.index('powerlaw_min')
        param[index] = r'$\mathregular{{log}}\,\mathregular{a}_\mathregular{min}/\mathregular{µm}$'

    if 'powerlaw_max' in param:
        index = param.index('powerlaw_max')
        param[index] = r'$\mathregular{{log}}\,\mathregular{a}_\mathregular{max}/\mathregular{µm}$'

    if 'powerlaw_exp' in param:
        index = param.index('powerlaw_exp')
        param[index] = r'$\beta$'

    if 'powerlaw_ext' in param:
        index = param.index('powerlaw_ext')
        param[index] = r'$\mathregular{A_V}$'

    if 'ism_ext' in param:
        index = param.index('ism_ext')
        param[index] = r'$\mathregular{A_V}$'

    if 'ism_red' in param:
        index = param.index('ism_red')
        param[index] = r'$\mathregular{R_V}$'

    if 'powerl_ext' in param:
        index = param.index('powerl_ext')
        param[index] = r'$\mathregular{A_V}$'

    if 'powerl_exp' in param:
        index = param.index('powerl_exp')
        param[index] = r'$\alpha_\mathregular{ext}$'

    if 'tint' in param:
        index = param.index('tint')
        param[index] = r'$\mathregular{T_{int}}$ (K)'

    for i in range(15):
        if f't{i}' in param:
            index = param.index(f't{i}')
            param[index] = rf'$\mathregular{{T}}_\mathregular{{{i}}}$ (K)'

    if 'alpha' in param:
        index = param.index('alpha')
        param[index] = r'$\alpha$'

    if 'log_delta' in param:
        index = param.index('log_delta')
        param[index] = r'$\mathregular{log}\,\mathregular{\delta}$'

    if 'log_p_quench' in param:
        index = param.index('log_p_quench')
        param[index] = r'$\mathregular{log}\,\mathregular{P}_\mathregular{quench}$'

    for i, item in enumerate(param):
        if item[0:8] == 'scaling_':
            param[i] = rf'$\mathregular{{a}}_\mathregular{{{item[8:]}}}$'

        elif item[0:6] == 'error_':
            param[i] = rf'$\mathregular{{b}}_\mathregular{{{item[6:]}}}$'

        elif item[0:6] == 'slope_':
            param[i] = rf'$\mathregular{{c}}_\mathregular{{{item[6:]}}}$'

        elif item[0:9] == 'log_corr_len_':
            param[i] = rf'$\mathregular{{log}}\,\ell_\mathregular{{{item[9:]}}}$'

        elif item[0:9] == 'corr_amp_':
            param[i] = rf'$\mathregular{{f}}_\mathregular{{{item[9:]}}}$'

        elif item[0:13] == 'log_glob_len_':
            param[i] = rf'$\mathregular{{log}}\,\rho_\mathregular{{{item[13:]}}}$'

        elif item[0:13] == 'log_glob_amp_':
            param[i] = rf'$\mathregular{{log}}\,\sigma^2_\mathregular{{{item[13:]}}}$'

    for i in range(100):
        if f'teff_{i}' in param:
            index = param.index(f'teff_{i}')
            param[index] = rf'$\mathregular{{T}}_\mathregular{{{i+1}}}$ (K)'

        else:
            break

    for i in range(100):
        if f'radius_{i}' in param:
            index = param.index(f'radius_{i}')
            param[index] = rf'$\mathregular{{R}}_\mathregular{{{i+1}}}$ ' + r'($\mathregular{R_J}$)'

        else:
            break

    for i in range(100):
        if f'luminosity_{i}' in param:
            index = param.index(f'luminosity_{i}')
            param[index] = rf'$\mathregular{{log}}\,\mathregular{{L}}_\mathregular{{{i+1}}}' \
                           rf'/\mathregular{{L}}_\mathregular{{\odot}}$'

        else:
            break

    if 'disk_teff' in param:
        index = param.index('disk_teff')
        param[index] = r'$\mathregular{T}_\mathregular{disk}$ (K)'

    if 'disk_radius' in param:
        index = param.index('disk_radius')
        param[index] = r'$\mathregular{R}_\mathregular{disk}$ ($\mathregular{R_J}$)'

    if 'log_powerlaw_a' in param:
        index = param.index('log_powerlaw_a')
        param[index] = r'$a_\mathregular{powerlaw}$'

    if 'log_powerlaw_b' in param:
        index = param.index('log_powerlaw_b')
        param[index] = r'$b_\mathregular{powerlaw}$'

    if 'log_powerlaw_c' in param:
        index = param.index('log_powerlaw_c')
        param[index] = r'$c_\mathregular{powerlaw}$'

    if 'veil_a' in param:
        index = param.index('veil_a')
        param[index] = r'$a_\mathregular{veil}$'

    if 'veil_b' in param:
        index = param.index('veil_b')
        param[index] = r'$b_\mathregular{veil}$'

    if 'veil_ref' in param:
        index = param.index('veil_ref')
        param[index] = r'$F_\mathregular{ref, veil}$'

    if 'gauss_amplitude' in param:
        index = param.index('gauss_amplitude')
        param[index] = r'$a$ (W m$^{-2}$ µm$^{-1}$)'

    if 'gauss_mean' in param:
        index = param.index('gauss_mean')
        param[index] = r'$\mu$ (nm)'

    if 'gauss_sigma' in param:
        index = param.index('gauss_sigma')
        param[index] = r'$\sigma$ (nm)'

    if 'gauss_fwhm' in param:
        index = param.index('gauss_fwhm')
        param[index] = r'$v$ (km s$^{-1}$)'

    if 'line_flux' in param:
        index = param.index('line_flux')
        param[index] = r'$F_\mathregular{line}$ (W m$^{-2}$)'

    if 'line_luminosity' in param:
        index = param.index('line_luminosity')
        param[index] = r'$L_\mathregular{line}$ (L$_\mathregular{\odot}$)'

    if 'line_eq_width' in param:
        index = param.index('line_eq_width')
        param[index] = r'EW ($\AA$)'

    return param


@typechecked
def model_name(key: str) -> str:
    """
    Function for updating a model name for use in plots.

    Parameters
    ----------
    key : str
        Model name as used by species.

    Returns
    -------
    str
        Updated model name for plots.
    """

    if key == 'drift-phoenix':
        name = 'DRIFT-PHOENIX'

    elif key == 'ames-cond':
        name = 'AMES-Cond'

    elif key == 'ames-dusty':
        name = 'AMES-Dusty'

    elif key == 'atmo':
        name = 'ATMO'

    elif key == 'bt-cond':
        name = 'BT-Cond'

    elif key == 'bt-cond-feh':
        name = 'BT-Cond'

    elif key == 'bt-settl':
        name = 'BT-Settl'

    elif key == 'bt-settl-cifist':
        name = 'BT-Settl'

    elif key == 'bt-nextgen':
        name = 'BT-NextGen'

    elif key == 'petitcode-cool-clear':
        name = 'petitCODE'

    elif key == 'petitcode-cool-cloudy':
        name = 'petitCODE'

    elif key == 'petitcode-hot-clear':
        name = 'petitCODE'

    elif key == 'petitcode-hot-cloudy':
        name = 'petitCODE'

    elif key == 'exo-rem':
        name = 'Exo-REM'

    elif key == 'planck':
        name = 'Blackbody radiation'

    elif key == 'zhu2015':
        name = 'Zhu (2015)'

    return name


@typechecked
def quantity_unit(param: List[str],
                  object_type: str) -> Tuple[List[str], List[Optional[str]], List[str]]:
    """
    Function for creating lists with quantities, units, and labels for fitted parameter.

    Parameters
    ----------
    param : list
        List with parameter names.
    object_type : str
        Object type (``'planet'`` or ``'star'``).

    Returns
    -------
    list
        List with the quantities.
    list
        List with the units.
    list
        List with the parameter labels for plots.
    """

    quantity = []
    unit = []
    label = []

    if 'teff' in param:
        quantity.append('teff')
        unit.append('K')
        label.append(r'$\mathregular{T}_\mathregular{eff}$')

    if 'logg' in param:
        quantity.append('logg')
        unit.append(None)
        label.append(r'$\mathregular{log}\,\mathregular{g}$')

    if 'feh' in param:
        quantity.append('feh')
        unit.append(None)
        label.append('[Fe/H]')

    if 'fsed' in param:
        quantity.append('fsed')
        unit.append(None)
        label.append(r'$\mathregular{f}_\mathregular{sed}$')

    if 'co' in param:
        quantity.append('co')
        unit.append(None)
        label.append('C/O')

    if 'radius' in param:
        quantity.append('radius')

        if object_type == 'planet':
            unit.append(r'$\mathregular{R}_\mathregular{J}$')

        elif object_type == 'star':
            unit.append(r'$\mathregular{R}_\mathregular{\odot}$')

        label.append(r'$\mathregular{R}$')

    for i in range(100):
        if f'teff_{i}' in param:
            quantity.append(f'teff_{i}')
            unit.append('K')
            label.append(rf'$\mathregular{{T}}_\mathregular{{{i+1}}}$')

        else:
            break

    for i in range(100):
        if f'radius_{i}' in param:
            quantity.append(f'radius_{i}')
            unit.append(r'$\mathregular{R}_\mathregular{J}$')
            label.append(rf'$\mathregular{{R}}_\mathregular{{{i+1}}}$')

        else:
            break

    if 'distance' in param:
        quantity.append('distance')
        unit.append('pc')
        label.append(r'$\mathregular{d}$')

    if 'mass' in param:
        quantity.append('mass')

        if object_type == 'planet':
            unit.append(r'$\mathregular{M}_\mathregular{J}$')

        elif object_type == 'star':
            unit.append(r'$\mathregular{M}_\mathregular{\odot}$')

        label.append('M')

    if 'luminosity' in param:
        quantity.append('luminosity')
        unit.append(None)
        label.append(r'$\mathregular{log}\,\mathregular{L}/\mathregular{L}_\mathregular{\odot}$')

    if 'ism_ext' in param:
        quantity.append('ism_ext')
        unit.append(None)
        label.append(r'$\mathregular{A}_\mathregular{V}$')

    if 'lognorm_ext' in param:
        quantity.append('lognorm_ext')
        unit.append(None)
        label.append(r'$\mathregular{A}_\mathregular{V}$')

    if 'powerlaw_ext' in param:
        quantity.append('powerlaw_ext')
        unit.append(None)
        label.append(r'$\mathregular{A}_\mathregular{V}$')

    return quantity, unit, label


def field_bounds_ticks(field_range):
    """
    Parameters
    ----------
    field_range : tuple(str, str), None
        Range of the discrete colorbar for the field dwarfs. The tuple should contain the lower
        and upper value ('early M', 'late M', 'early L', 'late L', 'early T', 'late T', 'early Y).
        The full range is used if set to None.

    Returns
    -------
    np.ndarray
    np.ndarray
    list(str, )
    """

    spectral_ranges = ['M0-M4', 'M5-M9', 'L0-L4', 'L5-L9', 'T0-T4', 'T5-T9', 'Y1-Y2']

    if field_range is None:
        index_start = 0
        index_end = 7

    else:
        if field_range[0] == 'early M':
            index_start = 0
        elif field_range[0] == 'late M':
            index_start = 1
        elif field_range[0] == 'early L':
            index_start = 2
        elif field_range[0] == 'late L':
            index_start = 3
        elif field_range[0] == 'early T':
            index_start = 4
        elif field_range[0] == 'late T':
            index_start = 5
        elif field_range[0] == 'early Y':
            index_start = 6

        if field_range[1] == 'early M':
            index_end = 1
        elif field_range[1] == 'late M':
            index_end = 2
        elif field_range[1] == 'early L':
            index_end = 3
        elif field_range[1] == 'late L':
            index_end = 4
        elif field_range[1] == 'early T':
            index_end = 5
        elif field_range[1] == 'late T':
            index_end = 6
        elif field_range[1] == 'early Y':
            index_end = 7

    index_range = index_end - index_start + 1

    bounds = np.linspace(index_start, index_end, index_range)
    ticks = np.linspace(index_start+0.5, index_end-0.5, index_range-1)
    labels = spectral_ranges[index_start:index_end]

    return bounds, ticks, labels


@typechecked
def remove_color_duplicates(object_names: List[str],
                            empirical_names: np.ndarray) -> List[int]:
    """"
    Function for deselecting young/low-gravity objects that will already be plotted individually
    as directly imaged objects.

    Parameters
    ----------
    object_names : list(str)
        List with names of directly imaged planets and brown dwarfs.
    empirical_names : np.ndarray
        Array with names of young/low-gravity objects.

    Returns
    -------
    list
        List with selected indices of the young/low-gravity objects.
    """

    indices = []

    for i, item in enumerate(empirical_names):
        if item == 'beta_Pic_b' and 'beta Pic b' in object_names:
            continue

        if item == 'HR8799b' and 'HR 8799 b' in object_names:
            continue

        if item == 'HR8799c' and 'HR 8799 c' in object_names:
            continue

        if item == 'HR8799d' and 'HR 8799 d' in object_names:
            continue

        if item == 'HR8799e' and 'HR 8799 e' in object_names:
            continue

        if item == 'kappa_And_B' and 'kappa And b' in object_names:
            continue

        if item == 'HD1160B' and 'HD 1160 B' in object_names:
            continue

        indices.append(i)

    return indices
