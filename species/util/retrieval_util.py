"""
Utility functions for atmospheric retrieval with ``petitRADTRANS``. This module was put together
with many contributions by Paul Mollière (MPIA).
"""

import copy

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import interp1d, CubicSpline, PchipInterpolator
from scipy.ndimage.filters import gaussian_filter
from typeguard import typechecked

from species.core import constants


@typechecked
def get_line_species() -> list:
    """
    Function to get the list of the molecular and atomic line species.

    Returns
    -------
    list
        List with the line species.
    """

    return ['CH4', 'CO', 'CO_all_iso', 'CO2', 'H2O', 'H2S', 'HCN', 'K', 'K_lor_cut', 'K_burrows',
            'NH3', 'Na', 'Na_lor_cut', 'Na_burrows', 'OH', 'PH3', 'TiO', 'VO', 'FeH',
            'H2O_main_iso', 'CH4_main_iso']


@typechecked
def pt_ret_model(temp_3: Optional[np.ndarray],
                 delta: float,
                 alpha: float,
                 tint: float,
                 press: np.ndarray,
                 metallicity: float,
                 c_o_ratio: float,
                 conv: bool = True) -> Tuple[np.ndarray, float, Optional[float]]:
    """
    Pressure-temperature profile for a self-luminous atmosphere (see Mollière et al. 2020).

    Parameters
    ----------
    temp_3 : np.ndarray, None
        Array with three temperature points that are added on top of the radiative Eddington
        structure (i.e. above tau = 0.1). The temperature nodes are connected with a spline
        interpolation and a prior is used such that t1 < t2 < t3 < t_connect. The three
        temperature points are not used if set to ``None``.
    delta : float
        Proportionality factor in tau = delta * press_cgs**alpha.
    alpha : float
        Power law index in tau = delta * press_cgs**alpha. For the tau model: use the proximity
        to the kappa_rosseland photosphere as prior.
    tint : float
        Internal temperature for the Eddington model.
    press : np.ndarray
        Pressure profile (bar).
    metallicity : float
        Metallicity [Fe/H]. Required for the ``nabla_ad`` interpolation.
    c_o_ratio : float
        Carbon-to-oxygen ratio. Required for the ``nabla_ad`` interpolation.
    conv : bool
        Enforce a convective adiabat.

    Returns
    -------
    np.ndarray
        Temperature profile (K) for ``press``.
    float
        Pressure (bar) where the optical depth is 1.
    float, None
        Pressure (bar) at the radiative-convective boundary.
    """

    # Go from bar to cgs units
    press_cgs = press*1e6

    # Calculate the optical depth
    tau = delta*press_cgs**alpha

    # Calculate the eddington temperature
    tedd = (3./4.*tint**4.*(2./3.+tau))**0.25

    # Import interpol_abundances here because it slows down importing species otherwise.
    # Importing interpol_abundances is only slow the first time, which occurs at the start
    # of the run_multinest method of AtmosphericRetrieval

    from poor_mans_nonequ_chem.poor_mans_nonequ_chem import interpol_abundances

    ab = interpol_abundances(np.full(tedd.shape[0], c_o_ratio),
                             np.full(tedd.shape[0], metallicity),
                             tedd,
                             press)

    nabla_ad = ab['nabla_ad']

    # Enforce convective adiabat
    if conv:
        # Calculate the current, radiative temperature gradient
        nab_rad = np.diff(np.log(tedd))/np.diff(np.log(press_cgs))

        # Extend to array of same length as pressure structure
        nabla_rad = np.ones_like(tedd)
        nabla_rad[0] = nab_rad[0]
        nabla_rad[-1] = nab_rad[-1]
        nabla_rad[1:-1] = (nab_rad[1:]+nab_rad[:-1])/2.

        # Where is the atmosphere convectively unstable?
        conv_index = nabla_rad > nabla_ad

        if np.argwhere(conv_index).size == 0:
            conv_press = None

        else:
            conv_bound = np.amin(np.argwhere(conv_index))
            conv_press = press[conv_bound]

        tfinal = None

        for i in range(10):
            if i == 0:
                t_take = copy.copy(tedd)
            else:
                t_take = copy.copy(tfinal)

            ab = interpol_abundances(np.full(t_take.shape[0], c_o_ratio),
                                     np.full(t_take.shape[0], metallicity),
                                     t_take,
                                     press)

            nabla_ad = ab['nabla_ad']

            # Calculate the average nabla_ad between the layers
            nabla_ad_mean = nabla_ad
            nabla_ad_mean[1:] = (nabla_ad[1:]+nabla_ad[:-1])/2.

            # What are the increments in temperature due to convection
            tnew = nabla_ad_mean[conv_index]*np.mean(np.diff(np.log(press_cgs)))

            # What is the last radiative temperature?
            tstart = np.log(t_take[~conv_index][-1])

            # Integrate and translate to temperature from log(temperature)
            tnew = np.exp(np.cumsum(tnew)+tstart)

            # Add upper radiative and lower conective part into one single array
            tfinal = copy.copy(t_take)
            tfinal[conv_index] = tnew

            if np.max(np.abs(t_take-tfinal)/t_take) < 0.01:
                # print('n_ad', 1./(1.-nabla_ad[conv_index]))
                break

    else:
        tfinal = tedd
        conv_press = None

    # Add the three temperature-point P-T description above tau = 0.1

    @typechecked
    def press_tau(tau: float) -> float:
        """
        Function to return the pressure in cgs units at a given optical depth.

        Parameters
        ----------
        tau : float
            Optical depth.

        Returns
        -------
        float
            Pressure (cgs) at optical depth ``tau``.
        """

        return (tau/delta)**(1./alpha)

    # Where is the uppermost pressure of the Eddington radiative structure?
    p_bot_spline = press_tau(0.1)

    if temp_3 is None:
        tret = tfinal

    else:
        for i_intp in range(2):
            if i_intp == 0:

                # Create the pressure coordinates for the spline support nodes at low pressure
                support_points_low = np.logspace(np.log10(press_cgs[0]), np.log10(p_bot_spline), 4)

                # Create the pressure coordinates for the spline support nodes at high pressure,
                # the corresponding temperatures for these nodes will be taken from the
                # radiative-convective solution
                support_points_high = 10.**np.arange(np.log10(p_bot_spline),
                                                     np.log10(press_cgs[-1]),
                                                     np.diff(np.log10(support_points_low))[0])

                # Combine into one support node array, don't add the p_bot_spline point twice.
                support_points = np.zeros(len(support_points_low)+len(support_points_high)-1)

                support_points[:4] = support_points_low
                support_points[4:] = support_points_high[1:]

            else:

                # Create the pressure coordinates for the spline support nodes at low pressure
                support_points_low = np.logspace(np.log10(press_cgs[0]), np.log10(p_bot_spline), 7)

                # Create the pressure coordinates for the spline support nodes at high pressure,
                # the corresponding temperatures for these nodes will be taken from the
                # radiative-convective solution
                support_points_high = np.logspace(np.log10(p_bot_spline), np.log10(press_cgs[-1]), 7)

                # Combine into one support node array, don't add the p_bot_spline point twice.
                support_points = np.zeros(len(support_points_low)+len(support_points_high)-1)
                support_points[:7] = support_points_low
                support_points[7:] = support_points_high[1:]

            # Define the temperature values at the node points.
            t_support = np.zeros_like(support_points)

            if i_intp == 0:
                tfintp = interp1d(press_cgs, tfinal)

                # The temperature at p_bot_spline (from the radiative-convective solution)
                t_support[len(support_points_low)-1] = tfintp(p_bot_spline)

                # if temp_3 is not None:
                # The temperature at pressures below p_bot_spline (free parameters)
                t_support[:len(support_points_low)-1] = temp_3

                # else:
                #     t_support[:3] = tfintp(support_points_low[:3])

                # The temperature at pressures above p_bot_spline (from the
                # radiative-convective solution)
                t_support[len(support_points_low):] = \
                    tfintp(support_points[len(support_points_low):])

            else:
                tfintp1 = interp1d(press_cgs, tret)

                t_support[:len(support_points_low)-1] = \
                    tfintp1(support_points[:len(support_points_low)-1])

                tfintp = interp1d(press_cgs, tfinal)

                # The temperature at p_bot_spline (from the radiative-convective solution)
                t_support[len(support_points_low)-1] = tfintp(p_bot_spline)

                # print('diff', t_connect_calc - tfintp(p_bot_spline))
                t_support[len(support_points_low):] = \
                    tfintp(support_points[len(support_points_low):])

            # Make the temperature spline interpolation to be returned to the user
            # tret = spline(np.log10(support_points), t_support, np.log10(press_cgs), order = 3)
            cs = PchipInterpolator(np.log10(support_points), t_support)
            tret = cs(np.log10(press_cgs))

    # Return the temperature, the pressure at tau = 1
    # The temperature at the connection point: tfintp(p_bot_spline)
    # The last two are needed for the priors on the P-T profile.
    return tret, press_tau(1.)/1e6, conv_press


@typechecked
def pt_spline_interp(knot_press: np.ndarray,
                     knot_temp: np.ndarray,
                     pressure: np.ndarray,
                     pt_smooth: float = 0.3) -> np.ndarray:
    """
    Function for interpolating the P-T nodes with a PCHIP 1-D monotonic cubic interpolation. The
    interpolated temperature is smoothed with a Gaussian kernel of width 0.3 dex in pressure
    (see Piette & Madhusudhan 2020).

    Parameters
    ----------
    knot_press : np.ndarray
        Pressure knots (bar).
    knot_temp : np.ndarray
        Temperature knots (K).
    pressure : np.ndarray
        Pressure points (bar) at which the temperatures is interpolated.
    pt_smooth : float
        Standard deviation of the Gaussian kernel that is used for smoothing the sampled
        temperature nodes of the P-T profile. The argument should be given as log10(P/bar) with the
        default value in :func:`~species.analysis.retrieval.AtmosphericRetrieval.run_multinest`
        set to 0.3 dex.

    Returns
    -------
    np.ndarray
        Interpolated, smoothed temperature points (K).
    """

    pt_interp = PchipInterpolator(np.log10(knot_press), knot_temp)

    temp_interp = pt_interp(np.log10(pressure))

    log_press = np.log10(pressure)
    log_diff = np.mean(np.diff(log_press))

    if np.std(np.diff(log_press))/log_diff > 1e-6:
        raise ValueError('Expecting equally spaced pressures in log space.')

    return gaussian_filter(temp_interp, sigma=pt_smooth/log_diff, mode='nearest')


@typechecked
def create_pt_profile(cube,
                      cube_index: Dict[str, float],
                      pt_profile: str,
                      pressure: np.ndarray,
                      knot_press: Optional[np.ndarray],
                      metallicity: float,
                      c_o_ratio: float,
                      pt_smooth: float = 0.3) -> Tuple[np.ndarray,
                                                       Optional[np.ndarray],
                                                       float,
                                                       Optional[float]]:
    """
    Function for creating the P-T profile.

    Parameters
    ----------
    cube : LP_c_double
        Unit cube.
    cube_index : dict
        Dictionary with the index of each parameter in the ``cube``.
    pt_profile : str
        The parametrization for the pressure-temperature profile ('molliere', 'free', or
        'monotonic').
    pressure : np.ndarray
        Pressure points (bar) at which the temperatures is interpolated.
    knot_press : np.ndarray, None
        Pressure knots (bar), which are required when the argument of ``pt_profile`` is either
        'free' or 'monotonic'.
    metallicity : float
        Metallicity [Fe/H].
    c_o_ratio : float
        Carbon-to-oxgen ratio.
    pt_smooth : float
        Standard deviation of the Gaussian kernel that is used for smoothing the sampled
        temperature nodes of the P-T profile. The argument should be given as log10(P/bar) with the
        default value in :func:`~species.analysis.retrieval.AtmosphericRetrieval.run_multinest`
        set to 0.3 dex.

    Returns
    -------
    np.ndarray
        Temperatures (K).
    np.ndarray, None
        Temperature at the knots (K). A None is returned if ``pt_profile`` is set to 'molliere'.
    float
        Pressure (bar) where the optical depth is 1.
    float, None
        Pressure (bar) at the radiative-convective boundary.
    """

    knot_temp = None

    if pt_profile == 'molliere':
        temp, phot_press, conv_press = pt_ret_model(np.array([cube[cube_index['t1']],
                                                              cube[cube_index['t2']],
                                                              cube[cube_index['t3']]]),
                                                    10.**cube[cube_index['log_delta']],
                                                    cube[cube_index['alpha']],
                                                    cube[cube_index['tint']],
                                                    pressure,
                                                    metallicity,
                                                    c_o_ratio)

    elif pt_profile == 'mod-molliere':
        temp, phot_press, conv_press = pt_ret_model(None,
                                                    10.**cube[cube_index['log_delta']],
                                                    cube[cube_index['alpha']],
                                                    cube[cube_index['tint']],
                                                    pressure,
                                                    metallicity,
                                                    c_o_ratio)

    elif pt_profile in ['free', 'monotonic']:
        knot_temp = []
        for i in range(15):
            knot_temp.append(cube[cube_index[f't{i}']])

        knot_temp = np.asarray(knot_temp)

        temp = pt_spline_interp(knot_press, knot_temp, pressure, pt_smooth)

    return temp, knot_temp, phot_press, conv_press


@typechecked
def make_half_pressure_better(p_base: Dict[str, float],
                              pressure: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for reducing the number of pressure layers from 1440 to ~100 (depending on the number
    of cloud species) with a refinement around the cloud decks.

    Parameters
    ----------
    p_base : dict
        Dictionary with the base of the cloud deck for all cloud species. The keys in the
        dictionary are included for example as MgSiO3(c).
    pressure : np.ndarray
        Pressures (bar) at high resolution (1440 points).

    Returns
    -------
    np.ndarray
        Pressures (bar) at lower resolution (60 points) but with a refinement around the position
        of the cloud decks.
    np.ndarray, None
        The indices of the pressures that have been selected from the input array ``pressure``.
    """

    press_plus_index = np.zeros(len(pressure)*2).reshape(len(pressure), 2)
    press_plus_index[:, 0] = pressure
    press_plus_index[:, 1] = range(len(pressure))

    press_small = press_plus_index[::24, :]
    press_plus_index = press_plus_index[::2, :]

    indexes_small = press_small[:, 0] > 0.
    indexes = press_plus_index[:, 0] > 0.

    for key, P_cloud in p_base.items():
        indexes_small = indexes_small & ((np.log10(press_small[:, 0]/P_cloud) > 0.05) |
                                         (np.log10(press_small[:, 0]/P_cloud) < -0.3))

        indexes = indexes & ((np.log10(press_plus_index[:, 0]/P_cloud) > 0.05) |
                             (np.log10(press_plus_index[:, 0]/P_cloud) < -0.3))

    press_cut = press_plus_index[~indexes, :]
    press_small_cut = press_small[indexes_small, :]

    press_out = np.zeros((len(press_cut)+len(press_small_cut))*2).reshape(
        (len(press_cut)+len(press_small_cut)), 2)

    press_out[:len(press_small_cut), :] = press_small_cut
    press_out[len(press_small_cut):, :] = press_cut

    press_out = np.sort(press_out, axis=0)

    return press_out[:, 0], press_out[:, 1].astype('int')


@typechecked
def create_abund_dict(abund_in: dict,
                      line_species: list,
                      chemistry: str,
                      pressure_grid: str = 'smaller',
                      indices: Optional[np.array] = None) -> dict:
    """
    Function to update the names in the abundance dictionary.

    Parameters
    ----------
    abund_in : dict
        Dictionary with the mass fractions.
    line_species : list
        List with the line species.
    chemistry : str
        Chemistry type ('equilibrium' or 'free').
    pressure_grid : str
        The type of pressure grid that is used for the radiative transfer. Either 'standard',
        to use 180 layers both for the atmospheric structure (e.g. when interpolating the
        abundances) and 180 layers with the radiative transfer, or 'smaller' to use 60 (instead
        of 180) with the radiative transfer, or 'clouds' to start with 1440 layers but resample
        to ~100 layers (depending on the number of cloud species) with a refinement around the
        cloud decks. For cloudless atmospheres it is recommended to use 'smaller', which runs
        faster than 'standard' and provides sufficient accuracy. For cloudy atmosphere, one can
        test with 'smaller' but it is recommended to use 'clouds' for improved accuracy fluxes.
    indices : np.ndarray, None
        Pressure indices from the adaptive refinement in a cloudy atmosphere. Only required with
        ``pressure_grid='clouds'``. Otherwise, the argument can be set to ``None``.

    Returns
    -------
    dict
        Dictionary with the updated names of the abundances.
    """

    # create a dictionary with the updated abundance names

    abund_out = {}

    if indices is not None:
        for item in line_species:
            if chemistry == 'equilibrium':
                item_replace = item.replace('_R_10', '')
                item_replace = item_replace.replace('_R_30', '')
                item_replace = item_replace.replace('_all_iso', '')
                item_replace = item_replace.replace('_main_iso', '')
                item_replace = item_replace.replace('_lor_cut', '')
                item_replace = item_replace.replace('_burrows', '')
                item_replace = item_replace.replace('_Plez', '')

                abund_out[item] = abund_in[item_replace][indices]

            elif chemistry == 'free':
                abund_out[item] = abund_in[item][indices]

        if 'Fe(c)' in abund_in:
            abund_out['Fe(c)'] = abund_in['Fe(c)'][indices]

        if 'MgSiO3(c)' in abund_in:
            abund_out['MgSiO3(c)'] = abund_in['MgSiO3(c)'][indices]

        if 'Al2O3(c)' in abund_in:
            abund_out['Al2O3(c)'] = abund_in['Al2O3(c)'][indices]

        if 'Na2S(c)' in abund_in:
            abund_out['Na2S(c)'] = abund_in['Na2S(c)'][indices]

        if 'KCL(c)' in abund_in:
            abund_out['KCL(c)'] = abund_in['KCL(c)'][indices]

        abund_out['H2'] = abund_in['H2'][indices]
        abund_out['He'] = abund_in['He'][indices]

    elif pressure_grid == 'smaller':
        for item in line_species:
            if chemistry == 'equilibrium':
                item_replace = item.replace('_R_10', '')
                item_replace = item_replace.replace('_R_30', '')
                item_replace = item_replace.replace('_all_iso', '')
                item_replace = item_replace.replace('_main_iso', '')
                item_replace = item_replace.replace('_lor_cut', '')
                item_replace = item_replace.replace('_burrows', '')
                item_replace = item_replace.replace('_Plez', '')

                abund_out[item] = abund_in[item_replace][::3]

            elif chemistry == 'free':
                abund_out[item] = abund_in[item][::3]

        if 'Fe(c)' in abund_in:
            abund_out['Fe(c)'] = abund_in['Fe(c)'][::3]

        if 'MgSiO3(c)' in abund_in:
            abund_out['MgSiO3(c)'] = abund_in['MgSiO3(c)'][::3]

        if 'Al2O3(c)' in abund_in:
            abund_out['Al2O3(c)'] = abund_in['Al2O3(c)'][::3]

        if 'Na2S(c)' in abund_in:
            abund_out['Na2S(c)'] = abund_in['Na2S(c)'][::3]

        if 'KCL(c)' in abund_in:
            abund_out['KCL(c)'] = abund_in['KCL(c)'][::3]

        abund_out['H2'] = abund_in['H2'][::3]
        abund_out['He'] = abund_in['He'][::3]

    else:
        for item in line_species:
            if chemistry == 'equilibrium':
                item_replace = item.replace('_R_10', '')
                item_replace = item_replace.replace('_R_30', '')
                item_replace = item_replace.replace('_all_iso', '')
                item_replace = item_replace.replace('_main_iso', '')
                item_replace = item_replace.replace('_lor_cut', '')
                item_replace = item_replace.replace('_burrows', '')
                item_replace = item_replace.replace('_Plez', '')

                abund_out[item] = abund_in[item_replace]

            elif chemistry == 'free':
                abund_out[item] = abund_in[item]

        if 'Fe(c)' in abund_in:
            abund_out['Fe(c)'] = abund_in['Fe(c)']

        if 'MgSiO3(c)' in abund_in:
            abund_out['MgSiO3(c)'] = abund_in['MgSiO3(c)']

        if 'Al2O3(c)' in abund_in:
            abund_out['Al2O3(c)'] = abund_in['Al2O3(c)']

        if 'Na2S(c)' in abund_in:
            abund_out['Na2S(c)'] = abund_in['Na2S(c)']

        if 'KCL(c)' in abund_in:
            abund_out['KCL(c)'] = abund_in['KCL(c)']

        abund_out['H2'] = abund_in['H2']
        abund_out['He'] = abund_in['He']

    # Correction for the nuclear spin degeneracy that was not included in the partition function
    # See Charnay et al. (2018)

    if 'FeH' in abund_out:
        abund_out['FeH'] = abund_out['FeH']/2.

    return abund_out


@typechecked
def calc_spectrum_clear(rt_object,
                        pressure: np.ndarray,
                        temperature: np.ndarray,
                        log_g: float,
                        c_o_ratio: Optional[float],
                        metallicity: Optional[float],
                        p_quench: Optional[float],
                        log_x_abund: Optional[dict],
                        chemistry: str,
                        pressure_grid: str = 'smaller',
                        contribution: bool = False) -> Tuple[np.ndarray,
                                                             np.ndarray,
                                                             Optional[np.ndarray]]:
    """
    Function to simulate an emission spectrum of a clear atmosphere. The function supports both
    equilibrium chemistry (``chemistry='equilibrium'``) and free abundances (``chemistry='free'``).

    rt_object : petitRADTRANS.radtrans.Radtrans
        Instance of ``Radtrans``.
    pressure : np.ndarray
        Array with the pressure points (bar).
    temperature : np.ndarray
        Array with the temperature points (K) corresponding to ``pressure``.
    log_g : float
        Log10 of the surface gravity (cm s-2).
    c_o_ratio : float, None
        Carbon-to-oxygen ratio.
    metallicity : float, None
        Metallicity.
    p_quench : float, None
        Quenching pressure (bar).
    log_x_abund : dict, None
        Dictionary with the log10 of the abundances. Only required when ``chemistry='free'``.
    chemistry : str
        Chemistry type (``'equilibrium'`` or ``'free'``).
    pressure_grid : str
        The type of pressure grid that is used for the radiative transfer. Either 'standard',
        to use 180 layers both for the atmospheric structure (e.g. when interpolating the
        abundances) and 180 layers with the radiative transfer, or 'smaller' to use 60 (instead
        of 180) with the radiative transfer, or 'clouds' to start with 1440 layers but resample
        to ~100 layers (depending on the number of cloud species) with a refinement around the
        cloud decks. For cloudless atmospheres it is recommended to use 'smaller', which runs
        faster than 'standard' and provides sufficient accuracy. For cloudy atmosphere, one can
        test with 'smaller' but it is recommended to use 'clouds' for improved accuracy fluxes.
    contribution : bool
        Calculate the emission contribution.

    Returns
    -------
    np.ndarray
        Wavelength (um).
    np.ndarray
        Flux (W m-2 um-1).
    np.ndarray, None
        Emission contribution.
    """

    # Import interpol_abundances here because it slows down importing species otherwise.
    # Importing interpol_abundances is only slow the first time, which occurs at the start
    # of the run_multinest method of AtmosphericRetrieval

    from poor_mans_nonequ_chem.poor_mans_nonequ_chem import interpol_abundances

    if chemistry == 'equilibrium':
        # Chemical equilibrium
        abund_in = interpol_abundances(np.full(pressure.shape, c_o_ratio),
                                       np.full(pressure.shape, metallicity),
                                       temperature,
                                       pressure,
                                       Pquench_carbon=p_quench)

        # Mean molecular weight
        mmw = abund_in['MMW']

    elif chemistry == 'free':
        # Free abundances

        # Create a dictionary with all mass fractions
        abund_in = mass_fractions(log_x_abund)

        # Mean molecular weight
        mmw = mean_molecular_weight(abund_in)

        # Create arrays of constant atmosphere abundance
        for item in abund_in:
            abund_in[item] *= np.ones_like(pressure)

        # Create an array of a constant mean molecular weight
        mmw *= np.ones_like(pressure)

    # Extract every three levels when pressure_grid is set to 'smaller'

    if pressure_grid == 'smaller':
        temperature = temperature[::3]
        pressure = pressure[::3]
        mmw = mmw[::3]

    abundances = create_abund_dict(abund_in,
                                   rt_object.line_species,
                                   chemistry,
                                   pressure_grid=pressure_grid,
                                   indices=None)

    # calculate the emission spectrum
    rt_object.calc_flux(temperature, abundances, 10.**log_g, mmw, contribution=contribution)

    # convert frequency (Hz) to wavelength (cm)
    wavel = constants.LIGHT*1e2/rt_object.freq

    # optionally return the emission contribution
    if contribution:
        contr_em = rt_object.contr_em
    else:
        contr_em = None

    # return wavelength (micron), flux (W m-2 um-1), and emission contribution
    return 1e4*wavel, 1e-7*rt_object.flux*constants.LIGHT*1e2/wavel**2., contr_em


@typechecked
def calc_spectrum_clouds(rt_object,
                         pressure: np.ndarray,
                         temperature: np.ndarray,
                         c_o_ratio: float,
                         metallicity: float,
                         p_quench: Optional[float],
                         log_x_abund: Optional[dict],
                         log_x_base: Optional[dict],
                         cloud_dict: Dict[str, Optional[float]],
                         log_g: float,
                         chemistry: str,
                         pressure_grid: str = 'smaller',
                         plotting: bool = False,
                         contribution: bool = False,
                         tau_cloud: Optional[float] = None,
                         cloud_wavel: Optional[Tuple[float, float]] = None) -> Tuple[
                             Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Function to simulate an emission spectrum of a cloudy atmosphere.

    Parameters
    ----------
    rt_object : petitRADTRANS.radtrans.Radtrans
        Instance of ``Radtrans``.
    pressure : np.ndarray
        Array with the pressure points (bar).
    temperature : np.ndarray
        Array with the temperature points (K) corresponding to ``pressure``.
    c_o_ratio : float
        Carbon-to-oxygen ratio.
    metallicity : float
        Metallicity.
    p_quench : float, None
        Quenching pressure (bar).
    log_x_abund : dict, None
        Dictionary with the log10 of the abundances. Only required when ``chemistry='free'``.
    log_x_base : dict, None
        Dictionary with the log10 of the mass fractions at the cloud base. Only required when the
        ``cloud_dict`` contains ``fsed``, ``log_kzz``, and ``sigma_lnorm``.
    cloud_dict : dict
        Dictionary with the cloud parameters. A parameter value is set to ``None`` if the parameter
        is not in use.
    log_g : float
        Log10 of the surface gravity (cm s-2).
    chemistry : str
        Chemistry type (only ``'equilibrium'`` is supported).
    pressure_grid : str
        The type of pressure grid that is used for the radiative transfer. Either 'standard',
        to use 180 layers both for the atmospheric structure (e.g. when interpolating the
        abundances) and 180 layers with the radiative transfer, or 'smaller' to use 60 (instead
        of 180) with the radiative transfer, or 'clouds' to start with 1440 layers but resample
        to ~100 layers (depending on the number of cloud species) with a refinement around the
        cloud decks. For cloudless atmospheres it is recommended to use 'smaller', which runs
        faster than 'standard' and provides sufficient accuracy. For cloudy atmosphere, one can
        test with 'smaller' but it is recommended to use 'clouds' for improved accuracy fluxes.
    plotting : bool
        Create plots.
    contribution : bool
        Calculate the emission contribution.
    tau_cloud : float, None
        Total cloud optical that will be used for scaling the cloud mass fractions. The mass
        fractions will not be scaled if the parameter is set to ``None``.
    cloud_wavel : tuple(float, float), None
        Tuple with the wavelength range (um) that is used for calculating the median optical
        depth of the clouds at the gas-only photosphere and then scaling the cloud optical
        depth to the value of ``log_tau_cloud``. The range of ``cloud_wavel`` should be
        encompassed by the range of ``wavel_range``.  The full wavelength range (i.e.
        ``wavel_range``) is used if the argument is set to ``None``.

    Returns
    -------
    np.ndarray, None
        Wavelength (um).
    np.ndarray, None
        Flux (W m-2 um-1).
    np.ndarray, None
        Emission contribution.
    """

    if chemistry == 'equilibrium':
        # Import interpol_abundances here because it slows down importing species otherwise.
        # Importing interpol_abundances is only slow the first time, which occurs at the start
        # of the run_multinest method of AtmosphericRetrieval

        from poor_mans_nonequ_chem.poor_mans_nonequ_chem import interpol_abundances

        # Interpolate the abundances, following chemical equilibrium
        abund_in = interpol_abundances(np.full(pressure.shape, c_o_ratio),
                                       np.full(pressure.shape, metallicity),
                                       temperature,
                                       pressure,
                                       Pquench_carbon=p_quench)

        # Extract the mean molecular weight
        mmw = abund_in['MMW']

    elif chemistry == 'free':
        # Free abundances

        # Create a dictionary with all mass fractions
        abund_in = mass_fractions(log_x_abund)

        # Mean molecular weight
        mmw = mean_molecular_weight(abund_in)

        # Create arrays of constant atmosphere abundance
        for item in abund_in:
            abund_in[item] *= np.ones_like(pressure)

        # Create an array of a constant mean molecular weight
        mmw *= np.ones_like(pressure)

    if log_x_base is not None:
        p_base = {}

        for item in log_x_base:
            p_base_item = find_cloud_deck(item,
                                          pressure,
                                          temperature,
                                          metallicity,
                                          c_o_ratio,
                                          mmw=np.mean(mmw),
                                          plotting=plotting)

            abund_in[f'{item}(c)'] = np.zeros_like(temperature)

            abund_in[f'{item}(c)'][pressure < p_base_item] = 10.**log_x_base[item] * \
                (pressure[pressure <= p_base_item] / p_base_item)**cloud_dict['fsed']

            p_base[f'{item}(c)'] = p_base_item

    # Adaptive pressure refinement around the cloud base
    if pressure_grid == 'clouds':
        _, indices = make_half_pressure_better(p_base, pressure)
    else:
        indices = None

    abundances = create_abund_dict(abund_in,
                                   rt_object.line_species,
                                   chemistry,
                                   pressure_grid=pressure_grid,
                                   indices=indices)

    # Create dictionary with sedimentation parameters
    # Use the same value for all cloud species

    fseds = {}
    for item in rt_object.cloud_species:
        # item has the form of e.g. MgSiO3(c)
        fseds[item] = cloud_dict['fsed']

    # Create an array with a constant eddy diffusion coefficient (cm2 s-1)

    if 'log_kzz' in cloud_dict:
        Kzz_use = np.full(pressure.shape, 10.**cloud_dict['log_kzz'])
    else:
        Kzz_use = None

    # Adjust number of atmospheric levels

    if pressure_grid == 'smaller':
        temperature = temperature[::3]
        pressure = pressure[::3]
        mmw = mmw[::3]

        if 'log_kzz' in cloud_dict:
            Kzz_use = Kzz_use[::3]

    elif pressure_grid == 'clouds':
        temperature = temperature[indices]
        pressure = pressure[indices]
        mmw = mmw[indices]

        if 'log_kzz' in cloud_dict:
            Kzz_use = Kzz_use[indices]

    # Optionally plot the cloud properties

    if plotting and Kzz_use is not None and \
            (rt_object.wlen_bords_micron[0] != 0.5 and rt_object.wlen_bords_micron[1] != 30.):
        if 'CO_all_iso' in abundances:
            plt.plot(abundances['CO_all_iso'], pressure, label='CO')
        if 'CH4' in abundances:
            plt.plot(abundances['CH4'], pressure, label='CH4')
        if 'H2O' in abundances:
            plt.plot(abundances['H2O'], pressure, label='H2O')
        plt.xlim(1e-10, 1.)
        plt.ylim(pressure[-1], pressure[0])
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Mass fraction')
        plt.ylabel('Pressure (bar)')
        if p_quench is not None:
            plt.axhline(p_quench, ls='--', color='black')
        plt.legend(loc='best')
        plt.savefig('abundances.pdf', bbox_inches='tight')
        plt.clf()

        plt.plot(temperature, pressure, 'o', ls='none', ms=2.)

        for item in log_x_base:
            plt.axhline(p_base[f'{item}(c)'], label=f'Cloud deck {item}', ls='--', color='black')

        plt.yscale('log')
        plt.ylim(1e3, 1e-6)
        plt.xlim(0., 4000.)
        plt.savefig('pt_cloud_deck.pdf', bbox_inches='tight')
        plt.clf()

        for item in log_x_base:
            plt.plot(abundances[f'{item}(c)'], pressure)
            plt.axhline(p_base[f'{item}(c)'])
            plt.yscale('log')
            if np.count_nonzero(abundances[f'{item}(c)']) > 0:
                plt.xscale('log')
            plt.ylim(1e3, 1e-6)
            plt.xlim(1e-10, 1.)
            log_x_base_item = log_x_base[item]
            fsed = cloud_dict['fsed']
            log_kzz = cloud_dict['log_kzz']
            plt.title(f'fsed = {fsed:.2f}, log(Kzz) = {log_kzz:.2f}, X_b = {log_x_base_item:.2f}')
            plt.savefig(f'{item.lower()}_clouds.pdf', bbox_inches='tight')
            plt.clf()

    # Turn clouds off
    # abundances['MgSiO3(c)'] = np.zeros_like(pressure)
    # abundances['Fe(c)'] = np.zeros_like(pressure)

    # Reinitiate the pressure layers after make_half_pressure_better
    if pressure_grid == 'clouds':
        rt_object.setup_opa_structure(pressure)

    # Width of cloud particle distribution
    if 'sigma_lnorm' in cloud_dict:
        sigma_lnorm = cloud_dict['sigma_lnorm']
    else:
        sigma_lnorm = None

    # Calculate the emission spectrum
    # TODO Remove try-except after merged PR in pRT repo

    try:
        rt_object.calc_flux(temperature,
                            abundances,
                            10.**log_g,
                            mmw,
                            sigma_lnorm=sigma_lnorm,
                            Kzz=Kzz_use,
                            fsed=fseds,
                            radius=None,
                            contribution=contribution,
                            gray_opacity=None,
                            Pcloud=None,
                            kappa_zero=None,
                            gamma_scat=None,
                            add_cloud_scat_as_abs=False,
                            hack_cloud_photospheric_tau=tau_cloud,
                            cloud_wlen=cloud_wavel,
                            new_simple_cloud_params=cloud_dict)

    except TypeError:
        rt_object.calc_flux(temperature,
                            abundances,
                            10.**log_g,
                            mmw,
                            sigma_lnorm=sigma_lnorm,
                            Kzz=Kzz_use,
                            fsed=fseds,
                            radius=None,
                            contribution=contribution,
                            gray_opacity=None,
                            Pcloud=None,
                            kappa_zero=None,
                            gamma_scat=None,
                            add_cloud_scat_as_abs=False,
                            hack_cloud_photospheric_tau=tau_cloud,
                            new_simple_cloud_params=cloud_dict)

    if hasattr(rt_object, 'scaling_physicality') and rt_object.scaling_physicality > 1.:
        # cloud_scaling_factor > 2 * (fsed + 1)
        # Set to None such that -inf will be returned as ln_like
        wavel = None
        f_lambda = None
        contr_em = None

    else:
        wavel = 1e6*constants.LIGHT/rt_object.freq  # (um)

        # (erg s-1 cm-2 Hz-1) -> (erg s-1 m-2 Hz-1)
        f_lambda = 1e4*rt_object.flux

        # (erg s-1 m-2 Hz-1) -> (erg s-1 m-2 m-1)
        f_lambda *= constants.LIGHT/(1e-6*wavel)**2.

        # (erg s-1 m-2 m-1) -> (erg s-1 m-2 um-1)
        f_lambda *= 1e-6

        # (erg s-1 m-2 um-1) -> (W m-2 um-1)
        f_lambda *= 1e-7

        # Optionally return the emission contribution
        if contribution:
            contr_em = rt_object.contr_em
        else:
            contr_em = None

    if plotting and Kzz_use is None:
        scat_opa = rt_object.ret_test_cloud_scat_plus_abs - rt_object.ret_test_cloud_abs
        plt.plot(wavel, rt_object.ret_test_cloud_scat_plus_abs[:, 0], label='Total opacity')
        plt.plot(wavel, rt_object.ret_test_cloud_abs[:, 0], label='Absorption opacity')
        plt.plot(wavel, scat_opa[:, 0], label='Scattering opacity')
        plt.xlabel(r'Wavelength ($\mu$m)')
        plt.ylabel('Opacity at smallest pressure')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.savefig('cloud_opacity.pdf', bbox_inches='tight')
        plt.clf()

    return wavel, f_lambda, contr_em


@typechecked
def mass_fractions(log_x_abund: dict) -> dict:
    """
    Function to return a dictionary with the mass fractions of all species.

    Parameters
    ----------
    log_x_abund : dict
        Dictionary with the log10 of the mass fractions of metals.

    Returns
    -------
    dict
        Dictionary with the mass fractions of all species.
    """

    # initiate abundance dictionary
    abund = {}

    # initiate the total mass fraction of the metals
    metal_sum = 0.

    for item in log_x_abund:
        # add the mass fraction to the dictionary
        abund[item] = 10.**log_x_abund[item]

        # update the total mass fraction of the metals
        metal_sum += abund[item]

    # mass fraction of H2 and He
    ab_h2_he = 1. - metal_sum

    # add H2 and He mass fraction to the dictionary
    abund['H2'] = ab_h2_he*0.75
    abund['He'] = ab_h2_he*0.25

    return abund


@typechecked
def calc_metal_ratio(log_x_abund: Dict[str, float]) -> Tuple[float, float, float]:
    """
    Function for calculating [C/H], [O/H], and C/O for a given set of abundances.

    Parameters
    ----------
    log_x_abund : dict
        Dictionary with the log10 mass fractions.

    Returns
    -------
    float
        Carbon-to-hydrogen ratio, relative to solar.
    float
        Oxygen-to-hydrogen ratio, relative to solar.
    float
        Carbon-to-oxygen ratio.
    """

    # Solar C/H from Asplund et al. (2009)
    c_h_solar = 10.**(8.43-12.)

    # Solar O/H from Asplund et al. (2009)
    o_h_solar = 10.**(8.69-12.)

    # Get the atomic masses
    masses = atomic_masses()

    # Create a dictionary with all mass fractions
    abund = mass_fractions(log_x_abund)

    # Calculate the mean molecular weight from the input mass fractions
    mmw = mean_molecular_weight(abund)

    # Initiate the C, H, and O abundance
    c_abund = 0.
    o_abund = 0.
    h_abund = 0.

    # Calculate the total C abundance

    if 'CO' in abund:
        c_abund += abund['CO'] * mmw/masses['CO']

    if 'CO_all_iso' in abund:
        c_abund += abund['CO_all_iso'] * mmw/masses['CO']

    if 'CO2' in abund:
        c_abund += abund['CO2'] * mmw/masses['CO2']

    if 'CO2_main_iso' in abund:
        c_abund += abund['CO2_main_iso'] * mmw/masses['CO2']

    if 'CH4' in abund:
        c_abund += abund['CH4'] * mmw/masses['CH4']

    if 'CH4_main_iso' in abund:
        c_abund += abund['CH4_main_iso'] * mmw/masses['CH4']

    # Calculate the total O abundance

    if 'CO' in abund:
        o_abund += abund['CO'] * mmw/masses['CO']

    if 'CO_all_iso' in abund:
        o_abund += abund['CO_all_iso'] * mmw/masses['CO']

    if 'CO2' in abund:
        o_abund += 2. * abund['CO2'] * mmw/masses['CO2']

    if 'CO2_main_iso' in abund:
        o_abund += 2. * abund['CO2_main_iso'] * mmw/masses['CO2']

    if 'H2O' in abund:
        o_abund += abund['H2O'] * mmw/masses['H2O']

    if 'H2O_main_iso' in abund:
        o_abund += abund['H2O_main_iso'] * mmw/masses['H2O']

    # Calculate the total H abundance

    h_abund += 2. * abund['H2'] * mmw/masses['H2']

    if 'CH4' in abund:
        h_abund += 4. * abund['CH4'] * mmw/masses['CH4']

    if 'CH4_main_iso' in abund:
        h_abund += 4. * abund['CH4_main_iso'] * mmw/masses['CH4']

    if 'H2O' in abund:
        h_abund += 2. * abund['H2O'] * mmw/masses['H2O']

    if 'H2O_main_iso' in abund:
        h_abund += 2. * abund['H2O_main_iso'] * mmw/masses['H2O']

    if 'NH3' in abund:
        h_abund += 3. * abund['NH3'] * mmw/masses['NH3']

    if 'NH3_main_iso' in abund:
        h_abund += 3. * abund['NH3_main_iso'] * mmw/masses['NH3']

    if 'H2S' in abund:
        h_abund += 2. * abund['H2S'] * mmw/masses['H2S']

    if 'H2S_main_iso' in abund:
        h_abund += 2. * abund['H2S_main_iso'] * mmw/masses['H2S']

    return np.log10(c_abund/h_abund/c_h_solar), \
        np.log10(o_abund/h_abund/o_h_solar), c_abund/o_abund


@typechecked
def mean_molecular_weight(abundances: dict) -> float:
    """
    Function to calculate the mean molecular weight from the abundances.

    Parameters
    ----------
    abundances : dict
        Dictionary with the mass fraction of each species.

    Returns
    -------
    float
        Mean molecular weight in atomic mass units.
    """

    masses = atomic_masses()

    mmw = 0.

    for key in abundances:
        if key == 'CO_all_iso':
            mmw += abundances[key]/masses['CO']

        elif key in ['Na_lor_cut', 'Na_burrows']:
            mmw += abundances[key]/masses['Na']

        elif key in ['K_lor_cut', 'K_burrows']:
            mmw += abundances[key]/masses['K']

        elif key == 'CH4_main_iso':
            mmw += abundances[key]/masses['CH4']

        elif key == 'H2O_main_iso':
            mmw += abundances[key]/masses['H2O']

        else:
            mmw += abundances[key]/masses[key]

    return 1./mmw


@typechecked
def potassium_abundance(log_x_abund: dict) -> float:
    """
    Function to calculate the mass fraction of potassium at a solar ratio of the sodium and
    potassium abundances.

    Parameters
    ----------
    log_x_abund : dict
        Dictionary with the log10 of the mass fractions.

    Returns
    -------
    float
        Log10 of the mass fraction of potassium.
    """

    # solar volume mixing ratios of Na and K (Asplund et al. 2009)
    n_na_solar = 1.60008694353205e-06
    n_k_solar = 9.86605611925677e-08

    # get the atomic masses
    masses = atomic_masses()

    # create a dictionary with all mass fractions
    x_abund = mass_fractions(log_x_abund)

    # calculate the mean molecular weight from the input mass fractions
    mmw = mean_molecular_weight(x_abund)

    # volume mixing ratio of sodium
    if 'Na' in log_x_abund:
        n_na_abund = x_abund['Na'] * mmw/masses['Na']

    elif 'Na_lor_cut' in log_x_abund:
        n_na_abund = x_abund['Na_lor_cut'] * mmw/masses['Na']

    elif 'Na_burrows' in log_x_abund:
        n_na_abund = x_abund['Na_burrows'] * mmw/masses['Na']

    # volume mixing ratio of potassium
    n_k_abund = n_na_abund * n_k_solar/n_na_solar

    return np.log10(n_k_abund * masses['K']/mmw)


@typechecked
def log_x_cloud_base(c_o_ratio: float,
                     metallicity: float,
                     cloud_fractions: dict) -> dict:
    """
    Function for returning a dictionary with the log10 mass fractions at the cloud base.

    Parameters
    ----------
    c_o_ratio : float
        C/O ratio.
    metallicity : float
        Metallicity, [Fe/H].
    cloud_fractions : dict
        Dictionary with the log10 mass fractions at the cloud base, relative to the maximum values
        allowed from elemental abundances. The dictionary keys are the cloud species without the
        structure and shape index (e.g. Na2S(c) instead of Na2S(c)_cd).

    Returns
    -------
    dict
        Dictionary with the log10 mass fractions at the cloud base. Compared to the keys of
        ``cloud_fractions``, the keys in the returned dictionary are provided without ``(c)``
        (e.g. Na2S instead of Na2S(c)).
    """

    log_x_base = {}

    for item in cloud_fractions:
        # Mass fraction
        x_cloud = cloud_mass_fraction(f'{item[:-3]}', metallicity, c_o_ratio)

        # Log10 of the mass fraction at the cloud base
        log_x_base[f'{item[:-3]}'] = np.log10(10.**cloud_fractions[item]*x_cloud)

    return log_x_base


@typechecked
def solar_mixing_ratios() -> dict:
    """
    Function which returns the volume mixing ratios for solar elemental abundances (i.e.
    [Fe/H] = 0); adopted from Asplund et al. (2009).

    Returns
    -------
    dict
        Dictionary with the solar number fractions (i.e. volume mixing ratios).
    """

    n_fracs = {}
    n_fracs['H'] = 0.9207539305
    n_fracs['He'] = 0.0783688694
    n_fracs['C'] = 0.0002478241
    n_fracs['N'] = 6.22506056949881e-05
    n_fracs['O'] = 0.0004509658
    n_fracs['Na'] = 1.60008694353205e-06
    n_fracs['Mg'] = 3.66558742055362e-05
    n_fracs['Al'] = 2.595e-06
    n_fracs['Si'] = 2.9795e-05
    n_fracs['P'] = 2.36670201997668e-07
    n_fracs['S'] = 1.2137900734604e-05
    n_fracs['Cl'] = 2.91167958499589e-07
    n_fracs['K'] = 9.86605611925677e-08
    n_fracs['Ca'] = 2.01439011429255e-06
    n_fracs['Ti'] = 8.20622804366359e-08
    n_fracs['V'] = 7.83688694089992e-09
    n_fracs['Fe'] = 2.91167958499589e-05
    n_fracs['Ni'] = 1.52807116806281e-06

    return n_fracs


@typechecked
def atomic_masses() -> dict:
    """
    Function which returns the atomic and molecular masses.

    Returns
    -------
    dict
        Dictionary with the atomic and molecular masses.
    """

    masses = {}

    # Atoms
    masses['H'] = 1.
    masses['He'] = 4.
    masses['C'] = 12.
    masses['N'] = 14.
    masses['O'] = 16.
    masses['Na'] = 23.
    masses['Na_lor_cur'] = 23.
    masses['Na_burrows'] = 23.
    masses['Mg'] = 24.3
    masses['Al'] = 27.
    masses['Si'] = 28.
    masses['P'] = 31.
    masses['S'] = 32.
    masses['Cl'] = 35.45
    masses['K'] = 39.1
    masses['K_lor_cut'] = 39.1
    masses['K_burrows'] = 39.1
    masses['Ca'] = 40.
    masses['Ti'] = 47.9
    masses['V'] = 51.
    masses['Fe'] = 55.8
    masses['Ni'] = 58.7

    # Molecules
    masses['H2'] = 2.
    masses['H2O'] = 18.
    masses['H2O_main_iso'] = 18.
    masses['CH4'] = 16.
    masses['CH4_main_iso'] = 16.
    masses['CO2'] = 44.
    masses['CO2_main_iso'] = 44.
    masses['CO'] = 28.
    masses['CO_all_iso'] = 28.
    masses['NH3'] = 17.
    masses['NH3_main_iso'] = 17.
    masses['HCN'] = 27.
    masses['C2H2,acetylene'] = 26.
    masses['PH3'] = 34.
    masses['PH3_main_iso'] = 34.
    masses['H2S'] = 34.
    masses['H2S_main_iso'] = 34.
    masses['VO'] = 67.
    masses['TiO'] = 64.
    masses['TiO_all_iso_Plez'] = 64.
    masses['FeH'] = 57.
    masses['FeH_main_iso'] = 57.
    masses['OH'] = 17.

    return masses


@typechecked
def cloud_mass_fraction(composition: str,
                        metallicity: float,
                        c_o_ratio: float) -> float:
    """
    Function to calculate the mass fraction for a cloud species.

    Parameters
    ----------
    composition : str
        Cloud composition ('Fe', 'MgSiO3', 'Al2O3', 'Na2S', or 'KCL').
    metallicity : float
        Metallicity [Fe/H].
    c_o_ratio : float
        Carbon-to-oxygen ratio.

    Returns
    -------
    float
        Mass fraction.
    """

    # Solar fractional number densities (i.e. volume mixing ratios; VMR)
    nfracs = solar_mixing_ratios()

    # Atomic masses
    masses = atomic_masses()

    # Make a copy of the dictionary with the solar number densities
    nfracs_use = copy.copy(nfracs)

    # Scale the solar number densities by the [Fe/H], except H and He
    for item in nfracs:
        if item != 'H' and item != 'He':
            nfracs_use[item] = nfracs[item]*10.**metallicity

    # Adjust the VMR of O with the C/O ratio
    nfracs_use['O'] = nfracs_use['C']/c_o_ratio

    if composition == 'Fe':
        nfrac_cloud = nfracs_use['Fe']
        mass_cloud = masses['Fe']

    elif composition == 'MgSiO3':
        nfrac_cloud = np.min([nfracs_use['Mg'], nfracs_use['Si'], nfracs_use['O']/3.])
        mass_cloud = masses['Mg'] + masses['Si'] + 3. * masses['O']

    elif composition == 'Al2O3':
        nfrac_cloud = np.min([nfracs_use['Al']/2., nfracs_use['O']/3.])
        mass_cloud = 2. * masses['Al'] + 3. * masses['O']

    elif composition == 'Na2S':
        nfrac_cloud = np.min([nfracs_use['Na']/2., nfracs_use['S']])
        mass_cloud = 2. * masses['Na'] + masses['S']

    elif composition == 'KCL':
        nfrac_cloud = np.min([nfracs_use['K'], nfracs_use['Cl']])
        mass_cloud = masses['K'] + masses['Cl']

    # Cloud mass fraction
    x_cloud = mass_cloud*nfrac_cloud

    mass_norm = 0.
    for item in nfracs_use:
        # Sum up the mass fractions of all species
        mass_norm += masses[item]*nfracs_use[item]

    # Normalize the cloud mass fraction by the total mass fraction
    return x_cloud/mass_norm


@typechecked
def find_cloud_deck(composition: str,
                    press: np.ndarray,
                    temp: np.ndarray,
                    metallicity: float,
                    c_o_ratio: float,
                    mmw: float = 2.33,
                    plotting: bool = False) -> float:
    """
    Function to find the base of the cloud deck by intersecting the P-T profile with the
    saturation vapor pressure.

    Parameters
    ----------
    composition : str
        Cloud composition ('Fe', 'MgSiO3', 'Al2O3', 'Na2S', or 'KCL').
    press : np.ndarray
        Pressures (bar).
    temp : np.ndarray
        Temperatures (K).
    metallicity : float
        Metallicity [Fe/H].
    c_o_ratio : float
        Carbon-to-oxygen ratio.
    mmw : float
        Mean molecular weight.
    plotting : bool
        Create a plot.

    Returns
    -------
    float
        Pressure (bar) at the base of the cloud deck.
    """

    if composition == 'Fe':
        Pc, Tc = return_T_cond_Fe_comb(metallicity, c_o_ratio, mmw)

    elif composition == 'MgSiO3':
        Pc, Tc = return_T_cond_MgSiO3(metallicity, c_o_ratio, mmw)

    elif composition == 'Al2O3':
        Pc, Tc = return_T_cond_Al2O3(metallicity, c_o_ratio, mmw)

    elif composition == 'Na2S':
        Pc, Tc = return_T_cond_Na2S(metallicity, c_o_ratio, mmw)

    elif composition == 'KCL':
        Pc, Tc = return_T_cond_KCl(metallicity, c_o_ratio, mmw)

    else:
        raise ValueError(f'The \'{composition}\' composition is not supported by find_cloud_deck.')

    index = (Pc > 1e-8) & (Pc < 1e5)
    Pc, Tc = Pc[index], Tc[index]

    tcond_p = interp1d(Pc, Tc)
    Tcond_on_input_grid = tcond_p(press)

    Tdiff = Tcond_on_input_grid - temp
    diff_vec = Tdiff[1:]*Tdiff[:-1]
    ind_cdf = (diff_vec < 0.)

    if len(diff_vec[ind_cdf]) > 0:
        P_clouds = (press[1:]+press[:-1])[ind_cdf]/2.
        P_cloud = float(P_clouds[-1])

    else:
        P_cloud = 1e-8

    if plotting:
        plt.plot(temp, press)
        plt.plot(Tcond_on_input_grid, press)
        plt.axhline(P_cloud, color='red', linestyle='--')
        plt.yscale('log')
        plt.xlim(0., 3000.)
        plt.ylim(1e2, 1e-6)
        plt.savefig(f'{composition.lower()}_clouds_cdf.pdf', bbox_inches='tight')
        plt.clf()

    return P_cloud


@typechecked
def scale_cloud_abund(params: Dict[str, float],
                      rt_object,
                      pressure: np.ndarray,
                      temperature: np.ndarray,
                      mmw: np.ndarray,
                      chemistry: str,
                      abund_in: Dict[str, np.ndarray],
                      composition: str,
                      tau_cloud: float,
                      pressure_grid: str) -> float:
    """
    Function to scale the mass fraction of a cloud species to the requested optical depth.

    Parameters
    ----------
    params : dict
        Dictionary with the model parameters.
    rt_object : petitRADTRANS.radtrans.Radtrans
        Instance of ``Radtrans``.
    pressure : np.ndarray
        Array with the pressure points (bar).
    temperature : np.ndarray
        Array with the temperature points (K) corresponding to ``pressure``.
    mmw : np.ndarray
        Array with the mean molecular weights corresponding to ``pressure``.
    chemistry : str
        Chemistry type (only ``'equilibrium'`` is supported).
    abund_in : dict
        Dictionary with arrays that contain the pressure-dependent, equilibrium mass fractions
        of the line species.
    composition : sr
        Cloud composition ('Fe(c)', 'MgSiO3(c)', 'Al2O3(c)', 'Na2S(c)', 'KCl(c)').
    tau_cloud : float
        Optical depth of the clouds. The returned mass fraction is scaled such that the optical
        depth at the shortest wavelength is equal to ``tau_cloud``.
    pressure_grid : str
        The type of pressure grid that is used for the radiative transfer. Either 'standard',
        to use 180 layers both for the atmospheric structure (e.g. when interpolating the
        abundances) and 180 layers with the radiative transfer, or 'smaller' to use 60 (instead
        of 180) with the radiative transfer, or 'clouds' to start with 1440 layers but resample
        to ~100 layers (depending on the number of cloud species) with a refinement around the
        cloud decks. For cloudless atmospheres it is recommended to use 'smaller', which runs
        faster than 'standard' and provides sufficient accuracy. For cloudy atmosphere, one can
        test with 'smaller' but it is recommended to use 'clouds' for improved accuracy fluxes.

    Returns
    -------
    float
        Mass fraction relative to the maximum value allowed from elemental abundances. The value
        has been scaled to the requested optical depth ``tau_cloud`` (at the shortest wavelength).
    """

    # Dictionary with the requested cloud composition and setting the log10 of the mass fraction
    # (relative to the maximum value allowed from elemental abundances) equal to zero
    cloud_fractions = {composition: 0.}

    # Create a dictionary with the log10 of the mass fraction at the cloud base
    log_x_base = log_x_cloud_base(params['c_o_ratio'],
                                  params['metallicity'],
                                  cloud_fractions)

    # Get the pressure (bar) of the cloud base
    p_base = find_cloud_deck(composition[:-3],
                             pressure,
                             temperature,
                             params['metallicity'],
                             params['c_o_ratio'],
                             mmw=np.mean(mmw),
                             plotting=False)

    # Initialize the cloud abundance in the dictionary with mass fractions
    abund_in[composition] = np.zeros_like(temperature)

    # Set the cloud abundances by scaling from the base with the f_sed parameter
    abund_in[composition][pressure < p_base] = 10.**log_x_base[composition[:-3]] * \
        (pressure[pressure <= p_base] / p_base)**params['fsed']

    # Adaptive pressure refinement around the cloud base
    if pressure_grid == 'clouds':
        _, indices = make_half_pressure_better({composition: p_base}, pressure)
    else:
        indices = None

    # Update the abundance dictionary
    abundances = create_abund_dict(abund_in,
                                   rt_object.line_species,
                                   chemistry,
                                   pressure_grid=pressure_grid,
                                   indices=indices)

    # Interpolate the line opacities to the temperature structure

    if pressure_grid == 'standard':
        rt_object.interpolate_species_opa(temperature)

        mmw_select = mmw.copy()

        if 'log_kzz' in params:
            kzz_select = np.full(pressure.size, 10.**params['log_kzz'])
        else:
            # Backward compatibility
            kzz_select = np.full(pressure.size, 10.**params['kzz'])

    elif pressure_grid == 'smaller':
        rt_object.interpolate_species_opa(temperature[::3])

        mmw_select = mmw[::3]

        if 'log_kzz' in params:
            kzz_select = np.full(pressure[::3].size, 10.**params['log_kzz'])
        else:
            # Backward compatibility
            kzz_select = np.full(pressure[::3].size, 10.**params['kzz'])

    elif pressure_grid == 'clouds':
        # Reinitiate the pressure structure after make_half_pressure_better
        rt_object.setup_opa_structure(pressure[indices])
        rt_object.interpolate_species_opa(temperature[indices])

        mmw_select = mmw[indices]

        if 'log_kzz' in params:
            kzz_select = np.full(pressure[indices].size, 10.**params['log_kzz'])
        else:
            # Backward compatibility
            kzz_select = np.full(pressure[indices].size, 10.**params['kzz'])

    # Set the continuum opacities to zero because calc_cloud_opacity adds to existing opacities
    rt_object.continuum_opa = np.zeros_like(rt_object.continuum_opa)
    rt_object.continuum_opa_scat = np.zeros_like(rt_object.continuum_opa_scat)
    rt_object.continuum_opa_scat_emis = np.zeros_like(rt_object.continuum_opa_scat_emis)

    # Calculate the cloud opacities for the defined atmospheric structure
    rt_object.calc_cloud_opacity(abundances,
                                 mmw_select,
                                 10.**params['logg'],
                                 params['sigma_lnorm'],
                                 fsed=params['fsed'],
                                 Kzz=kzz_select,
                                 radius=None,
                                 add_cloud_scat_as_abs=False)

    # Calculate the cloud optical depth and set the tau_cloud attribute
    rt_object.calc_tau_cloud(10.**params['logg'])

    # Extract the wavelength-averaged optical depth at the largest pressure
    tau_current = np.mean(rt_object.tau_cloud[0, :, 0, -1])

    # Set the continuum opacities again to zero
    rt_object.continuum_opa = np.zeros_like(rt_object.continuum_opa)
    rt_object.continuum_opa_scat = np.zeros_like(rt_object.continuum_opa_scat)
    rt_object.continuum_opa_scat_emis = np.zeros_like(rt_object.continuum_opa_scat_emis)

    if tau_current > 0.:
        # Scale the mass fraction
        log_x_scaled = np.log10(tau_cloud/tau_current)
    else:
        log_x_scaled = 100.

    return log_x_scaled


@typechecked
def cube_to_dict(cube,
                 cube_index: Dict[str, float]) -> Dict[str, float]:
    """
    Function to convert the parameter cube into a dictionary.

    Parameters
    ----------
    cube : LP_c_double
        Cube with the parameters.
    cube_index : dict
        Dictionary with the index of each parameter in the ``cube``.

    Returns
    -------
    dict
        Dictionary with the parameters.
    """

    params = {}

    for key, value in cube_index.items():
        params[key] = cube[value]

    return params


@typechecked
def list_to_dict(param_list: List[str],
                 sample_val: np.ndarray) -> Dict[str, float]:
    """
    Function to convert the parameter cube into a dictionary.

    Parameters
    ----------
    param_list : list(str)
        List with the parameter labels.
    sample_val : np.ndarray
        Array with the parameter values, in the same order as ``param_list``.

    Returns
    -------
    dict
        Dictionary with the parameters.
    """

    sample_dict = {}

    for item in param_list:
        sample_dict[item] = sample_val[param_list.index(item)]

    return sample_dict


@typechecked
def return_T_cond_Fe(FeH: float,
                     CO: float,
                     MMW: float = 2.33) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for calculating the saturation pressure for solid Fe.

    Parameters
    ----------
    FeH : float
        Metallicity.
    CO : float
        Carbon-to-oxygen ratio.
    MMW : float
        Mean molecular weight.

    Returns
    -------
    np.ndarray
        Saturation pressure (bar).
    np.ndarray
        Temperature (K).
    """

    masses = atomic_masses()

    T = np.linspace(100., 10000., 1000)

    # Taken from Ackerman & Marley (2001)
    # including erratum (P_vap is in bar, not cgs!)
    P_vap = lambda x: np.exp(15.71 - 47664./x)

    XFe = cloud_mass_fraction('Fe', FeH, CO)

    return P_vap(T)/(XFe*MMW/masses['Fe']), T


@typechecked
def return_T_cond_Fe_l(FeH: float,
                       CO: float,
                       MMW: float = 2.33) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for calculating the saturation pressure for liquid Fe.

    Parameters
    ----------
    FeH : float
        Metallicity.
    CO : float
        Carbon-to-oxygen ratio.
    MMW : float
        Mean molecular weight.

    Returns
    -------
    np.ndarray
        Saturation pressure (bar).
    np.ndarray
        Temperature (K).
    """

    masses = atomic_masses()

    T = np.linspace(100., 10000., 1000)

    # Taken from Ackerman & Marley (2001)
    # including erratum (P_vap is in bar, not cgs!)
    P_vap = lambda x: np.exp(9.86 - 37120./x)

    XFe = cloud_mass_fraction('Fe', FeH, CO)

    return P_vap(T)/(XFe*MMW/masses['Fe']), T


@typechecked
def return_T_cond_Fe_comb(FeH: float,
                          CO: float,
                          MMW: float = 2.33) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for calculating the saturation pressure for Fe.

    Parameters
    ----------
    FeH : float
        Metallicity.
    CO : float
        Carbon-to-oxygen ratio.
    MMW : float
        Mean molecular weight.

    Returns
    -------
    np.ndarray
        Saturation pressure (bar).
    np.ndarray
        Temperature (K).
    """

    P1, T1 = return_T_cond_Fe(FeH, CO, MMW)
    P2, T2 = return_T_cond_Fe_l(FeH, CO, MMW)

    retP = np.zeros_like(P1)
    index = P1 < P2
    retP[index] = P1[index]
    retP[~index] = P2[~index]

    return retP, T2


@typechecked
def return_T_cond_MgSiO3(FeH: float,
                         CO: float,
                         MMW: float = 2.33) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for calculating the saturation pressure for MgSiO3.

    Parameters
    ----------
    FeH : float
        Metallicity.
    CO : float
        Carbon-to-oxygen ratio.
    MMW : float
        Mean molecular weight.

    Returns
    -------
    np.ndarray
        Saturation pressure (bar).
    np.ndarray
        Temperature (K).
    """

    masses = atomic_masses()

    T = np.linspace(100., 10000., 1000)

    # Taken from Ackerman & Marley (2001)
    # including erratum (P_vap is in bar, not cgs!)
    P_vap = lambda x: np.exp(25.37 - 58663./x)

    Xmgsio3 = cloud_mass_fraction('MgSiO3', FeH, CO)

    m_mgsio3 = masses['Mg'] + masses['Si'] + 3. * masses['O']

    return P_vap(T)/(Xmgsio3*MMW/m_mgsio3), T


@typechecked
def return_T_cond_Al2O3(FeH: float,
                        CO: float,
                        MMW: float = 2.33) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for calculating the condensation temperature for Al2O3.

    Parameters
    ----------
    FeH : float
        Metallicity.
    CO : float
        Carbon-to-oxygen ratio.
    MMW : float
        Mean molecular weight.

    Returns
    -------
    np.ndarray
        Saturation pressure (bar).
    np.ndarray
        Temperature (K).
    """

    # Return dictionary with atomic masses
    # masses = atomic_masses()

    # Create pressures (bar)
    pressure = np.logspace(-6, 3, 1000)

    # Equilibrium mass fraction of Al2O3
    # Xal2o3 = cloud_mass_fraction('Al2O3', FeH, CO)

    # Molecular mass of Al2O3
    # m_al2o3 = 3. * masses['Al'] + 2. * masses['O']

    # Partial pressure of Al2O3
    # part_press = pressure/(Xal2o3*MMW/m_al2o3)

    # Condensation temperature of Al2O3 (see Eq. 4 in Wakeford et al. 2017)
    t_cond = 1e4 / (5.014 - 0.2179*np.log10(pressure) + 2.264e-3*np.log10(pressure)**2 - 0.580*FeH)

    return pressure, t_cond


@typechecked
def return_T_cond_Na2S(FeH: float,
                       CO: float,
                       MMW: float = 2.33) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for calculating the saturation pressure for Na2S.

    Parameters
    ----------
    FeH : float
        Metallicity.
    CO : float
        Carbon-to-oxygen ratio.
    MMW : float
        Mean molecular weight.

    Returns
    -------
    np.ndarray
        Saturation pressure (bar).
    np.ndarray
        Temperature (K).
    """

    masses = atomic_masses()

    # Taken from Charnay+2018
    T = np.linspace(100., 10000., 1000)

    # This is the partial pressure of Na, so divide by factor 2 to get the partial pressure of
    # the hypothetical Na2S gas particles, this is OK: there are more S than Na atoms at solar
    # abundance ratios.
    P_vap = lambda x: 1e1**(8.55 - 13889./x - 0.5*FeH)/2.

    Xna2s = cloud_mass_fraction('Na2S', FeH, CO)

    m_na2s = 2.*masses['Na'] + masses['S']

    return P_vap(T)/(Xna2s*MMW/m_na2s), T


@typechecked
def return_T_cond_KCl(FeH: float,
                      CO: float,
                      MMW: float = 2.33) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for calculating the saturation pressure for KCl.

    Parameters
    ----------
    FeH : float
        Metallicity.
    CO : float
        Carbon-to-oxygen ratio.
    MMW : float
        Mean molecular weight.

    Returns
    -------
    np.ndarray
        Saturation pressure (bar).
    np.ndarray
        Temperature (K).
    """

    masses = atomic_masses()

    T = np.linspace(100., 10000., 1000)

    # Taken from Charnay+2018
    P_vap = lambda x: 1e1**(7.611 - 11382./T)

    Xkcl = cloud_mass_fraction('KCL', FeH, CO)

    m_kcl = masses['K'] + masses['Cl']

    return P_vap(T)/(Xkcl*MMW/m_kcl), T


@typechecked
def convolve(input_wavel: np.ndarray,
             input_flux: np.ndarray,
             spec_res: float) -> np.ndarray:
    """
    Function to convolve a spectrum with a Gaussian filter.

    Parameters
    ----------
    input_wavel : np.ndarray
        Input wavelengths.
    input_flux : np.ndarray
        Input flux
    spec_res : float
        Spectral resolution of the Gaussian filter.

    Returns
    -------
    np.ndarray
        Convolved spectrum.
    """

    # From talking to Ignas: delta lambda of resolution element
    # is FWHM of the LSF's standard deviation, hence:
    sigma_lsf = 1./spec_res/(2.*np.sqrt(2.*np.log(2.)))

    # The input spacing of petitRADTRANS is 1e3, but just compute
    # it to be sure, or more versatile in the future.
    # Also, we have a log-spaced grid, so the spacing is constant
    # as a function of wavelength
    spacing = np.mean(2.*np.diff(input_wavel)/(input_wavel[1:]+input_wavel[:-1]))

    # Calculate the sigma to be used in the gauss filter in units
    # of input wavelength bins
    sigma_lsf_gauss_filter = sigma_lsf/spacing

    return gaussian_filter(input_flux, sigma=sigma_lsf_gauss_filter, mode='nearest')


@typechecked
def quench_pressure(pressure: np.ndarray,
                    temperature: np.ndarray,
                    metallicity: float,
                    c_o_ratio: float,
                    log_g: float,
                    log_kzz: float) -> Optional[float]:
    """
    Function to determine the CO/CH4 quenching pressure by intersecting the pressure-dependent
    timescales of the vertical mixing and the CO/CH4 reaction rates.

    Parameters
    ----------
    pressure : np.ndarray
        Array with the pressures (bar).
    temperature : np.ndarray
        Array with the temperatures (K) corresponding to ``pressure``.
    metallicity : float
        Metallicity [Fe/H].
    c_o_ratio : float
        Carbon-to-oxygen ratio.
    log_g : float
        Log10 of the surface gravity (cm s-2).
    log_kzz : float
        Log10 of the eddy diffusion coefficient (cm2 s-1).

    Returns
    -------
    float, None
        Quenching pressure (bar)
    """

    # Interpolate the equilibbrium abundances

    co_array = np.full(pressure.shape[0], c_o_ratio)
    feh_array = np.full(pressure.shape[0], metallicity)

    from poor_mans_nonequ_chem.poor_mans_nonequ_chem import interpol_abundances

    abund_eq = interpol_abundances(
        co_array, feh_array, temperature, pressure, Pquench_carbon=None)

    # Surface gravity (m s-2)
    gravity = 1e-2 * 10.**log_g

    # Mean molecular weight (kg)
    mmw = abund_eq['MMW'] * constants.ATOMIC_MASS

    # Pressure scale height (m)
    h_scale = constants.BOLTZMANN * temperature / (mmw * gravity)

    # Diffusion coefficient (m2 s-1)
    kzz = 1e-4 * 10.**log_kzz

    # Mixing timescale (s)
    t_mix = h_scale**2 / kzz

    # Chemical timescale (see Eq. 12 from Zahnle & Marley 2014)
    metal = 10.**metallicity
    t_chem = 1.5e-6 * pressure**-1. * metal**-0.7 * np.exp(42000./temperature)

    # Determine pressure at which t_mix = t_chem

    t_diff = t_mix - t_chem
    diff_product = t_diff[1:] * t_diff[:-1]

    # If t_mix and t_chem intersect then there is 1 negative value in diff_product
    indices = diff_product < 0.

    if np.sum(indices) == 1:
        p_quench = (pressure[1:]+pressure[:-1])[indices]/2.
        p_quench = p_quench[0]

    elif np.sum(indices) == 0:
        p_quench = None

    else:
        raise ValueError(f'Encountered unexpected number of indices '
                         f'({np.sum(indices)}) when determining the intersection of '
                         f't_mix and t_chem.')

    return p_quench
