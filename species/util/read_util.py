"""
Utility functions for reading data.
"""

import math
import warnings

from typing import Dict, Optional, Tuple, Union

import numpy as np

from scipy.integrate import simps
from scipy.ndimage import gaussian_filter
from typeguard import typechecked

from species.core import box, constants
from species.read import read_model, read_planck


@typechecked
def get_mass(
    logg: Union[float, np.ndarray], radius: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Function for converting a :math:`\\log(g)` and a radius into
    a mass.

    Parameters
    ----------
    logg : float, np.ndarray
        Log10 of the surface gravity (cgs).
    radius : float, np.ndarray
        Radius (Rjup).

    Returns
    -------
    float, np.ndarray
        Mass (Mjup).
    """

    surface_grav = 1e-2 * 10.0 ** logg  # (m s-2)
    radius *= constants.R_JUP  # (m)
    mass = surface_grav * radius ** 2 / constants.GRAVITY  # (kg)

    return mass / constants.M_JUP


@typechecked
def get_radius(
    logg: Union[float, np.ndarray], mass: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Function for converting a :math:`\\log(g)` and a mass into
    a radius.

    Parameters
    ----------
    logg : float, np.ndarray
        Log10 of the surface gravity (cgs).
    mass : float, np.ndarray
        Mass (Mjup).

    Returns
    -------
    float, np.ndarray
        Radius (Rjup).
    """

    surface_grav = 1e-2 * 10.0 ** logg  # (m s-2)
    mass *= constants.M_JUP  # (kg)
    radius = np.sqrt(mass * constants.GRAVITY / surface_grav)  # (m)

    return radius / constants.R_JUP


def add_luminosity(modelbox):
    """
    Function to add the luminosity of a model spectrum to the parameter
    dictionary of the box.

    Parameters
    ----------
    modelbox : species.core.box.ModelBox
        Box with the model spectrum. Should also contain the dictionary
        with the model parameters, the radius in particular.

    Returns
    -------
    species.core.box.ModelBox
        The input box with the luminosity added in the parameter
        dictionary.
    """

    print("Calculating the luminosity...", end="", flush=True)

    if modelbox.model == "planck":
        readmodel = read_planck.ReadPlanck(wavel_range=(1e-1, 1e3))
        fullspec = readmodel.get_spectrum(
            model_param=modelbox.parameters, spec_res=1000.0
        )

    else:
        readmodel = read_model.ReadModel(modelbox.model)
        fullspec = readmodel.get_model(modelbox.parameters)

    flux = simps(fullspec.flux, fullspec.wavelength)

    if "distance" in modelbox.parameters:
        luminosity = (
            4.0
            * np.pi
            * (fullspec.parameters["distance"] * constants.PARSEC) ** 2
            * flux
        )  # (W)

        # Analytical solution for a single-component Planck function
        # luminosity = 4.*np.pi*(modelbox.parameters['radius']*constants.R_JUP)**2* \
        #     constants.SIGMA_SB*modelbox.parameters['teff']**4.

    else:
        luminosity = (
            4.0 * np.pi * (fullspec.parameters["radius"] * constants.R_JUP) ** 2 * flux
        )  # (W)

    modelbox.parameters["luminosity"] = luminosity / constants.L_SUN  # (Lsun)

    print(" [DONE]")

    print(
        f"Wavelength range (um): {fullspec.wavelength[0]:.2e} - "
        f"{fullspec.wavelength[-1]:.2e}"
    )

    print(f"Luminosity (Lsun): {luminosity/constants.L_SUN:.2e}")

    return modelbox


@typechecked
def update_spectra(
    objectbox: box.ObjectBox, model_param: Dict[str, float], model: Optional[str] = None
) -> box.ObjectBox:
    """
    Function for applying a flux scaling and/or error inflation to the
    spectra of an :class:`~species.core.box.ObjectBox`.

    Parameters
    ----------
    objectbox : species.core.box.ObjectBox
        Box with the object's data, including the spectra.
    model_param : dict
        Dictionary with the model parameters. Should contain the
        value(s) of the flux scaling and/or the error inflation.
    model : str, None
        Name of the atmospheric model. Only required for inflating the
        errors of spectra. Otherwise, the argument can be set to
        ``None``. Not required when ``model='petitradtrans'`` because
        the error inflation is differently implemented with
        :class:`~species.analysis.retrieval.AtmosphericRetrieval`.

    Returns
    -------
    species.core.box.ObjectBox
        The input box which includes the spectra with the scaled fluxes
        and/or inflated errors.
    """

    warnings.warn(
        "The update_spectra function is deprecated and "
        "will be removed in a future release. Please use "
        "the update_objectbox function instead.",
        DeprecationWarning,
    )

    if objectbox.flux is not None:

        for key, value in objectbox.flux.items():
            if f"{key}_error" in model_param:
                var_add = model_param[f"{key}_error"] ** 2 * value[0] ** 2

                message = (
                    f"Inflating the error of {key} "
                    + f"(W m-2 um-1): {np.sqrt(var_add):.2e}..."
                )

                print(message, end="", flush=True)

                value[1] = np.sqrt(value[1] ** 2 + var_add)

                print(" [DONE]")

            objectbox.flux[key] = value

    if objectbox.spectrum is not None:
        # Check if there are any spectra

        for key, value in objectbox.spectrum.items():
            # Get the spectrum (3 columns)
            spec_tmp = value[0]

            if f"scaling_{key}" in model_param:
                # Scale the flux of the spectrum
                scaling = model_param[f"scaling_{key}"]

                print(
                    f"Scaling the flux of {key}: {scaling:.2f}...", end="", flush=True
                )
                spec_tmp[:, 1] *= model_param[f"scaling_{key}"]
                print(" [DONE]")

            if f"error_{key}" in model_param:
                if model is None:
                    warnings.warn(
                        f"The dictionary with model parameters contains the error "
                        f"inflation for {key} but the argument of 'model' is set "
                        f"to None. Inflation of the errors is therefore not possible."
                    )

                if model == "petitradtrans":
                    # Increase the errors by a constant value
                    add_error = 10.0 ** model_param[f"error_{key}"]
                    log_msg = (
                        f"Inflating the error of {key} (W m-2 um-1): {add_error:.2e}..."
                    )

                    print(log_msg, end="", flush=True)
                    spec_tmp[:, 2] += add_error
                    print(" [DONE]")

                else:
                    # Calculate the model spectrum
                    wavel_range = (0.9 * spec_tmp[0, 0], 1.1 * spec_tmp[-1, 0])
                    readmodel = read_model.ReadModel(model, wavel_range=wavel_range)

                    model_box = readmodel.get_model(
                        model_param,
                        spec_res=value[3],
                        wavel_resample=spec_tmp[:, 0],
                        smooth=True,
                    )

                    # Scale the errors relative to the model spectrum
                    err_scaling = model_param[f"error_{key}"]
                    log_msg = f"Inflating the error of {key}: {err_scaling:.2e}..."

                    print(log_msg, end="", flush=True)
                    spec_tmp[:, 2] = np.sqrt(
                        spec_tmp[:, 2] ** 2 + (err_scaling * model_box.flux) ** 2
                    )
                    print(" [DONE]")

            # Store the spectra with the scaled fluxes and/or errors
            # The other three elements (i.e. the covariance matrix,
            # the inverted covariance matrix, and the spectral
            # resolution) remain unaffected
            objectbox.spectrum[key] = (spec_tmp, value[1], value[2], value[3])

    return objectbox


@typechecked
def update_objectbox(
    objectbox: box.ObjectBox, model_param: Dict[str, float], model: Optional[str] = None
) -> box.ObjectBox:
    """
    Function for updating the spectra and/or photometric fluxes in
    an :class:`~species.core.box.ObjectBox`, for example by applying
    a flux scaling and/or error inflation.

    Parameters
    ----------
    objectbox : species.core.box.ObjectBox
        Box with the object's data, including the spectra and/or
        photometric fluxes.
    model_param : dict
        Dictionary with the model parameters. Should contain the
        value(s) of the flux scaling and/or the error inflation.
    model : str, None
        Name of the atmospheric model. Only required for inflating the
        errors of spectra. Otherwise, the argument can be set to
        ``None``. Not required when ``model='petitradtrans'`` because
        the error inflation is differently implemented with
        :class:`~species.analysis.retrieval.AtmosphericRetrieval`.

    Returns
    -------
    species.core.box.ObjectBox
        The input box which includes the spectra with the scaled fluxes
        and/or inflated errors.
    """

    if objectbox.flux is not None:

        for key, value in objectbox.flux.items():
            instr_name = key.split(".")[0]

            if f"{key}_error" in model_param:
                # Inflate photometric error of filter
                var_add = model_param[f"{key}_error"] ** 2 * value[0] ** 2

            elif f"{instr_name}_error" in model_param:
                # Inflate photometric error of instrument
                var_add = model_param[f"{instr_name}_error"] ** 2 * value[0] ** 2

            else:
                # No inflation required
                var_add = None

            if var_add is not None:
                message = (
                    f"Inflating the error of {key} "
                    + f"(W m-2 um-1): {np.sqrt(var_add):.2e}..."
                )

                print(message, end="", flush=True)

                value[1] = np.sqrt(value[1] ** 2 + var_add)

                print(" [DONE]")

                objectbox.flux[key] = value

    if objectbox.spectrum is not None:
        # Check if there are any spectra

        for key, value in objectbox.spectrum.items():
            # Get the spectrum (3 columns)
            spec_tmp = value[0]

            if f"scaling_{key}" in model_param:
                # Scale the flux of the spectrum
                scaling = model_param[f"scaling_{key}"]

                print(
                    f"Scaling the flux of {key}: {scaling:.2f}...", end="", flush=True
                )
                spec_tmp[:, 1] *= model_param[f"scaling_{key}"]
                print(" [DONE]")

            if f"error_{key}" in model_param:
                if model is None:
                    warnings.warn(
                        f"The dictionary with model parameters contains the error "
                        f"inflation for {key} but the argument of 'model' is set "
                        f"to None. Inflation of the errors is therefore not possible."
                    )

                elif model == "petitradtrans":
                    # Increase the errors by a constant value
                    add_error = 10.0 ** model_param[f"error_{key}"]
                    log_msg = (
                        f"Inflating the error of {key} (W m-2 um-1): {add_error:.2e}..."
                    )

                    print(log_msg, end="", flush=True)
                    spec_tmp[:, 2] += add_error
                    print(" [DONE]")

                else:
                    # Calculate the model spectrum
                    wavel_range = (0.9 * spec_tmp[0, 0], 1.1 * spec_tmp[-1, 0])
                    readmodel = read_model.ReadModel(model, wavel_range=wavel_range)

                    model_box = readmodel.get_model(
                        model_param,
                        spec_res=value[3],
                        wavel_resample=spec_tmp[:, 0],
                        smooth=True,
                    )

                    # Scale the errors relative to the model spectrum
                    err_scaling = model_param[f"error_{key}"]
                    log_msg = f"Inflating the error of {key}: {err_scaling:.2e}..."

                    print(log_msg, end="", flush=True)
                    spec_tmp[:, 2] = np.sqrt(
                        spec_tmp[:, 2] ** 2 + (err_scaling * model_box.flux) ** 2
                    )
                    print(" [DONE]")

            # Store the spectra with the scaled fluxes and/or errors
            # The other three elements (i.e. the covariance matrix,
            # the inverted covariance matrix, and the spectral
            # resolution) remain unaffected
            objectbox.spectrum[key] = (spec_tmp, value[1], value[2], value[3])

    return objectbox


@typechecked
def create_wavelengths(
    wavel_range: Tuple[Union[float, np.float32], Union[float, np.float32]],
    spec_res: float,
) -> np.ndarray:
    """
    Function for creating logarithmically-spaced wavelengths at a
    constant spectral resolution :math:`R`.

    Parameters
    ----------
    wavel_range : tuple(float, float)
        Wavelength range (um). Tuple with the minimum and maximum
        wavelength.
    spec_res : float
        Spectral resolution at which the wavelengths are sampled.

    Returns
    -------
    np.ndarray
        Array with the wavelength points and a fixed spectral
        resolution. Since the wavelength boundaries are fixed, the
        output spectral resolution is slightly different from the
        ``spec_res`` value.
    """

    n_test = 100

    wavel_test = np.logspace(np.log10(wavel_range[0]), np.log10(wavel_range[1]), n_test)

    res_test = 0.5 * (wavel_test[1:] + wavel_test[:-1]) / np.diff(wavel_test)

    # R = lambda / delta_lambda / 2, because twice as many points as
    # R are required to resolve two features that are lambda / R apart

    wavelength = np.logspace(
        np.log10(wavel_range[0]),
        np.log10(wavel_range[1]),
        math.ceil(2.0 * n_test * spec_res / np.mean(res_test)) + 1,
    )

    # res_out = np.mean(0.5*(wavelength[1:]+wavelength[:-1])/np.diff(wavelength)/2.)

    return wavelength


@typechecked
def smooth_spectrum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    spec_res: float,
    size: int = 11,
    force_smooth: bool = False,
) -> np.ndarray:
    """
    Function for smoothing a spectrum with a Gaussian kernel to a
    fixed spectral resolution. The kernel size is set to 5 times the
    FWHM of the Gaussian. The FWHM of the Gaussian is equal to the
    ratio of the wavelength and the spectral resolution. If the
    kernel does not fit within the available wavelength grid (i.e.
    at the edge of the array) then the flux values are set to NaN.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength points (um). Should be sampled with a uniform
        spectral resolution or a uniform wavelength spacing (slow).
    flux : np.ndarray
        Flux (W m-2 um-1).
    spec_res : float
        Spectral resolution.
    size : int
        Kernel size (odd integer).
    force_smooth : bool
        Force smoothing for constant spectral resolution

    Returns
    -------
    np.ndarray
        Smoothed spectrum (W m-2 um-1).
    """

    def _gaussian(size, sigma):
        pos = range(-(size - 1) // 2, (size - 1) // 2 + 1)
        kernel = [
            np.exp(-float(x) ** 2 / (2.0 * sigma ** 2)) / (sigma * np.sqrt(2.0 * np.pi))
            for x in pos
        ]

        return np.asarray(kernel / sum(kernel))

    spacing = np.mean(2.0 * np.diff(wavelength) / (wavelength[1:] + wavelength[:-1]))
    spacing_std = np.std(2.0 * np.diff(wavelength) / (wavelength[1:] + wavelength[:-1]))

    if spacing_std / spacing < 1e-2 or force_smooth:
        # see retrieval_util.convolve
        sigma_lsf = 1.0 / spec_res / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        flux_smooth = gaussian_filter(flux, sigma=sigma_lsf / spacing, mode="nearest")

    else:
        if size % 2 == 0:
            raise ValueError("The kernel size should be an odd number.")

        flux_smooth = np.zeros(flux.shape)  # (W m-2 um-1)

        spacing = np.mean(np.diff(wavelength))  # (um)
        spacing_std = np.std(np.diff(wavelength))  # (um)

        if spacing_std / spacing > 1e-2:
            warnings.warn(
                f"The wavelength spacing is not uniform ({spacing} +/- {spacing_std}). "
                f"The smoothing with the Gaussian kernel requires either the spectral "
                f"resolution or the wavelength spacing to be uniformly sampled."
            )

        for i, item in enumerate(wavelength):
            fwhm = item / spec_res  # (um)
            sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # (um)

            size = int(
                5.0 * sigma / spacing
            )  # Kernel size 5 times the width of the LSF
            if size % 2 == 0:
                size += 1

            gaussian = _gaussian(size, sigma / spacing)

            try:
                flux_smooth[i] = np.sum(
                    gaussian * flux[i - (size - 1) // 2 : i + (size - 1) // 2 + 1]
                )

            except ValueError:
                flux_smooth[i] = np.nan

    return flux_smooth


@typechecked
def powerlaw_spectrum(
    wavel_range: Union[Tuple[float, float], Tuple[np.float32, np.float32]],
    model_param: Dict[str, float],
    spec_res: float = 100.0,
) -> box.ModelBox:
    """
    Function for calculating a power-law spectrum. The power-law
    function is calculated in log(wavelength)-log(flux) space but
    stored in the :class:`~species.core.box.ModelBox` in linear
    wavelength-flux space.

    Parameters
    ----------
    wavel_range : tuple(float, float)
        Tuple with the minimum and maximum wavelength (um).
    model_param : dict
        Dictionary with the model parameters. Should contain
        `'log_powerlaw_a'`, `'log_powerlaw_b'`, and `'log_powerlaw_c'`.
    spec_res : float
        Spectral resolution (default: 100).

    Returns
    -------
    species.core.box.ModelBox
        Box with the power-law spectrum.
    """

    wavel = create_wavelengths((wavel_range[0], wavel_range[1]), spec_res)

    log_flux = (
        model_param["log_powerlaw_a"]
        + model_param["log_powerlaw_b"]
        * np.log10(wavel) ** model_param["log_powerlaw_c"]
    )

    model_box = box.create_box(
        boxtype="model",
        model="powerlaw",
        wavelength=wavel,
        flux=10.0 ** log_flux,
        parameters=model_param,
        quantity="flux",
    )

    return model_box


@typechecked
def gaussian_spectrum(
    wavel_range: Union[Tuple[float, float], Tuple[np.float32, np.float32]],
    model_param: Dict[str, float],
    spec_res: float = 100.0,
    double_gaussian: bool = False,
) -> box.ModelBox:
    """
    Function for calculating a Gaussian spectrum (i.e. for an emission
    line).

    Parameters
    ----------
    wavel_range : tuple(float, float)
        Tuple with the minimum and maximum wavelength (um).
    model_param : dict
        Dictionary with the model parameters. Should contain
        ``'gauss_amplitude'``, ``'gauss_mean'``, ``'gauss_sigma'``,
        and optionally ``'gauss_offset'``.
    spec_res : float
        Spectral resolution (default: 100).
    double_gaussian : bool
        Set to ``True`` for returning a double Gaussian function.
        In that case, ``model_param`` should also contain
        ``'gauss_amplitude_2'``, ``'gauss_mean_2'``, and
        ``'gauss_sigma_2'``.

    Returns
    -------
    species.core.box.ModelBox
        Box with the Gaussian spectrum.
    """

    wavel = create_wavelengths((wavel_range[0], wavel_range[1]), spec_res)

    gauss_exp = np.exp(
        -0.5
        * (wavel - model_param["gauss_mean"]) ** 2
        / model_param["gauss_sigma"] ** 2
    )

    flux = model_param["gauss_amplitude"] * gauss_exp

    if double_gaussian:
        gauss_exp = np.exp(
            -0.5
            * (wavel - model_param["gauss_mean_2"]) ** 2
            / model_param["gauss_sigma_2"] ** 2
        )

        flux += model_param["gauss_amplitude_2"] * gauss_exp

    if "gauss_offset" in model_param:
        flux += model_param["gauss_offset"]

    model_box = box.create_box(
        boxtype="model",
        model="gaussian",
        wavelength=wavel,
        flux=flux,
        parameters=model_param,
        quantity="flux",
    )

    return model_box


@typechecked
def binary_to_single(param_dict: Dict[str, float], star_index: int) -> Dict[str, float]:
    """
    Function for converting a dictionary with atmospheric parameters
    of a binary system to a dictionary of parameters for one of the
    two stars.

    Parameters
    ----------
    param_dict : dict
        Dictionary with the atmospheric parameters of both stars. The
        keywords end either with ``_0`` or ``_1`` that correspond with
        ``star_index=0`` or ``star_index=1``.
    star_index : int
        Star index (0 or 1) that is used for the parameters in
        ``param_dict``.

    Returns
    -------
    dict
        Dictionary with the parameters of the selected star.
    """

    new_dict = {}

    for key, value in param_dict.items():
        if star_index == 0 and key[-1] == "0":
            new_dict[key[:-2]] = value

        elif star_index == 1 and key[-1] == "1":
            new_dict[key[:-2]] = value

        elif key in ["teff", "logg", "feh", "c_o_ratio", "fsed", "radius", "distance"]:
            new_dict[key] = value

    return new_dict
