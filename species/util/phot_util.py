"""
Utility functions for photometry.
"""

import math
import warnings

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import spectres

from typeguard import typechecked

from species.analysis import photometry
from species.core import box
from species.read import (
    read_calibration,
    read_filter,
    read_model,
    read_planck,
    read_radtrans,
)
from species.util import read_util


@typechecked
def multi_photometry(
    datatype: str,
    spectrum: str,
    filters: List[str],
    parameters: Dict[str, float],
    radtrans: Optional[read_radtrans.ReadRadtrans] = None,
) -> box.SynphotBox:
    """
    Parameters
    ----------
    datatype : str
        Data type ('model' or 'calibration').
    spectrum : str
        Spectrum name (e.g., 'drift-phoenix', 'planck', 'powerlaw',
        'petitradtrans').
    filters : list(str)
        List with the filter names.
    parameters : dict
        Dictionary with the model parameters.
    radtrans : read_radtrans.ReadRadtrans, None
        Instance of :class:`~species.read.read_radtrans.ReadRadtrans`.
        Only required with ``spectrum='petitradtrans'`. Make sure that
        the ``wavel_range`` of the ``ReadRadtrans`` instance is
        sufficiently broad to cover all the ``filters``. Not used if
        set to `None`.

    Returns
    -------
    species.core.box.SynphotBox
        Box with synthetic photometry.
    """

    print("Calculating synthetic photometry...", end="", flush=True)

    flux = {}

    if datatype == "model":
        if spectrum == "petitradtrans":
            # Calculate the petitRADTRANS spectrum only once
            radtrans_box = radtrans.get_model(parameters)

        for item in filters:

            if spectrum == "petitradtrans":
                # Use an instance of SyntheticPhotometry instead
                # of get_flux from ReadRadtrans in order to not
                # recalculate the spectrum
                syn_phot = photometry.SyntheticPhotometry(item)

                flux[item], _ = syn_phot.spectrum_to_flux(
                    radtrans_box.wavelength, radtrans_box.flux
                )

            elif spectrum == "powerlaw":
                synphot = photometry.SyntheticPhotometry(item)

                # Set the wavel_range attribute
                synphot.zero_point()

                powerl_box = read_util.powerlaw_spectrum(
                    synphot.wavel_range, parameters
                )
                flux[item] = synphot.spectrum_to_flux(
                    powerl_box.wavelength, powerl_box.flux
                )[0]

            else:
                if spectrum == "planck":
                    readmodel = read_planck.ReadPlanck(filter_name=item)

                else:
                    readmodel = read_model.ReadModel(spectrum, filter_name=item)

                try:
                    if "teff_0" in parameters and "teff_1" in parameters:
                        # Binary system

                        param_0 = read_util.binary_to_single(parameters, 0)
                        model_flux_0 = readmodel.get_flux(param_0)[0]

                        param_1 = read_util.binary_to_single(parameters, 1)
                        model_flux_1 = readmodel.get_flux(param_1)[0]

                        flux[item] = (
                            parameters["spec_weight"] * model_flux_0
                            + (1.0 - parameters["spec_weight"]) * model_flux_1
                        )

                    else:
                        # Single object

                        flux[item] = readmodel.get_flux(parameters)[0]

                except IndexError:
                    flux[item] = np.nan

                    warnings.warn(
                        f"The wavelength range of the {item} filter does not "
                        f"match with the wavelength range of {spectrum}. The "
                        f"flux is set to NaN."
                    )

    elif datatype == "calibration":
        for item in filters:
            readcalib = read_calibration.ReadCalibration(spectrum, filter_name=item)
            flux[item] = readcalib.get_flux(parameters)[0]

    print(" [DONE]")

    return box.create_box("synphot", name="synphot", flux=flux)


@typechecked
def apparent_to_absolute(
    app_mag: Union[
        Tuple[float, Optional[float]], Tuple[np.ndarray, Optional[np.ndarray]]
    ],
    distance: Union[
        Tuple[float, Optional[float]], Tuple[np.ndarray, Optional[np.ndarray]]
    ],
) -> Union[Tuple[float, Optional[float]], Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Function for converting an apparent magnitude into an absolute
    magnitude. The uncertainty on the distance is propagated into the
    uncertainty on the absolute magnitude.

    Parameters
    ----------
    app_mag : tuple(float, float), tuple(np.ndarray, np.ndarray)
        Apparent magnitude and uncertainty (mag). The returned error
        on the absolute magnitude is set to None if the error on the
        apparent magnitude is set to None, for example
        ``app_mag=(15., None)``.
    distance : tuple(float, float), tuple(np.ndarray, np.ndarray)
        Distance and uncertainty (pc). The error is not propagated
        into the error on the absolute magnitude if set to None, for
        example ``distance=(20., None)``.

    Returns
    -------
    float, np.ndarray
        Absolute magnitude (mag).
    float, np.ndarray, None
        Uncertainty (mag).
    """

    abs_mag = app_mag[0] - 5.0 * np.log10(distance[0]) + 5.0

    if app_mag[1] is not None and distance[1] is not None:
        dist_err = distance[1] * (5.0 / (distance[0] * math.log(10.0)))
        abs_err = np.sqrt(app_mag[1] ** 2 + dist_err ** 2)

    elif app_mag[1] is not None and distance[1] is None:
        abs_err = app_mag[1]

    else:
        abs_err = None

    return abs_mag, abs_err


@typechecked
def absolute_to_apparent(
    abs_mag: Union[
        Tuple[float, Optional[float]], Tuple[np.ndarray, Optional[np.ndarray]]
    ],
    distance: Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]],
) -> Union[Tuple[float, Optional[float]], Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Function for converting an absolute magnitude into an apparent
    magnitude.

    Parameters
    ----------
    abs_mag : tuple(float, float), tuple(np.ndarray, np.ndarray)
        Absolute magnitude and uncertainty (mag). The same uncertainty
        is used for the apparent magnitude.
    distance : tuple(float, float), tuple(np.ndarray, np.ndarray)
        Distance and uncertainty (pc).

    Returns
    -------
    float, np.ndarray
        Apparent magnitude (mag).
    float, np.ndarray, None
        Uncertainty (mag).
    """

    app_mag = abs_mag[0] + 5.0 * np.log10(distance[0]) - 5.0

    return app_mag, abs_mag[1]


@typechecked
def get_residuals(
    datatype: str,
    spectrum: str,
    parameters: Dict[str, float],
    objectbox: box.ObjectBox,
    inc_phot: Union[bool, List[str]] = True,
    inc_spec: Union[bool, List[str]] = True,
    radtrans: Optional[read_radtrans.ReadRadtrans] = None,
) -> box.ResidualsBox:

    """
    Function for calculating the residuals from fitting model or
    calibration spectra to a set of spectra and/or photometry.

    Parameters
    ----------
    datatype : str
        Data type ('model' or 'calibration').
    spectrum : str
        Name of the atmospheric model or calibration spectrum.
    parameters : dict
        Parameters and values for the spectrum
    objectbox : species.core.box.ObjectBox
        Box with the photometry and/or spectra of an object. A scaling
        and/or error inflation of the spectra should be applied with
        :func:`~species.util.read_util.update_spectra` beforehand.
    inc_phot : bool, list(str)
        Include photometric data in the fit. If a boolean, either all
        (``True``) or none (``False``) of the data are selected. If a
        list, a subset of filter names (as stored in the database) can
        be provided.
    inc_spec : bool, list(str)
        Include spectroscopic data in the fit. If a boolean, either all
        (``True``) or none (``False``) of the data are selected. If a
        list, a subset of spectrum names (as stored in the database
        with :func:`~species.data.database.Database.add_object`) can be
        provided.
    radtrans : read_radtrans.ReadRadtrans, None
        Instance of :class:`~species.read.read_radtrans.ReadRadtrans`.
        Only required with ``spectrum='petitradtrans'`. Make sure that
        the ``wavel_range`` of the ``ReadRadtrans`` instance is
        sufficiently broad to cover all the photometric and
        spectroscopic data of ``inc_phot`` and ``inc_spec``. Not used
        if set to ``None``.

    Returns
    -------
    species.core.box.ResidualsBox
        Box with the residuals.
    """

    if isinstance(inc_phot, bool) and inc_phot:
        inc_phot = objectbox.filters

    if inc_phot:
        model_phot = multi_photometry(
            datatype=datatype,
            spectrum=spectrum,
            filters=inc_phot,
            parameters=parameters,
            radtrans=radtrans,
        )

        res_phot = {}

        for item in inc_phot:
            transmission = read_filter.ReadFilter(item)
            res_phot[item] = np.zeros(objectbox.flux[item].shape)

            if objectbox.flux[item].ndim == 1:
                res_phot[item][0] = transmission.mean_wavelength()
                res_phot[item][1] = (
                    objectbox.flux[item][0] - model_phot.flux[item]
                ) / objectbox.flux[item][1]

            elif objectbox.flux[item].ndim == 2:
                for j in range(objectbox.flux[item].shape[1]):
                    res_phot[item][0, j] = transmission.mean_wavelength()
                    res_phot[item][1, j] = (
                        objectbox.flux[item][0, j] - model_phot.flux[item]
                    ) / objectbox.flux[item][1, j]

    else:
        res_phot = None

    if inc_spec:
        res_spec = {}

        if spectrum == "petitradtrans":
            # Calculate the petitRADTRANS spectrum only once
            model = radtrans.get_model(parameters)

        for key in objectbox.spectrum:

            if isinstance(inc_spec, bool) or key in inc_spec:
                wavel_range = (
                    0.9 * objectbox.spectrum[key][0][0, 0],
                    1.1 * objectbox.spectrum[key][0][-1, 0],
                )

                wl_new = objectbox.spectrum[key][0][:, 0]
                spec_res = objectbox.spectrum[key][3]

                if spectrum == "planck":
                    readmodel = read_planck.ReadPlanck(wavel_range=wavel_range)

                    model = readmodel.get_spectrum(
                        model_param=parameters, spec_res=1000.0
                    )

                    # Separate resampling to the new wavelength points

                    flux_new = spectres.spectres(
                        wl_new,
                        model.wavelength,
                        model.flux,
                        spec_errs=None,
                        fill=0.0,
                        verbose=True,
                    )

                elif spectrum == "petitradtrans":
                    # Separate resampling to the new wavelength points
                    flux_new = spectres.spectres(
                        wl_new,
                        model.wavelength,
                        model.flux,
                        spec_errs=None,
                        fill=0.0,
                        verbose=True,
                    )

                else:
                    # Resampling to the new wavelength points
                    # is done by the get_model method

                    readmodel = read_model.ReadModel(spectrum, wavel_range=wavel_range)

                    if "teff_0" in parameters and "teff_1" in parameters:
                        # Binary system

                        param_0 = read_util.binary_to_single(parameters, 0)

                        model_spec_0 = readmodel.get_model(
                            param_0,
                            spec_res=spec_res,
                            wavel_resample=wl_new,
                            smooth=True,
                        )

                        param_1 = read_util.binary_to_single(parameters, 1)

                        model_spec_1 = readmodel.get_model(
                            param_1,
                            spec_res=spec_res,
                            wavel_resample=wl_new,
                            smooth=True,
                        )

                        flux_comb = (
                            parameters["spec_weight"] * model_spec_0.flux
                            + (1.0 - parameters["spec_weight"]) * model_spec_1.flux
                        )

                        model_spec = box.create_box(
                            boxtype="model",
                            model=spectrum,
                            wavelength=wl_new,
                            flux=flux_comb,
                            parameters=parameters,
                            quantity="flux",
                        )

                    else:
                        # Single object

                        model_spec = readmodel.get_model(
                            parameters,
                            spec_res=spec_res,
                            wavel_resample=wl_new,
                            smooth=True,
                        )

                    flux_new = model_spec.flux

                data_spec = objectbox.spectrum[key][0]
                res_tmp = (data_spec[:, 1] - flux_new) / data_spec[:, 2]

                res_spec[key] = np.column_stack([wl_new, res_tmp])

    else:
        res_spec = None

    print("Calculating residuals... [DONE]")

    print("Residuals (sigma):")

    if res_phot is not None:
        for item in inc_phot:
            if res_phot[item].ndim == 1:
                print(f"   - {item}: {res_phot[item][1]:.2f}")

            elif res_phot[item].ndim == 2:
                for j in range(res_phot[item].shape[1]):
                    print(f"   - {item}: {res_phot[item][1, j]:.2f}")

    if res_spec is not None:
        for key in objectbox.spectrum:
            if isinstance(inc_spec, bool) or key in inc_spec:
                print(
                    f"   - {key}: min: {np.nanmin(res_spec[key]):.2f}, "
                    f"max: {np.nanmax(res_spec[key]):.2f}"
                )

    chi2_stat = 0
    n_dof = 0

    if res_phot is not None:
        for key, value in res_phot.items():
            chi2_stat += value[1] ** 2
            n_dof += 1

    if res_spec is not None:
        for key, value in res_spec.items():
            chi2_stat += np.sum(value[:, 1] ** 2)
            n_dof += value.shape[0]

    for item in parameters:
        if item not in ["mass", "luminosity", "distance"]:
            n_dof -= 1

    chi2_red = chi2_stat / n_dof

    print(f"Reduced chi2 = {chi2_red:.2f}")
    print(f"Number of degrees of freedom = {n_dof}")

    return box.create_box(
        boxtype="residuals",
        name=objectbox.name,
        photometry=res_phot,
        spectrum=res_spec,
        chi2_red=chi2_red,
    )
