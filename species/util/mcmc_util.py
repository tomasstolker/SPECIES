"""
Utility functions for data processing.
"""

from typing import Optional

import numpy as np

from scipy.linalg import cho_factor, cho_solve, inv
from typeguard import typechecked
from astropy.io import fits


@typechecked
def sqexp_kernel(wavel: np.ndarray,
                 sigma: np.ndarray,
                 corr_amp: float,
                 log_corr_len: float) -> np.ndarray:
    """
    Function for modeling the covariances with a squared exponential kernel (see Wang et al. 2020).

    Parameters
    ----------
    wavel : np.ndarray
        Array with the wavelengths (um).
    sigma : np.ndarray
        Array with the flux uncertainties (W m-2 um-1).
    corr_amp : float
        The fractional amplitude (see Wang et al. 2020 for details).
    log_corr_len : float
        Log10 of the correlation length (um) (see Wang et al. 2020 for details).

    Returns
    -------
    np.ndarray
        Array with the covariances (W2 m-4 um-2).
    """

    corr_len = 10.**log_corr_len  # (um)

    wavel_i, wavel_j = np.meshgrid(wavel, wavel)
    sigma_i, sigma_j = np.meshgrid(sigma, sigma)

    cov_matrix = corr_amp**2 * sigma_i * sigma_j * \
        np.exp(-(wavel_i-wavel_j)**2 / (2.*corr_len**2)) + \
        (1.-corr_amp**2) * np.diag(sigma**2)

    return cov_matrix


@typechecked
def matern32_kernel(wavel: np.ndarray,
                    obs_var: np.ndarray,
                    log_corr_amp: float,
                    log_corr_len: float) -> np.ndarray:
    """
    Function for modeling the covariances with a Matérn-3/2 kernel (see Czekala et al. 2015).

    Parameters
    ----------
    wavel : np.ndarray
        Array with the wavelengths (um).
    obs_var : np.ndarray
        Array with the variances (W2 m-4 um-2).
    log_corr_amp : float
        Log10 of the amplitude (see Czekala et al. 2015 for details).
    log_corr_len : float
        Log10 of the length scale (see Czekala et al. 2015 for details).

    Returns
    -------
    np.ndarray
        Array with the covariances (W2 m-4 um-2).
    """

    corr_amp = 10.**log_corr_amp
    corr_len = 10.**log_corr_len  # (um)

    wavel_diff = wavel[:, None] - wavel[None, :]

    r_ij = np.sqrt(wavel_diff**2 + wavel_diff**2)

    cov_matrix = np.diag(obs_var)

    cov_term = np.sqrt(3.) * r_ij / corr_len
    cov_matrix = corr_amp * (1.+cov_term) * np.exp(-cov_term)

    # fits.writeto('cov_term.fits', cov_term, overwrite=True)
    # fits.writeto('cov_matrix.fits', cov_matrix, overwrite=True)
    # fits.writeto('inv_cov.fits', inv(cov_matrix), overwrite=True)

    return cov_matrix


@typechecked
def lnlike_phot(obs_flux: float,
                obs_var: float,
                model_flux: float,
                weight: Optional[float] = None) -> float:
    """
    Function to

    Parameters
    ----------

    Returns
    -------
    """

    if weight is None:
        weight = 1.

    ln_like = weight * (obs_flux - model_flux)**2 / obs_var
    ln_like += weight * np.log(2.*np.pi*obs_var)

    return ln_like


@typechecked
def lnlike_spec(obs_flux: np.ndarray,
                obs_var: np.ndarray,
                model_flux: np.ndarray,
                weight: Optional[float] = None,
                cov_matrix: Optional[np.ndarray] = None,
                inv_cov_matrix: Optional[np.ndarray] = None) -> float:
    """
    Function to

    Parameters
    ----------

    Returns
    -------
    """

    if weight is None:
        weight = 1.

    residual = obs_flux - model_flux

    if cov_matrix is None and inv_cov_matrix is None:
        ln_like = weight * np.nansum(residual**2/obs_var)
        ln_like += weight * np.nansum(np.log(2.*np.pi*obs_var))

    elif inv_cov_matrix is not None:
        sign, log_det_cov = np.linalg.slogdet(cov_matrix)

        if sign != 1.:
            raise ValueError(f'Unexpected sign ({sign}) returned by np.linalg.slogdet.')

        ln_like = weight * np.dot(residual, np.dot(inv_cov_matrix, residual))
        ln_like += weight * log_det_cov
        ln_like += weight * obs_flux.shape[0] * np.log(2.*np.pi)

    elif cov_matrix is not None:
        factor, flag = cho_factor(cov_matrix, overwrite_a=True)

        ln_like = weight * np.dot(residual, cho_solve((factor, flag), residual))
        ln_like += weight * 2. * np.nansum(np.log(np.diag(factor)))
        ln_like += weight * obs_flux.shape[0] * np.log(2.*np.pi)

    return -0.5*ln_like
