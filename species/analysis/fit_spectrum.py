"""
Module with functionalities for photometric and spectroscopic calibration. The fitting routine
can be used to fit photometric data with a calibration spectrum (e.g. extracted with
:func:`~species.read.read_model.ReadModel.get_model`) by simply fitting a scaling parameter.
"""

import math

from typing import Dict, List, Optional, Tuple, Union

from multiprocessing import cpu_count, Pool

import emcee
import numpy as np

from typeguard import typechecked

from species.analysis import photometry
from species.data import database
from species.read import read_calibration, read_object


@typechecked
def lnprob(param: np.ndarray,
           bounds: Dict[str, Tuple[float, float]],
           modelpar: List[str],
           objphot: List[np.ndarray],
           specphot: Union[List[float],
                           List[Tuple[photometry.SyntheticPhotometry,
                                      Tuple[np.float64, np.float64]]]]) -> float:
    """
    Internal function for calculating the posterior probability.

    Parameters
    ----------
    param : np.ndarray
        Value of the scaling parameter.
    bounds : dict
        Boundaries of the main scaling parameter.
    modelpar : list(str)
        Parameter names.
    objphot : list(tuple(float, float))
        Photometry of the object.
    specphot : list(float), photometry.SyntheticPhotometry
        Synthetic photometry of the calibration spectrum for the same filters as the photometry
        of the object.

    Returns
    -------
    float
        Log posterior probability.
    """

    ln_prob = 0.

    for i, item in enumerate(modelpar):

        if bounds[item][0] <= param[i] <= bounds[item][1]:
            ln_prob += 0.

        else:
            ln_prob += -np.inf
            break

    if not math.isinf(ln_prob):

        for i, obj_item in enumerate(objphot):
            if obj_item.ndim == 1:
                ln_prob += -0.5 * (obj_item[0] - param[0]*specphot[i])**2 / obj_item[1]**2

            else:

                for j in range(obj_item.shape[1]):
                    ln_prob += -0.5 * (obj_item[0, j] - param[0]*specphot[i])**2 / obj_item[1, j]**2

    return ln_prob


class FitSpectrum:
    """
    Class for fitting a calibration spectrum to photometric data.
    """

    @typechecked
    def __init__(self,
                 object_name: str,
                 filters: Optional[List[str]],
                 spectrum: str,
                 bounds: Dict[str, Tuple[float, float]]) -> None:
        """
        Parameters
        ----------
        object_name : str
            Object name in the database.
        filters : list(str)
            Filter names for which the photometry is selected. All available photometry of the
            object is selected if set to ``None``.
        spectrum : str
            Calibration spectrum as labelled in the database. The calibration spectrum can be
            stored in the database with :func:`~species.data.database.Database.add_calibration`.
        bounds : dict
            Boundaries of the scaling parameter, as ``{'scaling':(min, max)}``.

        Returns
        -------
        NoneType
            None
        """

        self.object = read_object.ReadObject(object_name)

        self.spectrum = spectrum
        self.bounds = bounds

        self.objphot = []
        self.specphot = []

        if filters is None:
            species_db = database.Database()

            objectbox = species_db.get_object(object_name, inc_phot=True, inc_spec=False)
            filters = objectbox.filters

        for item in filters:
            readcalib = read_calibration.ReadCalibration(self.spectrum, item)
            calibspec = readcalib.get_spectrum()

            synphot = photometry.SyntheticPhotometry(item)
            spec_phot = synphot.spectrum_to_flux(calibspec.wavelength, calibspec.flux)
            self.specphot.append(spec_phot[0])

            obj_phot = self.object.get_photometry(item)
            self.objphot.append(np.array([obj_phot[2], obj_phot[3]]))

        self.modelpar = ['scaling']

    @typechecked
    def run_mcmc(self,
                 nwalkers: int,
                 nsteps: int,
                 guess: Union[Dict[str, float], Dict[str, None]],
                 tag: str) -> None:
        """
        Function to run the MCMC sampler.

        Parameters
        ----------
        nwalkers : int
            Number of walkers.
        nsteps : int
            Number of steps per walker.
        guess : dict(str, float), dict(str, None)
            Guess of the scaling parameter.
        tag : str
            Database tag where the MCMC samples will be stored.

        Returns
        -------
        NoneType
            None
        """

        print('Running MCMC...')

        ndim = 1

        initial = np.zeros((nwalkers, ndim))

        for i, item in enumerate(self.modelpar):
            if guess[item] is not None:
                width = min(abs(guess[item] - self.bounds[item][0]),
                            abs(guess[item] - self.bounds[item][1]))

                initial[:, i] = guess[item] + np.random.normal(0, 0.1*width, nwalkers)

            else:
                initial[:, i] = np.random.uniform(low=self.bounds[item][0],
                                                  high=self.bounds[item][1],
                                                  size=nwalkers)

        with Pool(processes=cpu_count()):
            ens_sampler = emcee.EnsembleSampler(nwalkers,
                                                ndim,
                                                lnprob,
                                                args=([self.bounds,
                                                       self.modelpar,
                                                       self.objphot,
                                                       self.specphot]))

            ens_sampler.run_mcmc(initial, nsteps, progress=True)

        species_db = database.Database()

        species_db.add_samples(sampler='emcee',
                               samples=ens_sampler.get_chain(),
                               ln_prob=ens_sampler.get_log_prob(),
                               ln_evidence=None,
                               mean_accept=np.mean(ens_sampler.acceptance_fraction),
                               spectrum=('calibration', self.spectrum),
                               tag=tag,
                               modelpar=self.modelpar,
                               distance=None,
                               spec_labels=None)
