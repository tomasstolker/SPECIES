.. _overview:

Overview
========

Introduction
------------

*species* provides a coherent framework for spectral and photometric analysis of directly imaged planets and brown dwarfs. This page contains a short overview of the various data that are supported and some of the tools and features that have been implemented.

Supported data
--------------

The *species* toolkit benefits from publicly available data resources such as photometric and spectral libraries, atmospheric model spectra, evolutionary tracks, and photometry of directly imaged planets and brown dwarfs. The relevant data are automatically downloaded and added to the HDF5 database, which acts as the central data storage for a workflow. All data are stored in a fixed format such that the analysis and plotting tools can easily access and manipulate the data.

The following data and models are currently supported:

**Atmospheric model spectra**

- `AMES-Cond <https://phoenix.ens-lyon.fr/Grids/AMES-Cond/>`_
- `AMES-Dusty <https://phoenix.ens-lyon.fr/Grids/AMES-Dusty/>`_
- `ATMO <http://svo2.cab.inta-csic.es/svo/theory/newov2/index.php?models=atmo2020_ceq>`_
- `BT-Cond <http://svo2.cab.inta-csic.es/svo/theory/newov2/index.php?models=bt-cond>`_
- `BT-NextGen <https://phoenix.ens-lyon.fr/Grids/BT-NextGen/SPECTRA/>`_
- `BT-Settl <http://svo2.cab.inta-csic.es/svo/theory/newov2/index.php?models=bt-settl>`_
- `BT-Settl-CIFIST <http://svo2.cab.inta-csic.es/svo/theory/newov2/index.php?models=bt-settl-cifist>`_
- `DRIFT-PHOENIX <http://svo2.cab.inta-csic.es/theory/newov2/index.php?models=drift>`_
- `Exo-REM <https://ui.adsabs.harvard.edu/abs/2018ApJ...854..172C/abstract>`_
- `petitCODE <http://www.mpia.de/~molliere/#petitcode>`_

**Spectral libraries**

- `IRTF Spectral Library <http://irtfweb.ifa.hawaii.edu/~spex/IRTF_Spectral_Library/>`_
- `SpeX Prism Spectral Libraries <http://pono.ucsd.edu/~adam/browndwarfs/spexprism/index_old.html>`_
- `SDSS spectra by Kesseli et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017ApJS..230...16K/abstract>`_

**Photometric libraries**

- `Database of Ultracool Parallaxes <http://www.as.utexas.edu/~tdupuy/plx/Database_of_Ultracool_Parallaxes.html>`_
- Photometry from `S. Leggett <http://www.gemini.edu/staff/sleggett>`_
- Directly imaged planets and brown dwarfs (see dictionary of :class:`~species.data.companions`)

**Evolutionary tracks**

- All isochrones from the `Phoenix grids <https://phoenix.ens-lyon.fr/Grids/>`_

**Calibration**

- All filters from the `Filter Profile Service <http://svo2.cab.inta-csic.es/svo/theory/fps/>`_
- `Flux-calibrated spectrum of Vega <http://ssb.stsci.edu/cdbs/calspec/>`_

**Dust extinction**

- ISM empirical relation from `Cardelli et al. (1989) <https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C/abstract>`_
- Extinction cross sections computed with `PyMieScatt <https://pymiescatt.readthedocs.io>`_
- Optical constants compiled by `Mollière et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_

Please give credit to the relevant authors when using any of the external data in a publication. More information is available on the respective websites. Support for other datasets can be requested by creating an `issue <https://github.com/tomasstolker/species/issues>`_ on the Github page.

Analysis tools
--------------

After adding the relevant data to the database, the user can take advantage of the suite of tools that have been implemented for spectral and photometric analysis. Here is an incomplete list of available features and tools:

- Converting between fluxes and magnitudes (see :class:`~species.analysis.photometry.SyntheticPhotometry`).
- Calculating synthetic photometry spectra (see :class:`~species.analysis.photometry.SyntheticPhotometry`).
- Interpolating and plotting model spectra (see :class:`~species.read.read_model.ReadModel` and :func:`~species.plot.plot_spectrum.plot_spectrum`).
- Grid retrievals with Bayesian inference (see :class:`~species.analysis.fit_model.FitModel` and :class:`~species.plot.plot_mcmc`).
- Free retrievals through a frontend for `petitRADTRANS <https://petitradtrans.readthedocs.io>`_  (see `AtmosphericRetrieval <https://github.com/tomasstolker/species/blob/retrieval/species/analysis/retrieval.py>`_ on the `retrieval branch <https://github.com/tomasstolker/species/tree/retrieval>`_).
- Creating color-magnitude diagrams (see :class:`~species.read.read_color.ReadColorMagnitude` and :class:`~species.plot.plot_color.plot_color_magnitude`).
- Creating color-color diagrams (see :class:`~species.read.read_color.ReadColorColor` and :class:`~species.plot.plot_color.plot_color_color`).
- Computing synthetic fluxes from isochrones and model spectra (see :class:`~species.read.read_isochrone.ReadIsochrone`)
- Flux calibration of photometric and spectroscopic data (see :class:`~species.read.read_calibration.ReadCalibration`, :class:`~species.analysis.fit_model.FitModel`, and :class:`~species.analysis.fit_spectrum.FitSpectrum`).
- Empirical comparison of spectra to infer the spectral type (see :class:`~species.analysis.empirical.CompareSpectra`).
