.. _overview:

Overview
========

Introduction
------------

*species* provides a coherent framework for spectral and photometric analysis of directly imaged planets and brown dwarfs. This page contains a short overview of the various data that are supported and some of the tools and features that are provided.

Supported data
--------------

The toolkit benefits from publicly available data resources such as photometric and spectral libraries, atmospheric model spectra, evolutionary tracks, and photometry of directly imaged, low-mass objects. The relevant data are automatically downloaded and added to the HDF5 database, which acts as the central data storage for a workflow. All data are stored in a fixed format such that the analysis and plotting tools can easily access and process the data.

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
- `petitRADTRANS <https://petitradtrans.readthedocs.io>`_
- `T/Y dwarf spectra from Morley et al. (2012) <https://ui.adsabs.harvard.edu/abs/2012ApJ...756..172M/abstract>`_

**Spectral libraries**

- `IRTF Spectral Library <http://irtfweb.ifa.hawaii.edu/~spex/IRTF_Spectral_Library/>`_
- `SpeX Prism Spectral Libraries <http://pono.ucsd.edu/~adam/browndwarfs/spexprism/index_old.html>`_
- `SDSS spectra by Kesseli et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017ApJS..230...16K/abstract>`_
- NIR spectra of young M/L dwarfs by `Allers & Liu (2013) <https://ui.adsabs.harvard.edu/abs/2013ApJ...772...79A/abstract>`_
- NIR spectra of young M/L dwarfs by `Bonnefoy et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014A%26A...562A.127B/abstract>`_
- `Spectra of directly imaged planets and brown dwarfs <https://species.readthedocs.io/en/latest/species.data.html#species.data.companions.get_spec_data>`_

**Photometric libraries**

- `Database of Ultracool Parallaxes <http://www.as.utexas.edu/~tdupuy/plx/Database_of_Ultracool_Parallaxes.html>`_
- Photometry from `S. Leggett <http://www.gemini.edu/staff/sleggett>`_
- `Magnitudes of directly imaged planets and brown dwarfs <https://species.readthedocs.io/en/latest/species.data.html#species.data.companions.get_data>`_

**Evolutionary tracks**

- All isochrones from the `Phoenix grids <https://phoenix.ens-lyon.fr/Grids/>`_

**Calibration**

- All filters from the `Filter Profile Service <http://svo2.cab.inta-csic.es/svo/theory/fps/>`_
- `Flux-calibrated spectrum of Vega <http://ssb.stsci.edu/cdbs/calspec/>`_

**Dust extinction**

- ISM relation from `Cardelli et al. (1989) <https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C/abstract>`_
- Extinction cross sections computed with `PyMieScatt <https://pymiescatt.readthedocs.io>`_
- Optical constants compiled by `Mollière et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_

Please give credit to the relevant references when using any of the external data in a publication. More information is available on the respective websites. Support for other datasets can be requested by creating an `issue <https://github.com/tomasstolker/species/issues>`_ on the Github page.

Analysis tools
--------------

After adding the relevant data to the database, the user can take advantage of the suite of tools that have been implemented for spectral and photometric analysis. Here is an incomplete list of available features and tools:

- Converting between fluxes and magnitudes (see :class:`~species.analysis.photometry.SyntheticPhotometry`).
- Calculating synthetic photometry spectra (see :class:`~species.analysis.photometry.SyntheticPhotometry`).
- Interpolating and plotting model spectra (see :class:`~species.read.read_model.ReadModel` and :func:`~species.plot.plot_spectrum.plot_spectrum`).
- Grid retrievals with Bayesian inference (see :class:`~species.analysis.fit_model.FitModel` and :mod:`~species.plot.plot_mcmc`).
- Comparing a spectrum with a full grid of model spectra (see :meth:`~species.analysis.compare_spectra.CompareSpectra.compare_model`).
- Free retrievals with a frontend for `petitRADTRANS <https://petitradtrans.readthedocs.io>`_  (see :class:`~species.analysis.retrieval.AtmosphericRetrieval`).
- Creating color-magnitude diagrams (see :class:`~species.read.read_color.ReadColorMagnitude` and :class:`~species.plot.plot_color.plot_color_magnitude`).
- Creating color-color diagrams (see :class:`~species.read.read_color.ReadColorColor` and :class:`~species.plot.plot_color.plot_color_color`).
- Computing synthetic fluxes from isochrones and model spectra (see :class:`~species.read.read_isochrone.ReadIsochrone`)
- Flux calibration of photometric and spectroscopic data (see :class:`~species.read.read_calibration.ReadCalibration`, :class:`~species.analysis.fit_model.FitModel`, and :class:`~species.analysis.fit_spectrum.FitSpectrum`).
- Empirical comparison of spectra to infer the spectral type (see :meth:`~species.analysis.compare_spectra.CompareSpectra.spectral_type`).
- Analyzing emission lines from accreting planets (see :class:`~species.analysis.emission_line.EmissionLine`).
