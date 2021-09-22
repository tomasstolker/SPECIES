.. _tutorials:

Tutorials
=========

This page contains a list of tutorials which highlight some of the functionalities of `species`. The tutorials can be executed on Binder in an `interactive environment <https://mybinder.org/v2/gh/tomasstolker/species/HEAD>`_ or they can be downloaded as `Jupyter notebook <https://github.com/tomasstolker/species/tree/main/docs/tutorials>`_. Please `create an issue <https://github.com/tomasstolker/species/issues>`_ if you encounter any problems with the tutorials.

.. tip::
   Details on the various classes, functions, and parameters can be found in the `API documentation <https://species.readthedocs.io/en/latest/modules.html>`_.

.. toctree::
   :maxdepth: 1

   tutorials/flux_magnitude.ipynb
   tutorials/synthetic_photometry.ipynb
   tutorials/color_magnitude_broadband.ipynb
   tutorials/color_magnitude_narrowband.ipynb
   tutorials/atmospheric_models.ipynb
   tutorials/data_model.ipynb
   tutorials/spectral_library.ipynb
   tutorials/fitting_model_spectra.ipynb
   tutorials/flux_calibration.ipynb
   tutorials/emission_line.ipynb
   tutorials/free_retrieval.ipynb

.. important::
   A flux calibrated spectrum of Vega is used for the conversion between a flux density and magnitude. The magnitude of Vega is set to 0.03 for all filters. If needed, the magnitude of Vega can be changed with the ``vega_mag`` attribute of a ``SyntheticPhotometry`` object:

   .. code-block:: python

       >>> synphot = species.SyntheticPhotometry('MKO/NSFCam.K')
       >>> synphot.vega_mag = 0.01
