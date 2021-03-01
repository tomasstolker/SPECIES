"""
Module for extracting data of directly imaged planets and brown dwarfs.
"""

from typing import Dict, List, Tuple, Union

from typeguard import typechecked


@typechecked
def get_data() -> Dict[str, Dict[str, Union[Tuple[float, float],
                       Dict[str, Union[Tuple[float, float], List[Tuple[float, float]]]]]]]:
    """
    Function for extracting a dictionary with the distances (pc) and apparent magnitudes of
    directly imaged planets and brown dwarfs.

    Returns
    -------
    dict
        Dictionary with the distances and apparent magnitudes of directly imaged companions.
        Distances are from GAIA DR2 unless indicated as comment.
    """

    data = {'beta Pic b': {'distance': (19.75, 0.13),
                           'app_mag': {'Magellan/VisAO.Ys': (15.53, 0.34),  # Males et al. 2014,
                                       'Paranal/NACO.J': (14.11, 0.21),  # Currie et al. 2013
                                       'Gemini/NICI.ED286': (13.18, 0.15),  # Males et al. 2014
                                       'Paranal/NACO.H': (13.32, 0.14),  # Currie et al. 2013
                                       'Paranal/NACO.Ks': (12.64, 0.11),  # Bonnefoy et al. 2011
                                       'Paranal/NACO.NB374': (11.25, 0.23),  # Stolker et al. 2020
                                       'Paranal/NACO.Lp': (11.30, 0.06),  # Stolker et al. 2019
                                       'Paranal/NACO.NB405': (10.98, 0.05),  # Stolker et al. 2020
                                       'Paranal/NACO.Mp': (11.10, 0.12)}},  # Stolker et al. 2019

            'HIP 65426 b': {'distance': (109.21, 0.75),
                            'app_mag': {'Paranal/SPHERE.IRDIS_D_H23_2': (17.94, 0.05),  # Chauvin et al. 2017
                                        'Paranal/SPHERE.IRDIS_D_H23_3': (17.58, 0.06),  # Chauvin et al. 2017
                                        'Paranal/SPHERE.IRDIS_D_K12_1': (17.01, 0.09),  # Chauvin et al. 2017
                                        'Paranal/SPHERE.IRDIS_D_K12_2': (16.79, 0.09),  # Chauvin et al. 2017
                                        'Paranal/NACO.Lp': (15.33, 0.12),  # Stolker et al. 2020
                                        'Paranal/NACO.NB405': (15.23, 0.22),  # Stolker et al. 2020
                                        'Paranal/NACO.Mp': (14.65, 0.29)}},  # Stolker et al. 2020

            '51 Eri b': {'distance': (29.78, 0.12),
                         'app_mag': {'MKO/NSFCam.J': (19.04, 0.40),  # Rajan et al. 2017
                                     'MKO/NSFCam.H': (18.99, 0.21),  # Rajan et al. 2017
                                     'MKO/NSFCam.K': (18.67, 0.19),  # Rajan et al. 2017
                                     'Paranal/SPHERE.IRDIS_B_H': (19.45, 0.29),  # Samland et al. 2017
                                     'Paranal/SPHERE.IRDIS_D_H23_2': (18.41, 0.26),  # Samland et al. 2017
                                     'Paranal/SPHERE.IRDIS_D_K12_1': (17.55, 0.14),  # Samland et al. 2017
                                     'Keck/NIRC2.Lp': (16.20, 0.11),  # Rajan et al. 2017
                                     'Keck/NIRC2.Ms': (16.1, 0.5)}},  # Rajan et al. 2017

            'HR 8799 b': {'distance': (41.29, 0.15),
                          'app_mag': {'Subaru/CIAO.z': (21.22, 0.29),  # Currie et al. 2011
                                      'Paranal/SPHERE.IRDIS_B_J': (19.78, 0.09),  # Zurlo et al. 2016
                                      'Keck/NIRC2.H': (18.05, 0.09),  # Currie et al. 2012
                                      'Paranal/SPHERE.IRDIS_D_H23_2': (18.08, 0.14),  # Zurlo et al. 2016
                                      'Paranal/SPHERE.IRDIS_D_H23_3': (17.78, 0.10),  # Zurlo et al. 2016
                                      'Keck/NIRC2.Ks': (17.03, 0.08),  # Marois et al. 2010
                                      'Paranal/SPHERE.IRDIS_D_K12_1': (17.15, 0.06),  # Zurlo et al. 2016
                                      'Paranal/SPHERE.IRDIS_D_K12_2': (16.97, 0.09),  # Zurlo et al. 2016
                                      'Paranal/NACO.Lp': (15.52, 0.10),  # Currie et al. 2014
                                      'Paranal/NACO.NB405': (14.82, 0.18),  # Currie et al. 2014
                                      'Keck/NIRC2.Ms': (16.05, 0.30)}},  # Galicher et al. 2011

            'HR 8799 c': {'distance': (41.29, 0.15),
                          'app_mag': {'Paranal/SPHERE.IRDIS_B_J': (18.60, 0.13),  # Zurlo et al. 2016
                                      'Keck/NIRC2.H': (17.06, 0.13),  # Currie et al. 2012
                                      'Paranal/SPHERE.IRDIS_D_H23_2': (17.09, 0.12),  # Zurlo et al. 2016
                                      'Paranal/SPHERE.IRDIS_D_H23_3': (16.78, 0.10),  # Zurlo et al. 2016
                                      'Keck/NIRC2.Ks': (16.11, 0.08),  # Marois et al. 2010
                                      'Paranal/SPHERE.IRDIS_D_K12_1': (16.19, 0.05),  # Zurlo et al. 2016
                                      'Paranal/SPHERE.IRDIS_D_K12_2': (15.86, 0.07),  # Zurlo et al. 2016
                                      'Paranal/NACO.Lp': (14.65, 0.11),  # Currie et al. 2014
                                      'Paranal/NACO.NB405': (13.97, 0.11),  # Currie et al. 2014
                                      'Keck/NIRC2.Ms': (15.03, 0.14)}},  # Galicher et al. 2011

            'HR 8799 d': {'distance': (41.29, 0.15),
                          'app_mag': {'Paranal/SPHERE.IRDIS_B_J': (18.59, 0.37),  # Zurlo et al. 2016
                                      'Keck/NIRC2.H': (16.71, 0.24),  # Currie et al. 2012
                                      'Paranal/SPHERE.IRDIS_D_H23_2': (17.02, 0.17),  # Zurlo et al. 2016
                                      'Paranal/SPHERE.IRDIS_D_H23_3': (16.85, 0.16),  # Zurlo et al. 2016
                                      'Keck/NIRC2.Ks': (16.09, 0.12),  # Marois et al. 2010
                                      'Paranal/SPHERE.IRDIS_D_K12_1': (16.20, 0.07),  # Zurlo et al. 2016
                                      'Paranal/SPHERE.IRDIS_D_K12_2': (15.84, 0.10),  # Zurlo et al. 2016
                                      'Paranal/NACO.Lp': (14.55, 0.14),  # Currie et al. 2014
                                      'Paranal/NACO.NB405': (13.87, 0.15),  # Currie et al. 2014
                                      'Keck/NIRC2.Ms': (14.65, 0.35)}},  # Galicher et al. 2011

            'HR 8799 e': {'distance': (41.29, 0.15),
                          'app_mag': {'Paranal/SPHERE.IRDIS_B_J': (18.40, 0.21),  # Zurlo et al. 2016
                                      'Paranal/SPHERE.IRDIS_D_H23_2': (16.91, 0.20),  # Zurlo et al. 2016
                                      'Paranal/SPHERE.IRDIS_D_H23_3': (16.68, 0.21),  # Zurlo et al. 2016
                                      'Keck/NIRC2.Ks': (15.91, 0.22),  # Marois et al. 2010
                                      'Paranal/SPHERE.IRDIS_D_K12_1': (16.12, 0.10),  # Zurlo et al. 2016
                                      'Paranal/SPHERE.IRDIS_D_K12_2': (15.82, 0.11),  # Zurlo et al. 2016
                                      'Paranal/NACO.Lp': (14.49, 0.21),  # Currie et al. 2014
                                      'Paranal/NACO.NB405': (13.72, 0.20)}},  # Currie et al. 2014

            'HD 95086 b': {'distance': (86.44, 0.24),
                           'app_mag': {'Gemini/GPI.H': (20.51, 0.25),  # De Rosa et al. 2016
                                       'Gemini/GPI.K1': (18.99, 0.20),  # De Rosa et al. 2016
                                       'Paranal/NACO.Lp': (16.27, 0.19)}},  # De Rosa et al. 2016

            'PDS 70 b': {'distance': (113.43, 0.52),
                         'app_mag': {'Paranal/SPHERE.IRDIS_D_H23_2': (18.12, 0.21),  # Stolker et al. 2020.
                                     'Paranal/SPHERE.IRDIS_D_H23_3': (17.97, 0.18),  # Stolker et al. 2020.
                                     'Paranal/SPHERE.IRDIS_D_K12_1': (16.66, 0.04),  # Stolker et al. 2020.
                                     'Paranal/SPHERE.IRDIS_D_K12_2': (16.37, 0.06),  # Stolker et al. 2020.
                                     'MKO/NSFCam.J': (20.04, 0.09),  # Stolker et al. 2020 / Müller et al. 2017
                                     'MKO/NSFCam.H': (18.24, 0.04),  # Stolker et al. 2020 / Müller et al. 2017
                                     'Paranal/NACO.Lp': (14.68, 0.22),  # Stolker et al. 2020.
                                     'Paranal/NACO.NB405': (14.68, 0.27),  # Stolker et al. 2020
                                     'Paranal/NACO.Mp': (13.80, 0.27),  # Stolker et al. 2020
                                     'Keck/NIRC2.Lp': (14.64, 0.18)}},  # Wang et al. 2020

            'PDS 70 c': {'distance': (113.43, 0.52),
                         'app_mag': {'Paranal/NACO.NB405': (14.91, 0.35),  # Stolker et al. 2020
                                     'Keck/NIRC2.Lp': (15.5, 0.46)}},  # Wang et al. 2020

            '2M1207 b': {'distance': (64.42, 0.65),
                         'app_mag': {'HST/NICMOS1.F090M': (22.58, 0.35),  # Song et al. 2006
                                     'HST/NICMOS1.F110M': (20.61, 0.15),  # Song et al. 2006
                                     'HST/NICMOS1.F145M': (19.05, 0.03),  # Song et al. 2006
                                     'HST/NICMOS1.F160W': (18.27, 0.02),  # Song et al. 2006
                                     'Paranal/NACO.J': (20.0, 0.2),  # Mohanty et al. 200z
                                     'Paranal/NACO.H': (18.09, 0.21),  # Chauvin et al. 2004
                                     'Paranal/NACO.Ks': (16.93, 0.11),  # Chauvin et al. 2004
                                     'Paranal/NACO.Lp': (15.28, 0.14)}},  # Chauvin et al. 2004

            'AB Pic B': {'distance': (50.12, 0.07),
                         'app_mag': {'Paranal/NACO.J': (16.18, 0.10),  # Chauvin et al. 2005
                                     'Paranal/NACO.H': (14.69, 0.10),  # Chauvin et al. 2005
                                     'Paranal/NACO.Ks': (14.14, 0.08)}},  # Chauvin et al. 2005

            'HD 206893 B': {'distance': (40.81, 0.11),
                            'app_mag': {'Paranal/SPHERE.IRDIS_B_H': (16.79, 0.06),  # Milli et al. 2017
                                        'Paranal/SPHERE.IRDIS_D_K12_1': (15.2, 0.10),  # Delorme et al. 2017
                                        'Paranal/SPHERE.IRDIS_D_K12_2': (14.88, 0.09),  # Delorme et al. 2017
                                        'Paranal/NACO.Lp': (13.79, 0.31),  # Stolker et al. 2020
                                        'Paranal/NACO.NB405': (13.16, 0.34),  # Stolker et al. 2020
                                        'Paranal/NACO.Mp': (12.77, 0.27)}},  # Stolker et al. 2020

            'RZ Psc B': {'distance': (195.86, 4.03),
                         'app_mag': {'Paranal/SPHERE.IRDIS_B_H': [(13.71, 0.14),  # Kennedy et al. 2020
                                                                  (13.85, 0.26)],  # Kennedy et al. 2020
                                     'Paranal/SPHERE.IRDIS_B_Ks': (13.51, 0.20)}},  # Kennedy et al. 2020

            'GQ Lup B': {'distance': (151.82, 1.10),
                         'app_mag': {'HST/WFPC2-PC.F606W': (19.19, 0.07),  # Marois et al. 2007
                                     'HST/WFPC2-PC.F814W': (17.67, 0.05),  # Marois et al. 2007
                                     'HST/NICMOS2.F171M': (13.84, 0.13),  # Marois et al. 2007
                                     'HST/NICMOS2.F190N': (14.08, 0.20),  # Marois et al. 2007
                                     'HST/NICMOS2.F215N': (13.40, 0.15),  # Marois et al. 2007
                                     'Magellan/VisAO.ip': (18.89, 0.24),  # Wu et al. 2017
                                     'Magellan/VisAO.zp': (16.40, 0.10),  # Wu et al. 2017
                                     'Magellan/VisAO.Ys': (15.88, 0.10),  # Wu et al. 2017
                                     'Paranal/NACO.Ks': [(13.474, 0.031),  # Ginski et al. 2014
                                                         (13.386, 0.032),  # Ginski et al. 2014
                                                         (13.496, 0.050),  # Ginski et al. 2014
                                                         (13.501, 0.028)],  # Ginski et al. 2014
                                     'Subaru/CIAO.CH4s': (13.76, 0.26),  # Marois et al. 2007
                                     'Subaru/CIAO.K': (13.37, 0.12),  # Marois et al. 2007
                                     'Subaru/CIAO.Lp': (12.44, 0.22)}},  # Marois et al. 2007

            'PZ Tel B': {'distance': (47.13, 0.13),
                         'app_mag': {'Paranal/SPHERE.ZIMPOL_R_PRIM': (17.84, 0.31),  # Maire et al. 2015
                                     'Paranal/SPHERE.ZIMPOL_I_PRIM': (15.16, 0.12),  # Maire et al. 2015
                                     'Paranal/SPHERE.IRDIS_D_H23_2': (11.78, 0.19),  # Maire et al. 2015
                                     'Paranal/SPHERE.IRDIS_D_H23_3': (11.65, 0.19),  # Maire et al. 2015
                                     'Paranal/SPHERE.IRDIS_D_K12_1': (11.56, 0.09),  # Maire et al. 2015
                                     'Paranal/SPHERE.IRDIS_D_K12_2': (11.29, 0.10),  # Maire et al. 2015
                                     'Paranal/NACO.J': (12.47, 0.20),  # Biller et al. 2010
                                     'Paranal/NACO.H': (11.93, 0.14),  # Biller et al. 2010
                                     'Paranal/NACO.Ks': (11.53, 0.07),  # Biller et al. 2010
                                     'Paranal/NACO.Lp': (11.04, 0.22),  # Stolker et al. 2020
                                     'Paranal/NACO.NB405': (10.94, 0.07),  # Stolker et al. 2020
                                     'Paranal/NACO.Mp': (10.93, 0.03),  # Stolker et al. 2020
                                     'Gemini/NICI.ED286': (11.68, 0.14),  # Biller et al. 2010
                                     'Gemini/NIRI.H2S1v2-1-G0220': (11.39, 0.14)}},  # Biller et al. 2010

            'kappa And b': {'distance': (50.06, 0.87),
                            'app_mag': {'Subaru/CIAO.J': (15.86, 0.21),  # Bonnefoy et al. 2014
                                        'Subaru/CIAO.H': (14.95, 0.13),  # Bonnefoy et al. 2014
                                        'Subaru/CIAO.Ks': (14.32, 0.09),  # Bonnefoy et al. 2014
                                        'Keck/NIRC2.Lp': (13.12, 0.1),  # Bonnefoy et al. 2014
                                        # 'Keck/NIRC2.NB_4.05': (13.0, 0.2),  # Bonnefoy et al. 2014
                                        'LBT/LMIRCam.M_77K': (13.3, 0.3)}},  # Bonnefoy et al. 2014

            'HD 1160 B': {'distance': (125.9, 1.2),
                          'app_mag': {'MKO/NSFCam.J': (14.69, 0.05),  # Victor Garcia et al. 2017
                                      'MKO/NSFCam.H': (14.21, 0.02),  # Victor Garcia et al. 2017
                                      'MKO/NSFCam.Ks': (14.12, 0.05),  # Nielsen et al. 2012
                                      'Paranal/NACO.Lp': (13.60, 0.10),  # Maire et al. 2016
                                      'Keck/NIRC2.Ms': (13.81, 0.24)}},  # Victor Garcia et al. 2017

            'ROXs 42 Bb': {'distance': (144.16, 1.53),
                           'app_mag': {'Keck/NIRC2.J': (16.91, 0.11),  # Daemgen et al. 2017
                                       'Keck/NIRC2.H': (15.88, 0.05),  # Daemgen et al. 2017
                                       'Keck/NIRC2.Ks': (15.01, 0.06),  # Daemgen et al. 2017
                                       'Keck/NIRC2.Lp': (13.97, 0.06),  # Daemgen et al. 2017
                                       # 'Keck/NIRC2.NB_4.05': (13.90, 0.08),  # Daemgen et al. 2017
                                       'Keck/NIRC2.Ms': (14.01, 0.23)}},  # Daemgen et al. 2017

            'GJ 504 b': {'distance': (17.54, 0.08),
                         'app_mag': {'Paranal/SPHERE.IRDIS_D_Y23_2': (20.98, 0.20),  # Bonnefoy et al. 2018
                                     'Paranal/SPHERE.IRDIS_D_Y23_3': (20.14, 0.09),  # Bonnefoy et al. 2018
                                     'Paranal/SPHERE.IRDIS_D_J23_3': (19.01, 0.17),  # Bonnefoy et al. 2018
                                     'Paranal/SPHERE.IRDIS_D_H23_2': (18.95, 0.30),  # Bonnefoy et al. 2018
                                     'Paranal/SPHERE.IRDIS_D_H23_3': (21.81, 0.35),  # Bonnefoy et al. 2018
                                     'Paranal/SPHERE.IRDIS_D_K12_1': (18.77, 0.20),  # Bonnefoy et al. 2018
                                     'Subaru/CIAO.J': (19.78, 0.10),  # Janson et al. 2013
                                     'Subaru/CIAO.H': (20.01, 0.14),  # Janson et al. 2013
                                     'Subaru/CIAO.Ks': (19.38, 0.11),  # Janson et al. 2013
                                     'Subaru/CIAO.CH4s': (19.58, 0.13),  # Janson et al. 2013
                                     'Subaru/IRCS.Lp': (16.70, 0.17)}},  # Kuzuhara et al. 2013

            'GU Psc b': {'distance': (47.61, 0.16),
                         'app_mag': {'Gemini/GMOS-S.z': (21.75, 0.07),  # Naud et al. 2014
                                     'CFHT/Wircam.Y': (19.4, 0.05),  # Naud et al. 2014
                                     'CFHT/Wircam.J': (18.12, 0.03),  # Naud et al. 2014
                                     'CFHT/Wircam.H': (17.70, 0.03),  # Naud et al. 2014
                                     'CFHT/Wircam.Ks': (17.40, 0.03),  # Naud et al. 2014
                                     'WISE/WISE.W1': (17.17, 0.33),  # Naud et al. 2014
                                     'WISE/WISE.W2': (15.41, 0.22)}},  # Naud et al. 2014

            '2M0103 ABb': {'distance': (47.2, 3.1),  # Delorme et al. 2013
                           'app_mag': {'Paranal/NACO.J': (15.47, 0.30),  # Delorme et al. 2013
                                       'Paranal/NACO.H': (14.27, 0.20),  # Delorme et al. 2013
                                       'Paranal/NACO.Ks': (13.67, 0.20),  # Delorme et al. 2013
                                       'Paranal/NACO.Lp': (12.67, 0.10)}},  # Delorme et al. 2013

            '1RXS 1609 B': {'distance': (139.67, 1.33),
                            'app_mag': {'Gemini/NIRI.J-G0202w': (17.90, 0.12),  # Lafreniere et al. 2008
                                        'Gemini/NIRI.H-G0203w': (16.87, 0.07),  # Lafreniere et al. 2008
                                        'Gemini/NIRI.K-G0204w': (16.17, 0.18),  # Lafreniere et al. 2008
                                        'Gemini/NIRI.Lprime-G0207w': (14.8, 0.3)}},  # Lafreniere et al. 2010

            'GSC 06214 B': {'distance': (108.84, 0.51),
                            'app_mag': {'MKO/NSFCam.J': (16.24, 0.04),  # Ireland et al. 2011
                                        'MKO/NSFCam.H': (15.55, 0.04),  # Ireland et al. 2011
                                        'MKO/NSFCam.Kp': (14.95, 0.05),  # Ireland et al. 2011
                                        'MKO/NSFCam.Lp': (13.75, 0.07),  # Ireland et al. 2011
                                        'LBT/LMIRCam.M_77K': (13.75, 0.3)}},  # Bailey et al. 2013

            'HD 72946 B': {'distance': (25.87, 0.03),
                           'app_mag': {'Paranal/SPHERE.IRDIS_D_H23_2': (14.56, 0.07),  # Maire et al. 2019
                                       'Paranal/SPHERE.IRDIS_D_H23_3': (14.40, 0.07)}},  # Maire et al. 2019

            'HIP 64892 B': {'distance': (125.20, 1.42),
                            'app_mag': {'Paranal/SPHERE.IRDIS_D_H23_2': (14.21, 0.17),  # Cheetham et al. 2018
                                        'Paranal/SPHERE.IRDIS_D_H23_3': (13.94, 0.17),  # Cheetham et al. 2018
                                        'Paranal/SPHERE.IRDIS_D_K12_1': (13.77, 0.17),  # Cheetham et al. 2018
                                        'Paranal/SPHERE.IRDIS_D_K12_2': (13.45, 0.19),  # Cheetham et al. 2018
                                        'Paranal/NACO.Lp': (13.09, 0.17)}},  # Cheetham et al. 2018

            'TYC 8988 b': {'distance': (94.6, 0.3),
                           'app_mag': {'Paranal/SPHERE.IRDIS_D_Y23_2': (17.03, 0.21),  # Bohn et al. 2019
                                       'Paranal/SPHERE.IRDIS_D_Y23_3': (16.67, 0.16),  # Bohn et al. 2019
                                       'Paranal/SPHERE.IRDIS_D_J23_2': (16.27, 0.08),  # Bohn et al. 2019
                                       'Paranal/SPHERE.IRDIS_D_J23_3': (15.73, 0.07),  # Bohn et al. 2019
                                       'Paranal/SPHERE.IRDIS_D_H23_2': (15.11, 0.08),  # Bohn et al. 2019
                                       'Paranal/SPHERE.IRDIS_D_H23_3': (14.78, 0.07),  # Bohn et al. 2019
                                       'Paranal/SPHERE.IRDIS_D_K12_1': (14.44, 0.04),  # Bohn et al. 2019
                                       'Paranal/SPHERE.IRDIS_D_K12_2': (14.07, 0.04),  # Bohn et al. 2019
                                       'Paranal/SPHERE.IRDIS_B_J': (15.73, 0.38),  # Bohn et al. 2019
                                       'Paranal/SPHERE.IRDIS_B_H': (15.87, 0.38),  # Bohn et al. 2019
                                       'Paranal/SPHERE.IRDIS_B_Ks': (14.70, 0.14),  # Bohn et al. 2019
                                       'Paranal/NACO.Lp': (13.30, 0.08),  # Bohn et al. 2019
                                       'Paranal/NACO.Mp': (13.08, 0.20)}},  # Bohn et al. 2019

            'TYC 8988 c': {'distance': (94.6, 0.3),
                           'app_mag': {'Paranal/SPHERE.IRDIS_D_Y23_3': (22.37, 0.31),  # Bohn et al. 2020
                                       'Paranal/SPHERE.IRDIS_D_J23_2': (21.81, 0.22),  # Bohn et al. 2020
                                       'Paranal/SPHERE.IRDIS_D_J23_3': (21.17, 0.15),  # Bohn et al. 2020
                                       'Paranal/SPHERE.IRDIS_D_H23_2': (19.78, 0.08),  # Bohn et al. 2020
                                       'Paranal/SPHERE.IRDIS_D_H23_3': (19.32, 0.06),  # Bohn et al. 2020
                                       'Paranal/SPHERE.IRDIS_D_K12_1': (18.34, 0.04),  # Bohn et al. 2020
                                       'Paranal/SPHERE.IRDIS_D_K12_2': (17.85, 0.09),  # Bohn et al. 2020
                                       'Paranal/SPHERE.IRDIS_B_H': (19.69, 0.23),  # Bohn et al. 2020
                                       'Paranal/NACO.Lp': (16.29, 0.21)}}}  # Bohn et al. 2020

    return data
