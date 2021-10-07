"""
Text
"""

import os
import urllib.request

import h5py
import numpy as np
import pandas as pd

from species.util import data_util


def add_leggett(input_path, database):
    """
    Function for adding the Database of Ultracool Parallaxes to the database.

    Parameters
    ----------
    input_path : str
        Path of the data folder.
    database : h5py._hl.files.File
        Database.

    Returns
    -------
    NoneType
        None
    """

    data_file1 = os.path.join(input_path, "2010_phot.xls")
    url1 = "http://staff.gemini.edu/~sleggett/2010_phot.xls"

    data_file2 = os.path.join(input_path, "datafile8.txt")
    url2 = "http://staff.gemini.edu/~sleggett/datafile8.txt"

    if not os.path.isfile(data_file1):
        print("Downloading Leggett L and T Dwarf Data (88 kB)...", end="", flush=True)
        urllib.request.urlretrieve(url1, data_file1)
        print(" [DONE]")

    if not os.path.isfile(data_file2):
        print("Downloading Leggett T6+ and Y Dwarf Data (44 kB)...", end="", flush=True)
        urllib.request.urlretrieve(url2, data_file2)
        print(" [DONE]")

    print("Adding Leggett L and T Dwarf Data...", end="", flush=True)

    group = "photometry/leggett"

    database.create_group(group)

    dataframe = pd.pandas.read_excel(data_file1)
    dataframe.columns = dataframe.columns.str.replace("'", "")

    modulus = np.asarray(dataframe["M-m"])  # M-m (mag)
    modulus_error = np.asarray(dataframe["Mmerr"])  # M-m (mag)

    distance = 10.0 ** (-modulus / 5.0 + 1.0)  # (pc)
    distance_lower = distance - 10.0 ** (-(modulus + modulus_error) / 5.0 + 1.0)  # (pc)
    distance_upper = 10.0 ** (-(modulus - modulus_error) / 5.0 + 1.0) - distance  # (pc)
    distance_error = (distance_lower + distance_upper) / 2.0

    name = np.asarray(dataframe["Name"])

    # Near-infrared spectral type
    sptype = np.asarray(dataframe["Type"])
    sptype = data_util.update_sptype(sptype)
    sptype = np.asarray(sptype)

    mag_y = np.asarray(dataframe["Y"])
    mag_j = np.asarray(dataframe["J"])
    mag_h = np.asarray(dataframe["H"])
    mag_k = np.asarray(dataframe["K"])
    mag_lp = np.asarray(dataframe["L"])
    mag_mp = np.asarray(dataframe["M"])
    mag_ch1 = np.asarray(dataframe["Ch1"])
    mag_ch2 = np.asarray(dataframe["Ch2"])
    mag_ch3 = np.asarray(dataframe["Ch3"])
    mag_ch4 = np.asarray(dataframe["Ch4"])
    mag_w1 = np.repeat(np.nan, np.size(name))
    mag_w2 = np.repeat(np.nan, np.size(name))
    mag_w3 = np.repeat(np.nan, np.size(name))
    mag_w4 = np.repeat(np.nan, np.size(name))

    print(" [DONE]")
    print("Adding Leggett T6+ and Y Dwarf Data...", end="", flush=True)

    file_io = open(data_file2, "r")
    lines = file_io.readlines()[69:]

    for item in lines:
        name = np.append(name, item[0:16])

        spt_tmp = item[62:66]
        if spt_tmp[0] == "2":
            spt_tmp = "T" + spt_tmp[1]
        elif spt_tmp[0] == "3":
            spt_tmp = "Y" + spt_tmp[1]

        sptype = np.append(sptype, spt_tmp)

        modulus = float(item[67:73])  # M-m (mag)
        if modulus == 999.0:
            modulus = np.nan

        distance = np.append(distance, 10.0 ** (-modulus / 5.0 + 1.0))  # (pc)

        mag = np.zeros(14)

        mag[0] = float(item[95:101])  # MKO Y
        mag[1] = float(item[102:107])  # MKO J
        mag[2] = float(item[108:114])  # MKO H
        mag[3] = float(item[115:121])  # MKO K
        mag[4] = float(item[122:128])  # MKO L'
        mag[5] = float(item[129:135])  # MKO M'
        mag[6] = float(item[136:142])  # Spitzer/IRAC 3.6 um
        mag[7] = float(item[143:149])  # Spitzer/IRAC 4.5 um
        mag[8] = float(item[150:156])  # Spitzer/IRAC 5.8 um
        mag[9] = float(item[157:163])  # Spitzer/IRAC 8.0 um
        mag[10] = float(item[164:170])  # WISE W1
        mag[11] = float(item[171:176])  # WISE W2
        mag[12] = float(item[177:183])  # WISE W3
        mag[13] = float(item[184:190])  # WISE W4

        for j, mag_item in enumerate(mag):
            if mag_item == 999.0:
                mag[j] = np.nan

        mag_y = np.append(mag_y, mag[0])
        mag_j = np.append(mag_j, mag[1])
        mag_h = np.append(mag_h, mag[2])
        mag_k = np.append(mag_k, mag[3])
        mag_lp = np.append(mag_lp, mag[4])
        mag_mp = np.append(mag_mp, mag[5])
        mag_ch1 = np.append(mag_ch1, mag[6])
        mag_ch2 = np.append(mag_ch2, mag[7])
        mag_ch3 = np.append(mag_ch3, mag[8])
        mag_ch4 = np.append(mag_ch4, mag[9])
        mag_w1 = np.append(mag_w1, mag[10])
        mag_w2 = np.append(mag_w2, mag[11])
        mag_w3 = np.append(mag_w3, mag[12])
        mag_w4 = np.append(mag_w4, mag[13])

    file_io.close()

    dtype = h5py.special_dtype(vlen=str)

    dset = database.create_dataset(group + "/name", (np.size(name),), dtype=dtype)
    dset[...] = name

    dset = database.create_dataset(group + "/sptype", (np.size(sptype),), dtype=dtype)
    dset[...] = sptype

    flag = np.repeat("null", np.size(name))

    dset = database.create_dataset(group + "/flag", (np.size(flag),), dtype=dtype)
    dset[...] = flag

    database.create_dataset(group + "/distance", data=distance)
    database.create_dataset(group + "/distance_error", data=distance_error)
    database.create_dataset(group + "/MKO/NSFCam.Y", data=mag_y)
    database.create_dataset(group + "/MKO/NSFCam.J", data=mag_j)
    database.create_dataset(group + "/MKO/NSFCam.H", data=mag_h)
    database.create_dataset(group + "/MKO/NSFCam.K", data=mag_k)
    database.create_dataset(group + "/MKO/NSFCam.Lp", data=mag_lp)
    database.create_dataset(group + "/MKO/NSFCam.Mp", data=mag_mp)
    database.create_dataset(group + "/Spitzer/IRAC.I1", data=mag_ch1)
    database.create_dataset(group + "/Spitzer/IRAC.I2", data=mag_ch2)
    database.create_dataset(group + "/Spitzer/IRAC.I3", data=mag_ch3)
    database.create_dataset(group + "/Spitzer/IRAC.I4", data=mag_ch4)
    database.create_dataset(group + "/WISE/WISE.W1", data=mag_w1)
    database.create_dataset(group + "/WISE/WISE.W2", data=mag_w2)
    database.create_dataset(group + "/WISE/WISE.W3", data=mag_w3)
    database.create_dataset(group + "/WISE/WISE.W4", data=mag_w4)

    print(" [DONE]")

    database.close()
