import sys
import os
import math
import time
import pandas as pd
import numpy as np
from scipy import constants
import seaborn as sb
import matplotlib.pyplot as plt
from spectral import *
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from tensorly.decomposition import parafac

h = constants.h
c = constants.c
e = math.e
pi = math.pi
k = constants.k

def load_flux(fluxname):

    with open(fluxname, 'r') as file:
        data = file.readlines()
        size = len(data)
        flux,wavelengths = [],[]
        for i in range(size):
            flux.append(float(data[i].strip().split("\t")[1]))
            wavelengths.append(float(data[i].strip().split("\t")[0]))

    return flux,wavelengths

def rad_list(xp,yp):
    rad = np.zeros(len(wavelengths))
    for i in range(len(wavelengths)):
        rad[i] = img_open[xp,yp,i+5]
    
    return rad

filename = '20191202/ch2_iir_nci_20191202T0639493114_d_img_d18'

# Load solar flux
flux,wavelengths = load_flux('IIRS_ConvFlux.txt')
print
wavelengths = wavelengths[:250]
# Load spectrum data
img = envi.open(filename + '.hdr', image = filename + '.qub')
# print(img)

# Data in 3D-array 
img_open = img.open_memmap(writeable = False)[:,:,:250]




for R in range(12,20):

    # Create a folder of each R and Note down start time
    start_time = time.time()





    # Do parafac decomposition and obtain weights and factors for each R
    weights, factors = parafac(img_open, R)

    # Save weights and factors in a sub-folder
    filename = "Results/R"+ str(R) + '/factors/'
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    np.savetxt("Results/R"+ str(R) + "/weights.csv", weights, delimiter=",")

    for i in range(3):
        np.savetxt("Results/R"+ str(R) + "/factors/A" + str(i+1) + ".csv", factors[i], delimiter=",")


    # Calculate X_tilda by outer products and save file (memmap)
    xr = weights[0] * np.einsum('i,j,k', factors[0][:,0], factors[1][:,0], factors[1][:,0])

    for i in range(1,R):
        xr = xr + (weights[i] * np.einsum('i,j,k', factors[0][:,i], factors[1][:,i], factors[1][:,i]))
    

    filename = "Results/R"+ str(R) + '/X/'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    envi.save_image("Results/R"+ str(R) + "/X/X.hdr", xr, force = True, interleave = 'bsq')




    # Save a snapshot in a snapshot folder

    # Calculate M (noise) by substraction (X - X_tilda)
    filename = "Results/R"+ str(R) + '/Noise/'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    envi.save_image("Results/R"+ str(R) + "/Noise/Noise.hdr", (img_open - xr), force = True, interleave = 'bsq')







    # Note down end time

    filename = "Results/R"+ str(R) + "/runtime.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write("--- %s seconds ---" % (time.time() - start_time))
        f.close()
    
    print(str(R) + " done.....")