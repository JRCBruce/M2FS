############ IMPORTS
####################
# Here we are importing any packages that we will be using
from __future__ import (division, print_function, absolute_import,unicode_literals)
import pandas as pd
import astropy.units as u
from alexmods import specutils
from astropy.time import Time
from alexmods.robust_polyfit import polyfit
from astropy.io import fits, ascii
import sys, os, glob, time
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from astropy.coordinates import SkyCoord
from alexmods.specutils import Spectrum1D
from alexmods.specutils import motions
from alexmods.specutils.rv import cross_correlate
from alexmods.specutils.utils import fast_find_peaks, fast_find_noise, fast_smooth
from alexmods.specutils.utils import fast_find_continuum, fast_find_continuum_polyfit
from alexmods.specutils.utils import find_peaks, cosmic_ray_reject
from astropy.stats import median_absolute_deviation
from scipy import signal, optimize, stats
import pickle
from astropy.modeling import models, fitting
from astropy.table import Table

############
## Inputs ##
############
# This is the section that contains parameters that must be changed for different runs

# This is the path to the directory where we will read and store all the files
basedir = "/home/t/tingli/jbruce/M2FS"

# The run_name contains the date of the observation and is associated with what folder the data is located in
run_name = '20220506'

# This makes the path to the folder for the data run we are looking at
dir_data = basedir+'/%s'%run_name

# First we have the file_nums file, which contains the exposure numbers that were used to observe the target we are looking at
# Note this is for the runs where SINGLE = 0 and it is the directory rather than a specific file.
file_nums_path = basedir+'/file_nums/%s/'%run_name

# Second we have the path to the fibermaps that will be used for the object we are looking at
# Note this is for the runs where SINGLE = 0 and it is the directory rather than a specific file.
path_fibermap = basedir+'/Fibermaps/%s/'%run_name

# Here we are choosing the option on whether to use the code to create the mjd files or to use mjd files that were already created
# Currently we are using files already created since the fits files contained incorrect mjd data
create_mjd = 0

# This is to choose whether we are running a single obervations, or through all the obervations in. 0 is to run through all observations, 1 is to run just a single observation
# Note that if you choose to run through all the obervations it will automatically reduce the individual exposures as well as the coadded exposures
single = 0

# Note that if we are using an mjd file already created, it must follow the same format
if create_mjd == 0:
    if single ==1:
        input_mjd_file_single = basedir+'/downloaded_mjds/obs_mjds_20190809.txt'
    else:
        input_mjd_files = basedir+'/downloaded_mjds/'
        # these files are all titled obs_mjd_ and then then run_name.txt

# IF SINGLE = 1. Here is for naming files, and should be specified to include the name, date, and exposure or epoch if available so that files are not overwritten
# Note this is only for when single = 1, when we are running over a directory the field name is taken from the name of the file_nums file
if single == 1:
    field_name = "hyd1_1_test"
    
########## EVERYTHING BELOW IS ALSO ONLY MODIFIED IF SINGLE == 1    
    
# First we have the file_nums file, which contains the exposure numbers that were used to observe the target we are looking at
# Note this is for the runs where SINGLE = 1.
file_nums_path_single = '/home/t/tingli/jbruce/M2FS/file_nums/20190809/hyd1_1.txt'

# Second we have the path to the fibermaps that will be used for the object we are looking at
# This is for a single file, if SINGLE = 1
path_fibermap_single = basedir+"/Fibermaps/%s/Li_Ret2Hyd1_2019B-hyd1_binar-48.fibermap"%run_name
    
# IF SINGLE = 1: this is whether the individual exposures are to be coadded or not, select 0 for indiivdual exposures, or 1 for the ivar caodd
coadd_exp = 1
# IF SINGLE = 1: this is to choose which exposure to run this file for, should not exceed the len of the file_nums text file
exp_num = 0

############
## Outputs #
############
# This sections is choosing the locations of where files will be saved, this only needs to be changed if you want to store files in a different directory

# here we are making the folders if they do not exist already to store the data and the files
if single == 1: # when single = 0 we do this at the bottom of the script for each field
    if not os.path.exists('%s/objnames/%s'%(basedir, run_name)): # making the objnames directory
        os.makedirs('%s/objnames/%s'%(basedir, run_name))
    if not os.path.exists(basedir+"/reduced_data/%s/%s"%(run_name, field_name)): # making output spectra directory
        os.makedirs(basedir+"/reduced_data/%s/%s"%(run_name, field_name))
        
    npydir = basedir+"/npy_data_files/%s/%s"%(run_name, field_name)
    if not os.path.exists(npydir): # making output directory for the intermediate .npy files for record keeping
        os.makedirs(npydir)
        
    reductiondir = basedir+"/%s/reduction_figures/"%(run_name) 
    if not os.path.exists(reductiondir): # making output directory for the intermediate .npy files for record keeping
        os.makedirs(reductiondir)
    
    # These two files are where the objnames of the observed objects are saved. This also contains R.A., decl., magnitudes, fiber number etc.
    file_objB = open('%s/objnames/%s/objnames_b_%s.txt'%(basedir, run_name, field_name),'w') # for the blue fibers
    file_objR = open('%s/objnames/%s/objnames_r_%s.txt'%(basedir, run_name, field_name),'w') # for the red fibers
    
    # This directory is where to save the output files including the spectra, master skies and other sky files
    outdir = basedir+"/reduced_data/%s/%s"%(run_name, field_name)
    # Here we are making the directories if they do not already exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Here is the locations for the mjd files
    mjd_output = basedir+'/mjds/%s/%s'%(run_name, field_name)
    if not os.path.exists(mjd_output):
        os.makedirs(mjd_output)

#############################
## Functions ################
#############################

def translate_fibermap(fibermap_file): # The input is the fibermap path
    """This function is used to extract information of which targets were observed by which fibers.
    This includes looking at both the red and blue fibers and then writing the important information into a datafile.
    There is no output to this function, and instead the data is save to the objnames directory.
    """
    print("Translating fibermap of %s ..."%(field_name))
    
    file_objB.write("iobj,F1,F2,OBJECT,FIB,RA,DEC,MAGNITUDE\n")
    file_objR.write("iobj,F1,F2,OBJECT,FIB,RA,DEC,MAGNITUDE\n")
    
    # Now we are going to read the fibermap and extract the information we want
    for line in open(fibermap_file,'r'): # reading single line of fibermap, 'r' refers to reading
        if line[0]=='B': # if it is a blue line
            line = line.split() # separate the lines
            if line[1]!='unplugged': # this states that we will not read the unpluggled lines
                # The next three lines are getting the data from the fibermap
                F1 = np.int32(line[0][1]) 
                F2 = np.int32(line[0][3:5])
                iobj = np.int32(((8-F1)*16+F2+1)/2-1) # 
                # Then we are writing the information we extracted into the file to save
                # All the information is stored in a single line which we separated using line.split, and now we are accessing the split line
                file_objB.write("%d,%d,%d,%s,%s,%s,%s,%s\n"%(iobj,F1,F2,line[1],line[0],line[2],line[3],line[13]))
        elif line[0]=='R': # REPEAT the same process for the red fibers
            line = line.split()
            if line[1]!='unplugged':
                F1 = np.int32(line[0][1])
                F2 = np.int32(line[0][3:5])
                iobj = np.int32(((F1-1)*16+F2+1)/2-1)
                file_objR.write("%d,%d,%d,%s,%s,%s,%s,%s\n"%(iobj,F1,F2,line[1],line[0],line[2],line[3],line[13]))

    # Close the files since we are done writing the information to them
    file_objB.close()
    file_objR.close()
    
    print("Done Translating Fibermap!")
    print("Output: objnames_blue_...txt and objnames_red_...txt in "+'%s/objnames/%s/'%(basedir, run_name)+' folder')
    
def get_mjd(a_filenum_path):
    """
    Summary: Here we are reading the mjds from the header of the data files, and then computing the middle mjd of the exposure based on the exposure time.
    We then save the information into text files.
    There is no output to this file an instead we save the data to a directory.
    
    If there are mjd files provided already we extract the mjd of the files we want to look at from these files instead.
    """
    print("Getting mjds from run %s"%run_name)

    # First we are determining which files we want to look at or which exposures
    files1 = []
    file_txt = open(a_filenum_path) # opening the file_nums txt file
    for line in file_txt: # for each file listed in the txt file
        files1.append(line[:4]) # append the exposure number, EX: 0073
    file_txt.close()

    if coadd_exp == 0:
        files = files1[exp_num] # if we only want to look at a specific exposure then we are limiting the files we look at to the single exposure
    if coadd_exp == 1:
        files = files1 # otherwise we look at all the exposures listed in the .txt file

    # Here we are now getting the date for the exposures we are looking at
    fname_all = []
    for file in os.listdir(dir_data): # looking at all the data files
        if file.endswith('fox_specs.fits'): # making sure there are no other files we accidentaly include
            if file[1:5] in files: # here we are only considering the files we are interested in
                fname_all.append(file) # adding the data to the list
    fname_all.sort() # sort them to make it easier

    # Here we are creating three mjd files
    # The first is a general file that contains the mjd_start time, the computed middle time by adding half the exposure time, and the file name
    # The second and third files only contain the middle mjd to use in calculations
    if create_mjd == 1:
        file_mjds = open('%s/%s_all_mjds_%s.txt'%(mjd_output, field_name, run_name),'w')
        file_mjds.write("# mjd_middle, mjd_start, file\n") # adding column titles
    if create_mjd == 0:
        if single == 1:
            file_mjds = Table.read(input_mjd_file_single, format='ascii')
        else:
            file_mjds = Table.read(input_mjd_files+'obs_mjds_%s.txt'%(run_name), format='ascii')
    new_mjds_r = open('%s/%s_r_mjds_%s.txt'%(mjd_output, field_name, run_name),'w')
    new_mjds_b = open('%s/%s_b_mjds_%s.txt'%(mjd_output, field_name, run_name),'w')

    # here we are computing the middle mjd and adding the value to the txt files we created, if create_mjd = 1
    if create_mjd == 1:
        for i_file in range(len(fname_all)):
            hdul_temp = fits.open('%s/%s'%(dir_data,fname_all[i_file])) # opening a single file
            mjd_start =  hdul_temp[0].header['mjd'] # getting the start MJD from the header
            mjd_middle = (Time(mjd_start, format='mjd') + 0.5*hdul_temp[0].header['exptime']*u.s).mjd # adding half the eposure time to the MJD
            file_mjds.write("%.05f, %.05f,# %s\n"%(mjd_middle,mjd_start,fname_all[i_file])) # writing the information to the overall observation file

            # here we are then only taking the middle mjd for the exposure in each fiber 
            if (fname_all[i_file][1:5] in files) and (fname_all[i_file][0] == 'r'): # if it is from the red fiber
                new_mjds_r.write("%.05f\n"%(mjd_middle)) # add the middle mjd
            if (fname_all[i_file][1:5] in files) and (fname_all[i_file][0] == 'b'): # if it is from the blue fiber
                new_mjds_b.write("%.05f\n"%(mjd_middle)) # add the middle mjd
                
    if create_mjd == 0:
        for i in range(len(file_mjds)):
            if (file_mjds[i]['file'][1:5] in files):
                new_mjds_r.write("%.05f\n"%(file_mjds[i]['mjd_middle'])) # add the middle mjd
                new_mjds_b.write("%.05f\n"%(file_mjds[i]['mjd_middle'])) # add the middle mjd, since they are the same for b or r

    # Close all the files
    new_mjds_r.close() 
    new_mjds_b.close()
    if create_mjd == 1:
        file_mjds.close()

    print("Done getting mjds!")
    print("Output: the three mjd files have been saved to the "+mjd_output+' folder')
    
def get_data(Nobjs, Nords, a_filenum_path):
    print('Starting get_data.py')

    # here is defining the number of objects and the number of orders
    Nobj = Nobjs
    Nord = Nords

    # this is the datatype of the files we are using, this may need to be altered if a different datatype is used in the future
    datatype = 'fox'
    
    # First we are determining which files we want to look at or which exposures
    files1 = []
    file_txt = open(a_filenum_path) # opening the file_nums txt file
    for line in file_txt: # for each file listed in the txt file
        files1.append(line[:4]) # append the exposure number, EX: 0073
    file_txt.close()

    if coadd_exp == 0:
        files = files1[exp_num] # if we only want to look at a specific exposure then we are limiting the files we look at to the single exposure
    if coadd_exp == 1:
        files = files1 # otherwise we look at all the exposures listed in the .txt file

    # Here we are creating the data .npy files to use later
    outfname_r = npydir+"/{}_{}_data_{}_{}.npy".format(field_name,'r', datatype, run_name)
    outfname_b = npydir+"/{}_{}_data_{}_{}.npy".format(field_name,'b', datatype, run_name)


    # here we are getting the path to the data that we are using
    fnames_r1 = glob.glob(os.path.join(dir_data, "{}*ds_{}_specs.fits".format('r',datatype)))
    fnames_b1 = glob.glob(os.path.join(dir_data, "{}*ds_{}_specs.fits".format('b',datatype)))

    # here we are only taking the path to the files we want to look at
    fnames_r = []
    fnames_b = []
    for i in range(len(fnames_r1)): # for red fibers
        if fnames_r1[i][-21:-17] in files:
            fnames_r.append(fnames_r1[i])
    for i in range(len(fnames_b1)): # for blue fibers
        if fnames_b1[i][-21:-17] in files:
            fnames_b.append(fnames_b1[i])

    # opening the first file to get the shape, if coadd_exp = 0 then this is opening the only file
    with fits.open(fnames_r[0]) as hdulist:
        Nband, Npix, Nord2, Nobj2 = hdulist[0].data.shape # here we get the number of orders, objects, bands and pix.

    # here is a check to make sure that the shapes are correct, will raise error if incorrect
    assert Nord==Nord2 # this is confirming there are only two orders
    assert Nobj==Nobj2 # this is confirming that there are 64 objects

    # here we are making an array to hold the data, the shape is based on the number of objects, number of exposures, number of orders, and number of pixels
    alloutput = np.zeros((Nobj, len(fnames_r), Nord, Npix, 5))

    # here is a for loop for each exposure we are looking at, this gets the data for all the objects and orders in each exposure as well as the flatflux and err
    for iframe,fname in enumerate(fnames_r):
        hdulist = fits.open(fname) # opening each file
        data = hdulist[0].data # getting the data from the file, wave, flux and error
        Fdata = hdulist[1].data # getting the flat flux and error
        for iobj in range(Nobj): # for each object
            for iord in range(Nord): # for each order
                for iband in range(3): 
                    alloutput[iobj, iframe, iord, :, iband] = data[iband, :, iord, iobj] # adding the science data for the given object, order and band
                alloutput[iobj, iframe, iord, :, 3] = Fdata[1, :, iord, iobj] # adding the flat data
                alloutput[iobj, iframe, iord, :, 4] = Fdata[2, :, iord, iobj] # adding more flat data

    # here we are saving the alloutput data into the red file that we defined above
    np.save(outfname_r,alloutput)
    print("Saved red data to "+npydir)

    # now we are repeating the process for the blue fibers, MORE IN DEPTH DOCUMENTATION ABOVE
    with fits.open(fnames_b[0]) as hdulist: 
        Nband, Npix, Nord2, Nobj2 = hdulist[0].data.shape # getting shape
    assert Nord==Nord2 # confirming num of orders
    assert Nobj==Nobj2 # confirming num of objects

    # getting the science and flat data
    alloutput = np.zeros((Nobj, len(fnames_b), Nord, Npix, 5))
    for iframe,fname in enumerate(fnames_b):
        hdulist = fits.open(fname)
        data = hdulist[0].data
        Fdata = hdulist[1].data
        for iobj in range(Nobj):
            for iord in range(Nord):
                for iband in range(3):
                    alloutput[iobj, iframe, iord, :, iband] = data[iband, :, iord, iobj]
                alloutput[iobj, iframe, iord, :, 3] = Fdata[1, :, iord, iobj]
                alloutput[iobj, iframe, iord, :, 4] = Fdata[2, :, iord, iobj]
    np.save(outfname_b,alloutput)
    print("Saved blue data to "+npydir)

    print('Done get_data.py!')

def get_wave(rb, iord): # loading in the array containing the wavelenghts for the data, takes 'r' or 'b' and the order as 0 or 1
    return np.load(npydir+"/{}_{}_wave_{}_{}.npy".format(field_name ,rb,iord, run_name))
def get_master_sky_fname(rb, iord): # getting the master sky filename
    return outdir+"/{}_{}_master_skies_{}.fits".format(field_name,rb, iord)
def get_all_files(rb, iord): # getting all the individual object .fits files for 'r' or 'b'
    fnames = glob.glob(outdir+"/{}_{}_*_{}.fits".format(field_name, rb, iord))
    try: fnames.remove(outdir+"/{}_{}_master_skies_{}.fits".format(field_name,rb, iord)) # removing master skies files that we dont want to look at
    except: pass
    return fnames # return list of file names
def load_objnames(rb): # loading the object names files we created earlier
    if rb=="b": # blue
        objnames = ascii.read('%s/objnames/%s/objnames_b_%s.txt'%(basedir, run_name, field_name))
        objnames.sort('iobj')
    elif rb=="r": # 'red'
        objnames = ascii.read('%s/objnames/%s/objnames_r_%s.txt'%(basedir, run_name, field_name))
        objnames.sort('iobj')
    objnames = objnames["OBJECT"]
    return objnames
def get_table_value(rb, iobj, key): # getting a certain table file using the key and the index
    if rb=="b":
        tab = ascii.read('%s/objnames/%s/objnames_b_%s.txt'%(basedir, run_name, field_name))
        tab.sort('iobj')
    elif rb=="r":
        tab = ascii.read('%s/objnames/%s/objnames_r_%s.txt'%(basedir, run_name, field_name))
        tab.sort('iobj')
    return tab[iobj][key] # return the value we are looking for

def interpolate_onto_common_dispersion(rb,extracttype): # linearly interpolate onto a common dispersion
    alloutput = np.load(npydir+"/{}_{}_data_{}_{}.npy".format(field_name,rb, extracttype, run_name)) # loading the output data file
    Nobj, Nframe, Nord, Npix, Nband = alloutput.shape # getting the shape of this output file to show the number of objects, orders, pix etc
    
    Npix_interp_max = int(Npix*2) # getting the max number of pixels used for the interpolation

    wave_interp = np.zeros((Nord, Npix_interp_max)) + np.nan # making a blank array for the wavelengths
    flux_interp = np.zeros((Nobj, Nframe, Nord, Npix_interp_max)) + np.nan # blank for flux
    ivar_interp = np.zeros((Nobj, Nframe, Nord, Npix_interp_max)) # blank for ivar
    Fflux_interp = np.zeros((Nobj, Nframe, Nord, Npix_interp_max)) + np.nan # blank for the flat flux
    Fivar_interp = np.zeros((Nobj, Nframe, Nord, Npix_interp_max)) # blank for the flat ivar
    for iord in range(Nord): # for each order
        data = alloutput[:,:,iord,:,:] # getting data for the single order
        wave = data[:,:,:,0] # getting the wavelengths of the data
        wave[wave < 1] = np.nan # putting nan values where the data is < 1
        finite_wave = np.where(np.isfinite(wave))[2] # getting the location where the wavelengths are finite
        wave_ix_min, wave_ix_max = np.min(finite_wave), np.max(finite_wave) + 1 # getting the min and max of the finite wavelengths

        Npix_used = wave_ix_max - wave_ix_min # getting the number of pixels used within the range we just defined
        Npix_interp = int(Npix_used) # getting the number of pixels used for interpolation, it is the same as the number wihthin the above range
        interp_ix_min, interp_ix_max = int(wave_ix_min), int(wave_ix_max) # the interpolation minimum and maximum

        wmin, wmax = np.nanmin(wave), np.nanmax(wave) # min and max wavelength
        this_wave_interp = np.linspace(wmin, wmax, Npix_interp) #  generating a linspace for this the range of wavelengths using the num of pixels
        wave_interp[iord, interp_ix_min:interp_ix_max] = this_wave_interp #  filling the zeroes with the interpolation base
        print(iord, Npix_used, Npix_interp, np.diff(this_wave_interp)[0]) # printing inforomation

        for iobj in range(Nobj): # for each object
            for iframe in range(Nframe): # for each frame
                # using np.interp to interpolate the data for the flux at the points of the wavelengths we defined above
                
                # filling in the data into this array
                flux_interp[iobj,iframe,iord,interp_ix_min:interp_ix_max] = \
                    np.interp(this_wave_interp, # evaluating the interpolated values over the range of wavelengths
                              data[iobj,iframe,wave_ix_min:wave_ix_max,0], # the x-coordinate of the data we are interpolating
                              data[iobj,iframe,wave_ix_min:wave_ix_max,1], # the y-componenet
                              left=np.nan, right=np.nan) # here we are putting a nan value for areas not within the range we are interested in
                total_ivar1 = np.nansum(data[iobj,iframe,wave_ix_min:wave_ix_max,2]**-2.) # getting the total ivar, treating na values as 0
                
                # interpolating again to get the ivar, same method as above just using the ivar index of the data
                ivar_interp[iobj,iframe,iord, interp_ix_min:interp_ix_max] = \
                    np.interp(this_wave_interp,
                              data[iobj,iframe,wave_ix_min:wave_ix_max,0],
                              data[iobj,iframe,wave_ix_min:wave_ix_max,2]**-2.,
                              left=np.nan, right=np.nan)
                total_ivar2 = np.nansum(ivar_interp[iobj,iframe,iord, interp_ix_min:interp_ix_max]) # getting the total ivar again 
                ivar_interp[iobj,iframe,iord, interp_ix_min:interp_ix_max] = ivar_interp[iobj,iframe,iord, interp_ix_min:interp_ix_max]*total_ivar1/total_ivar2 # writing this to the zeroes array we made
                
                # doing the same for flat files, same process just using different index of data file
                Fflux_interp[iobj,iframe,iord,interp_ix_min:interp_ix_max] = \
                    np.interp(this_wave_interp,
                              data[iobj,iframe,wave_ix_min:wave_ix_max,0],
                              data[iobj,iframe,wave_ix_min:wave_ix_max,3],
                              left=np.nan, right=np.nan)
                total_ivar1 = np.nansum(data[iobj,iframe,wave_ix_min:wave_ix_max,4]**-2.)
                # same for ivar of flats
                Fivar_interp[iobj,iframe,iord, interp_ix_min:interp_ix_max] = \
                    np.interp(this_wave_interp,
                              data[iobj,iframe,wave_ix_min:wave_ix_max,0],
                              data[iobj,iframe,wave_ix_min:wave_ix_max,4]**-2.,
                              left=np.nan, right=np.nan)
                total_ivar2 = np.nansum(Fivar_interp[iobj,iframe,iord, interp_ix_min:interp_ix_max]) # same
                Fivar_interp[iobj,iframe,iord, interp_ix_min:interp_ix_max] = Fivar_interp[iobj,iframe,iord, interp_ix_min:interp_ix_max]*total_ivar1/total_ivar2 # same
                
            # Below are comments from when I received the original codes
            ## This was cutting like 2/3 of the pixels, not really sure why but probably that my CRR algorithm is just wrong!
            #new_flux, new_ivar, new_mask = cosmic_ray_reject(flux_interp[iobj,:,iord,:],
            #                                                 ivar_interp[iobj,:,iord,:],
            #                                                 sigma=10., minflux=-100, verbose=True, use_mad=True)
            #print("Cosmic ray rejection for obj {} rejected {}/{}".format(iobj,new_mask.sum(),new_mask.size))
            # END
            
            # Below we are saving the data we interpolated above into arrays for us to use
            new_flux, new_ivar = flux_interp[iobj,:,iord,:], ivar_interp[iobj,:,iord,:] # getting the new flux and ivar for the certain order based on the interpolated values we determined above
            thisflat = Fflux_interp[iobj,:,iord,:] # getting the flat
            thisflaterr = Fivar_interp[iobj,:,iord,:]**-0.5 # error in the flat
            flux_interp[iobj,:,iord,:] = new_flux/thisflat # using the flat to correct the flux
            ivar_interp[iobj,:,iord,:] = new_ivar*(thisflat**2) # using the flat to fix the errors
    np.save(npydir+"/{}_{}_interpolated_{}_{}.npy".format(field_name,rb, extracttype, run_name), [wave_interp, flux_interp, ivar_interp]) # saving the wave, flux, and ivar

def make_master_sky(all_sky, method="weighted_mean", fp=sys.stdout): # making the master sky files
    """
    method in ["mean", "median", "sigma_clip", "weighted_mean", "kelson03"]
    kelson03 is a least squares fit including clipping (not actually the 2D subtraction)
    """
    wl = all_sky[0].dispersion # getting the wavelengths 
    Nsky, Nwl = len(all_sky), len(wl) # the size of the wavelength array and the number of sky fibers
    fp.write("{} sky fibers found\n".format(Nsky)) # stating how many sky fibers were found

    assert method in ["mean", "median", "sigma_clip", "weighted_mean"], method # making sure the method we selected is one of the available options

    skyflux = np.zeros((Nsky, Nwl)) + np.nan # making an array full of nan values rather than zeroes
    for i,sky in enumerate(all_sky): # running a for loop with i as the index, and sky as the value at that index in all_sky
        iigood = np.isfinite(sky.flux) & (sky.ivar > 1e-7) # making sure that this index is good, the flux must be finite and the error greater than a very small number
        skyflux[i,iigood] = sky.flux[iigood] # listing whether it is good or not

    if method == "mean": # mean method
        master_sky = np.nanmean(skyflux, axis=0)  # getting the mean sky flux for the master sky
        ivar = 1./master_sky # getting the ivar value
    elif method == "median": # median method
        master_sky = np.nanmedian(skyflux, axis=0) # getting the median value
        ivar = 1./master_sky # getting the ivar value
        error = 1.4826 * np.sqrt(np.pi/2.) * median_absolute_deviation(skyflux, axis=0, ignore_nan=True) # the error is adding this constant to the median absolute deviation
        number_skies = np.sum(np.isfinite(skyflux), axis=0) # the number of finite skyfluxes
        ivar = number_skies / error**2. # getting ivar based on the number of finite fluxes
    if method == "sigma_clip": # sigma_clip method, THIS IS WHAT IS USED LATER ON
        master_sky = np.zeros(Nwl) # getting a zeroes array first
        for j in range(Nwl): # for each wavelength
            
            ### master_sky[j] = np.nanmean(stats.sigmaclip(skyflux[:,j])[0])
            ### modified by YYS on Feb 18, 2022
            
            temp_sky = skyflux[:,j] # getting all the skyfiber data for the wavelength we are looking at in this for loop iteration
            temp_sky = temp_sky[~np.isnan(temp_sky)] # Only keeping the fiber data that is not a nan value
            
            # In the next line, we are finding the median (of non nan values) while also sigma-clipping or removing outliers using high & low as bounds
            master_sky[j] = np.nanmedian(stats.sigmaclip(temp_sky,low=10,high=1)[0], axis=0) # saving this median value of sky fibers at this wavelength to the master sky file
        ivar = 1./master_sky # getting ivar
    elif method == "weighted_mean": # weighed mean method
        skyivar = np.zeros((Nsky, Nwl)) # getting zeroes to start
        for i,sky in enumerate(all_sky): # for index and element in all_sky
            iigood = np.isfinite(sky.flux) # checking if finite
            skyivar[i,iigood] = sky.ivar[iigood] # writing whether it is finite
        total_weight = np.nansum(skyivar, axis=0) # getting the total weight, the sum of the ivars
        master_sky = np.nansum(skyflux * skyivar, axis=0)/total_weight # getting the masker sky by summing the flux weighted by the ivar
        ivar = total_weight # ivar is equal to the total weight
        if np.any(np.isnan(master_sky)): # checking if any are nan in the master sky file
            nan_mask = np.isnan(master_sky) # getting nan_mask of the sky
            master_sky[nan_mask] = 0. # putting the nan values as zero on the master sky
            ivar[nan_mask] = 1e-8 # putting the ivars as extremely small on the master sky for nan values
        assert np.all(~np.isnan(master_sky)) # confirming there are no nan values

    return Spectrum1D(wl, master_sky, ivar) # returning the results as a 1D spectrum containing the wavelength, master sky flux and ivar

def get_ix_minmax(wave): # getting the min and max of wave
    ix_min, ix_max = np.where(np.isfinite(wave))[0][[0,-1]] # getting an array of indices where the wavelength is finite
    ix_max += 1 # adding one to the max
    return ix_min, ix_max
def sort_by_object(rb, iord,coadding_def, extracttype="splineghlb"): # sorting by the object
    #COMMENTS FROM ORIGINAL CODE
    # I'm purposefully being a bit inefficient here so it's processed the same way as the GIRAFFE data
    # Process one order at a time
    #rb = "b"
    #extracttype = "splineghlb"
    #iord = 3
    # END
    
    lco_lat = -29 - 0.2/60 # latitude
    lco_lon = 70 + 42.1/60 # longitude
    lco_alt = 2282. # altitude
    
    frame_mjds = np.loadtxt('%s/%s_%s_mjds_%s.txt'%(mjd_output, field_name, rb, run_name)) # importing the mjd we created earlier
    
    if coadding_def == 0: # here since if there is only a single file we need to put it into list format
        temp_list = [] # making the list
        temp_list.append(frame_mjds) # appending the single mjd
        frame_mjds1 = temp_list # making this the new value of the variable
    if coadding_def == 1:
        frame_mjds1 = frame_mjds
    
    # getting the interpolated wavelenghts, flux and ivar
    allwave, allflux, allivar = np.load(npydir+"/{}_{}_interpolated_{}_{}.npy".format(field_name,rb, extracttype, run_name),allow_pickle=True)
    ix_min, ix_max = np.where(np.isfinite(allwave[iord,:]))[0][[0,-1]] # getting the min and max of the finite wavelengths
    ix_max += 1 # adding 1 to the max
    wave = allwave[iord, ix_min:ix_max] # getting the wavelengths of the order we are interested in within the bounds of the min and max
    allflux = allflux[:,:,iord,ix_min:ix_max] # getting the flux for the same range and order
    allivar = allivar[:,:,iord,ix_min:ix_max] #  getting the ivar for the same range and order
    Npix = ix_max - ix_min # getting the numper of pixels using the min and max
    assert Npix == len(wave) # Making sure the Npix matches with the wave length

    objnames = load_objnames(rb) # getting the object names of either red or blue objects
    sky_indices = np.where(np.array(["Sky" in name for name in objnames]))[0] # getting the indices for skylines, changed by YYS, on Feb 03, 2022

    bigdict = {} # initiate dictionary
    skydict = {} # initiate skyline dictionary
    allskyflux = [] # initiate flux list
    allskyivar = [] # initiate ivar list
    master_skies = np.zeros((Npix,Nframe,3)) # make zeroes array to put master skies info into

    for iobj, obj in enumerate(objnames): # for index and the object in the objnames files
        data = np.zeros((Npix,Nframe,3)) # making zeroes array for the data
        data[:,:,0] = allflux[iobj,:,:].T # getting the flux of the object
        data[:,:,1] = allivar[iobj,:,:].T # getting the ivar
        data[:,:,2] = (allivar[iobj,:,:] < 1e-7).T # getting the areas where the ivar is extremely small, most likely where it was nan and we placed this above
        meta = {} # making a blank dictionary
        for key in ["OBJECT","RA","DEC","MAGNITUDE"]: # listing the keys for the dictionary
            meta[key] = get_table_value(rb, iobj, key) # making the elements of the dictionary using the values from the objnames files
            
        coo = SkyCoord(ra=meta["RA"],dec=meta["DEC"],unit=("hourangle","degree")) # getting the Skycoord location in RA and DEC, also changed to input to degrees
        metalist = [meta.copy() for iframe in range(Nframe)] # making a copy of the dict as a list
        hcorrs = []
        mjd_list = []
        for iframe in range(Nframe): # for each file or exposure
            mjd = frame_mjds1[iframe] # getting the mjd of the frame
            hcorr = motions.corrections(lco_lon, lco_lat, lco_alt, coo.ra, coo.dec, mjd)[1] # getting the helio correction based on the location
            hcorrs.append(hcorr.to("km/s").value) # adding the hcorr to a list to average
            mjd_list.append(mjd) # adding the mjd to a list to average
        hcorr_coadd = np.mean(hcorrs) # getting average
        mjd_coadd = np.mean(mjd_list) # getting average
        for iframe in range(Nframe):
            metalist[iframe]["RA-D"] = coo.ra.deg # adding RA in degrees to make it easier later
            metalist[iframe]["DEC-D"] = coo.dec.deg # adding DEC in degrees to make it easier later
            metalist[iframe]["mjd Average"] = mjd_coadd  # adding the average to the header
            metalist[iframe]["HCORR Average"] = hcorr_coadd # adding the average to the header
            for j in range(len(hcorrs)):
                metalist[iframe]["HCORR exposure "+str(j)] = hcorrs[j]
                if coadding_def == 0:
                    metalist[iframe]["mjd exposure "+str(j)] = float(mjd_list[j]) # this line has to be added sice if only one exposure it becomes a 0-dimensional array and messes up the format
                else:
                    metalist[iframe]["mjd exposure "+str(j)] = mjd_list[j] # this is the normal format
        if "Sky" not in obj: # if there is no sky in the name of the object
            bigdict[obj] = [data, metalist] # add the obj to the big dictionary using the meta list data
        else:
            skydict[obj] = [data, metalist] # else add the sky obj to the sky dictionary
    for iframe in range(Nframe): # for each file or exposure
        fluxs = allflux[:,iframe,:] # the fluxes for the exposure
        ivars = allflux[:,iframe,:] # the ivars for the exposure
        all_sky = [Spectrum1D(wave, fluxs[ix], ivars[ix]) for ix in sky_indices] # getting the all_sky spectrum for all the skylines
        #master = make_master_sky(all_sky, method="median")
        master = make_master_sky(all_sky, method="sigma_clip") ### modified by YYS on Feb 18, 2022 # making the master sky frame using the sigma clip method
        master_skies[:,iframe,0] = master.flux # adding the master flux
        master_skies[:,iframe,1] = master.ivar # adding master ivar
        master_skies[:,iframe,2] = master.ivar < 1e-7 # adding master ivar where it is extremely small

    # Spectrum Output 
    npydir+"/{}_{}_interpolated_{}_{}.npy".format(field_name,rb, extracttype, run_name)
    
    np.save(npydir+"/{}_{}_wave_{}_{}.npy".format(field_name ,rb,iord, run_name), wave) # saving the file
    for obj in np.sort(list(bigdict.keys())+list(skydict.keys())): # for each object in all the keys
        try: x = bigdict[obj] # trying to see if this causes an error
        except KeyError: x = skydict[obj] # handles the error as a key error

        hdu = fits.PrimaryHDU(x[0]) # getting the primary Header data unit
        keys_list = ["OBJECT","RA","DEC","MAGNITUDE", 'RA-D', 'DEC-D', "mjd Average", "HCORR Average"] # Making the list of headers that will be listed in the fits file
        for iframe in range(Nframe): # for each exposure add the Hcorr for that exposure
            keys_list.append("HCORR exposure "+str(iframe))
        for iframe in range(Nframe): # for each exposure add the mjd for that exposure
            keys_list.append("mjd exposure "+str(iframe))
        for key in keys_list: # for each key
            hdu.header[key] = x[1][0][key] # add the key to the header data
        hdu.header.add_comment("Extension 1: raw data (flux, ivar, mask)") # add a comment abount the first extension 
        outfname = os.path.join(outdir, "{}_{}_{}_{}.fits".format(field_name,rb, obj, iord)) # the output file name 
        hdu.writeto(outfname, overwrite=True) # write the hdu onto the output file
        
    # Master Sky Output 
    hdu = fits.PrimaryHDU(master_skies) # getting the master skies
    hdu.header["OBJECT"] = "Master Sky" # labelling it
    hdu.header.add_comment("Extension 1: raw data (flux, ivar, mask)") # adding extension description 
    hdu.writeto(get_master_sky_fname(rb, iord), overwrite=True) # writing the information onto the file

def subtract_sky_simple(rb, iord): # subtracting the master sky from all the fibers
    master_skies = fits.open(get_master_sky_fname(rb, iord))[0].data # getting the master sky for a given fiber and order
    allfiles = get_all_files(rb, iord) # getting all files for the same color fiber and order

    start = time.time() # timing it
    for fname in allfiles: # for loop through the files
        hdul = fits.open(fname) # opening the file
        hdu = hdul[0] # getting primary data
        data = np.zeros_like(hdu.data)+np.nan # make an array of nans to start
        
        # subtract master sky from flux
        data[:,:,0] = hdu.data[:,:,0] - master_skies[:,:,0] # subtracting the master skies from the data
        data[:,:,1] = (hdu.data[:,:,1]**-1. + master_skies[:,:,1]**-1.)**-1. # getting the new ivar with the master skies
        data[:,:,2] = np.logical_or(hdu.data[:,:,2], master_skies[:,:,2]) # this inputs True if one of the two (data or master skies) is True

        hdu2 = fits.ImageHDU(data) # making an IMAGE HDU
        hdu.header.add_comment("Extension 2: raw - unscaled master sky (flux, ivar, mask)") # desribing the second extension
        hdulist = fits.HDUList([hdu, hdu2]) # making an HDU List for both extrensions 
        hdulist.writeto(fname, overwrite=True) # rewriting the extensions to the file
    print(time.time()-start) # print time it took

def process_master_sky(rb, iord): # identifying the line pixels in the master sky
    """
    Find peaks in sky spectrum
    Create sky pixel mask
    """
    wave = get_wave(rb, iord) # getting wavelengths for the fiber color and order
    hdul = fits.open(get_master_sky_fname(rb, iord)) # opening master sky file for this order and fiber color
    master_skies = hdul[0].data # getting the data
    ix_min, ix_max = get_ix_minmax(wave) # getting min and max ix
    Npix = ix_max - ix_min # getting number of pixels between the min and the max

    maskwindow = 7 # the size of half the mask window used to mask the peaks
    fig, axes = plt.subplots(M,1,figsize=(16,5*M)) ### modified by YYS on Jan 27, 2022 # starting the figures
    if coadd_exp == 1: # if we are coadding the data then we need to confirm the flattened length is correct
        assert len(axes.flat) == M # testing the axes match M
    pixel_mask = np.zeros((Npix, M), dtype=int) # making a zeroes array to use as the pixel mask
    continuums = np.zeros((Npix, M)) # making a zeroes array to use for the continuums

    max_Npeak = 0
    skylineparams = [] # starting list to store the parameters
    for i in range(M): # for each exposure
        start = time.time() # timing it
        if coadd_exp == 1: # if we are coadding then we need to flatten it
            ax = axes.flat[i] # taking only a single exposure
        else:
            ax = axes # for a single exposure we do not need to flatten it

        flux = master_skies[:,i,0] # getting the master skies flux for this exposure
        cont0 = fast_find_continuum(flux) # finding continuum 
        cont = fast_find_continuum_polyfit(flux, 3) # finding better contiuum fit using polyfit
        noise = fast_find_noise(flux-cont) # finding the noise
        print("noise={:.1f}".format(noise)) # stating noise level
        fluxsmooth = fast_smooth(flux, 11) # making it smoother so noisy adjacent peaks are not found
        peaks = fast_find_peaks(fluxsmooth, cont, noise, detection_sigma=2.0) # finding the peaks of the flux
        ax.plot(wave, flux-cont, 'k-', lw=1) # plotting the noise
        ax.plot(wave, cont, 'r-') # plotting continuum
        ax.plot(wave, cont0, ':', color='orange') # plotting fast continuum
        for x in peaks: # for each peak
            ax.axvline(wave[x], color='c', lw=.5) # plot a vline for each of the peaks
            _min, _max = max(0,x-maskwindow), min(Npix-1,x+maskwindow) # making a min and max location around the peak to mask so that it is not included
            ax.axvspan(wave[_min], wave[_max], color='r', alpha=.4) #  adding a vertical rectangle spanning the peak that we want to mask
            pixel_mask[_min:_max, i] = True # making the mask in this area
        continuums[:,i] = cont # adding the continuum fit into the array that we created earlier
        ax.set_title("Npeak = {}, noise={:.1f}".format(len(peaks), noise)) # stating the number of peaks and the noise in the title of the plot
        ax.set_ylim(-noise, np.nanmax(cont)+5*noise) # ylim to show in the plot
        ax.set_xlim(wave[0], wave[-1]) # xlim to show in the plot to cover the whole wavelength range
    fig.tight_layout()
    fig.savefig(basedir+"/reduction_figures/{}_skylineloc_{}_{}_{}.pdf".format(rb,iord, field_name, run_name), bbox_inches="tight") # saving the figure into a folder
    plt.close(fig) # closingthe figure

    # COMMENT FROM ORIGINAL CODE
    # Assume wl calibration is good enough that all of these pixels are sky!
    # Only include things that were sky lines in >half of the sky frames
    # END
    
    print("pixel_mask counts: {}".format(np.nansum(pixel_mask, axis=0))) # printing the sum of the pixel masks
    pixel_mask = (np.nansum(pixel_mask, axis=1) > 0.25*M).astype(int) # making the mask by summing
    print("pixel_mask counts: {}".format(np.nansum(pixel_mask))) # printing count again
    hdu = hdul[0] # saving extension 1
    hdu2 = fits.ImageHDU(pixel_mask) # making Image data for the pixel mask
    hdu.header.add_comment("Extension 2: pixel mask generated from full master sky") # adding info
    hdu3 = fits.ImageHDU(continuums) # storing continuums in the fits file
    hdu.header.add_comment("Extension 3: continuum for master sky") # adding info
    hdulist = fits.HDUList([hdu, hdu2, hdu3]) # making list to add into the fits file
    hdulist.writeto(get_master_sky_fname(rb, iord), overwrite=True) # saving the data for the raw data, pixel mask, and continuum into a file

def scaled_sky_subtraction(rb, iord): # scale master sky to match the variable sky
    start = time.time()
    with fits.open(get_master_sky_fname(rb, iord)) as hdul: # retreiving the master sky filename based on the order and fiber color
        master_skies = hdul[0].data # getting the data for the master skies
        sky_line_mask = hdul[1].data.astype(bool) # getting the skyline mask as an array of bool values
        master_skies_contsub = master_skies[:,:,0].copy() - hdul[2].data  # subtracting the continuum from the master skies data

    wave = get_wave(rb, iord) # getting the wavelength for this order and fiber color
    ix_min, ix_max = get_ix_minmax(wave) # getting the bounds for the wavelengths
    Npix = ix_max - ix_min # finding the number of pix between the bounds
    fnames = get_all_files(rb, iord) # retreiving all the files we need
    rvarr = np.zeros((len(fnames),M)) # making an array of zeroes to store data inside
    for ifname, fname in enumerate(fnames): # for the index and file in the list of files we retrieved
        start2 = time.time()
        hdul = fits.open(fname) # open the file
        data = hdul[0].data # get the data
        newdata = np.zeros((Npix,M,4)) + np.nan # makeing an array of nan values to store the data inside
        scales = np.ones(M) # making an array of ones to put the scales into
        contdata = np.zeros((Npix,M)) + np.nan # making an array of nan values to store the continuum data in
        for iframe in range(M): # for each exposure
            flux = data[:,iframe,0] # get the flux data
            if np.all(np.isnan(flux)): # if all the data is nan
                scales[iframe] = np.nan # then make the scale nan
                continue # if its nan then we can skip the rest of this for loop using continue
            skyflux = master_skies[:,iframe,0] # getting the master sky flux
            skyivar = master_skies[:,iframe,1] # getting the master sky ivar
            skymask = master_skies[:,iframe,2] # getting the master sky mask
            skyflux_contsub = master_skies_contsub[:,iframe] # getting the master skyflux with the continuum subtracted 
            Niter = 3
            flux_fit = flux.copy() # copying the flux to alter
            flux_fit[sky_line_mask] = 0. # replacing the masked areas of the flux with 0
            flux_fit[np.isnan(flux_fit)] = 0. # replacing any nan values with 0 as well
            cont = fast_find_continuum_polyfit(flux, 3) # finding the continuum of the original flux, before the alterations we just made
            for it in range(Niter):
                flux_fit[sky_line_mask] = cont[sky_line_mask] # replacing the altered flux with the continuum value in the mask
                cont = fast_find_continuum_polyfit(flux_fit, 3) # finding the contiuum of this new altered flux
            line = flux - cont # subtracting the continuum from the original flux
            contdata[:,iframe] = cont # adding the continuum into the data array
            
            ## BELOW WAS FROM THE ORIGINAL CODE AND WAS ALSO COMMENTED IN THE ORIGINAL CODE
            ## cross correlation of object and sky lines. This matches the sky to <1 km/s, so we won't readjust
            ## because when an order has no sky lines/the lines are noisy then things go crazy
            ## Assume the master sky is correct and shift the object to that position
            #line_mask_sky = skyflux_contsub.copy()
            #line_mask_sky[~sky_line_mask] = 0.
            #line_mask_flux = line.copy()
            #line_mask_flux[~sky_line_mask] = 0.
            ## The ivar doesn't really matter, but calculate it anyway
            #line_mask_skyivar = skyivar.copy()
            #line_mask_skyivar[~sky_line_mask] = 0.
            #line_mask_ivar = data[:,iframe,1].copy()
            #line_mask_ivar[~sky_line_mask] = 0.
            #skylinespec = Spectrum1D(wave, line_mask_sky, line_mask_skyivar)
            #objlinespec = Spectrum1D(wave, line_mask_flux, line_mask_ivar)
            #rv, e_rv, ccf = cross_correlate(objlinespec, skylinespec)
            ##print(fname,iframe,rv)
            #rvarr[ifname,iframe] = rv
            #objlinespec.redshift(-rv, reinterpolate=True)
            #line = objlinespec.flux # do not bother to redo the ivar
            # END

            # find optimum scale factor to match with soft L1 norm to 5 digits precision
            finite = np.isfinite(skyflux_contsub) & np.isfinite(line) # finding the areas where the skyline flux and the spectra flux are finite
            fit_line_mask = sky_line_mask.copy() # copying the skyline mask and naming it as the fitline mask
            masked_sky_contsub = skyflux_contsub[fit_line_mask & finite] # masking the skyline flux using the skyline mask and also only choosing finite values 
            masked_line= line[fit_line_mask & finite] # finding the flux using the fitline mask and also only choosing finite values
            
            def resid_func(theta): 
                # defining a function to multiply the masked sky data (with the continuum subtracted) by a given numbers, and then subtract the flux (that is masked and has the continuum subtracted)
                return theta[0]*masked_sky_contsub - masked_line
            
            res_lsq = optimize.least_squares(resid_func, [1.0], bounds=(0.5,2.0), xtol=1e-5, loss="soft_l1") # here we are using a least squares method to optimize the function we just defined above
            scale = res_lsq.x # making the scale the solution to the above optimization
            scales[iframe] = scale # saving this scale to the array
            
            # Next we subtract the scaled master sky from the flux
            newdata[:,iframe,0] = data[:,iframe,0] - scale * skyflux # subtracting the rescaled skyline flux
            newdata[:,iframe,1] = 1./(data[:,iframe,1]**-1. + scale**2. * skyivar**-1.) # getting the new ivar
            newdata[:,iframe,2] = (data[:,iframe,2] + skymask) > 0 # finding the areas where the sum of the data mask and the skymask is > 0
            newdata[:,iframe,3] = scale * skyflux # saving the scaled skyline flux
        hdul.close() 
        hdul = fits.open(fname)
        hdu1, hdu2 = hdul[0], hdul[1] # getting the first two extensions data before we overwrite the file

        hdu3 = fits.ImageHDU(newdata) # appending the new data that has the rescaled skyline subtracted from ti
        hdu4 = fits.ImageHDU(scales) # appending the scaling factors that we used for each exposure
        hdu5 = fits.ImageHDU(contdata) # appending the continuum data we used for each exposure
        hdu1.header.add_comment("Extension 3: raw - scaled-to-line master sky (flux, ivar, mask, sky)") # naming the extensions
        hdu1.header.add_comment("Extension 4: raw - scaled-to-line master sky scaling factors") # naming the extensions
        hdu1.header.add_comment("Extension 5: continuum fit with masked sky") # naming the extensions
        hdulist = fits.HDUList([hdu1, hdu2, hdu3, hdu4, hdu5]) # appending all the data into a list
        hdulist.writeto(fname, overwrite=True) # writing the data to the fits file
        print("{} took {:.1f}s".format(fname, time.time()-start2))
        np.save(npydir+"/{}_{}_{}_rvarr_{}.npy".format(rb,iord, field_name, run_name), rvarr)
    print("Total scaled sky subtraction took {:.1f}s".format(time.time()-start))

def svd_sky_subtraction(rb, iord): # Dr. Songs attempt to use SVD to model the sky, not working at the moment and so it is not Implemented
    print("SVD sky subtraction not implemented. Adding a dummy extension")  # adding a dummy extension that could be updated in the future
    fnames = get_all_files(rb, iord)
    for fname in fnames:
        hdul = fits.open(fname)
        hdu1, hdu2, hdu3, hdu4, hdu5 = hdul[0:5]
        hdu6 = fits.ImageHDU(np.zeros(1))
        hdu1.header.add_comment("Extension 6: svd sky (not implemented!)")
        hdulist = fits.HDUList([hdu1, hdu2, hdu3, hdu4, hdu5, hdu6])
        hdulist.writeto(fname, overwrite=True)

def coadd(rb, iord, hduindex=2): # coadding the results, or if it is a single exposure appending the results to the seventh extension to remain consistent with data location
    wave = get_wave(rb, iord) # getting wavelength for the order and fiber color we are looking at
    fnames = get_all_files(rb, iord) # getting the files associated with the order and fiber color

    start = time.time()
    for i, fname in enumerate(fnames): # for each file that we are looking at
        if coadd_exp == 1: # if we want to coadd the exposures from a single night for the data then we will run this section
            hdul = fits.open(fname) # getting the data from the file
            data = hdul[hduindex].data # extracting the data from the second index of the fits file, this is extension 3 or the flux after the scaled skyline subtraction
            flux = data[:,:,0] # getting the flux
            ivar = data[:,:,1] # getting the ivar
            mask = data[:,:,2].astype(bool) # getting the mask
            flux[mask] = np.nan # masking values with nan values
            ivar[mask] = 0. # masking values with na values
            ivarnorm = np.nansum(ivar, axis=1) # summing the ivar values in order to get a norm

            flux, ivar, mask = cosmic_ray_reject(flux.T,ivar.T,
                                                 sigma=10., minflux=-1000, verbose=True, use_mad=False) # using a function to remove cosmic rays from the data
            flux, ivar, mask = flux.T, ivar.T, mask.T # here we are transposing the data so that we can easily coadd the exposures

            overall_signal = np.nanmedian(flux, axis=0) # the median flux or overall signal of the each exposure
            total_overall_signal = np.sum(overall_signal) # the overall sum of the median of each exposure

            rescaled_flux = flux/overall_signal[np.newaxis,:] # rescaling the flux based on the overall signal
            rescaled_ivar = overall_signal[np.newaxis,:]**2 * ivar # rescaling the ivar based on the overall signal
            rescaled_ivarnorm = np.nansum(rescaled_ivar, axis=1) # getting the new summed ivar after rescaling
            rescaled_ivarnorm2 = np.nansum(rescaled_ivar**2, axis=1) # getting the sum of the squared ivars after rescaling
            
            # determining the coadded flux by summing the values that are not nan with a weighting of their inverse variance, and then dividing by the ivar norm we measured above
            coadd_flux = np.nansum(rescaled_flux*rescaled_ivar, axis=1)/rescaled_ivarnorm
            Mnonzero = np.nansum(rescaled_ivar > 1e-5, axis=1) # determining which exposures provide nonzero data by summing the rescaled ivar and seeing if it is large enough to be considered
            
            # BELOW WAS COMMENTED IN THE ORIGINAL CODE AND IS KEPT IN CASE IT BECOMES RELEVANT IN THE FUTURE
            #coadd_ivar = ((Mnonzero-1)/Mnonzero) * rescaled_ivarnorm / np.sum(rescaled_ivar*(rescaled_flux-coadd_flux[:,np.newaxis])**2.,axis=1)
            #coadd_ivar = Mnonzero * rescaled_ivarnorm / np.sum(rescaled_ivar*(rescaled_flux-coadd_flux[:,np.newaxis])**2.,axis=1)
            #coadd_ivar = rescaled_ivarnorm**2 / np.sum(rescaled_ivar**2 / rescaled_ivar, axis=1)
            #END
            
            # determiningthe coadded ivar by using several of the arrays we defined above
            coadd_ivar = (Mnonzero-1) * (rescaled_ivarnorm - rescaled_ivarnorm2/rescaled_ivarnorm) / np.nansum(rescaled_ivar*(rescaled_flux-coadd_flux[:,np.newaxis])**2.,axis=1) 
            
            coadd_flux = coadd_flux * total_overall_signal # multiplying the coadded flux by the total overall signal
            coadd_ivar = coadd_ivar/(total_overall_signal**2) # adjusting the coadded ivar accoringly using the total signal
            
            # BELOW WAS COMMENTED IN THE ORIGINAL CODE AND IS KEPT IN CASE IT BECOMES RELEVANT IN THE FUTURE
            #coadd_flux = np.nansum(flux*ivar, axis=1)/ivarnorm
            ### Type 1: wrong
            #coadd_ivar = ivarnorm
            ### Type 2: correct
            #Mnonzero = np.nansum(ivar > 1e-5, axis=1)
            #coadd_ivar = ((Mnonzero-1)/Mnonzero) * ivarnorm / np.sum(ivar*(flux-coadd_flux[:,np.newaxis])**2.,axis=1)
            ### Type 3: correct without the unbiased
            #coadd_ivar = ivarnorm / np.sum(ivar*(flux-coadd_flux[:,np.newaxis])**2.,axis=1)
            ### Type 4: direct stderr
            #Mnonzero = np.nansum(ivar > 1e-5, axis=1)
            #coadd_ivar = Mnonzero*(np.nanstd(flux, axis=1))**-2.
            #END

            coadd_data = np.vstack([wave, coadd_flux, coadd_ivar]).T # stacking the coadded and transposing it back into its original shape
            hdul.close()
            hdul = fits.open(fname)
            hdu1, hdu2, hdu3, hdu4, hdu5, hdu6 = hdul[0:6] # saving the first 6 extensions
            hdu1.header.add_comment("Extension 7: ivar-weighted coadd (wave, flux, ivar)") # adding a title
            hdu7 = fits.ImageHDU(coadd_data) # saving the coadded data
            hdulist = fits.HDUList([hdu1, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7]) # appending all the data to the fits file
            hdulist.writeto(fname, overwrite=True) # saving the data
        else: # if the data does not need to be coadded because there is only one exposure, then we are just reformatting the third extension
            hdul = fits.open(fname) # opening the file
            data = hdul[hduindex].data # getting the data after the scaled sky subtraction
            flux = data[:,:,0] # getting the flux
            ivar = data[:,:,1] # getting the ivar
            mask = data[:,:,2].astype(bool) # getting the mask
            flux[mask] = np.nan # masking the flux values with nan
            ivar[mask] = 0. # masking values with 0 in the ivar
            hdul.close() # closing the data
            
            coadd_data = np.vstack([wave, flux[:, 0], ivar[:, 0]]).T # here we are stacking the data to be appended into the seventh extension, also transpose it to get the proper shape
            
            hdul = fits.open(fname) # opening the fits file
            hdu1, hdu2, hdu3, hdu4, hdu5, hdu6 = hdul[0:6] # saving earlier extensions
            hdu1.header.add_comment("Extension 7: Data for single exposure (wave, flux, ivar)") # naming the new extension
            hdu7 = fits.ImageHDU(coadd_data) # saving the data into fits format
            hdulist = fits.HDUList([hdu1, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7]) # appending the data to a list
            hdulist.writeto(fname, overwrite=True) # saving the data to a fits file
            
    if coadd_exp == 1:
        print("Coadd took {:.1f}s".format(time.time()-start))

if __name__=="__main__":
    # here is what happens when we run the file, we are going to use all the functions above to get the data, subtract skylines and then save the output spectra
    start = time.time() # timing
    datatype = 'fox' # this can be changed in the future if a different data type is used
    
    if single == 1:
        # First we are determining which files we want to look at or which exposures
        files1 = []
        file_txt = open(file_nums_path_single) # opening the file_nums txt file
        for line in file_txt: # for each file listed in the txt file
            files1.append(line[:4]) # append the exposure number, EX: 0073
        file_txt.close()

        if coadd_exp == 0:
            files = files1[exp_num] # if we only want to look at a specific exposure then we are limiting the files we look at to the single exposure
        if coadd_exp == 1:
            files = files1 # otherwise we look at all the exposures listed in the .txt file

        # here we are getting the path to the data that we are using
        fnames_r1 = glob.glob(os.path.join(dir_data, "{}*ds_{}_specs.fits".format('r',datatype)))
        fnames_b1 = glob.glob(os.path.join(dir_data, "{}*ds_{}_specs.fits".format('b',datatype)))

        # here we are only taking the path to the files we want to look at
        fnames_r = []
        fnames_b = []
        for i in range(len(fnames_r1)): # for red fibers
            if fnames_r1[i][-21:-17] in files:
                fnames_r.append(fnames_r1[i])
        for i in range(len(fnames_b1)): # for blue fibers
            if fnames_b1[i][-21:-17] in files:
                fnames_b.append(fnames_b1[i])
                
        M = len(fnames_r) # the number of exposures we are looking at, this is used in some of the functions
        Nframe = len(fnames_r) # once again the number of exposures we are looking at

        translate_fibermap(path_fibermap_single) # first we are translating the fibermap using the function, this also gets the object names and information
        get_mjd(file_nums_path_single) # then we are getting the mjds of the exposures that we are looking at, and computing the mjd at the middle of the exposure
        get_data(64, 2, file_nums_path_single) # here we are extracting the 'raw' data from the files which we will then perform a sky subtraction and coadd on if we choose

        rb, Nord, hduindex = "r", 2, 2 # here we are stating that we want to start with the red fibers, where there are two orders, and we will using the second index (extensions 3) in the coadd function
        for extracttype in ["fox"]: # only looking at the _fox_ files, if this changes the list can be altered
            print("Running",rb,extracttype) # print what it is running, for what color and what extracttype
            interpolate_onto_common_dispersion(rb, extracttype) # interpolatiing the data for this order
            for iord in range(Nord): # for each order we run this for loop
                sort_by_object(rb,iord,coadd_exp,extracttype) # creates output arrays and sorts the objects for easier use in the following functions
                subtract_sky_simple(rb,iord) # subtract master sky from all fibers
                process_master_sky(rb,iord) # identify sky line pixes in the master sky and generate the sky masks
                scaled_sky_subtraction(rb,iord) # scale the master sky to match the variable sky, then subtract this scaled sky subtraction from the data
                svd_sky_subtraction(rb,iord) # Dr. Song workin on this and it is a dummy extension for now, NOT IMPLEMENTED
                coadd(rb,iord,hduindex=hduindex) # coadding the results if there are multiple exposures, if there are not then this is just reformatting the data

            print("Time is {:.1f}".format(time.time()-start)) # showing the timing so far
            start = time.time() # timing
        # repeating the same process for blue fibers, See above documentation for descriptions
        rb, Nord, hduindex = "b", 2, 2 
        for extracttype in ["fox"]:
            print("Running",rb,extracttype)
            interpolate_onto_common_dispersion(rb, extracttype) 
            for iord in range(Nord):
                sort_by_object(rb,iord,coadd_exp,extracttype) 
                subtract_sky_simple(rb,iord) 
                process_master_sky(rb,iord)
                scaled_sky_subtraction(rb,iord) 
                svd_sky_subtraction(rb,iord) # Dr. Song workin on this and it is a dummy extension for now
                coadd(rb,iord,hduindex=hduindex) 
                
    else: # Here this is for when we are running over a directory rather than a single object
        file_nums_all = glob.glob(os.path.join(file_nums_path+'*')) # here we get all the objects we are looking at
        file_nums_all.sort()
        num_objects = len(file_nums_all)
        for i in range(num_objects):
            coadd_exp = 1
            field_name = file_nums_all[i].split('/')[-1][:-4]+'_'+run_name
            filenum_file = file_nums_all[i] # getting the field file that we are looking at
            
            files1 = []
            file_txt = open(file_nums_all[i]) # opening the file_nums txt file
            for line in file_txt: # for each file listed in the txt file
                files1.append(line[:4]) # append the exposure number, EX: 0073
            file_txt.close()
            
            files = files1 # Here we are coadding first
            if len(files) > 1: # this is making sure there is more than one exposure to caodd
                if not os.path.exists('%s/objnames/%s'%(basedir, run_name)): # making the objnames directory
                    os.makedirs('%s/objnames/%s'%(basedir, run_name))
                if not os.path.exists(basedir+"/reduced_data/%s/%s"%(run_name, field_name)): # making output spectra directory
                    os.makedirs(basedir+"/reduced_data/%s/%s"%(run_name, field_name))
                npydir = basedir+"/npy_data_files/%s/%s"%(run_name, field_name)
                if not os.path.exists(npydir): # making output directory for the intermediate .npy files for record keeping
                    os.makedirs(npydir)
                reductiondir = basedir+"/%s/reduction_figures/"%(run_name) 
                if not os.path.exists(reductiondir): # making output directory for the intermediate .npy files for record keeping
                    os.makedirs(reductiondir)

                # Here is the locations for the mjd files
                mjd_output = basedir+'/mjds/%s/%s'%(run_name, field_name)

                # Here we are making the directories if they do not already exist
                if not os.path.exists(mjd_output):
                    os.makedirs(mjd_output)

                # These two files are where the objnames of the observed objects are saved. This also contains R.A., decl., magnitudes, fiber number etc.
                file_objB = open('%s/objnames/%s/objnames_b_%s.txt'%(basedir, run_name, field_name),'w') # for the blue fibers
                file_objR = open('%s/objnames/%s/objnames_r_%s.txt'%(basedir, run_name, field_name),'w') # for the red fibers

                # This directory is where to save the output files including the spectra, master skies and other sky files
                outdir = basedir+"/reduced_data/%s/%s"%(run_name, field_name)
                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                # here we are getting the path to the data that we are using
                fnames_r1 = glob.glob(os.path.join(dir_data, "{}*ds_{}_specs.fits".format('r',datatype)))
                fnames_b1 = glob.glob(os.path.join(dir_data, "{}*ds_{}_specs.fits".format('b',datatype)))

                # here we are only taking the path to the files we want to look at
                fnames_r = []
                fnames_b = []
                for i in range(len(fnames_r1)): # for red fibers
                    if fnames_r1[i][-21:-17] in files:
                        fnames_r.append(fnames_r1[i])
                for i in range(len(fnames_b1)): # for blue fibers
                    if fnames_b1[i][-21:-17] in files:
                        fnames_b.append(fnames_b1[i])

                M = len(fnames_r) # the number of exposures we are looking at, this is used in some of the functions
                Nframe = len(fnames_r) # once again the number of exposures we are looking at

                # now we need to choose the correct fibermap 
                fibermaps_all = glob.glob(os.path.join(path_fibermap+'*')) # here we are getting all the fibermaps for the given run
                for i in range(len(fibermaps_all)): # looping through all the maps
                    if (fibermaps_all[i][-1] == 'p') and (field_name.upper()[:3] in fibermaps_all[i].upper()): # this selects the files that end in .fibermap, and contain the field name
                        fibermap_path = fibermaps_all[i]

                translate_fibermap(fibermap_path) # first we are translating the fibermap using the function, this also gets the object names and information
                get_mjd(filenum_file) # then we are getting the mjds of the exposures that we are looking at, and computing the mjd at the middle of the exposure
                get_data(64, 2, filenum_file) # here we are extracting the 'raw' data from the files which we will then perform a sky subtraction and coadd on if we choose

                rb, Nord, hduindex = "r", 2, 2 # here we are stating that we want to start with the red fibers, where there are two orders, and we will using the second index (extensions 3) in the coadd function
                for extracttype in ["fox"]: # only looking at the _fox_ files, if this changes the list can be altered
                    print("Running",rb,extracttype) # print what it is running, for what color and what extracttype
                    interpolate_onto_common_dispersion(rb, extracttype) # interpolatiing the data for this order
                    for iord in range(Nord): # for each order we run this for loop
                        sort_by_object(rb,iord,coadd_exp,extracttype) # creates output arrays and sorts the objects for easier use in the following functions
                        subtract_sky_simple(rb,iord) # subtract master sky from all fibers
                        process_master_sky(rb,iord) # identify sky line pixes in the master sky and generate the sky masks
                        scaled_sky_subtraction(rb,iord) # scale the master sky to match the variable sky, then subtract this scaled sky subtraction from the data
                        svd_sky_subtraction(rb,iord) # Dr. Song workin on this and it is a dummy extension for now, NOT IMPLEMENTED
                        coadd(rb,iord,hduindex=hduindex) # coadding the results if there are multiple exposures, if there are not then this is just reformatting the data

                    print("Time is {:.1f}".format(time.time()-start)) # showing the timing so far
                    start = time.time() # timing
                # repeating the same process for blue fibers, See above documentation for descriptions
                rb, Nord, hduindex = "b", 2, 2 
                for extracttype in ["fox"]:
                    print("Running",rb,extracttype)
                    interpolate_onto_common_dispersion(rb, extracttype) 
                    for iord in range(Nord):
                        sort_by_object(rb,iord,coadd_exp,extracttype) 
                        subtract_sky_simple(rb,iord) 
                        process_master_sky(rb,iord)
                        scaled_sky_subtraction(rb,iord) 
                        svd_sky_subtraction(rb,iord) # Dr. Song workin on this and it is a dummy extension for now
                        coadd(rb,iord,hduindex=hduindex) 

                original_name = field_name

                # now that we are done the coadding we will rerun with the single exposures for each field
                for j in range(len(files)):
                    coadd_exp = 0
                    field_name = original_name+'_exp'+str(j)
                    files = files1[j] # Here we are coadding first

                    if not os.path.exists('%s/objnames/%s'%(basedir, run_name)): # making the objnames directory
                        os.makedirs('%s/objnames/%s'%(basedir, run_name))
                    if not os.path.exists(basedir+"/reduced_data/%s/%s"%(run_name, field_name)): # making output spectra directory
                        os.makedirs(basedir+"/reduced_data/%s/%s"%(run_name, field_name))
                    npydir = basedir+"/npy_data_files/%s/%s"%(run_name, field_name)
                    if not os.path.exists(npydir): # making output directory for the intermediate .npy files for record keeping
                        os.makedirs(npydir)
                    reductiondir = basedir+"/%s/reduction_figures/"%(run_name) 
                    if not os.path.exists(reductiondir): # making output directory for the intermediate .npy files for record keeping
                        os.makedirs(reductiondir)
                    # Here is the locations for the mjd files
                    mjd_output = basedir+'/mjds/%s/%s'%(run_name, field_name)

                    # Here we are making the directories if they do not already exist
                    if not os.path.exists(mjd_output):
                        os.makedirs(mjd_output)

                    # These two files are where the objnames of the observed objects are saved. This also contains R.A., decl., magnitudes, fiber number etc.
                    file_objB = open('%s/objnames/%s/objnames_b_%s.txt'%(basedir, run_name, field_name),'w') # for the blue fibers
                    file_objR = open('%s/objnames/%s/objnames_r_%s.txt'%(basedir, run_name, field_name),'w') # for the red fibers

                    # This directory is where to save the output files including the spectra, master skies and other sky files
                    outdir = basedir+"/reduced_data/%s/%s"%(run_name, field_name)
                    if not os.path.exists(outdir):
                        os.makedirs(outdir)

                    # here we are getting the path to the data that we are using
                    fnames_r1 = glob.glob(os.path.join(dir_data, "{}*ds_{}_specs.fits".format('r',datatype)))
                    fnames_b1 = glob.glob(os.path.join(dir_data, "{}*ds_{}_specs.fits".format('b',datatype)))

                    # here we are only taking the path to the files we want to look at
                    fnames_r = []
                    fnames_b = []
                    for i in range(len(fnames_r1)): # for red fibers
                        if fnames_r1[i][-21:-17] in files:
                            fnames_r.append(fnames_r1[i])
                    for i in range(len(fnames_b1)): # for blue fibers
                        if fnames_b1[i][-21:-17] in files:
                            fnames_b.append(fnames_b1[i])

                    M = len(fnames_r) # the number of exposures we are looking at, this is used in some of the functions
                    Nframe = len(fnames_r) # once again the number of exposures we are looking at

                    # now we need to choose the correct fibermap 
                    fibermaps_all = glob.glob(os.path.join(path_fibermap+'*')) # here we are getting all the fibermaps for the given run
                    for i in range(len(fibermaps_all)): # looping through all the maps
                        if (fibermaps_all[i][-1] == 'p') and (field_name.upper()[:3] in fibermaps_all[i].upper()): # this selects the files that end in .fibermap, and contain the field name
                            fibermap_path = fibermaps_all[i]

                    translate_fibermap(fibermap_path) # first we are translating the fibermap using the function, this also gets the object names and information
                    get_mjd(filenum_file) # then we are getting the mjds of the exposures that we are looking at, and computing the mjd at the middle of the exposure
                    get_data(64, 2, filenum_file) # here we are extracting the 'raw' data from the files which we will then perform a sky subtraction and coadd on if we choose

                    rb, Nord, hduindex = "r", 2, 2 # here we are stating that we want to start with the red fibers, where there are two orders, and we will using the second index (extensions 3) in the coadd function
                    for extracttype in ["fox"]: # only looking at the _fox_ files, if this changes the list can be altered
                        print("Running",rb,extracttype) # print what it is running, for what color and what extracttype
                        interpolate_onto_common_dispersion(rb, extracttype) # interpolatiing the data for this order
                        for iord in range(Nord): # for each order we run this for loop
                            sort_by_object(rb,iord,coadd_exp,extracttype) # creates output arrays and sorts the objects for easier use in the following functions
                            subtract_sky_simple(rb,iord) # subtract master sky from all fibers
                            process_master_sky(rb,iord) # identify sky line pixes in the master sky and generate the sky masks
                            scaled_sky_subtraction(rb,iord) # scale the master sky to match the variable sky, then subtract this scaled sky subtraction from the data
                            svd_sky_subtraction(rb,iord) # Dr. Song workin on this and it is a dummy extension for now, NOT IMPLEMENTED
                            coadd(rb,iord,hduindex=hduindex) # coadding the results if there are multiple exposures, if there are not then this is just reformatting the data

                        print("Time is {:.1f}".format(time.time()-start)) # showing the timing so far
                        start = time.time() # timing
                    # repeating the same process for blue fibers, See above documentation for descriptions
                    rb, Nord, hduindex = "b", 2, 2 
                    for extracttype in ["fox"]:
                        print("Running",rb,extracttype)
                        interpolate_onto_common_dispersion(rb, extracttype) 
                        for iord in range(Nord):
                            sort_by_object(rb,iord,coadd_exp,extracttype) 
                            subtract_sky_simple(rb,iord) 
                            process_master_sky(rb,iord)
                            scaled_sky_subtraction(rb,iord) 
                            svd_sky_subtraction(rb,iord) # Dr. Song workin on this and it is a dummy extension for now
                            coadd(rb,iord,hduindex=hduindex) 
                            
            else:
                coadd_exp = 0
                if not os.path.exists('%s/objnames/%s'%(basedir, run_name)): # making the objnames directory
                    os.makedirs('%s/objnames/%s'%(basedir, run_name))
                if not os.path.exists(basedir+"/reduced_data/%s/%s"%(run_name, field_name)): # making output spectra directory
                    os.makedirs(basedir+"/reduced_data/%s/%s"%(run_name, field_name))
                npydir = basedir+"/npy_data_files/%s/%s"%(run_name, field_name)
                if not os.path.exists(npydir): # making output directory for the intermediate .npy files for record keeping
                    os.makedirs(npydir)
                reductiondir = basedir+"/%s/reduction_figures/"%(run_name) 
                if not os.path.exists(reductiondir): # making output directory for the intermediate .npy files for record keeping
                    os.makedirs(reductiondir)
                # Here is the locations for the mjd files
                mjd_output = basedir+'/mjds/%s/%s'%(run_name, field_name)

                # Here we are making the directories if they do not already exist
                if not os.path.exists(mjd_output):
                    os.makedirs(mjd_output)

                # These two files are where the objnames of the observed objects are saved. This also contains R.A., decl., magnitudes, fiber number etc.
                file_objB = open('%s/objnames/%s/objnames_b_%s.txt'%(basedir, run_name, field_name),'w') # for the blue fibers
                file_objR = open('%s/objnames/%s/objnames_r_%s.txt'%(basedir, run_name, field_name),'w') # for the red fibers

                # This directory is where to save the output files including the spectra, master skies and other sky files
                outdir = basedir+"/reduced_data/%s/%s"%(run_name, field_name)
                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                # here we are getting the path to the data that we are using
                fnames_r1 = glob.glob(os.path.join(dir_data, "{}*ds_{}_specs.fits".format('r',datatype)))
                fnames_b1 = glob.glob(os.path.join(dir_data, "{}*ds_{}_specs.fits".format('b',datatype)))

                # here we are only taking the path to the files we want to look at
                fnames_r = []
                fnames_b = []
                for i in range(len(fnames_r1)): # for red fibers
                    if fnames_r1[i][-21:-17] in files:
                        fnames_r.append(fnames_r1[i])
                for i in range(len(fnames_b1)): # for blue fibers
                    if fnames_b1[i][-21:-17] in files:
                        fnames_b.append(fnames_b1[i])

                M = len(fnames_r) # the number of exposures we are looking at, this is used in some of the functions
                Nframe = len(fnames_r) # once again the number of exposures we are looking at

                # now we need to choose the correct fibermap 
                fibermaps_all = glob.glob(os.path.join(path_fibermap+'*')) # here we are getting all the fibermaps for the given run
                for i in range(len(fibermaps_all)): # looping through all the maps
                    if (fibermaps_all[i][-1] == 'p') and (field_name.upper()[:3] in fibermaps_all[i].upper()): # this selects the files that end in .fibermap, and contain the field name
                        fibermap_path = fibermaps_all[i]

                translate_fibermap(fibermap_path) # first we are translating the fibermap using the function, this also gets the object names and information
                get_mjd(filenum_file) # then we are getting the mjds of the exposures that we are looking at, and computing the mjd at the middle of the exposure
                get_data(64, 2, filenum_file) # here we are extracting the 'raw' data from the files which we will then perform a sky subtraction and coadd on if we choose

                rb, Nord, hduindex = "r", 2, 2 # here we are stating that we want to start with the red fibers, where there are two orders, and we will using the second index (extensions 3) in the coadd function
                for extracttype in ["fox"]: # only looking at the _fox_ files, if this changes the list can be altered
                    print("Running",rb,extracttype) # print what it is running, for what color and what extracttype
                    interpolate_onto_common_dispersion(rb, extracttype) # interpolatiing the data for this order
                    for iord in range(Nord): # for each order we run this for loop
                        sort_by_object(rb,iord,coadd_exp,extracttype) # creates output arrays and sorts the objects for easier use in the following functions
                        subtract_sky_simple(rb,iord) # subtract master sky from all fibers
                        process_master_sky(rb,iord) # identify sky line pixes in the master sky and generate the sky masks
                        scaled_sky_subtraction(rb,iord) # scale the master sky to match the variable sky, then subtract this scaled sky subtraction from the data
                        svd_sky_subtraction(rb,iord) # Dr. Song workin on this and it is a dummy extension for now, NOT IMPLEMENTED
                        coadd(rb,iord,hduindex=hduindex) # coadding the results if there are multiple exposures, if there are not then this is just reformatting the data

                    print("Time is {:.1f}".format(time.time()-start)) # showing the timing so far
                    start = time.time() # timing
                # repeating the same process for blue fibers, See above documentation for descriptions
                rb, Nord, hduindex = "b", 2, 2 
                for extracttype in ["fox"]:
                    print("Running",rb,extracttype)
                    interpolate_onto_common_dispersion(rb, extracttype) 
                    for iord in range(Nord):
                        sort_by_object(rb,iord,coadd_exp,extracttype) 
                        subtract_sky_simple(rb,iord) 
                        process_master_sky(rb,iord)
                        scaled_sky_subtraction(rb,iord) 
                        svd_sky_subtraction(rb,iord) # Dr. Song workin on this and it is a dummy extension for now
                        coadd(rb,iord,hduindex=hduindex) 

    print("Time is {:.1f}".format(time.time()-start))

################## END
