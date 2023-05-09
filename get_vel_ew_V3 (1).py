################# IMPORTS
from astropy.io import fits
from math import pi
import numpy as np
import astropy.io.fits as pyfits
from mpfit import mpfit
import matplotlib.pyplot as plt
import emcee
from scipy import stats
import glob
import os
from scipy import ndimage
from numpy.polynomial.legendre import legfit, legval
from scipy.interpolate import CubicSpline
from schwimmbad import MultiPool
import time

import warnings
warnings.filterwarnings("ignore")

#######################
######## Input ########
#######################
# Here are the inputs we will run the code for, this section must be altered for different fields

# This is the path to the directory where we will read and store all the files
basedir = "/home/t/tingli/jbruce/M2FS"

#save output catalog or not
savedata = 1

#show plots for how normalization is done, if 1 then it will save the normalization image
show_normalization = 0

#save the plot containing the template fitting to the CaT lines
savervplot = 1

# save the plot containing the fitting of the equivalent widths of the CaT lines
saveewplot = 1

# Here is the input for what order we want to fit/plot. 0 is order 0, 1 is order 1, and 2 is for both orders together.
order = 2

objdir = ['/home/t/tingli/jbruce/M2FS/reduced_data/20190809/hyd1_1_20190809_exp0/'] # input spectra folder, this contains multiple objects

# here we have the name we will use to store the files.
field_name = 'hyd1_1_testing_exp0_1'

single = 0 # this decides whether we are looking at a single objects spectra

# Here we are listing the spectra we would like to analyze for a single spectra, rather than a full directory
# if running single, put order 0 first, then 1.
object_fname_single = ['/home/t/tingli/jbruce/M2FS/reduced_data/20190809/hyd1_1_20190809/hyd1_1_20190809_b_4632209949149051392_0.fits', \
                      '/home/t/tingli/jbruce/M2FS/reduced_data/20190809/hyd1_1_20190809/hyd1_1_20190809_b_4632209949149051392_1.fits']

# this is to choose whether to run over the entire folder of reduced data from a given run. This will take a very long time
run_whole_folder = 1
run_name = '20200102' # this is selecting which run
path_to_folder = '/home/t/tingli/jbruce/M2FS/reduced_data/%s/'%(run_name) # the path to the data

#### Here are a few functions that were implemented in IMACS pipeline, but are currently not implemented
bhb = 0 # bhb star or not (NOT YET IMPLEMENTED DO NOT USE)

# just single epoch for now, expand to multiple in future? 
multiple_epoch = 0 # (NOT YET IMPLEMENTED DO NOT USE)

#######################
######### End Input ###
#######################

#######################
####### Output ########
#######################

if run_whole_folder == 0: # when we are running the whole folder we create these directories at the bottom of the file
    #directory for output catalog and figure
    outputdir = basedir+'/vel_ew/'
    if not os.path.exists(outputdir): # here we are making the output directory if it doesnt exist already
        os.makedirs(outputdir)

    #directory for saved figures
    figdir = outputdir+"fig/%s/"%(field_name)
    if not os.path.exists(figdir): # here we are making the output directory if it doesnt exist already
        os.makedirs(figdir)

    #file name for the output catalog
    outputfile = outputdir+"catalog/"+field_name+'.txt'
    if not os.path.exists(outputdir+"catalog/"): # here we are making the output directory if it doesnt exist already
        os.makedirs(outputdir+"catalog/")

#path for the rv templates, note that o1 (order 1) is the shorter wavlength range ~8400 to 8600, o2 is 8600 to 8800
rv_fname_1_b_o1 = "/home/t/tingli/jbruce/M2FS/std_lib/std1_o1_b_restframe_v4.fits"
rv_fname_1_b_o2 = "/home/t/tingli/jbruce/M2FS/std_lib/std1_o2_b_restframe_v4.fits"
rv_fname_1_r_o1 = "/home/t/tingli/jbruce/M2FS/std_lib/std1_o1_r_restframe_v4.fits"
rv_fname_1_r_o2 = "/home/t/tingli/jbruce/M2FS/std_lib/std1_o2_r_restframe_v4.fits"
rv_fname_2_b_o1 = "/home/t/tingli/jbruce/M2FS/std_lib/std2_o1_b_restframe_v4.fits"
rv_fname_2_b_o2 = "/home/t/tingli/jbruce/M2FS/std_lib/std2_o2_b_restframe_v4.fits"
rv_fname_2_r_o1 = "/home/t/tingli/jbruce/M2FS/std_lib/std2_o1_r_restframe_v4.fits"
rv_fname_2_r_o2 = "/home/t/tingli/jbruce/M2FS/std_lib/std2_o2_r_restframe_v4.fits"
rv_fname_3_b_o1 = "/home/t/tingli/jbruce/M2FS/std_lib/std3_o1_b_restframe_v4.fits"
rv_fname_3_b_o2 = "/home/t/tingli/jbruce/M2FS/std_lib/std3_o2_b_restframe_v4.fits"
rv_fname_3_r_o1 = "/home/t/tingli/jbruce/M2FS/std_lib/std3_o1_r_restframe_v4.fits"
rv_fname_3_r_o2 = "/home/t/tingli/jbruce/M2FS/std_lib/std3_o2_r_restframe_v4.fits"

#######################
##### End Output ######
#######################

#######################
######## Parameters ###
#######################

# number of mcmc steps. 1000 for official runs, 100 for test runs to save time.
nsam=100

# snr threshold, any spectra with snr below snr_min will be skipped for RV or EW fit.
# currently when running both orders if either order is below this minimum then the object is skipped
snr_min = 3

# if doing cubicspline interpolation, then cubic = 1. If linear then cubic = 0
cubic = 1

#display the parameters from the EW fit in the terminal (but not saved)
dispara = 0

# spectra display window (just for plotting)
CaT1min=8480
CaT1max=8520

CaT2min=8520
CaT2max=8565

CaT3min=8640
CaT3max=8680

#speed of light
c = 2.99792458e5

#########################
### End paramters########
#########################

#########################
#### Functions ##########
#########################

def normalize_spec(wl, spec, dspec, order): # here is the function to normalize the spectra
    """
    normalize the spectra with a legendre polynomial
    """
    idx = (np.isnan(spec)) # getting the index where the values are nan in the spectra flux
    spec[idx] = 0 # replacing nan vals with 0
    dspec[idx] = 1e15 # replacing nan vals uncertainty with very large number
    idx = np.isnan(dspec) # getting the values where the uncertainties are nan values
    dspec[idx] = 1e15 # replacing all nan uncertainties with large number
    idx = spec < 0  # getting the values where the flux is below 0
    dspec[idx] = 1e15 # all flux less than 0 have very large uncertainty
    idx = dspec < 0 # getting the areas where the uncertainty is less than 0
    dspec[idx] = 1e15 # same for uncertainty less than 0
    idx = dspec == 0 # where uncertainty is = 0
    dspec[idx] == 1e15 # making the uncertainty very large rather than 0
    idx = (np.isinf(spec)) # finding the areas where is it infinite
    spec[idx] = 0 # replacing inf vals with 0
    dspec[idx] = 1e15 # replacing inf vals uncertainty with very large number
    for i in range(len(dspec)): # here we are looking once again to confirm that all the uncertainties and flux have correct values
        if dspec[i] == 0:
            dspec[i] = 1e15
        if dspec[i] < 0:
            dspec[i] = 1e15
        if spec[i] < 0:
            dspec[i] = 1e15
            spec[i] = 0

    snr = np.median(spec)/np.median(dspec) # the snr is the median of the spec divided by uncertainty
    thlow = 0.8 # lower nomralized limit
    thhigh = 1.2 # upper --
    maxiter = 10 # max num of iterations
    
    cont = np.median(spec) # the median flux
        
    idx1 = (spec/cont > thlow) & (spec/cont < thhigh) # where the spec falls within the normalized limits after dividing by median
        
    if order == 0: # here is the noramlization for the first order, which is bigger and includes two CaT dips
        if snr > 7: # for the higher snr spectra we do a polynomial fitting of degree 2 rather than a linear fitting
            for i in range(maxiter):
                idx1 = idx1 # the index we listed above
                z = legfit(wl[idx1], spec[idx1], 2, w = 1./dspec[idx1]) # performing the legendre fit of degree two using the inverse error as a weight over this index (idx1)
                cont = legval(wl,z) # getting the continuum over the wl range using the fit from above
                idx2 = (spec/cont > thlow) & (spec/cont < thhigh) # redefining the range to include areas within the bounds we specified above
                if all(idx1 == idx2): # if everything is within the limits then it is normalized and we can end the loop
                    break
                else:
                    idx1 = idx2 # otherwise we rerun the loop using the new index
        else: # if the snr is lower and the spectra is a lot nosier then we do a linear fitting instead of degree 0
            z = legfit(wl[idx1], spec[idx1], 0, w = 1./dspec[idx1])
            cont = legval(wl,z)
            
    if order == 1: # for the first order since it is much smaller I noticed that the order 0 fit was the best overall and provided the best normalization
        z = legfit(wl[idx1], spec[idx1], 0, w = 1./dspec[idx1]) 
        cont = legval(wl,z)

    if show_normalization: # if we want to see the normalization plot this runs
        plt.figure()
        plt.plot(wl, spec) # plot the spectra
        plt.plot(wl[idx1], spec[idx1], lw = 2) # plot the range we looked at 
        plt.plot(wl, cont) # plot the continuum line that we are using to normalize the fit
        plt.title(str(snr)) # showing the SNR in the title
        plt.savefig(figdir+str(object)+'_'+str(order)+'_normalizing.png') # saving the plot
        plt.close()
        
    spec = spec/cont # normalized spec using the continuum value we found
    dspec = dspec/cont # normalized uncertainty as well

    return spec,dspec # return the normalized spec and dspec

def lp_postv(rv, rvmin, rvmax, mask, wl, model, obj, objerr, helio): # here is the log likelihood function for fitting the velocity template, for a single order
    lp = -np.inf # settign the starting value
    rv1 = rv  - helio # accounts for helio correction by subtracting it from the rv we are looking at
    if rv1 < rvmax and rv1 > rvmin: # if the rv is within the bounds we set then we use this loop
        z = rv1/c # accounting for the doppler effect
        lp_prior=0.0 # setting the prior

        new_wl = wl*(1+z) # the new wavelength after accounting for the doppler effect
        if cubic: # if we are doing a cubic fitting
            p = CubicSpline(new_wl,model) # doing a cubic interpolation of the wavelength and the rv template
            model = p(wl) # getting the new model over the correct wavelength range
        else: # if we are doing a linear fitting
            model = np.interp(wl,new_wl,model) # doing a linear interpolation of the rv tempaltes
        model = model[mask] # getting the new template
        obj = obj[mask] # getting the new science spectra within the mask
        objerr = objerr[mask] # getting the spectra error

        lp_post= - np.sum((obj-model)**2/(2.0*(objerr**2))) # using the likelihood function for this given rv and the rv template
        if np.isfinite(lp_post): # if the lp_post is finite (not the starting value)
            lp=lp_post+lp_prior # then the likelihood is what we determined above + the prior

    return lp # return the likelihood

def lp_postV3(rv, rvmin, rvmax, mask, wl, model, obj, objerr, helio): # expaning the likelihood function to both orders
    result = [] 
    # here we are getting the likelihood of the template on both orders for the given rv and adding them both to a list
    result.append(lp_postv(rv, rvmin, rvmax, mask[0], wl[0], model[0], obj[0], objerr[0], helio[0]))
    result.append(lp_postv(rv, rvmin, rvmax, mask[1], wl[1], model[1], obj[1], objerr[1], helio[1]))
    return np.sum(result) # here we sum the log likelihoods to get the overal likelihood

def chi2cal(theta, mask, wl, model, obj, objerr): # here is the chi squared calculation function
    rv = theta # radial v
    z = rv/c # doppler effect

    new_wl = wl*(1+z) # account for doppler to get new wl
    if cubic: # interpolating again using the cubic interpolation
        p = CubicSpline(new_wl,model)
        model = p(wl)
    else: # a linear interpolation if we choose not to do the cubic
        model = np.interp(wl,new_wl,model)
    model = model[mask] # new rv template within mask after interpolation
    obj = obj[mask] # new spectra in mask
    objerr = objerr[mask] # new error in mask
    chi2 = np.sum((obj-model)**2/(objerr**2)) # here we measure the chi squared value of the template and the science spectra at this rv
    return chi2 # returning the corresponding chi2 value

def read_rv_template(filename): # here is reading in the rv templates that we will use and also masking some areas that we dont want to fit
    temp = fits.open(filename) # opening the template file
    start = temp[0].header['CRVAL1'] # Getting the starting value from the template file
    step = temp[0].header['CDELT1'] # getting the step size from the template file
    rvspec = temp[0].data # getting the flux data
    rvwl = [] # making an empty list to append the wl values
    fixed_spec = rvspec # getting the flux 
    for i in range(len(rvspec)): # for each element in the length of the flux
        rvwl.append(start + step*i) # get the wl value based on the start and the step size
        
    # here is masking two of the templates that contain the 'ghost' regions that we do not want to fit
    if filename == rv_fname_2_b_o2: # this is one of the templates we use
        fixed_spec = []
        start = rvspec[np.where(np.array(rvwl) > 8635)[0][0]] # start bound of the flux values for the area we want to mask
        end = rvspec[np.where(np.array(rvwl) > 8655)[0][0]] # end bound of the area we want to mask
        start_wl = rvwl[np.where(np.array(rvwl) > 8635)[0][0]] # start wl we want to mask
        end_wl = rvwl[np.where(np.array(rvwl) > 8655)[0][0]] # end wl we want to mask
        for i in range(len(rvwl)): # for each element within the range
            if (rvwl[i] > 8635) and (rvwl[i] < 8655): # if the value is in the masking range
                fixed_spec.append(1.0) # replace it with unity flux
            else:
                fixed_spec.append(rvspec[i]) # otherwise it remains the same
    if filename == rv_fname_3_b_o2:  # same process for another template that we are using
        fixed_spec = []
        start = rvspec[np.where(np.array(rvwl) > 8635)[0][0]]
        end = rvspec[np.where(np.array(rvwl) > 8650)[0][0]]
        start_wl = rvwl[np.where(np.array(rvwl) > 8635)[0][0]]
        end_wl = rvwl[np.where(np.array(rvwl) > 8650)[0][0]]
        for i in range(len(rvwl)):
            if (rvwl[i] > 8635) and (rvwl[i] < 8650):
                fixed_spec.append(1.0)
            else:
                fixed_spec.append(rvspec[i])
                
    return rvwl, fixed_spec # return the template spectra and wavlength

def get_rv_order(wl, spec, dspec, rvwl, rvspec, object, rvstar, order, make_plot, helio): # getting the RV of a single order
    if single and bhb: # this is not implemented
        fitstart = (np.abs(wl-8400)).argmin()
        fitend = (np.abs(wl-9000)).argmin()
    else:
        if order == 0: # for the first order we are choosing the area that we want to fit over. This is not the template fitting region, just the overal range we are normalizing and using to plot
            fitstart = (np.abs(wl-8400)).argmin() # 8460 also works # the fitting start, this can be modified to test different fitting areas
            fitend = (np.abs(wl-8585)).argmin() # 8570 works # the fitting end, this can be modified to test different fitting areas
        if order == 1: # same for the second order, getting the range to fit over
            fitstart = (np.abs(wl-8600)).argmin() # 8620 Works
            fitend = (np.abs(wl-8695)).argmin() # 8690 works
            

    spec = spec[fitstart:fitend] # getting the flux within the bounds
    dspec = dspec[fitstart:fitend] # error within bounds
    wl = wl[fitstart:fitend] # wl within bounds
    
    spec,dspec = normalize_spec(wl, spec, dspec, order) # normalizing the spectra within the bounds

    ndim=1 # choosing the number of dimensions
    nwalkers=20 # choosing the number of walkers for MCMC
    rvmin = -800 # choosing a minimum rv
    rvmax = 800 # choosing a maximum rv 
    
    nstars = len(rvstar) # here is the number of star tempaltes we will be fitting
    rvdist = np.zeros([nstars, nwalkers * nsam]) # making a zero array for storing data on each template
    chi2rv = np.zeros(nstars) # making a zeroes array to store the chi squared values 

    rvspec_temp = np.zeros([nstars, len(wl)]) # making a zeroes array to store the template flux data

    # MCMC needs some time to produce reasonable "d" from the likelihood, which is called the "burn-in" period.
    # Adjusting the "burn-in" period is quite empirical.
    nburn=50 # choosing the burn in period
    
    for kk in range(0, nstars, 1): # for each template
        if cubic: # performing cubic interpolation
            p = CubicSpline(rvwl[kk], rvspec[kk])
            tempspec = p(wl)
        else: # performing linear interpolation
            tempspec = np.interp(wl, rvwl[kk], rvspec[kk])
        rvspec_temp[kk] = tempspec # appending the data to the zeroes array we made

    rvspec = rvspec_temp # getting the new spectra after interpolation for all templates
    
    if single and bhb: # NOT IMPLEMENTED
        wlmask = (wl > wlmaskmin_bhb)  & (wl < wlmaskmax_bhb)
    else:
        if order == 0: # here we are choosing the wl mask for which we will fit the templates
            wlmaskmin = 8475 # lower bound
            wlmaskmax = 8575 #  upper bound
        if order == 1: # for the second order
            wlmaskmin = 8645 # lower bound
            wlmaskmax = 8685 # upper bound
        wlmask = (wl > wlmaskmin)  & (wl < wlmaskmax) # making the overall mask within these bounds
    
    snr = np.nanmedian(spec[wlmask]/dspec[wlmask]) # getting the SNR of the spectra within the bounds that we set
    print('SNR = '+str(snr)) # printing the SNR

    for kk in range(0, nstars, 1): # for each template
        # here we use the chi-square minimization to find the starting p0
        rvarr =  np.arange(rvmin,rvmax) # getting an array of possible RV values within our limits
        likearr = np.array([lp_postv(i,rvmin, rvmax, wlmask, wl, rvspec[kk], spec, dspec, helio) for i in rvarr]) # getting an array of likelihood values for these RVs
        p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim)) # getting random values in the shape we want based on dimension and number of walkers
        p0 = p0 + rvarr[max(likearr) == likearr][0] # getting teh starting value based on the rv that maximizes the likelihood function

        with MultiPool() as pool: # this is using a multiprocessing package for when we run the MCMC sampler
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lp_postv, args=(rvmin, rvmax, wlmask, wl, rvspec[kk], spec, dspec, helio), pool=pool) # defining the sampler and likleihood functions
            pos, prob, state = sampler.run_mcmc(p0, nburn) # running the MCMC sampler
            sampler.reset() # resetting the bookkeeping parameters
            sampler.run_mcmc(pos, nsam) # running the sampler again 

        rvdist[kk, :] = sampler.flatchain[:, 0] # getting the stored chain of MCMC samples in a flattened format
        rv_mean = np.nanmedian(rvdist[kk, :]) # getting the mean rv value as the median of the MCMC samples, this is the best fit rv for this template
        rv_std = np.std(rvdist[kk, :]) # getting the uncertainty based on the samples
        masklen = len(wl[wlmask]) # getting the length of the mask that we used
        chi2rv[kk] = chi2cal(rv_mean, wlmask, wl, rvspec[kk], spec, dspec) / masklen # getting the chi squared value by calculating the overal chi squared value and then dividing by the mask length
        print(rv_mean, rv_std, chi2rv[kk], rvstar[kk]) # printing the values we just determined

    chi2rv[chi2rv == 0] = 1e10 # replacing any chi squared values that are zero with a large number so these indices not chosen as the best fit RV value
    
    jj = np.where(chi2rv == np.min(chi2rv))[0][0] # here we are getting the id for the tempalte with the lowest chi squared value
    temp = rvdist[jj] # getting the samples for the best fit template
    temp = stats.sigmaclip(temp, low=5, high=5)[0] # sigmaclipping the data
    rv_mean1 = np.nanmedian(temp) # getting the rv value for the best fit template as the median
    rv_std = 0.5 * (np.percentile(temp, 84) - np.percentile(temp, 16)) # getting the error of this rv value

    print('best fit', rv_mean1, rv_std, chi2rv[jj], rvstar[jj]) # printing the results
    rv_mean = rv_mean1 - helio # here we are accounting for the helio correction to be used in plotting since the helio correction was accounted for in the likelihood function only
    
    if make_plot == True: # if we want to see the fitting plot
        if single and bhb: # NOT IMPLEMENTED
            fig, axarr = plt.subplots(1, 2, figsize=(15,6))
            axarr[0].hist(temp, 100, color="k", histtype="step", range = [rv_mean1 - 5*rv_std, rv_mean1 + 5*rv_std])
            axarr[0].set_title('RV Histogram', fontsize=16)
            axarr[0].xaxis.set_major_locator(plt.MultipleLocator(rv_std*3))
            axarr[0].set_xlabel('RV')
            axarr[0].set_xlim(rv_mean1 - 5*rv_std, rv_mean1 + 5*rv_std)
            #axarr[0].axvline(np.percentile(temp, 16), ls='--', color='r')
            #axarr[0].axvline(np.percentile(temp, 84), ls='--', color='r')
            axarr[0].axvline(np.percentile(temp, 50), ls='--', color='r')

            axarr[1].plot(wl[wlmask], spec[wlmask], 'lime',lw=0.5)
            axarr[1].plot(wl[wlmask]*(1+rv_mean/c), rvspec[jj][wlmask], 'b')
            axarr[1].set_xlim(wlmaskmin_bhb,wlmaskmax_bhb)
            axarr[1].set_ylim(-0.5,1.5)
            axarr[1].set_title(object+'+'+rvstar[jj])

        else:
            if order == 0: # if we are looking at the first order this the the plot
                fig, axarr = plt.subplots(1, 3, figsize=(15,6)) # initiating the plot
                axarr[0].hist(temp, 100, color="k", histtype="step", range = [rv_mean1 - 5*rv_std, rv_mean1 + 5*rv_std]) # plotting the rv histogram within a 5sigma range
                axarr[0].set_title('RV Histogram (helio corrected) SNR = '+str(np.round(snr, 2)), fontsize=16)
                axarr[0].xaxis.set_major_locator(plt.MultipleLocator(rv_std*3)) # setting tick locations
                axarr[0].set_xlabel('RV')
                axarr[0].set_xlim(rv_mean1 - 5*rv_std, rv_mean1 + 5*rv_std) # xlimits
                axarr[0].axvline(np.percentile(temp, 50), ls='--', color='r') # adding vertical lines in the histogram to show median value we are choosing

                axarr[1].plot(wl[wlmask], spec[wlmask], 'lime',lw=0.5) # plotting the first CaT line
                axarr[1].plot(wl[wlmask]*(1+rv_mean/c), rvspec[jj][wlmask], 'b', lw=0.5, ls='--') # plotting the template fit
                axarr[1].set_xlim(CaT1min*(1+rv_mean/c),CaT1max*(1+rv_mean/c))
                axarr[1].set_ylim(-0.5,1.5)
                axarr[1].xaxis.set_major_locator(plt.MultipleLocator(10))
                axarr[1].axvline(8498.03*(1+rv_mean/c), ls='--', color='r') # plotting the location of the CaT dip center

                axarr[2].plot(wl[wlmask], spec[wlmask], 'lime',lw=0.5) # second CaT line, same as above
                axarr[2].plot(wl[wlmask]*(1+rv_mean/c), rvspec[jj][wlmask], 'b', lw=0.5, ls='--')
                axarr[2].set_xlim(CaT2min*(1+rv_mean/c),CaT2max*(1+rv_mean/c))
                axarr[2].set_ylim(-0.5,1.5)
                axarr[2].xaxis.set_major_locator(plt.MultipleLocator(10))
                axarr[2].axvline(8542.09*(1+rv_mean/c), ls='--', color='r')
                axarr[2].set_title(object+'+'+rvstar[jj])
                axarr[2].set_xlabel('Wavelength')
                
            if order == 1: # for the first order we only show a single CaT line, using the same process as above
                fig, axarr = plt.subplots(1, 2, figsize=(15,6))
                axarr[0].hist(temp, 100, color="k", histtype="step", range = [rv_mean1 - 5*rv_std, rv_mean1 + 5*rv_std])
                axarr[0].set_title('RV Histogram (helio corrected) SNR = '+str(np.round(snr, 2)), fontsize=16)
                axarr[0].xaxis.set_major_locator(plt.MultipleLocator(rv_std*3))
                axarr[0].set_xlabel('RV')
                axarr[0].set_xlim(rv_mean1 - 5*rv_std, rv_mean1 + 5*rv_std) 
                axarr[0].axvline(np.percentile(temp, 50), ls='--', color='r')

                axarr[1].plot(wl[wlmask], spec[wlmask], 'lime',lw=0.5)
                axarr[1].plot(wl[wlmask]*(1+rv_mean/c), rvspec[jj][wlmask], 'b', lw=0.5, ls='--')
                axarr[1].set_xlim(CaT3min*(1+rv_mean/c),CaT3max*(1+rv_mean/c))
                axarr[1].set_ylim(-0.5,1.5)
                axarr[1].set_xlabel('Wavelength')
                axarr[1].axvline(8662.14*(1+rv_mean/c), ls='--', color='r')
                axarr[1].xaxis.set_major_locator(plt.MultipleLocator(10))

        if savervplot: # saving the plot if we choose
            plt.savefig(figdir+str(object)+'_'+str(order)+'_rv.png')
 
    return temp, rv_mean1, rv_std, chi2rv[jj], snr, rvstar[jj], spec, wl, rvspec, object, rvstar, wlmask, jj, dspec # returns the values

def get_rv_order_special(wl, spec, dspec, rvwl, rvspec, object, rvstar, order, make_plot, helio, wllow, wlhigh, combined = 0): # This is a special version for plotting only
    # this function is very similar to the normal get_rv_order function but it has some additions that can be modified to test different fitting parameters
    # The main difference is that two arguments (wllow and wlhigh) show the range that we are fitting the tempaltes over and are called directly
    # There is also a section below that allows for the BHB template to not be considered. This was used when I was testing for very small wavelength ranges when I already knew the star was not BHB
    
    # Do not use this function other than for testing 
    
    if single and bhb:
        fitstart = (np.abs(wl-8400)).argmin()
        fitend = (np.abs(wl-9000)).argmin()
    else:
        if order == 0:
            fitstart = (np.abs(wl-8400)).argmin() # 8460 works 
            fitend = (np.abs(wl-8585)).argmin() # 8570 works
        if order == 1:
            fitstart = (np.abs(wl-8600)).argmin() # 8620 Works
            fitend = (np.abs(wl-8695)).argmin() # 8690 works

    spec = spec[fitstart:fitend]
    dspec = dspec[fitstart:fitend]
    wl = wl[fitstart:fitend]
    
    spec,dspec = normalize_spec(wl, spec, dspec, order)

    ndim=1
    nwalkers=20
    rvmin = -800
    rvmax = 800
    
    nstars = len(rvstar)
    rvdist = np.zeros([nstars, nwalkers * nsam])
    chi2rv = np.zeros(nstars)

    rvspec_temp = np.zeros([nstars, len(wl)])

    # MCMC needs some time to produce reasonable "d" from the likelihood, which is called the "burn-in" period.
    # Adjusting the "burn-in" period is quite empirical.
    nburn=50
    
    for kk in range(0, nstars, 1):
        if cubic:
            p = CubicSpline(rvwl[kk], rvspec[kk])
            tempspec = p(wl)
        else:
            tempspec = np.interp(wl, rvwl[kk], rvspec[kk])
        rvspec_temp[kk] = tempspec

    rvspec = rvspec_temp
    
    if single and bhb:
        wlmask = (wl > wlmaskmin_bhb)  & (wl < wlmaskmax_bhb)
    else:
        if order == 0:
            wlmaskmin = wllow # 8475
            wlmaskmax = wlhigh # 8565
        if order == 1:
            wlmaskmin = wllow
            wlmaskmax = wlhigh
        wlmask = (wl > wlmaskmin)  & (wl < wlmaskmax)
    
    snr = np.nanmedian(spec[wlmask]/dspec[wlmask])
    print('SNR = '+str(snr))


    for kk in range(0, nstars, 1):
        # here we use the chi-square minimization to find the starting p0
        rvarr =  np.arange(rvmin,rvmax)
        likearr = np.array([lp_postv(i,rvmin, rvmax, wlmask, wl, rvspec[kk], spec, dspec, helio) for i in rvarr])
        p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
        #p0= p0 * rvmax * 2 - rvmax
        p0 = p0 + rvarr[max(likearr) == likearr][0]

        with MultiPool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lp_postv, args=(rvmin, rvmax, wlmask, wl, rvspec[kk], spec, dspec, helio), pool=pool)
            pos, prob, state = sampler.run_mcmc(p0, nburn)
            sampler.reset()
            sampler.run_mcmc(pos, nsam)

        rvdist[kk, :] = sampler.flatchain[:, 0]
        rv_mean = np.nanmedian(rvdist[kk, :])
        rv_std = np.std(rvdist[kk, :])
        masklen = len(wl[wlmask])
        chi2rv[kk] = chi2cal(rv_mean, wlmask, wl, rvspec[kk], spec, dspec) / masklen
        print(rv_mean, rv_std, chi2rv[kk], rvstar[kk])


    chi2rv[chi2rv == 0] = 1e10
    
    # here I am confirming whether it is BHB or not. This is the addition, where the value marked below can be changed to an arbitrarty number to set a cutoff for bhb templates
    if combined == 0:
        red_chi = []
        red_chi.append(chi2rv[1])
        red_chi.append(chi2rv[2])
        red_id = (red_chi == np.min(red_chi))
        red_jj = np.arange(0, len(red_chi))[red_id][0]

        bhb_chi = []
        bhb_chi.append(chi2rv[0])
        bhb_id = (bhb_chi == np.min(bhb_chi))
        bhb_jj = np.arange(0, len(bhb_chi))[bhb_id][0]

        if ((red_chi[red_jj]/bhb_chi[bhb_jj]) > 1): # setting this 1 to a higher number prevents BHB template fitting unless it is very clearly a bhb star
            if bhb_jj == 0:
                rvidx = 0
        elif ((red_chi[red_jj]/bhb_chi[bhb_jj]) < 1):
            if red_jj == 0:
                rvidx = 1
            elif red_jj == 1:
                rvidx = 2
        else:
            rvidx = 0
    if combined == 1:
        rvidx = 0
    
    jj = rvidx
    temp = rvdist[jj]
    #temp_mean = np.nanmedian(temp)
    #if np.std(temp) > 10: temp = temp[(temp < temp_mean + 150) & (temp > temp_mean - 150)]
    temp = stats.sigmaclip(temp, low=5, high=5)[0]
    rv_mean1 = np.nanmedian(temp)
    #rv_std = np.std(temp)
    rv_std = 0.5 * (np.percentile(temp, 84) - np.percentile(temp, 16))

    print('best fit', rv_mean1, rv_std, chi2rv[jj], rvstar[jj])
    rv_mean = rv_mean1 - helio
    
    # Plot the result.
    if make_plot == True:
        if single and bhb:
            fig, axarr = plt.subplots(1, 2, figsize=(15,6))
            axarr[0].hist(temp, 100, color="k", histtype="step", range = [rv_mean1 - 5*rv_std, rv_mean1 + 5*rv_std])
            axarr[0].set_title('RV Histogram', fontsize=16)
            axarr[0].xaxis.set_major_locator(plt.MultipleLocator(rv_std*3))
            axarr[0].set_xlabel('RV')
            axarr[0].set_xlim(rv_mean1 - 5*rv_std, rv_mean1 + 5*rv_std)
            #axarr[0].axvline(np.percentile(temp, 16), ls='--', color='r')
            #axarr[0].axvline(np.percentile(temp, 84), ls='--', color='r')
            axarr[0].axvline(np.percentile(temp, 50), ls='--', color='r')

            axarr[1].plot(wl[wlmask], spec[wlmask], 'lime',lw=0.5)
            axarr[1].plot(wl[wlmask]*(1+rv_mean/c), rvspec[jj][wlmask], 'b')
            axarr[1].set_xlim(wlmaskmin_bhb,wlmaskmax_bhb)
            axarr[1].set_ylim(-0.5,1.5)
            axarr[1].set_title(object+'+'+rvstar[jj])

        else:
            if order == 0:
                fig, axarr = plt.subplots(1, 3, figsize=(15,6))
                axarr[0].hist(temp, 100, color="k", histtype="step", range = [rv_mean1 - 5*rv_std, rv_mean1 + 5*rv_std])
                axarr[0].set_title('RV Histogram (helio corrected) SNR = '+str(np.round(snr, 2)), fontsize=16)
                axarr[0].xaxis.set_major_locator(plt.MultipleLocator(rv_std*3))
                axarr[0].set_xlabel('RV')
                axarr[0].set_xlim(rv_mean1 - 5*rv_std, rv_mean1 + 5*rv_std)
                #axarr[0].axvline(np.percentile(temp, 16), ls='--', color='r')
                #axarr[0].axvline(np.percentile(temp, 84), ls='--', color='r')
                axarr[0].axvline(np.percentile(temp, 50), ls='--', color='r')

                axarr[1].plot(wl[wlmask], spec[wlmask], 'lime',lw=0.5)
                axarr[1].plot(wl[wlmask]*(1+rv_mean/c), rvspec[jj][wlmask], 'b', lw=0.5, ls='--')
                axarr[1].set_xlim(8498.03*(1+rv_mean/c)-5,8498.03*(1+rv_mean/c)+5)
                axarr[1].set_ylim(-0.5,1.5)
                axarr[1].xaxis.set_major_locator(plt.MultipleLocator(10))
                axarr[1].axvline(8498.03*(1+rv_mean/c), ls='--', color='r')

                axarr[2].plot(wl[wlmask], spec[wlmask], 'lime',lw=0.5)
                axarr[2].plot(wl[wlmask]*(1+rv_mean/c), rvspec[jj][wlmask], 'b', lw=0.5, ls='--')
                axarr[2].set_xlim(8542.09*(1+rv_mean/c)-5,8542.09*(1+rv_mean/c)+5)
                axarr[2].set_ylim(-0.5,1.5)
                axarr[2].xaxis.set_major_locator(plt.MultipleLocator(10))
                axarr[2].axvline(8542.09*(1+rv_mean/c), ls='--', color='r')
                axarr[2].set_title(object+'+'+rvstar[jj])
                axarr[2].set_xlabel('Wavelength')
                plt.savefig(figdir+str(object)+'_'+str(order)+str(wllow)+'-'+str(wlhigh)+'_rv.png')
                plt.close()
                
            if order == 1:
                fig, axarr = plt.subplots(1, 3, figsize=(15,6))
                
                axarr[0].hist(temp, 100, color="k", histtype="step", range = [rv_mean1 - 5*rv_std, rv_mean1 + 5*rv_std])
                axarr[0].set_title('RV Histogram (helio corrected) SNR = '+str(np.round(snr, 2))+' WL '+str(wllow)+' '+str(wlhigh), fontsize=16)
                axarr[0].xaxis.set_major_locator(plt.MultipleLocator(rv_std*3))
                axarr[0].set_xlabel('RV = '+str(rv_mean1))
                axarr[0].set_xlim(rv_mean1 - 5*rv_std, rv_mean1 + 5*rv_std) # changed lower limit to 1
                #axarr[0].axvline(np.percentile(temp, 16), ls='--', color='r')
                #axarr[0].axvline(np.percentile(temp, 84), ls='--', color='r')
                axarr[0].axvline(np.percentile(temp, 50), ls='--', color='r')

                #axarr[1].plot(wl[wlmask], spec[wlmask], 'lime',lw=0.5)
                axarr[1].plot(wl, spec, 'lime',lw=0.5)
                axarr[1].plot(wl, dspec, 'y',lw=0.25)
                #axarr[1].plot(wl[wlmask]*(1+rv_mean/c), rvspec[0][wlmask], 'b', lw=0.5, ls='--')
                #axarr[1].plot(wl[wlmask]*(1+rv_mean/c), rvspec[1][wlmask], 'm', lw=0.5, ls='--')
                #axarr[1].plot(wl[wlmask]*(1+rv_mean/c), rvspec[2][wlmask], 'g', lw=0.5, ls='--')
                axarr[1].plot(wl[wlmask]*(1+rv_mean/c), rvspec[jj][wlmask], 'k', lw=0.5, ls='--')
                axarr[1].set_xlim(CaT3min*(1+rv_mean/c),CaT3max*(1+rv_mean/c))
                axarr[1].set_ylim(-0.5,1.5)
                axarr[1].set_xlabel('Wavelength')
                axarr[1].axvline(8662.14*(1+rv_mean/c), ls='--', color='r')
                axarr[1].xaxis.set_major_locator(plt.MultipleLocator(10))
                
                axarr[2].plot(wl, spec, 'lime',lw=0.5)
                axarr[2].plot(wl, dspec, 'y',lw=0.5)
                #axarr[1].plot(wl[wlmask]*(1+rv_mean/c), rvspec[0][wlmask], 'b', lw=0.5, ls='--')
                #axarr[1].plot(wl[wlmask]*(1+rv_mean/c), rvspec[1][wlmask], 'm', lw=0.5, ls='--')
                #axarr[1].plot(wl[wlmask]*(1+rv_mean/c), rvspec[2][wlmask], 'g', lw=0.5, ls='--')
                axarr[2].plot(wl[wlmask]*(1+rv_mean/c), rvspec[jj][wlmask], 'k', lw=0.5, ls='--')
                axarr[2].set_xlim(8662.14*(1+rv_mean/c)-6,8662.14*(1+rv_mean/c)+8)
                axarr[2].set_ylim(-0.5,1.5)
                axarr[2].set_xlabel('Wavelength')
                axarr[2].axvline(8662.14*(1+rv_mean/c), ls='--', color='r')
                axarr[2].xaxis.set_major_locator(plt.MultipleLocator(10))
                plt.savefig(figdir+str(object)+'_'+str(order)+str(wllow)+'-'+str(wlhigh)+'_rv.png')
                plt.close()

    return temp, rv_mean1, rv_std, chi2rv[jj], snr, rvstar[jj], spec, wl, rvspec, object, rvstar, wlmask, jj, dspec

def get_rv_both_ordersV3(wl, spec, dspec, rvwl, rvspec, objects, rvstar, helio): # this is the combined order fitting function that takes inputs as lists containing both orders data
    # The figure output of this code consists of the top two rows showing the individual fitting of the orders without considering the other order
    # the bottom two rows are the results of the combined velocity template fitting, showing the overall histogram and how this best fit RV fits to each order when it is used as the template
    
    length = len(wl) # defining the number of orders that we are working with
    text_add = '' # a blank text 
    objects_new = [] # next we have several lists to append the data for each order to for calling later in the function
    rv_means = []
    rv_stds = []
    rvspecs = []
    wlmasks = []
    wls = []
    snrs = []
    specs = []
    dspecs = []
    rvstars = []
    jjs = []
    rvmin = -800 # here we have some parameters that we set for runnning the MCMC
    rvmax = 800
    ndim=1
    nwalkers=20
    nburn=50
    nstars = len(rvstar) # the number of templates used
    rvdist = np.zeros([nstars, nwalkers * nsam]) # blank arrays to place data
    chi2rv_order = np.zeros([len(wl), nstars]) # getting an array for the chi2values for each template for each order
    if single and bhb: # NOT IMPLEMENTED
        fig, axarr = plt.subplots(length*2, 2, figsize=(20*length, 10*length))
    else:
        fig, axarr = plt.subplots(length*2, 5, figsize=(20*length, 10*length))  # making the subplots based on how many orders we are using
    for i in range(len(wl)): # for each order, since wl contains two arrays (one for each order)
        # first we use the get_rv_order to get the data fitting the order we are looking at by itself
        temp1, rv_mean1, rv_std1, chi21, snr1, template1, spec1, wl1, rvspec1, object1, rvstar1, wlmask1, jj1, dspec1 =  get_rv_order(wl[i],spec[i],dspec[i],rvwl[i],rvspec[i],objects[i], rvstar, i, False, helio[i])
        # then we save the data we found in list format so we can fit templates at the same time
        rvspecs.append(rvspec1)
        wlmasks.append(wlmask1)
        wls.append(wl1)
        specs.append(spec1)
        snrs.append(snr1)
        dspecs.append(dspec1)
        rvstars.append(rvstar1)
        jjs.append(jj1)
        rv_means.append(rv_mean1)
        rv_stds.append(rv_std1)
        objects_new.append(object1)

    for kk in range(0, nstars, 1): # this is the same documentation as get_rv_order, refer to that function for more detail
        # here we use the chi-square minimization to find the starting p0 and then find likelihood
        rvspec_kk = []
        for i in range(len(rvspecs)):
            rvspec_kk.append(rvspecs[i][kk]) # getting the rvspec for each order
        rvarr =  np.arange(rvmin,rvmax)
        likearr = np.array([lp_postV3(i,rvmin, rvmax, wlmasks, wls, rvspec_kk, specs, dspecs, helio) for i in rvarr]) # this is using the lp_postV3 function that combines the likelihood
        p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
        p0 = p0 + rvarr[max(likearr) == likearr][0] # getting the initial value

        with MultiPool() as pool: # running the mcmc sampler for this template
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lp_postV3, args=(rvmin, rvmax, wlmasks, wls, rvspec_kk, specs, dspecs, helio), pool=pool) # this is using the lp_postV3 function that combines the likelihood
            pos, prob, state = sampler.run_mcmc(p0, nburn)
            sampler.reset()
            sampler.run_mcmc(pos, nsam)

        rvdist[kk, :] = sampler.flatchain[:, 0] # getting data
        rv_mean = np.nanmedian(rvdist[kk, :]) # finding best rv
        rv_std = np.std(rvdist[kk, :]) # finding error
        masklen = []
        for i in range(len(wl)): # this is getting the masklength for each order
            masklen.append(len(wls[i][wlmasks[i]])) # recording masklength
            chi2rv_order[i][kk] = chi2cal(rv_mean, wlmasks[i], wls[i], rvspec_kk[i], specs[i], dspecs[i]) / masklen[i] # here is the chi2 calculation for each order individually
            if chi2rv_order[i][kk] == 0: # making sure the value isnt zero
                chi2rv_order[i][kk] = 1e10
            print(rv_mean, rv_std, chi2rv_order[i][kk], rvstar[kk]) # printing the values
    
    chi2rv = np.zeros(nstars) # making a zeroes array to store the chi squared values for the combined order
    for kk in range(0, nstars, 1): # for each template
        chi2rv[kk] = (chi2rv_order[0][kk]*masklen[0] + chi2rv_order[1][kk]*masklen[1])/(masklen[0] + masklen[1])
        # in the above line we revert the chi2rv_order value back to the chi2 value overall by multiplying by the mask length
        # then we find the overall by adding the total chi2 value over both orders and dividing by the total mask length over both orders
        
    jj = np.where(chi2rv == np.min(chi2rv))[0][0] # here we are getting the id for the template with the lowest chi squared value
    
    rvspec_com = []
    tempt = rvdist[jj] # getting samples for the best fit two templates
    tempt = stats.sigmaclip(tempt, low=5, high=5)[0]# sigmaclipping the data
    rv_meant = np.nanmedian(tempt) # getting the rv value for the best fit tem# getting the error of this rv valueplate as the median
    rv_stdt = 0.5 * (np.percentile(tempt, 84) - np.percentile(tempt, 16))
    for i in range(len(wl)):
        rvspec_com.append(rvspecs[i][jj]) # recording the templates over the ranges used for each order

    print('best fit', rv_meant, rv_stdt, chi2rv[jj], rvstar[jj]) # print the values we determined for the best overall fit and best template
    
    rv_mean_new = []
    rv_std_new = []

    for i in range(len(wl)):
        rv_mean_new.append(rv_meant- helio[i]) # removing the helio correction for each order, again should be the same but just in case each order has different heliocorr
  
    new_rvwl = [[rvwl[0][jj]], [rvwl[1][jj]]] # combining the best fit values in a list for plotting
    new_rvspec = [[rvspec[0][jj]], [rvspec[1][jj]]] # same for the rv templates 
    new_rvstar = [rvstar[jj]] # recording the name of the template used
    
    # From here on is plotting
    length = len(wl) # making lists
    text_add = ''
    objects_new = []
    rv_means = []
    rv_stds = []
    rvspecs = []
    wlmasks = []
    chi2s = []
    wls = []
    snrs = []
    specs = []
    dspecs = []
    rvstars = []
    jjs = []
    rvmin = -800
    rvmax = 800
    ndim=1
    nwalkers=20
    nburn=50
    for i in range(len(wl)):
        temp1, rv_mean1, rv_std1, chi21, snr1, template1, spec1, wl1, rvspec1, object1, rvstar1, wlmask1, jj1, dspec1 =  get_rv_order(wl[i],spec[i],dspec[i],new_rvwl[i],new_rvspec[i],objects[i], new_rvstar, i, False, helio[i]) # here we are fitting each order using the best fitting template and range that we found above, this is to test how the combined RV fits to each order individually
        rvspecs.append(rvspec1)
        wlmasks.append(wlmask1)
        chi2s.append(chi21)
        wls.append(wl1)
        specs.append(spec1)
        snrs.append(snr1)
        dspecs.append(dspec1)
        rvstars.append(rvstar1)
        jjs.append(jj1)
        rv_means.append(rv_mean1)
        rv_plot = rv_mean1 - helio[i] # removing helio correction for plotting
        rv_stds.append(rv_std1)
        objects_new.append(object1)
        if single and bhb: # NOT IMPLEMENTED
            axarr[i,0].hist(temp1, 100, color="k", histtype="step", range = [rv_mean1 - 5*rv_std1, rv_mean1 + 5*rv_std1])
            axarr[i,0].set_title('Observed RV ' + str(i+1)+' Histogram', fontsize=16)
            axarr[i,0].xaxis.set_major_locator(plt.MultipleLocator(rv_std1*3))
            axarr[i,0].set_xlabel('RV')
            axarr[i,0].set_xlim(rv_mean1 - 5*rv_std1, rv_mean1 + 5*rv_std1)
            #axarr[i,0].axvline(np.percentile(temp, 16), ls='--', color='r')
            #axarr[i,0].axvline(np.percentile(temp, 84), ls='--', color='r')
            axarr[i,0].axvline(np.percentile(temp1, 50), ls='--', color='r')
    
            axarr[i,1].plot(wl1[wlmask1], spec1[wlmask1], 'lime',lw=0.5)
            axarr[i,1].plot(wl1[wlmask1]*(1+rv_plot/c), rvspec1[jj1], 'b')
            axarr[i,1].set_xlim(wlmaskmin_bhb,wlmaskmax_bhb)
            axarr[i,1].set_ylim(-0.5,1.5)
            axarr[i,1].set_title(object1+'+'+rvstar1[jj1])
    
        else:
            if i == 0: # plotting the first order
                axarr[i, 0].hist(temp1, 100, color="k", histtype="step", range = [rv_mean1 - 5*rv_std1, rv_mean1 + 5*rv_std1])
                axarr[i, 0].set_title('RV Histogram (helio corrected)', fontsize=16)
                axarr[i, 0].xaxis.set_major_locator(plt.MultipleLocator(rv_std1*3))
                axarr[i, 0].set_xlabel('RV')
                axarr[i, 0].set_xlim(rv_mean1 - 5*rv_std1, rv_mean1 + 5*rv_std1)
                axarr[i, 0].axvline(np.percentile(temp1, 50), ls='--', color='r')

                axarr[i, 1].plot(wl1[wlmask1], spec1[wlmask1], 'lime',lw=0.5)
                axarr[i, 1].plot(wl1[wlmask1]*(1+rv_plot/c), rvspec1[jj1][wlmask1], 'b')
                axarr[i, 1].set_xlim(CaT1min*(1+rv_plot/c),CaT1max*(1+rv_plot/c))
                axarr[i, 1].set_ylim(-0.5,1.5)
                axarr[i, 1].xaxis.set_major_locator(plt.MultipleLocator(10))
                axarr[i, 1].axvline(8498.03*(1+rv_plot/c), ls='--', color='r')

                axarr[i, 2].plot(wl1[wlmask1], spec1[wlmask1], 'lime',lw=0.5)
                axarr[i, 2].plot(wl1[wlmask1]*(1+rv_plot/c), rvspec1[jj1][wlmask1], 'b')
                axarr[i, 2].set_xlim(CaT2min*(1+rv_plot/c),CaT2max*(1+rv_plot/c))
                axarr[i, 2].set_ylim(-0.5,1.5)
                axarr[i, 2].xaxis.set_major_locator(plt.MultipleLocator(10))
                axarr[i, 2].axvline(8542.09*(1+rv_plot/c), ls='--', color='r')
                axarr[i, 2].set_title(object+'+'+rvstar1[jj1])
                axarr[i, 2].set_xlabel('Wavelength')
                
                # below we are replotting the same as above just increasing the magnification so we can more clearly see the fit
                axarr[i, 3].plot(wl1[wlmask1], spec1[wlmask1], 'lime',lw=0.5)
                axarr[i, 3].plot(wl1[wlmask1]*(1+rv_plot/c), rvspec1[jj1][wlmask1], 'b', linestyle='--', lw=0.4)
                axarr[i, 3].set_xlim(CaT1min*(1+rv_plot/c) + 10,CaT1max*(1+rv_plot/c) - 10)
                axarr[i, 3].set_ylim(-0.5,1.5)
                axarr[i, 3].set_title('Zoomed in')
                axarr[i, 3].xaxis.set_major_locator(plt.MultipleLocator(10))
                axarr[i, 3].axvline(8498.03*(1+rv_plot/c), ls='--', color='r')

                axarr[i, 4].plot(wl1[wlmask1], spec1[wlmask1], 'lime',lw=0.5)
                axarr[i, 4].plot(wl1[wlmask1]*(1+rv_plot/c), rvspec1[jj1][wlmask1], 'b', linestyle='--', lw=0.4)
                axarr[i, 4].set_xlim(CaT2min*(1+rv_plot/c) + 10,CaT2max*(1+rv_plot/c) - 10)
                axarr[i, 4].set_ylim(-0.5,1.5)
                axarr[i, 4].xaxis.set_major_locator(plt.MultipleLocator(10))
                axarr[i, 4].axvline(8542.09*(1+rv_plot/c), ls='--', color='r')
                axarr[i, 4].set_title('Zoomed in')
                axarr[i, 4].set_xlabel('Wavelength')
                
            if i == 1: # same for the second order
                axarr[i, 0].hist(temp1, 100, color="k", histtype="step", range = [rv_mean1 - 5*rv_std1, rv_mean1 + 5*rv_std1])
                axarr[i, 0].set_title('RV Histogram (helio corrected)', fontsize=16)
                axarr[i, 0].xaxis.set_major_locator(plt.MultipleLocator(rv_std1*3))
                axarr[i, 0].set_xlabel('RV')
                axarr[i, 0].set_xlim(rv_mean1 - 5*rv_std1, rv_mean1 + 5*rv_std1)
                axarr[i, 0].axvline(np.percentile(temp1, 50), ls='--', color='r')

                axarr[i ,1].plot(wl1[wlmask1], spec1[wlmask1], 'lime',lw=0.5)
                axarr[i, 1].plot(wl1[wlmask1]*(1+rv_plot/c), rvspec1[jj1][wlmask1], 'b')
                axarr[i, 1].set_xlim(CaT3min*(1+rv_plot/c),CaT3max*(1+rv_plot/c))
                axarr[i, 1].set_ylim(-0.5,1.5)
                axarr[i, 1].set_xlabel('Wavelength')
                axarr[i, 1].axvline(8662.14*(1+rv_plot/c), ls='--', color='r')
                axarr[i, 1].xaxis.set_major_locator(plt.MultipleLocator(10))
                
                # increase magnification
                axarr[i, 2].plot(wl1[wlmask1], spec1[wlmask1], 'lime',lw=0.5)
                axarr[i, 2].plot(wl1[wlmask1]*(1+rv_plot/c), rvspec1[jj1][wlmask1], 'b', linestyle='--', lw=0.4)
                axarr[i, 2].set_xlim(CaT3min*(1+rv_plot/c) + 20,CaT3max*(1+rv_plot/c) - 10)
                axarr[i, 2].set_ylim(-0.5,1.5)
                axarr[i, 2].set_title('Zoomed in')
                axarr[i, 2].set_xlabel('Wavelength')
                axarr[i, 2].axvline(8662.14*(1+rv_plot/c), ls='--', color='r')
                axarr[i, 2].xaxis.set_major_locator(plt.MultipleLocator(10))
                
                axarr[i,3].axis('off')
                axarr[i,4].axis('off')
                
        # below we are adding information to display for each order
        text_add = text_add + 'SNR ' + str(i+1) + ' = '+str(np.round(snr1, 2))+'\n'
        text_add = text_add + 'RV ' + str(i+1) + ' = '+str(np.round(rv_mean1, 2))+'\n'
        text_add = text_add + 'RV Error ' + str(i+1) + ' = '+str(np.round(rv_std1, 2))+'\n'
        text_add = text_add+'Heliocentric Correction '+str(i+1)+' = '+str(np.round(helio[i], 2))+'\n'
        text_add = text_add+'Template for order '+str(i)+' = '+str(template1)+'\n'
        
    if single and bhb: # Not implemented
        axarr[length, 0].hist(tempt, 100, color="k", histtype="step", range = [rv_meant - 5*rv_stdt, rv_meant + 5*rv_stdt])
        axarr[length, 0].set_title('Combined RV Histogram', fontsize=16)
        axarr[length, 0].xaxis.set_major_locator(plt.MultipleLocator(rv_stdt*3))
        axarr[length, 0].set_xlabel('RV')
        axarr[length, 0].set_xlim(rv_meant - 5*rv_stdt, rv_meant + 5*rv_stdt)
        #axarr[0, 0].axvline(np.percentile(tempt, 16), ls='--', color='r')
        #axarr[0, 0].axvline(np.percentile(tempt, 84), ls='--', color='r')
        axarr[length, 0].axvline(np.percentile(tempt, 50), ls='--', color='r')
            
        for i in range(length):
            j = i + length
            axarr[j, 1].plot(wls[i][wlmasks[i]], specs[i][wlmasks[i]], 'lime',lw=0.5)
            axarr[j, 1].plot(wls[i][wlmasks[i]]*(1+rv_mean_new[i]/c), rvspec_com[i][wlmasks[i]], 'b')
            axarr[j, 1].set_xlim(wlmaskmin_bhb,wlmaskmax_bhb)
            axarr[j, 1].set_ylim(-0.5,1.5)
            axarr[j, 1].set_title(objects[i]+'+'+rvstars[i])
    else: # plotting the combined histogram 
        axarr[length, 0].hist(tempt, 100, color="k", histtype="step", range = [rv_meant - 5*rv_stdt, rv_meant + 5*rv_stdt])
        axarr[length, 0].set_title('Combined RV Histogram', fontsize=16)
        axarr[length, 0].xaxis.set_major_locator(plt.MultipleLocator(rv_stdt*3))
        axarr[length, 0].set_xlabel('RV')
        axarr[length, 0].set_xlim(rv_meant - 5*rv_stdt, rv_meant + 5*rv_stdt)
        axarr[length, 0].axvline(np.percentile(tempt, 50), ls='--', color='r')
        
        # adding the overall information to display
        text_add = text_add + 'Combined Rv = '+str(np.round(rv_meant, 2))+'\n'
        text_add = text_add + 'Combined Rv Error = '+str(np.round(rv_stdt, 2))+'\n'
        text_add = text_add + 'Combined Template used = '+str(rvstar[jj])+'\n'
        
        axarr[length + 1, 0].text(0, 0, text_add, fontsize = 12) # displaying the information
        
        for i in range(length): # next we are showing how the template using the best RV fit to each order as opposed to when the code is run for a single order
            j = i + length
            if i > 0:
                axarr[j, 0].axis('off')
            if i < 1:
                axarr[j, 1].axvline(8498.03*(1+rv_mean_new[i]/c), ls='--', color='r')
                axarr[j, 2].axvline(8542.09*(1+rv_mean_new[i]/c), ls='--', color='r')
                
            axarr[j, 1].plot(wls[i][wlmasks[i]], specs[i][wlmasks[i]], 'lime',lw=0.5) # fitting the first CaT
            axarr[j, 1].plot(wls[i][wlmasks[i]]*(1+rv_mean_new[i]/c), rvspec_com[i][wlmasks[i]], 'b')
            axarr[j, 1].set_xlim(CaT1min*(1+rv_mean_new[i]/c),CaT1max*(1+rv_mean_new[i]/c))
            axarr[j, 1].set_ylim(-0.5,1.5)
            axarr[j, 1].xaxis.set_major_locator(plt.MultipleLocator(10))
    
            axarr[j, 2].plot(wls[i][wlmasks[i]], specs[i][wlmasks[i]], 'lime',lw=0.5) # fitting the second CaT
            axarr[j, 2].plot(wls[i][wlmasks[i]]*(1+rv_mean_new[i]/c), rvspec_com[i][wlmasks[i]], 'b')
            axarr[j, 2].set_xlim(CaT2min*(1+rv_mean_new[i]/c),CaT2max*(1+rv_mean_new[i]/c))
            axarr[j, 2].set_ylim(-0.5,1.5)
            axarr[j, 2].xaxis.set_major_locator(plt.MultipleLocator(10))
            axarr[j, 2].set_title('Epoch '+str(i+1) +' Reverting From Combined RV')
            axarr[j, 2].set_xlabel('Wavelength')
            
            if i == 0: # if we are looking at the first order, then we also include the zoomed in section
                axarr[j, 3].plot(wls[i][wlmasks[i]], specs[i][wlmasks[i]], 'lime',lw=0.5)
                axarr[j, 3].plot(wls[i][wlmasks[i]]*(1+rv_mean_new[i]/c), rvspec_com[i][wlmasks[i]], 'b', lw=0.5, ls='--')
                axarr[j, 3].set_xlim(CaT1min*(1+rv_mean_new[i]/c) + 10,CaT1max*(1+rv_mean_new[i]/c) - 10)
                axarr[j, 3].set_ylim(-0.5,1.5)
                axarr[j, 3].set_title('Zoomed in')
                axarr[j, 3].xaxis.set_major_locator(plt.MultipleLocator(10))
                axarr[j, 3].axvline(8498.03*(1+rv_mean_new[i]/c), ls='--', color='r')
    
                axarr[j, 4].plot(wls[i][wlmasks[i]], specs[i][wlmasks[i]], 'lime',lw=0.5)
                axarr[j, 4].plot(wls[i][wlmasks[i]]*(1+rv_mean_new[i]/c), rvspec_com[i][wlmasks[i]], 'b', lw=0.5, ls='--')
                axarr[j, 4].set_xlim(CaT2min*(1+rv_mean_new[i]/c) + 10,CaT2max*(1+rv_mean_new[i]/c) - 10)
                axarr[j, 4].set_ylim(-0.5,1.5)
                axarr[j, 4].set_title('Zoomed in')
                axarr[j, 4].xaxis.set_major_locator(plt.MultipleLocator(10))
                axarr[j, 4].axvline(8542.09*(1+rv_mean_new[i]/c), ls='--', color='r')
                axarr[j, 4].set_title('Epoch '+str(i+1) +' Reverting From Combined RV')
                axarr[j, 4].set_xlabel('Wavelength')
                
            if i == 1:
                axarr[j,1].axis('off')
                axarr[j,2].axis('off')
                
                axarr[j, 3].plot(wls[i][wlmasks[i]], specs[i][wlmasks[i]], 'lime',lw=0.5) # here we are plotting the third CaT line
                axarr[j, 3].plot(wls[i][wlmasks[i]]*(1+rv_mean_new[i]/c), rvspec_com[i][wlmasks[i]], 'b')
                axarr[j, 3].set_xlim(CaT3min*(1+rv_mean_new[i]/c),CaT3max*(1+rv_mean_new[i]/c))
                axarr[j, 3].set_ylim(-0.5,1.5)
                axarr[j, 3].axvline(8662.14*(1+rv_mean_new[i]/c), ls='--', color='r')
                axarr[j, 3].xaxis.set_major_locator(plt.MultipleLocator(10))
                
                axarr[j, 4].plot(wls[i][wlmasks[i]], specs[i][wlmasks[i]], 'lime',lw=0.5) # and the zoomed in section
                axarr[j, 4].plot(wls[i][wlmasks[i]]*(1+rv_mean_new[i]/c), rvspec_com[i][wlmasks[i]], 'b', lw=0.5, ls='--')
                axarr[j, 4].set_xlim(CaT3min*(1+rv_mean_new[i]/c) + 20,CaT3max*(1+rv_mean_new[i]/c) - 10)
                axarr[j, 4].set_ylim(-0.5,1.5)
                axarr[j, 4].set_title('Zoomed in')
                axarr[j, 4].axvline(8662.14*(1+rv_mean_new[i]/c), ls='--', color='r')
                axarr[j, 4].xaxis.set_major_locator(plt.MultipleLocator(10))
                
            
    plt.savefig(figdir+str(objects[i])+'_rv_V3.png') # saving the figure
         
            
    return rv_meant, rv_stdt, chi2rv[jj], snrs, rvstar[jj], snrs[0], snrs[1], rv_means[0], rv_means[1], rv_stds[0], rv_stds[1], rvstars[0], rvstars[1], chi2s[0], chi2s[1], wls, specs, dspecs # returning the values

def Flin(x,p): # for fitting ew, defining gaussian and lorentzian function

    #P[0] = CONTINUUM LEVEL
    #P[1] = GAUSSIAN HEIGHT/DEPTH FOR MIDDLE CAT LINE
    #P[2] = LINE POSITION
    #P[3] = GAUSSIAN WIDTH
    #P[4] = LORENTZIAN HEIGHT/DEPTH FOR MIDDLE CAT LINE
    #P[5] = LORENTZIAN WIDTH
    #P[6] = GAUSSIAN HEIGHT/DEPTH FOR 8498 CAT LINE
    #P[7] = GAUSSIAN HEIGHT/DEPTH FOR 8662 CAT LINE
    #P[8] = LORENTZIAN HEIGHT/DEPTH FOR 8498 CAT LINE
    #P[9] = LORENTZIAN HEIGHT/DEPTH FOR 8662 CAT LINE

    gauss = p[1]*np.exp(-0.5*((x-p[2])/p[3])**2)+ \
            p[6]*np.exp(-0.5*( (x-p[2]*0.994841)/p[3] )**2) + \
            p[7]*np.exp(-0.5*( (x-p[2]*1.01405)/p[3] )**2)
    lorentz = p[4]*p[5]/( (x-p[2])**2 + (p[5]/2.)**2 ) + \
              p[8]*p[5]/( (x-p[2]*0.994841)**2 + (p[5]/2.)**2 ) + \
              p[9]*p[5]/( (x-p[2]*1.01405)**2 + (p[5]/2.)**2 )

    return p[0] * (1 + gauss + lorentz)

def myfunctlin(p, fjac=None, x=None, y=None, err=None): # using the above function
    model = Flin(x, p)
    status = 0
    return [status, ((y-model)/err)]

def get_ew_both_orders(object, wls, specs, dspecs, rv, snr, gaussianonly = 0): # wls, specs and dspecs are lists with order 0 and 1 info, rv is combined value, object is only one    
    # this function is getting the equivalent widths of the CaT lines, and must have both orders in order to determine this
    # the process involves combining both orders into a single spectra and then fitting the function we defined above
    combined_wl = []
    combined_spec = []
    combined_dspec = []
    
    # after looking at the wavelength that is covered by both orders I chose an arbitrary cutoff that accounts for any overlap that would occur when we combine the orders
    start_wl_1 = wls[1][0] + 20
    end_wl_0 = wls[0][-1] - 10
    
    for i in range(len(wls[0])): # for the first order
        if wls[0][i] < end_wl_0: # if the wavelength does not extend into the area designated as the 'overlap' we can append it to our combined values
            combined_wl.append(wls[0][i]) # append the wavelength
            combined_spec.append(specs[0][i]) # append the flux
            combined_dspec.append(dspecs[0][i]) # append the error
    wl_num = end_wl_0
    step = wls[0][30] - wls[0][29] # here we are quantifying the distance between different wavelengths so that the data we use to cover the area of overlap is equally spaced
    while wl_num < start_wl_1: # for any number that is in the overlap
        combined_wl.append(wl_num) # append the wavelength that we are looking at
        combined_spec.append(1.0) # change the flux to 1
        combined_dspec.append(1e15) # change the error to be very large so this area is not looked at for fitting
        wl_num = wl_num + step # increase the wavelength we are looking at by a step and repeat
    for i in range(len(wls[1])): # Now for the second order at a higher wavelength
        if wls[1][i] > start_wl_1: # if it is outside of the 'overlap' region we can append the data to our lists
            combined_wl.append(wls[1][i])
            combined_spec.append(specs[1][i])
            combined_dspec.append(dspecs[1][i])
    
    # now we change the data type to an array to use the functions we are calling below
    combined_spec = np.array(combined_spec)
    combined_wl = np.array(combined_wl)
    combined_dspec = np.array(combined_dspec)
            
    #renaming the combined spectra to make it easier
    wl = combined_wl
    spec = combined_spec
    dspec = combined_dspec
            
    # here we can look at the entire wavelength not just a single order, and also remove the large dips at the end of each order
    fitstart = (np.abs(wl-8400)).argmin()
    fitend = (np.abs(wl-8700)).argmin()
    
    # now we alter the data to stay within this range
    spec = spec[fitstart:fitend]
    dspec = dspec[fitstart:fitend]
    wl = wl[fitstart:fitend]
    
    # now we choose another region to start the EW fitting
    fitstart = (np.abs(wl-8483*(1+rv/c))).argmin()
    fitend = (np.abs(wl-8677*(1+rv/c))).argmin()
    
    # again we are limiting the area we are looking at for the function fitting
    contstart = (np.abs(wl-8563*(1+rv/c))).argmin()
    contend = (np.abs(wl-8577*(1+rv/c))).argmin()
    
    # we are now defining an area to being looking for peaks in the spectra
    peakfindstart = (np.abs(wl-8538.09*(1+rv/c))).argmin()
    peakfindend = (np.abs(wl-8546.09*(1+rv/c))).argmin()

    sn = np.median(spec[fitstart:fitend]/dspec[fitstart:fitend]) # finding the SNR of the spectra within the bounds that we are looking at
    
    # if we wish to fit a Gaussian function only then we can alter this next line to consider Guassianonly
    if sn < -7:
        guassianonly = 1
    else:
        gaussianonly = 0

    contlevel = np.nanmedian(spec[contstart:contend][spec[contstart:contend]>0]) # finding the continuum level of the spectra as the median
    
    if np.isnan(contlevel) : # if the level we define is a nan value then this is another step to redefine the limits we are looking at and find a new value
        contstart = (np.abs(wl-8590*(1+rv/c))).argmin()
        contend = (np.abs(wl-8610*(1+rv/c))).argmin()
        contlevel = np.nanmedian(spec[contstart:contend][spec[contstart:contend]>0])

    # here we are altering the spec and its error using the continuum level we just found to make it level
    spec = spec / contlevel
    dspec = dspec / contlevel

    smoothspec = ndimage.filters.uniform_filter(spec,size=5) # here we are applying a filter to get a smooth spectra

    linepos = smoothspec[peakfindstart:peakfindend].argmin() # here we are finding the position of the absorption line we are trying to find
    depth = min(smoothspec[peakfindstart:peakfindend]) - np.median(spec[fitstart:fitend]) # here we measure the depth of the line

    initial_guesses = np.zeros(10) # making an array for our initial guesses
    param_control = [{'fixed':0, 'limited':[0,0], 'limits':[0.,0.]} for i in range(10)] # defining params dictionary
    
    # here we define a series of parameters as the initial guesses for each
    initial_guesses[0] = np.median(spec[contstart:contend])
    initial_guesses[1] = 0.5*depth
    #initial_guesses[2] = (wl[fitstart:fitend])[linepos+peakfindstart-fitstart]
    initial_guesses[2] = 8542.09*(1+rv/c)  # changed this to fix the chip gap issue
    initial_guesses[3] = 1.0
    initial_guesses[4] = 0.3*depth
    initial_guesses[5] = 1.0
    initial_guesses[6] = 0.25*depth
    initial_guesses[7] = 0.4*depth
    initial_guesses[8] = 0.15*depth
    initial_guesses[9] = 0.24*depth

    param_control[1]['limited'][1] = 1
    param_control[1]['limits'][1] = 0.
    param_control[1]['limited'][0] = 1
    param_control[1]['limits'][0] = -1.
    param_control[4]['limited'][1] = 1
    param_control[4]['limits'][1]  = 0.
    param_control[4]['limited'][0] = 1
    param_control[4]['limits'][0] = -1.
    param_control[6]['limited'][1] = 1
    param_control[6]['limits'][1]  = 0.
    param_control[6]['limited'][0] = 1
    param_control[6]['limits'][0] = -1.
    param_control[7]['limited'][1] = 1
    param_control[7]['limits'][1]  = 0.
    param_control[7]['limited'][0] = 1
    param_control[7]['limits'][0] = -1.
    param_control[8]['limited'][1] = 1
    param_control[8]['limits'][1]  = 0.
    param_control[8]['limited'][0] = 1
    param_control[8]['limits'][0] = -1.
    param_control[9]['limited'][1] = 1
    param_control[9]['limits'][1]  = 0.
    param_control[9]['limited'][0] = 1
    param_control[9]['limits'][0] = -1.

    #FORCE LINE WIDTHS TO BE AT LEAST 1 RESOLUTION ELEMENT (0.8AA??) AND LESS THAN 300 KM/S
    param_control[3]['limited'][0] = 1
    param_control[3]['limits'][0] = 0.2 #0.8/2.35
    param_control[3]['limited'][1] = 1
    param_control[3]['limits'][1] = 3.63
    param_control[5]['limited'][0] = 1
    param_control[5]['limits'][0] = 0.2 #0.8
    param_control[5]['limited'][1] = 1
    param_control[5]['limits'][1] = 3.63

    if gaussianonly: # if it is gaussian only then we change certain parameters to ignore the lorentzian function
        initial_guesses[4] = 0.
        initial_guesses[5] = 1.0
        param_control[4]['fixed'] = 1
        param_control[5]['fixed'] = 1
    fa = {'x':wl[fitstart:fitend], 'y': spec[fitstart:fitend], 'err':dspec[fitstart:fitend]} # getting the data in the form of x, y and error to use in the function

    try: # here we are using the function we defined earlier
        m = mpfit(myfunctlin, initial_guesses, functkw=fa, quiet=1, parinfo=param_control, xtol = 1.0e-15) # minimizes sum of squares of Gaussian+lorentzian and spectra
    except ValueError:
        print("Oops!  Something wrong.")

    modelgl = Flin(wl[fitstart:fitend], m.params) # using the model we defined earlier
    covargl = m.covar # getting covariance
    errmsg = m.errmsg # checking error messages
    status = m.status # checking status
    print('iter', m.niter) # prining the number of iterations used
    if m.niter <= 2:
        print('MPFIT STUCK! RESULTS MAY BE WRONG')
    niter = m.niter # geting the number of iterations
    perrorgl = m.perror # printing error if there is
    chisqgl = sum((spec[fitstart:fitend]-modelgl)**2/dspec[fitstart:fitend]**2) # getting the chi squared value for the best fit function that we determined
    outparams = m.params # getting the parameters used in the best fit function
    lineparams=outparams # saving these parameters

    if perrorgl is None: # then we generate zeroes arrays to store data
            perrorgl = np.zeros(10) + 1
            covargl = np.zeros([10, 10]) + 1
            niter = 2

    if dispara: # if we want to display parameters in the terminal it will print these lines
        print('  CHI-SQUARE = %10.1f' %chisqgl)
        print('  DOF = %10.1f' %(fitend-fitstart+1-len(outparams)))
        print('  P(0) = %7.3f +/- %7.3f' %(outparams[0],perrorgl[0]))
        print('  P(1) = %7.3f +/- %7.3f' %(outparams[1],perrorgl[1]))
        print('  P(2) = %10.4f +/- %10.4f' %(outparams[2],perrorgl[2]))
        print('  P(3) = %7.3f +/- %7.3f' %(outparams[3],perrorgl[3]))
        print('  P(4) = %7.3f +/- %7.3f' %(outparams[4],perrorgl[4]))
        print('  P(5) = %7.3f +/- %7.3f' %(outparams[5],perrorgl[5]))
        print('  P(6) = %7.3f +/- %7.3f' %(outparams[6],perrorgl[6]))
        print('  P(7) = %7.3f +/- %7.3f' %(outparams[7],perrorgl[7]))
        print('  P(8) = %7.3f +/- %7.3f' %(outparams[8],perrorgl[8]))
        print('  P(9) = %7.3f +/- %7.3f' %(outparams[9],perrorgl[9]))
        
    # here we define the gaussian integral as well as the uncertainty
    gaussian_integral = outparams[1] * outparams[3] * np.sqrt(2*np.pi)
    dgaussian_integral = np.sqrt((outparams[1] * perrorgl[3] * np.sqrt(2*np.pi))**2 + \
                               (perrorgl[1] * outparams[3] * np.sqrt(2*np.pi))**2)

    # here we define the lorentzian integral and the uncertainty
    lorentzian_integral = 2*np.pi*outparams[4]
    dlorentzian_integral = 2*np.pi*perrorgl[4]
    
    # Here we are gitting the second CaT line using the above integrals, and getting the uncertainty and covariance
    ew2_fit = gaussian_integral + lorentzian_integral
    dew2_fit = np.sqrt(dgaussian_integral**2 + dlorentzian_integral**2)
    dew2_fit_covar = np.sqrt(dgaussian_integral**2 + dlorentzian_integral**2 + \
                          2*np.pi*outparams[1]*outparams[3]*covargl[1,3]*perrorgl[1]*perrorgl[3] + \
                          (2*np.pi)**1.5*outparams[3]*covargl[1,4]*perrorgl[1]*perrorgl[4] + \
                          (2*np.pi)**1.5*outparams[1]*covargl[3,4]*perrorgl[3]*perrorgl[4])

    v2 = (outparams[2] - 8542.09)/8542.09*c
    dv2 = perrorgl[2]/8542.09*c
    if dispara: # displaying parameters
        print('V_CaT2: %10.3f +/- %10.3f' %(v2,dv2))
        print('CaT2 (fit): %10.3f +/- %10.3f' %(ew2_fit,dew2_fit))
        print('CaT2 (fit, covar) %10.3f +/- %10.3f' %(ew2_fit,dew2_fit_covar))

    # getting a new gaussian integral and uncertainty
    gaussian_integral = outparams[6] * outparams[3] * np.sqrt(2*np.pi)
    dgaussian_integral = np.sqrt( (outparams[6] * perrorgl[3] * np.sqrt(2*np.pi))**2 + \
                               (perrorgl[6] * outparams[3] * np.sqrt(2*np.pi))**2 )

    # getting a new lorentzian function and integral
    lorentzian_integral = 2*np.pi*outparams[8]
    dlorentzian_integral = 2*np.pi*perrorgl[8]
    # fitting the first CaT line using the integrals, and the error and covariance
    ew1_fit = gaussian_integral + lorentzian_integral
    dew1_fit = np.sqrt(dgaussian_integral**2 + dlorentzian_integral**2)
    dew1_fit_covar = np.sqrt(dgaussian_integral**2 + dlorentzian_integral**2 + \
                          2*np.pi*outparams[6]*outparams[3]*covargl[6,3]*perrorgl[6]*perrorgl[3] + \
                          (2*np.pi)**1.5*outparams[3]*covargl[6,8]*perrorgl[6]*perrorgl[8] + \
                          (2*np.pi)**1.5*outparams[6]*covargl[3,8]*perrorgl[3]*perrorgl[8])

    if dispara: # printing parameters
        print('CaT1 (fit): %10.3f +/- %10.3f' %(ew1_fit,dew1_fit))
        print('CaT1 (fit, covar) %10.3f +/- %10.3f' %(ew1_fit,dew1_fit_covar))
        print('CaT1 (fit): %10.3f +/- %10.3f' %(0.6 * ew2_fit, 0.6 * dew2_fit))

    # getting new gaussian integral and uncertainty
    gaussian_integral = outparams[7] * outparams[3] * np.sqrt(2*np.pi)
    dgaussian_integral = np.sqrt( (outparams[7] * perrorgl[3] * np.sqrt(2*np.pi))**2 + \
                               (perrorgl[7] * outparams[3] * np.sqrt(2*np.pi))**2 )

    # getting the new lorentzian integral
    lorentzian_integral = 2*np.pi*outparams[9]
    dlorentzian_integral = 2*np.pi*perrorgl[9]
    # fitting the third CaT line, and error, and covariance
    ew3_fit = gaussian_integral + lorentzian_integral
    dew3_fit = np.sqrt(dgaussian_integral**2 + dlorentzian_integral**2)
    dew3_fit_covar = np.sqrt(dgaussian_integral**2 + dlorentzian_integral**2 + \
                          2*np.pi*outparams[7]*outparams[3]*covargl[7,3]*perrorgl[7]*perrorgl[3] + \
                          (2*np.pi)**1.5*outparams[3]*covargl[7,9]*perrorgl[7]*perrorgl[9] + \
                          (2*np.pi)**1.5*outparams[7]*covargl[3,9]*perrorgl[3]*perrorgl[9])

    if dispara: # printing parameters
        print('CaT3 (fit): %10.3f +/- %10.3f' %(ew3_fit,dew3_fit))
        print('CaT3 (fit, covar) %10.3f +/- %10.3f' %(ew3_fit,dew3_fit_covar))
        print('CaT3 (fit): %10.3f +/- %10.3f' %(0.9 * ew2_fit, 0.9 * dew2_fit))

    # getting the total EW by summing the individual, and getting the error
    ews = ew1_fit+ew2_fit+ew3_fit
    dews = np.sqrt(dew1_fit**2+dew2_fit**2+dew3_fit**2)
    vcat = v2
    
    if True:
        # Plot the result.
        fig, axarr = plt.subplots(1, 3, figsize=(17,6))

        # plotting first CaT
        axarr[0].plot(wl[fitstart:fitend],spec[fitstart:fitend],c='lime')
        axarr[0].plot(wl[fitstart:fitend],spec[fitstart:fitend]-modelgl,c='y')
        axarr[0].plot(wl[fitstart:fitend],modelgl,lw=1,c='k', ls='--')
        axarr[0].set_xlim(CaT1min*(1+rv/c),CaT1max*(1+rv/c))
        axarr[0].set_ylim(-0.5,1.5)
        axarr[0].xaxis.set_major_locator(plt.MultipleLocator(10))

        # plotting second CaT
        axarr[1].plot(wl[fitstart:fitend],spec[fitstart:fitend],c='lime')
        axarr[1].plot(wl[fitstart:fitend],spec[fitstart:fitend]-modelgl,c='y')

        axarr[1].plot(wl[fitstart:fitend],modelgl,lw=1,c='k', ls='--')
        axarr[1].set_xlim(CaT2min*(1+rv/c),CaT2max*(1+rv/c))
        axarr[1].set_ylim(-0.5,1.5)
        axarr[1].set_title('SNR = '+str(np.round(snr, 2))+' '+str(object)+' EWs = '+str(np.round(ews, 2))+" +_ "+str(np.round(dews, 2)), fontsize=16)
        axarr[1].xaxis.set_major_locator(plt.MultipleLocator(10))
        axarr[1].set_xlabel('Wavelength')

        # plotting third CaT
        axarr[2].plot(wl[fitstart:fitend],spec[fitstart:fitend],c='lime')
        axarr[2].plot(wl[fitstart:fitend],spec[fitstart:fitend]-modelgl,c='y')
        axarr[2].plot(wl[fitstart:fitend],modelgl,lw=1,c='k', ls='--')
        axarr[2].set_xlim(CaT3min*(1+rv/c),CaT3max*(1+rv/c))
        axarr[2].set_ylim(-0.5,1.5)
        axarr[2].xaxis.set_major_locator(plt.MultipleLocator(10))

        if saveewplot: # saving the EW plot
            plt.savefig(figdir+str(object)+'_ew.png')

    return -ew1_fit, dew1_fit, -ew2_fit, dew2_fit, -ew3_fit, dew3_fit, -ews, dews, vcat, niter # returning the values that we determined
    
#########################
#### End Functions ######
#########################

#########################
###### Running ##########
#########################

if __name__ == "__main__":
    start_time = time.time() # Timing
    
    rvwl1_b_o1, rvspec1_b_o1 = read_rv_template(rv_fname_1_b_o1) # getting the first template first order
    rvwl1_b_o2, rvspec1_b_o2 = read_rv_template(rv_fname_1_b_o2) # second order
    
    rvwl2_b_o1, rvspec2_b_o1 = read_rv_template(rv_fname_2_b_o1) # second template
    rvwl2_b_o2, rvspec2_b_o2 = read_rv_template(rv_fname_2_b_o2)
    
    rvwl3_b_o1, rvspec3_b_o1 = read_rv_template(rv_fname_3_b_o1) # third template
    rvwl3_b_o2, rvspec3_b_o2 = read_rv_template(rv_fname_3_b_o2)
    
    #### Note all the templates have different wl ranges
        
    rvspec1_b_o1[rvspec1_b_o1 == 0] = 1 # if the template is zero at any point then we change this to unity
    rvspec1_b_o2[rvspec1_b_o2 == 0] = 1 # --
    
    rvspec2_b_o1[rvspec2_b_o1 == 0] = 1 # if the template is zero at any point then we change this to unity
    rvspec2_b_o2[rvspec2_b_o2 == 0] = 1 # --
            
    rvspec3_b_o1[rvspec3_b_o1 == 0] = 1 # if the template is zero at any point then we change this to unity
    rvspec3_b_o2[rvspec3_b_o2 == 0] = 1 # --
    
    # here is making sure that there are no nan values
    rv_specs = [rvspec1_b_o1, rvspec1_b_o2, \
            rvspec2_b_o1, rvspec2_b_o2, \
            rvspec3_b_o1, rvspec3_b_o2]
    
    for i in rv_specs:
        for j in range(len(i)):
            if np.isnan(i[j]):
                i[j] = 1 # if the value is nan then we replace it with 1
    
    rvstar = np.array(['HD161817', 'HD6268', 'HD21581']) # template names
    
    # order 0 array
    s1 = np.zeros(len(rvwl1_b_o1))
    s2 = np.zeros(len(rvwl2_b_o1))
    s3 = np.zeros(len(rvwl3_b_o1))
    l1 = list(s1)
    l2 = list(s2)
    l3 = list(s3)
    rvwl_0 = np.array([l1, l2, l3]) # combining all templates into a single array for easier use
    rvspec_0 = np.array([l1, l2, l3])
    
    # order 1 array, same as above
    s1 = np.zeros(len(rvwl1_b_o2))
    s2 = np.zeros(len(rvwl2_b_o2))
    s3 = np.zeros(len(rvwl3_b_o2))
    l1 = list(s1)
    l2 = list(s2)
    l3 = list(s3)
    rvwl_1 = np.array([l1, l2, l3])
    rvspec_1 = np.array([l1, l2, l3])
    
    rvwl_0[0] = rvwl1_b_o1 # filling the empty array for order 0
    rvspec_0[0] = rvspec1_b_o1
    rvwl_0[1] = rvwl2_b_o1
    rvspec_0[1] = rvspec2_b_o1
    rvwl_0[2] = rvwl3_b_o1
    rvspec_0[2] = rvspec3_b_o1
    
    rvwl_1[0] = rvwl1_b_o2 # filling the empty array for order 1
    rvspec_1[0] = rvspec1_b_o2
    rvwl_1[1] = rvwl2_b_o2
    rvspec_1[1] = rvspec2_b_o2
    rvwl_1[2] = rvwl3_b_o2
    rvspec_1[2] = rvspec3_b_o2

    if savedata: # if we are saving the data we name the columns we are going to fill
        if run_whole_folder == 0: # if we are running the whole folder we do this step later
            if order == 2: # if we are combining the orders
                f = open(outputfile,'a')
                f.write('#INDEX OBJECT    SNR_1    V_1    dV_1   template_1  chi2rv_1  zq1_1    SNR_1    V_1    dV_1   template_1  chi2rv_1  zq1_1    V_t    dV_t   template_t  chi2rv_t  zq1_t    EW1   DEW1   EW2   DEW2   EW3   DEW3   EW   DEW   VCaT   zq_ew   niter\n')
                f.close()
            else: # for single order 
                f = open(outputfile,'a')
                f.write('#INDEX OBJECT    SNR    V    dV   template  chi2rv  zq1\n')
                f.close()
    
    if single: # if we are looking at a single star 
        if multiple_epoch == 0: # currently this is the only option but leaving this line in case it needs to be expanded in the future
            if order == 0: # if we are looking only at the first order
                object_fname_0 = object_fname_single[0] # getting the spectra
                spec_0 = fits.open(object_fname_0) # opening the file
                data_0 = spec_0[6].data # get data
                helio_0 = spec_0[0].header['HCORR Average'] # getting the average HCORR data, 
                wl_0 = data_0[:, 0] # getting wavelength
                spec_0 = data_0[:, 1] # getting flux
                dspec_0_ivar = data_0[:, 2] # getting ivar
                dspec_0_var = 1/dspec_0_ivar # getting variance
                dspec_0 = np.sqrt(dspec_0_var) # getting the error
                object = object_fname_0.split('_')[-2] # spliting the file name to get the star name
                print('OBJECT ID = %s'% object) # printing the star we are looking at 
                
                # here we fit the rv templates to this order
                temp1, rv_mean1, rv_std1, chi21, snr1, template1, spec1, wl1, rvspec1, object1, rvstar1, wlmask1, jj1, dspec1 = get_rv_order(wl_0,spec_0,dspec_0,rvwl_0,rvspec_0,object, rvstar, 0, True, helio_0)
                print('RV = %8.3f +/- %5.3f' %(rv_mean1, rv_std1)) # print results
                print('chi-square = %5.2f' %chi21) # print the chi2 value
                print('EW fitting not supported for single order fitting') # since we only have one order we are not fitting for EW here
                
            if order == 1: # same as above but for the second order rather than the first order
                object_fname_1 = object_fname_single[1]
                spec_1 = fits.open(object_fname_1)
                data_1 = spec_1[6].data
                helio_1 = spec_1[0].header['HCORR Average']
                wl_1 = data_1[:, 0]
                spec_1 = data_1[:, 1]
                dspec_1_ivar = data_1[:, 2]
                dspec_1_var = 1/dspec_1_ivar
                dspec_1 = np.sqrt(dspec_1_var)
                # spec_1, dspec_1 = normalize_spec(wl_1, spec_1, dspec_1, 1)# already done in rv fitting
                object = object_fname_1.split('_')[-2]
                print('OBJECT ID = %s'% object)
                
                temp1, rv_mean1, rv_std1, chi21, snr1, template1, spec1, wl1, rvspec1, object1, rvstar1, wlmask1, jj1, dspec1 = get_rv_order(wl_1,spec_1,dspec_1,rvwl_1,rvspec_1,object, rvstar, 1, True, helio_1)
                print('RV = %8.3f +/- %5.3f' %(rv_mean1, rv_std1))
                print('chi-square = %5.2f' %chi21)
                print('EW fitting not supported for single order fitting')
                
            if order == 2: # if the single mode is set to run both orders at the same time this part is used
                object_fname_0 = object_fname_single[0] # here is getting the 2 spectra from the different orders
                object_fname_1 = object_fname_single[1] # second order
            
                spec_0 = fits.open(object_fname_0) # opening first order
                spec_1 = fits.open(object_fname_1) # second
            
                data_0 = spec_0[6].data # get data for first order
                data_1 = spec_1[6].data # second
             
                helio_0 = spec_0[0].header['HCORR Average'] # these should be the exact same but we get them just in case
                helio_1 = spec_1[0].header['HCORR Average']
                helio_both = [helio_0, helio_1] # save in a list
            
                wl_0 = data_0[:, 0] # wavelength for order 0
                wl_1 = data_1[:, 0] # order 1
            
                spec_0 = data_0[:, 1] # spec for order 0
                spec_1 = data_1[:, 1] # order 1
            
                # the errors are stored as ivar values so have to revert to std
            
                dspec_0_ivar = data_0[:, 2] # inverse variance
                dspec_1_ivar = data_1[:, 2] # second order

                dspec_0_var = 1/dspec_0_ivar # variance
                dspec_1_var = 1/dspec_1_ivar # second order
            
                dspec_0 = np.sqrt(dspec_0_var) # std
                dspec_1 = np.sqrt(dspec_1_var) # second order
            
                object = object_fname_0.split('_')[-2] # getting the name of the star
            
                print('OBJECT ID = %s'% object) # printing the object name
                    
                # then we use the combined get_rv function to find the combined RV using both orders
                rvt, rverrt, chi2t, snrt, templatet, snr_0, snr_1, rv_0, rv_1, rverr_0, rverr_1, template_0, template_1, chi2_0, chi2_1, wls_norm, specs_norm, dspecs_norm  = get_rv_both_ordersV3([wl_0, wl_1], [spec_0, spec_1], [dspec_0, dspec_1], [rvwl_0, rvwl_1], [rvspec_0, rvspec_1], [object, object], rvstar, helio_both)
                print('RV = %8.3f +/- %5.3f' %(rvt, rverrt)) # printing the results
                print('chi-square = %5.2f' %chi2t)

                # then we get the EW of the CaT lines since we have both orders data
                ew1, dew1, ew2, dew2, ew3, dew3, ews, dews, vcat, niter = get_ew_both_orders(object, wls_norm, specs_norm, \
                                                                                                      dspecs_norm, rvt, snrt, gaussianonly = 0)
                # print the results
                print('EW1 = %8.2f +/- %5.2f' %(ew1, dew1))
                print('EW2 = %8.2f +/- %5.2f' %(ew2, dew2))
                print('EW3 = %8.2f +/- %5.2f' %(ew3, dew3))
                print('EWs = %8.2f +/- %5.2f' %(ews, dews))
                print('V_CaT = %8.2f'%vcat)
                print('niter = '+ str(niter))

                if savedata: # if we want to save the data
                        zq1 = 2  # setting these quality parameters to 2 since we will change them later as we visually inspect
                        zq3 = 2 # same
                        f = open(outputfile, 'a')
                        f.write('%2d %22s %7.2f %7.2f %5.2f %10s %5.2f %3i %7.2f %7.2f %5.2f %10s %5.2f %3i %7.2f %5.2f %10s %5.2f %3i %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %7.2f %3i %3i \n'\
                                %(0, object, snr_0, rv_0, rverr_0, template_0[0], chi2_0, zq1, snr_1, rv_1, rverr_1, template_1[0], chi2_1, zq1, rvt, rverrt, templatet, chi2t, zq1, \
                                 ew1, dew1, ew2, dew2, ew3, dew3, ews, dews, vcat, zq1, niter)) # writing all the variabls
    else: # if we instead want to run over a directory rather than a single file this will run
        
        if run_whole_folder == 0:
            # here is to separate the spectra from the master skies and other files
            files = os.listdir(objdir[0]) # getting the files
            spec_files = []
            for i in range(len(files)):
                split_file = files[i].split('_')
                if split_file[-2].isnumeric():  # if it is a star file then we keep it (we reject master sky and other sky files)
                    spec_files.append(files[i])

            spec_files.sort() # then we sort the files to make it easier to use and this also groups the files for each order of the same star together

            # below is confirming that both orders are present for all spectra
            for i in range(0, len(spec_files), 2): # for every other file (since they should be in groups of two, since two orders)
                ord1 = spec_files[i].split('_')
                ord2 = spec_files[i+1].split('_')
                if ord1[-2] != ord2[-2]: # making sure that the name of adjacent files are identical
                    print('ERROR: The spectra orders files are causing an error, most likely one order for an object is unavailable.', ord1[-2], ord2[-2]) # print error if there is a file missing

            all_files = os.listdir(objdir[0]) # getting all the files from the directory
            all_files.sort() # sorting the files
            star_files = []
            for i in range(len(all_files)):
                temp_file = all_files[i].split('_')
                if (not temp_file[-2][:3] == 'Sky') and (not temp_file[-2] == 'skies') and (not temp_file[-2] == 'wave') and (not temp_file[-1] == 'checkpoints'):
                    star_files.append(all_files[i])

            # get order files
            files_0 = []
            files_1 = []
            for i in range(len(star_files)):
                if star_files[i].split('_')[-1][-1] == '0':
                    files_0.append(star_files[i])
                if star_files[i].split('_')[-1][-1] == '1':
                    files_1.append(star_files[i])

            if order == 0: # if we only want to look at the first order of each star in the directory this will run
                for p in range(int(len(star_files)/2)): # here we are dividing by two since we only need the first order
                    fi = 2*p # this is getting the indices for the first order of each star, ex: 0, 2, 4, etc since 1, 3, 5 are first order files
                    object_fname_0 = objdir[0]+star_files[fi] # getting the path to the first order of the object
                    spec_0 = fits.open(object_fname_0) # opening the file
                    data_0 = spec_0[6].data # get data
                    helio_0 = spec_0[0].header['HCORR Average'] # Hcorr
                    wl_0 = data_0[:, 0] # wavelength
                    spec_0 = data_0[:, 1] # flux
                    dspec_0_ivar = data_0[:, 2] # ivar
                    dspec_0_var = 1/dspec_0_ivar # variance
                    dspec_0 = np.sqrt(dspec_0_var) # error
                    object = object_fname_0.split('_')[-2] # getting object name
                    print('OBJECT ID = %s'% object) # printing name

                    # fitting the rv template
                    temp1, rv_mean1, rv_std1, chi21, snr1, template1, spec1, wl1, rvspec1, object1, rvstar1, wlmask1, jj1, dspec1 = get_rv_order(wl_0,spec_0,dspec_0,rvwl_0,rvspec_0,object, rvstar, 0, True, helio_0)
                    print('RV = %8.3f +/- %5.3f' %(rv_mean1, rv_std1))
                    print('chi-square = %5.2f' %chi21)
                    print('EW fitting not supported for single order fitting') # since we dont have all three CaT lines

                    if savedata: # saving the data into the text file
                        zq1 = 2
                        f = open(outputfile, 'a')
                        f.write('%2d %22s %7.2f %7.2f %5.2f %10s %5.2f %3i \n'\
                                %(p, object, snr1, rv_mean1, rv_std1, template1, chi21, zq1))

            if order == 1:
                for p in range(int(len(star_files)/2)):
                    fi = 2*p
                    se = fi + 1 # this is the second order of each star

                    object_fname_1 = objdir[0]+star_files[se]  # getting file
                    spec_1 = fits.open(object_fname_1) # opening file
                    data_1 = spec_1[6].data # getting data
                    helio_1 = spec_1[0].header['HCORR Average'] #Hcorr
                    wl_1 = data_1[:, 0] # wl
                    spec_1 = data_1[:, 1] # flux
                    dspec_1_ivar = data_1[:, 2] # ivar
                    dspec_1_var = 1/dspec_1_ivar # variance
                    dspec_1 = np.sqrt(dspec_1_var) # error
                    object = object_fname_1.split('_')[-2] # getting the object  name
                    print('OBJECT ID = %s'% object) # printing name

                    # fitting the rv template
                    temp1, rv_mean1, rv_std1, chi21, snr1, template1, spec1, wl1, rvspec1, object1, rvstar1, wlmask1, jj1, dspec1 = get_rv_order(wl_1,spec_1,dspec_1,rvwl_1,rvspec_1,object, rvstar, 1, True, helio_1)
                    print('RV = %8.3f +/- %5.3f' %(rv_mean1, rv_std1))
                    print('chi-square = %5.2f' %chi21)
                    print('EW fitting not supported for single order fitting') # since we only have one CaT line

            if order == 2: # if we are fitting both orders
                for p in range(int(len(star_files)/2)):
                    fi = 2*p # first order indices
                    se = fi + 1 # second order indices

                    object_fname_0 = objdir[0]+star_files[fi] # here is getting the first order file
                    object_fname_1 = objdir[0]+star_files[se] # second

                    spec_0 = fits.open(object_fname_0) # opening first order
                    spec_1 = fits.open(object_fname_1) # second

                    data_0 = spec_0[6].data # get data
                    data_1 = spec_1[6].data # second

                    helio_0 = spec_0[0].header['HCORR Average'] # these should be the exact same
                    helio_1 = spec_1[0].header['HCORR Average'] # but just in case this is the second order
                    helio_both = [helio_0, helio_1] # making them into a list

                    wl_0 = data_0[:, 0] # wavelength
                    wl_1 = data_1[:, 0] # second

                    spec_0 = data_0[:, 1] # flux
                    spec_1 = data_1[:, 1] # second

                    # the errors are stored as ivar values so have to revert to std
                    dspec_0_ivar = data_0[:, 2] # inverse variance
                    dspec_1_ivar = data_1[:, 2] # second order

                    dspec_0_var = 1/dspec_0_ivar # variance
                    dspec_1_var = 1/dspec_1_ivar # second order

                    dspec_0 = np.sqrt(dspec_0_var) # std
                    dspec_1 = np.sqrt(dspec_1_var) # second order

                    snr_0 = np.nanmedian(spec_0/dspec_0) # getting the first order snr
                    snr_1 = np.nanmedian(spec_1/dspec_1) # second order snr
                    print(snr_0, snr_1) # printing the values

                    if (snr_0 < snr_min) or (snr_1 < snr_min): # making sure the snr of these orders is above the limit that we set
                        print('LOW SNR')
                    else:
                        object = object_fname_0.split('_')[-2] # getting the name

                        print('OBJECT ID = %s'% object) # printing the name

                        # then we get the rv of the combined order fitting
                        rvt, rverrt, chi2t, snrt, templatet, snr_0, snr_1, rv_0, rv_1, rverr_0, rverr_1, template_0, template_1, chi2_0, chi2_1, wls_norm, specs_norm, dspecs_norm = get_rv_both_ordersV3([wl_0, wl_1], [spec_0, spec_1], [dspec_0, dspec_1], [rvwl_0, rvwl_1], [rvspec_0, rvspec_1], [object, object], rvstar, helio_both)
                        print('RV = %8.3f +/- %5.3f' %(rvt, rverrt))
                        print('chi-square = %5.2f' %chi2t)

                        # then we can determine the EW since we have both orders and all three CaT
                        ew1, dew1, ew2, dew2, ew3, dew3, ews, dews, vcat, niter = get_ew_both_orders(object, wls_norm, specs_norm, \
                                                                                                              dspecs_norm, rvt, snrt, gaussianonly = 0)
                        print('EW1 = %8.2f +/- %5.2f' %(ew1, dew1))
                        print('EW2 = %8.2f +/- %5.2f' %(ew2, dew2))
                        print('EW3 = %8.2f +/- %5.2f' %(ew3, dew3))
                        print('EWs = %8.2f +/- %5.2f' %(ews, dews))
                        print('V_CaT = %8.2f'%vcat)
                        print('niter = '+ str(niter))

                        if savedata: # saving the data
                            zq1 = 2
                            zq3 = 2
                            f = open(outputfile, 'a')
                            f.write('%2d %22s %7.2f %7.2f %5.2f %10s %5.2f %3i %7.2f %7.2f %5.2f %10s %5.2f %3i %7.2f %5.2f %10s %5.2f %3i %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %7.2f %3i %3i \n'\
                                    %(p, object, snr_0, rv_0, rverr_0, template_0[0], chi2_0, zq1, snr_1, rv_1, rverr_1, template_1[0], chi2_1, zq1, rvt, rverrt, templatet, chi2t, zq1, \
                                     ew1, dew1, ew2, dew2, ew3, dew3, ews, dews, vcat, zq1, niter))
                            
        if run_whole_folder == 1:
            folders = glob.glob(os.path.join(path_to_folder+'*'))
            for i in range(len(folders)):
                # getting the directory
                objdir = [folders[i]+'/']
                # getting the field_name
                field_name = folders[i].split('/')[-1]

                #directory for output catalog and figure
                outputdir = basedir+'/vel_ew/'
                if not os.path.exists(outputdir): # here we are making the output directory if it doesnt exist already
                    os.makedirs(outputdir)

                #directory for saved figures
                figdir = outputdir+"fig/%s/"%(field_name)
                if not os.path.exists(figdir): # here we are making the output directory if it doesnt exist already
                    os.makedirs(figdir)

                #file name for the output catalog
                outputfile = outputdir+"catalog/"+field_name+'.txt'
                if not os.path.exists(outputdir+"catalog/"): # here we are making the output directory if it doesnt exist already
                    os.makedirs(outputdir+"catalog/")
                    
                if savedata: # if we are saving the data we name the columns we are going to fill
                    if order == 2: # if we are combining the orders
                        f = open(outputfile,'a')
                        f.write('#INDEX OBJECT    SNR_1    V_1    dV_1   template_1  chi2rv_1  zq1_1    SNR_1    V_1    dV_1   template_1  chi2rv_1  zq1_1    V_t    dV_t   template_t  chi2rv_t  zq1_t    EW1   DEW1   EW2   DEW2   EW3   DEW3   EW   DEW   VCaT   zq_ew   niter\n')
                        f.close()
                    else: # for single order
                        f = open(outputfile,'a')
                        f.write('#INDEX OBJECT    SNR    V    dV   template  chi2rv  zq1\n')
                        f.close()
                    
                # here is to separate the spectra from the master skies and other files
                files = os.listdir(objdir[0]) # getting the files
                spec_files = []
                for i in range(len(files)):
                    split_file = files[i].split('_')
                    if split_file[-2].isnumeric():  # if it is a star file then we keep it (we reject master sky and other sky files)
                        spec_files.append(files[i])

                spec_files.sort() # then we sort the files to make it easier to use and this also groups the files for each order of the same star together

                # below is confirming that both orders are present for all spectra
                for i in range(0, len(spec_files), 2): # for every other file (since they should be in groups of two, since two orders)
                    ord1 = spec_files[i].split('_')
                    ord2 = spec_files[i+1].split('_')
                    if ord1[-2] != ord2[-2]: # making sure that the name of adjacent files are identical
                        print('ERROR: The spectra orders files are causing an error, most likely one order for an object is unavailable.', ord1[-2], ord2[-2]) # print error if there is a file missing

                all_files = os.listdir(objdir[0]) # getting all the files from the directory
                all_files.sort() # sorting the files
                star_files = []
                for i in range(len(all_files)):
                    temp_file = all_files[i].split('_')
                    if (not temp_file[-2][:3] == 'Sky') and (not temp_file[-2] == 'skies') and (not temp_file[-2] == 'wave') and (not temp_file[-1] == 'checkpoints'):
                        star_files.append(all_files[i])

                # get order files
                files_0 = []
                files_1 = []
                for i in range(len(star_files)):
                    if star_files[i].split('_')[-1][-1] == '0':
                        files_0.append(star_files[i])
                    if star_files[i].split('_')[-1][-1] == '1':
                        files_1.append(star_files[i])

                if order == 0: # if we only want to look at the first order of each star in the directory this will run
                    for p in range(int(len(star_files)/2)): # here we are dividing by two since we only need the first order
                        fi = 2*p # this is getting the indices for the first order of each star, ex: 0, 2, 4, etc since 1, 3, 5 are first order files
                        object_fname_0 = objdir[0]+star_files[fi] # getting the path to the first order of the object
                        spec_0 = fits.open(object_fname_0) # opening the file
                        data_0 = spec_0[6].data # get data
                        helio_0 = spec_0[0].header['HCORR Average'] # Hcorr
                        wl_0 = data_0[:, 0] # wavelength
                        spec_0 = data_0[:, 1] # flux
                        dspec_0_ivar = data_0[:, 2] # ivar
                        dspec_0_var = 1/dspec_0_ivar # variance
                        dspec_0 = np.sqrt(dspec_0_var) # error
                        object = object_fname_0.split('_')[-2] # getting object name
                        print('OBJECT ID = %s'% object) # printing name

                        # fitting the rv template
                        temp1, rv_mean1, rv_std1, chi21, snr1, template1, spec1, wl1, rvspec1, object1, rvstar1, wlmask1, jj1, dspec1 = get_rv_order(wl_0,spec_0,dspec_0,rvwl_0,rvspec_0,object, rvstar, 0, True, helio_0)
                        print('RV = %8.3f +/- %5.3f' %(rv_mean1, rv_std1))
                        print('chi-square = %5.2f' %chi21)
                        print('EW fitting not supported for single order fitting') # since we dont have all three CaT lines

                        if savedata: # saving the data into the text file
                            zq1 = 2
                            f = open(outputfile, 'a')
                            f.write('%2d %22s %7.2f %7.2f %5.2f %10s %5.2f %3i \n'\
                                    %(p, object, snr1, rv_mean1, rv_std1, template1, chi21, zq1))

                if order == 1:
                    for p in range(int(len(star_files)/2)):
                        fi = 2*p
                        se = fi + 1 # this is the second order of each star

                        object_fname_1 = objdir[0]+star_files[se]  # getting file
                        spec_1 = fits.open(object_fname_1) # opening file
                        data_1 = spec_1[6].data # getting data
                        helio_1 = spec_1[0].header['HCORR Average'] #Hcorr
                        wl_1 = data_1[:, 0] # wl
                        spec_1 = data_1[:, 1] # flux
                        dspec_1_ivar = data_1[:, 2] # ivar
                        dspec_1_var = 1/dspec_1_ivar # variance
                        dspec_1 = np.sqrt(dspec_1_var) # error
                        object = object_fname_1.split('_')[-2] # getting the object  name
                        print('OBJECT ID = %s'% object) # printing name

                        # fitting the rv template
                        temp1, rv_mean1, rv_std1, chi21, snr1, template1, spec1, wl1, rvspec1, object1, rvstar1, wlmask1, jj1, dspec1 = get_rv_order(wl_1,spec_1,dspec_1,rvwl_1,rvspec_1,object, rvstar, 1, True, helio_1)
                        print('RV = %8.3f +/- %5.3f' %(rv_mean1, rv_std1))
                        print('chi-square = %5.2f' %chi21)
                        print('EW fitting not supported for single order fitting') # since we only have one CaT line

                if order == 2: # if we are fitting both orders
                    for p in range(int(len(star_files)/2)):
                        fi = 2*p # first order indices
                        se = fi + 1 # second order indices

                        object_fname_0 = objdir[0]+star_files[fi] # here is getting the first order file
                        object_fname_1 = objdir[0]+star_files[se] # second

                        spec_0 = fits.open(object_fname_0) # opening first order
                        spec_1 = fits.open(object_fname_1) # second

                        data_0 = spec_0[6].data # get data
                        data_1 = spec_1[6].data # second

                        helio_0 = spec_0[0].header['HCORR Average'] # these should be the exact same
                        helio_1 = spec_1[0].header['HCORR Average'] # but just in case this is the second order
                        helio_both = [helio_0, helio_1] # making them into a list

                        wl_0 = data_0[:, 0] # wavelength
                        wl_1 = data_1[:, 0] # second

                        spec_0 = data_0[:, 1] # flux
                        spec_1 = data_1[:, 1] # second

                        # the errors are stored as ivar values so have to revert to std
                        dspec_0_ivar = data_0[:, 2] # inverse variance
                        dspec_1_ivar = data_1[:, 2] # second order

                        dspec_0_var = 1/dspec_0_ivar # variance
                        dspec_1_var = 1/dspec_1_ivar # second order

                        dspec_0 = np.sqrt(dspec_0_var) # std
                        dspec_1 = np.sqrt(dspec_1_var) # second order

                        snr_0 = np.nanmedian(spec_0/dspec_0) # getting the first order snr
                        snr_1 = np.nanmedian(spec_1/dspec_1) # second order snr
                        print(snr_0, snr_1) # printing the values

                        if (snr_0 < snr_min) or (snr_1 < snr_min): # making sure the snr of these orders is above the limit that we set
                            print('LOW SNR')
                        else:
                            object = object_fname_0.split('_')[-2] # getting the name

                            print('OBJECT ID = %s'% object) # printing the name

                            # then we get the rv of the combined order fitting
                            rvt, rverrt, chi2t, snrt, templatet, snr_0, snr_1, rv_0, rv_1, rverr_0, rverr_1, template_0, template_1, chi2_0, chi2_1, wls_norm, specs_norm, dspecs_norm = get_rv_both_ordersV3([wl_0, wl_1], [spec_0, spec_1], [dspec_0, dspec_1], [rvwl_0, rvwl_1], [rvspec_0, rvspec_1], [object, object], rvstar, helio_both)
                            print('RV = %8.3f +/- %5.3f' %(rvt, rverrt))
                            print('chi-square = %5.2f' %chi2t)

                            # then we can determine the EW since we have both orders and all three CaT
                            ew1, dew1, ew2, dew2, ew3, dew3, ews, dews, vcat, niter = get_ew_both_orders(object, wls_norm, specs_norm, \
                                                                                                                  dspecs_norm, rvt, snrt, gaussianonly = 0)
                            print('EW1 = %8.2f +/- %5.2f' %(ew1, dew1))
                            print('EW2 = %8.2f +/- %5.2f' %(ew2, dew2))
                            print('EW3 = %8.2f +/- %5.2f' %(ew3, dew3))
                            print('EWs = %8.2f +/- %5.2f' %(ews, dews))
                            print('V_CaT = %8.2f'%vcat)
                            print('niter = '+ str(niter))

                            if savedata: # saving the data
                                zq1 = 2
                                zq3 = 2
                                f = open(outputfile, 'a')
                                f.write('%2d %22s %7.2f %7.2f %5.2f %10s %5.2f %3i %7.2f %7.2f %5.2f %10s %5.2f %3i %7.2f %5.2f %10s %5.2f %3i %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %7.2f %3i %3i \n'\
                                        %(p, object, snr_0, rv_0, rverr_0, template_0[0], chi2_0, zq1, snr_1, rv_1, rverr_1, template_1[0], chi2_1, zq1, rvt, rverrt, templatet, chi2t, zq1, \
                                         ew1, dew1, ew2, dew2, ew3, dew3, ews, dews, vcat, zq1, niter))
                        
    if multiple_epoch == 1: # THIS IS NOT IMPLEMENTED SO IT RETURNS NOTHING
        print('MULTIPLE EPOCH FITTING IS NOT IMPLEMENTED YET')
                
    print("--- %s seconds ---" % (time.time() - start_time)) # end timing
            
#####################################
#################END#################
#####################################