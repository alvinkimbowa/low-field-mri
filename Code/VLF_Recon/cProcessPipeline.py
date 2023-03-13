#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 11:38:09 2022

@author: sairamgeethanath
"""
import numpy as np 
import keaDataProcessing as keaProc
import imageProcessing as imProc
import matplotlib.pyplot as plt
# from roipoly import RoiPoly
import matplotlib
# matplotlib.use("TkAgg")


# def prepData(dataFolder,subFolder, noiseScan,freqDrift,dim=3):
#     '''Read scanParams'''
#     scanParams = keaProc.readPar(r'%s/%s/acqu.par'%(dataFolder,subFolder))
#
#     '''Read k-space data'''
#     if(dim==3):
#         kSpace, scanParams = keaProc.readKSpace(r'%s/%s/data.3d'%(dataFolder,subFolder), scanParams = scanParams, correctOversampling = False)
#     else:
#         kSpace, scanParams = keaProc.readKSpace(r'%s/%s/data.2d'%(dataFolder,subFolder), scanParams = scanParams, correctOversampling = False)
#
#     ''' K-space centering '''
#     max_xyz = np.asarray((np.where(np.abs(kSpace) == np.abs(kSpace).max())))
#     dx = np.int32(max_xyz[0] - 0.5*kSpace.shape[0])
#     dy = np.int32(max_xyz[1] - 0.5*kSpace.shape[1])
#     dz = np.int32(max_xyz[2] - 0.5*kSpace.shape[2])
#
#     kSpace = np.roll(kSpace, -dx, 0)
#     kSpace = np.roll(kSpace, -dy, 1)
#     kSpace = np.roll(kSpace, -dz, 2)
#
#     '''Noise correction'''
#     if(noiseScan):
#         kSpace = imProc.noiseCorrection(kSpace, scanParams, dataFolder)
#
#     '''Drift correction'''
#     if(dim ==3):
#         trajectory      = np.genfromtxt(r'%s/%s/trajectory.csv'%(dataFolder,subFolder), delimiter = ',')
#         trajectory      = np.rot90(trajectory)
#         # freqDrift       = -300
#         kSpace          = imProc.driftCorrection(kSpace, scanParams, trajectory, freqDrift)
#     else:
#         trajectory =0
#     '''apply filter to k-space data'''
#     # kSpace = np.swapaxes(kSpace, 0,1)
#     # kSpace = keaProc.sineBellSquaredFilter(kSpace, filterStrength =0.2) # strength = 0 no filter, 1 = max
#     kSpace = keaProc.gaussianFilter(kSpace, 1/2, 1/3)
#     '''Zero fill data'''
#     # kSpace       = keaProc.zeroFill(kSpace, (120,120,32))
#
#
#
#     '''Fourier transform data'''
#     reconImage      = np.fft.fftshift(np.fft.fftn((np.fft.fftshift(kSpace))))
#
#
#     return reconImage, kSpace, scanParams, trajectory


def prepData(dataFolder, subFolder, noiseScan, freqDrift):
    '''Read scanParams'''
    scanParams = keaProc.readPar(r'%s/%s/acqu.par' % (dataFolder, subFolder))

    '''Read k-space data'''
    kSpace, scanParams = keaProc.readKSpace(r'%s/%s/data.3d' % (dataFolder, subFolder), scanParams=scanParams,
                                            correctOversampling=False)

    '''Noise correction'''
    if (noiseScan):
        kSpace = imProc.noiseCorrection(kSpace, scanParams, dataFolder)

    '''Drift correction'''
    trajectory = np.genfromtxt(r'%s/%s/trajectory.csv' % (dataFolder, subFolder), delimiter=',')
    trajectory = np.rot90(trajectory)
    # freqDrift       = -300
    kSpace = imProc.driftCorrection(kSpace, scanParams, trajectory, freqDrift)

    '''apply filter to k-space data'''
    kSpace = keaProc.sineBellSquaredFilter(kSpace, filterStrength=0.2)  # strength = 0 no filter, 1 = max

    '''Zero fill data'''
    # kSpace       = keaProc.zeroFill(kSpace, (120,120,32))

    '''Fourier transform data'''
    reconImage = np.fft.fftshift(np.fft.fftn((np.fft.fftshift(kSpace))))

    return reconImage, kSpace, scanParams, trajectory

    
def deconvolve(y,h):
    
    Y = np.fft.fftn(y)
    H = np.fft.fftn(h)
    X = np.divide(Y,H)
    x = np.fft.ifftn(X)
    return x


def get_mask(Im): #2D for now
    plt.imshow(Im)
    plt.title("left click: line segment         right click: close region")
    ROI_object = RoiPoly(color='r')  # let user draw first ROI
    mask = ROI_object.get_mask(Im)
    masked_im = np.multiply(Im,mask)
    return masked_im, mask
    
    