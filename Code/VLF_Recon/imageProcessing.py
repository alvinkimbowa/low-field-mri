# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 10:33:25 2021

@author: Low field
"""
import numpy as np
from pathlib import Path


def driftCorrection(kSpace, acquPar, trajectory, finalF0, initialF0 = 0):
    
    shotLayout = np.zeros(np.shape(trajectory))
    
    acqTime = float(acquPar['acqTime'])*1e-3
    try:
        echoTrain = int(acquPar['etLength'])
    except:
        echoTrain = 1
        
    numShots = int(np.ceil(np.size(trajectory)/echoTrain))
    
    driftPerShot = (finalF0 - initialF0)/numShots
    
    for shot in range(numShots):
        for echo in range(echoTrain):
            shotLayout[trajectory == shot + echo*numShots] = shot
    
    drift = shotLayout*driftPerShot + initialF0
    timeScale = np.linspace(-acqTime/2, acqTime/2, np.size(kSpace,0), endpoint = True)
    
    phaseCorrection = np.exp(-1j*2*np.pi*timeScale[:,np.newaxis,np.newaxis]*drift[np.newaxis,:,:])
            
    shiftedData = np.multiply(kSpace,phaseCorrection)
    return shiftedData  
  
def jitterCorrection(kSpace, acquPar, trajectory, jitter):
    
    jitterArray = np.zeros(np.shape(trajectory))
    
    acqTime = np.float(acquPar['acqTime'])*1e-3
    try:
        echoTrain = int(acquPar['etLength'])
    except:
        echoTrain = 1
        
    numShots = int(np.ceil(np.size(trajectory)/echoTrain))
        
    for shot in range(numShots):
        for echo in range(echoTrain):
            jitterArray[trajectory == shot + echo*numShots] = jitter[shot]


    timeScale = np.linspace(-acqTime/2, acqTime/2, np.size(kSpace,0), endpoint = True)
    
    phaseCorrection = np.exp(-1j*2*np.pi*timeScale[:,np.newaxis,np.newaxis]*jitterArray[np.newaxis,:,:])
     
    shiftedData = np.multiply(kSpace,phaseCorrection)
    return shiftedData  
    
def shiftImage(kSpace,scanParams, distance, axis):
    """ Shift image by adding a time dependent phase to one of the components,
        Distance defines how far in mm, axis the axis along which it should be moved"""
    if axis == 0:
        fov = float(scanParams['FOVread'])
        nrPts = int(scanParams['nrPnts']) 
    elif axis == 1: 
        fov = float(scanParams['FOVphase1'])
        nrPts = int(scanParams['nPhase1']) 
    elif axis == 2:
        fov = float(scanParams['FOVphase2'])
        nrPts = int(scanParams['nPhase2']) 
    else:
        raise("Invalid axis")
    
    hzPerMM = 1e3*float(scanParams['bandwidth'])/float(scanParams['FOVread'])
    freqShift = distance * hzPerMM
    acqTime = nrPts/float(scanParams['bandwidth'])
    timeScale = np.linspace(-acqTime/2, acqTime/2, nrPts)
    phaseTerm = np.exp(-1j*2*np.pi*timeScale*freqShift)
    
    if axis == 0:
        shiftedKspace = np.multiply(kSpace, phaseTerm[:,np.newaxis, np.newaxis])
    elif axis == 1: 
        shiftedKspace = np.multiply(kSpace, phaseTerm[np.newaxis,:, np.newaxis])
    elif axis == 2:
        shiftedKspace = np.multiply(kSpace, phaseTerm[np.newaxis, np.newaxis,:])
    return shiftedKspace

def noiseCorrection(kSpace, scanParams, dataFolder):
    try:
        '''store descriptor in parent folder to be accessed by other reconstruction methods'''
        dataPath            = Path(dataFolder)
        mainDir             = str(dataPath.parent)
        noiseDict           = np.load(r"%s/NoiseData.npy"%(mainDir), allow_pickle = True).item()
    except:
        print("Could not load noise scan data")
        return kSpace
    
    freqOffset = float(scanParams["b1Freq"][:-1]) - float(noiseDict["center_freq"][:-1])
    
    
    print("Freq difference: %.0f Hz"%(freqOffset*1e6))
    
    imageBandwidth  = np.linspace(-float(scanParams["reconBW"])/2,float(scanParams["reconBW"])/2, np.size(kSpace, 0))
    
    polyFunc        = np.poly1d(noiseDict["noise_fit"])
    noiseFit        = polyFunc(imageBandwidth+freqOffset)
    
    image           = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(kSpace)))
    image           /= noiseFit[:,np.newaxis,np.newaxis]
    
    return np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(image)))

def b0Correction(kSpace, scanParams, dataFolder, b0Dict, returnKspace = True):
    #check to see if b0 map data exists in main folder
    try:
        '''store descriptor in parent folder to be accessed by other reconstruction methods'''
        dataPath            = Path(dataFolder)
        mainDir             = str(dataPath.parent)
        # b0Dict              = np.load(r"%s/b0Data.npy"%(mainDir), allow_pickle = True).item()
    except:
        print("Could not load B0 map data")
        if returnKspace:
            return kSpace
        else:
            return np.fft.fftshift(np.fft.fftn(np.fft.fftshift(kSpace)))
        
    #check to see if spherical harmonics already exists in main folder
    try:
        '''store descriptor in parent folder to be accessed by other reconstruction methods'''
        dataPath            = Path(dataFolder)
        mainDir             = str(dataPath.parent)
        shdDict              = np.load(r"%s/sphericalHarmonics.npy"%(mainDir), allow_pickle = True).item()
    except:
        #generatae spherical harmonics

        pass

    
    
    