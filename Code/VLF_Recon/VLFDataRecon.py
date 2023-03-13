import numpy as np
import matplotlib.pyplot as plt
from nibabel.viewers import OrthoSlicer3D

import cProcessPipeline as cPP

# Read folder
dataFolder = r'../Data/S1/3DTSE'
raw_folder = '1'
noiseScan = 0
freqDrift = -60  # Frequency drift

'''Read k-space and reconstruct image'''

reconImage_raw, kSpace_raw, scanParams_raw, trajectory_raw = cPP.prepData(dataFolder, raw_folder, noiseScan, freqDrift)

# Display
OrthoSlicer3D(np.abs(reconImage_raw)).show()
