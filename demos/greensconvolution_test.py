import numpy as np

from  greensconvolution.greensconvolution_calc import read_greensconvolution
from  greensconvolution.greensconvolution_fast import greensconvolution_crosscheck

greensconvolution_params=read_greensconvolution('greensconvolution.nc')

greensconvolution_crosscheck(greensconvolution_params,np.float32(1e-3),np.float32(1.4e-3),np.float32(2),np.float32(0.0),np.float32(.138),np.float32(1.57e3),np.float32(730.0))
