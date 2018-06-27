from  greensconvolution.greensconvolution_calc import read_greensconvolution
from  greensconvolution.greensconvolution_fast import greensconvolution_crosscheck

greensconvolution_params=read_greensconvolution('greensconvolution.nc')

greensconvolution_crosscheck(greensconvolution_params,1e-3,1.4e-3,2,.138,1.57e3,730.0)
