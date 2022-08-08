#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 09:34:56 2021

@author: xiaohui
"""

import numpy as np

#gas volume mixing ratios defined in rrtmg_sw
ch4 = 1774*10**(-9)
n2o = 319*10**(-9)        
#Annual function for co2 after v4.2 in WRF
#    co2 = (280 + 90*np.exp(0.02*(yr-2000)))*10**(-6)
co2 = 0.000379
n2o = 319*10**(-9)
o2 = 0.209488    

#gas volume mixing ratios defined in rrtmg_lw
ccl4 = 0.093*10**(-9)
cfc11 = 0.251*10**(-9)
cfc12 = 0.538*10**(-9)
cfc22 = 0.169*10**(-9)   

#YYYYMMDD_HHMMSS.npz
fileName = '20210905_054000.npz'

ListofVar = ['emiss', 'solc', 'albedo', 'landfrac', 'sicefrac', 'snow', 
	         'cosz', 'tsfc', 'tlay', 'tlev', 'play', 'plev', 'qv', 'qc', 'qr',
	         'qi', 'qs', 'qg', 'o3vmr', 'cldfrac', 'swuflx', 'swdflx', 'lwuflx', \
             'lwdflx', 'lw_hr', 'sw_hr']
             
var = np.load(fileName)

aldif = var['albedo']
aldir = var['albedo']
asdif = var['albedo']
asdir = var['albedo']

 

