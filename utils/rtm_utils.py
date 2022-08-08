#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 13:57:25 2021

@author: xiaohui

This function is used to calculate heating rate given the flux

"""

# def calculate_hr():
#     """
#     uflx, dflux, plevel: B*H,

#     """
#     pass


def calculate_hr(uflx, dflx, plev, vdim=1, unit_pressure='hPa'):
    """
    source dir:   /home/admin/Code/Python/RTM/rrtmg_MPAS/calculate_hr.py
    """
    g = 9.8066  # m s^-2

    # reference to WRF/share/module_model_constants.F gas constant of dry air
    rgas = 287.0
    cp = 7.*rgas/2.

    if unit_pressure in {'hPa', 'mb'}:
        heatfac = g*8.64*10**4/(cp*100)

    elif unit_pressure == 'Pa':
        heatfac = g*8.64*10**4/cp

    fnet = uflx - dflx

    if vdim == 0:
        delta_fnet = fnet[0:-1, :] - fnet[1:, :]

        delta_p = plev[0:-1, :] - plev[1:, :]

    elif vdim == 1:
        delta_fnet = fnet[:, 0:-1] - fnet[:, 1:]

        delta_p = plev[:, 0:-1] - plev[:, 1:]

    elif vdim == 2:
        delta_fnet = fnet[:, :, 0:-1] - fnet[:, :, 1:]

        delta_p = plev[:, :, 0:-1] - plev[:, :, 1:]

    hr = delta_fnet/delta_p * heatfac

    return hr


'''
import netCDF4 as nc
from RTM.rrtmg_MPAS.calculate_hr import calculate_hr
import numpy as np
p = 'rrtmg4nn_orig.nc'
f=nc.Dataset(p)

lwuflx = np.array(f['lwuflx'][:])
lwdflx = np.array(f['lwdflx'][:])
lwhr = np.array(f['lwhr'][:])
plev = np.array(f['plev'][:])
hr = calculate_hr(lwuflx, lwdflx, plev, 2)
d = hr - lwhr
print(d)
'''
