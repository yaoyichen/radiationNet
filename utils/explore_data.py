import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import netCDF4 as nc
import os
import io
import numpy as np
import xarray as xr
import time


nc_file = "rrtmg4nn.nc"
root_dir = "/Users/yaoyichen"

full_file_path = os.path.join(root_dir, nc_file)

df = nc.Dataset(full_file_path)
print(df)

print(df.variables["lwdflx"])

test1 = df.variables["lwdflx"][0:10:2,0:10:2,0:10:2]

tt3 = xr.open_mfdataset(os.path.join(root_dir, nc_file), combine='by_coords')
print(type(tt2))

test2 = tt3.variables["lwdflx"][0:10:2,0:10:2,0:10:2]
print(test2)

tt4 = xr.open_dataset(os.path.join(root_dir, nc_file), engine = "netcdf4")

print(tt4["lwdflx"][0:10,0:10,:])
