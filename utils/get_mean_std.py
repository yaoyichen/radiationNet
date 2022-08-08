
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:54:07 2021

@author: yaoyichen
"""

import os
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from radiation_copy import get_filedetail_from_folder


def fun1():

    """
    获得不同变量的mean variance 
    """

    # nc_file = "rrtmg4nn.nc"
    # root_dir = "/Users/yaoyichen"
    # root_dir = "../../data/radiation/"


    # nc_file = "rrtmg4nn.nc"
    # root_dir = "/Users/yaoyichen/dataset/radiation/MPAS/data_1day"

    # nc_file = "rrtmg4nn_00000.000.nc"
    # root_dir = "/Users/yaoyichen/dataset/radiation/MPAS/data_1year"

    nc_file = "/home/eason.yyc/data/radiation/fullyear_data/fullyear_trainset_20_5_useful.nc"
    root_dir =  ""


    full_file_path = os.path.join(root_dir, nc_file)
    df = nc.Dataset(full_file_path)

    variable_list = ["aldif", "aldir", "asdif", "asdir", "cosz", "landfrac", "sicefrac", "snow", "solc", "tsfc", "emiss", "ccl4vmr", "cfc11vmr", "cfc12vmr", "cfc22vmr",
                     "ch4vmr", "cldfrac", "co2vmr", "n2ovmr", "o2vmr", "o3vmr", "play", "qc", "qg", "qi", "qr", "qs", "qv", "tlay", 'swuflx', 'swdflx', 'lwuflx', 'lwdflx']
    result = {}

    for variable_name in variable_list:

        data  = df.variables[variable_name][:]
        if variable_name in ["aldif", "aldir", "asdif", "asdir", "cosz", "solc", "swuflx", "swdflx","swhr"]:
            data[data < 0.0] = 0.0

        mean_value = ma.getdata(np.mean(data)).astype(float)
        scale_value = ma.getdata(np.std(data))
        stat = {"mean": float(mean_value), "scale": float(scale_value)}
        result[variable_name] = stat
    print(result)


def fun2():
    nc_file = "rrtmg4nn.nc"
    # root_dir = "/Users/yaoyichen"
    root_dir = "../../data/radiation/"

    full_file_path = os.path.join(root_dir, nc_file)
    df = nc.Dataset(full_file_path)

    temp = np.asarray(df.variables["ch4vmr"]).astype(float)
    plt.hist(temp.max(axis=0), bins=200)


def fun3():
    """
    获得文件夹下的所有文件
    """
    source_folder = "/home/eason.yyc/data/radiation/fullyear_data/"
    endswith_str = None
    startswith_str = "rrtmg"

    file_mapping_source = get_filedetail_from_folder(source_folder, endswith_str = endswith_str, startswith_str = startswith_str)

    for key,value in file_mapping_source.items():
        print(key, value)
    

def fun4():
    """
    获得文件夹下的所有文件
    """
    source_folder = "/home/eason.yyc/data/radiation/fullyear_data/"
    endswith_str = None
    startswith_str = "rrtmg"

    file_mapping_source = get_filedetail_from_folder(source_folder, endswith_str = endswith_str, startswith_str = startswith_str)


    variable_list = ["aldif", "aldir", "asdif", "asdir", "cosz", "landfrac", "sicefrac", "snow", "solc", "tsfc", "emiss", "ccl4vmr", "cfc11vmr", "cfc12vmr", "cfc22vmr",
                    "ch4vmr", "cldfrac", "co2vmr", "n2ovmr", "o2vmr", "o3vmr", "play", "qc", "qg", "qi", "qr", "qs", "qv", "tlay", 'swuflx', 'swdflx', 'lwuflx', 'lwdflx', 'swhr', 'lwhr']
    result = {}

            
    for key,value in file_mapping_source.items():
        
        df = nc.Dataset(key)

        for variable_name in variable_list:

            data  = df.variables[variable_name][:]
            if variable_name in ["aldif", "aldir", "asdif", "asdir", "cosz", "solc", "swuflx", "swdflx","swhr"]:
                data[data < 0.0] = 0.0

            mean_value = ma.getdata(np.mean(data)).astype(float)
            scale_value = ma.getdata(np.std(data))

            if(variable_name not in result):
                stat = {"mean": float(mean_value), "scale": float(scale_value), "count":1.0}
                result[variable_name] = stat

            else:
                result_mean = (result[variable_name]["mean"]* result[variable_name]["count"] + mean_value)/(result[variable_name]["count"] + 1.0) 

                """
                方差 + 均值绝对值之和
                """
                result_std =  (result[variable_name]["scale"]* result[variable_name]["count"] +     scale_value)/(result[variable_name]["count"] + 1.0) + np.abs(result[variable_name]["mean"] - mean_value)/5.0

                result_count = result[variable_name]["count"] + 1.0
                
                stat = {"mean": float(result_mean), "scale": float(result_std), "count":result_count}
                result[variable_name] = stat


    print(result)   



def fun_calculate_wrf_statistics():
    from make_wrfRRTMG_data import train_dict, WrfRRTMGDataset
    from config import norm_mapping_standard
    from torch.utils.data import Dataset, DataLoader
    import torch
    dateset =  WrfRRTMGDataset(vertical_layers = 57, type = "train", norm_mapping= norm_mapping_standard)
    dataLoader = DataLoader(dataset= dateset)
    counter = 0
    for batch_idx, (feature, targets, auxis) in enumerate(dataLoader): 
        if( batch_idx ==0):
            feature_mean = torch.mean(feature,dim = [0,1,3])
            targets_mean = torch.mean(targets,dim = [0,1,3])
            auxis_mean =   torch.mean(auxis,dim = [0,1,3])
        else:
            feature_mean += torch.mean(feature,dim = [0,1,3])
            targets_mean +=  torch.mean(targets,dim = [0,1,3])
            auxis_mean += torch.mean(auxis,dim = [0,1,3])
        counter += 1

    print(feature_mean/counter)
    print(targets_mean/counter)
    print(auxis_mean/counter)



    feature_mean_unsqueeze = (feature_mean/counter).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    targets_mean_unsqueeze = (targets_mean/counter).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    auxis_mean_unsqueeze = (auxis_mean/counter).unsqueeze(0).unsqueeze(0).unsqueeze(-1)

    counter = 0
    for batch_idx, (feature, targets, auxis) in enumerate(dataLoader): 
        if( batch_idx ==0):
            feature_mean = torch.mean(torch.square(feature - feature_mean_unsqueeze),dim = [0,1,3])
            targets_mean = torch.mean(torch.square(targets - targets_mean_unsqueeze),dim = [0,1,3])
            auxis_mean =   torch.mean(torch.square(auxis - auxis_mean_unsqueeze) ,dim = [0,1,3])
        else:
            feature_mean +=  torch.mean(torch.square(feature - feature_mean_unsqueeze),dim = [0,1,3])
            targets_mean +=  torch.mean(torch.square(targets - targets_mean_unsqueeze),dim = [0,1,3])
            auxis_mean +=  torch.mean(torch.square(auxis - auxis_mean_unsqueeze) ,dim = [0,1,3])
        counter += 1

    print(torch.sqrt(feature_mean/counter))
    print(torch.sqrt(targets_mean/counter))
    print(torch.sqrt(auxis_mean/counter))
    




        


if __name__ == "__main__":
    fun_calculate_wrf_statistics()
