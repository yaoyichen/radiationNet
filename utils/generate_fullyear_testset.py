#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:44:23 2022

@author: yaoyichen
"""

from sqlalchemy import false
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from radiation_copy import get_filedetail_from_folder
from netCDF4 import Dataset,num2date,date2num
import time


# nc_file = "rrtmg4nn_00000.000.nc"
# root_dir = "/Users/yaoyichen/dataset/radiation/MPAS/data_1year/"
# source_folder = "/mnt_det/eason.yyc/radiation/fullyear_data/20200719-20200722/"


"""
用于构建一份test数据集
"""

# file_list = [os.path.join(root_dir,nc_file), os.path.join(root_dir,nc_file)]
def get_filelist_method1():
    source_folder_list = ["/mnt_det/eason.yyc/radiation/fullyear_data/20200108-20200111/",
                          "/mnt_det/eason.yyc/radiation/fullyear_data/20200213-20200216/",
                          "/mnt_det/eason.yyc/radiation/fullyear_data/20200302-20200305/",
                          "/mnt_det/eason.yyc/radiation/fullyear_data/20200420-20200423/",
                          "/mnt_det/eason.yyc/radiation/fullyear_data/20200528-20200531/",
                          "/mnt_det/eason.yyc/radiation/fullyear_data/20200615-20200618/",
                          "/mnt_det/eason.yyc/radiation/fullyear_data/20200719-20200722/",
                          "/mnt_det/eason.yyc/radiation/fullyear_data/20200811-20200814/",
                          "/mnt_det/eason.yyc/radiation/fullyear_data/20200927-20200930/",
                          "/mnt_det/eason.yyc/radiation/fullyear_data/20201012-20201015/",
                          "/mnt_det/eason.yyc/radiation/fullyear_data/20201124-20201127/",
                          "/mnt_det/eason.yyc/radiation/fullyear_data/20201204-20201207/" ]
    
    file_list1 = []
    file_list2 = []

    for source_folder in source_folder_list:
        file_mapping_source = get_filedetail_from_folder(source_folder, endswith_str = None, startswith_str = "rrtmg4nn_000")
        file_list1 = file_list1 + list(file_mapping_source.keys())
    
    for source_folder in source_folder_list:
        file_mapping_source = get_filedetail_from_folder(source_folder, endswith_str = None, startswith_str = "rrtmg4nn_002")
        file_list2 = file_list2 + list(file_mapping_source.keys())

    # for source_folder in source_folder_list:
    #     file_mapping_source = get_filedetail_from_folder(source_folder, endswith_str = None, startswith_str = "rrtmg4nn_004")
    #     file_list2 = file_list2 + list(file_mapping_source.keys())

    return file_list1 + file_list2


def get_filelist_method2():
    source_folder_list = ["/mnt_det/eason.yyc/radiation/fullyear_data/20200108-20200111/"]
    
    file_list1 = []

    for source_folder in source_folder_list:
        file_mapping_source = get_filedetail_from_folder(source_folder, endswith_str = None, startswith_str = "rrtmg4nn_004")
        file_list1 = file_list1 + list(file_mapping_source.keys())
    # for source_folder in source_folder_list:
    #     file_mapping_source = get_filedetail_from_folder(source_folder, endswith_str = None, startswith_str = "rrtmg4nn_004")
    #     file_list2 = file_list2 + list(file_mapping_source.keys())

    return file_list1


def get_filelist_method3():
    """
    从snapshot中获取数据 
    """
    source_folder_list = ["/mnt_det/eason.yyc/radiation/fullyear_data/snapshot/"]
    file_list1 = []

    for source_folder in source_folder_list:
        file_mapping_source = get_filedetail_from_folder(source_folder,
        endswith_str = ".nc")
        file_list1 = file_list1 + list(file_mapping_source.keys())
    # for source_folder in source_folder_list:
    #     file_mapping_source = get_filedetail_from_folder(source_folder, endswith_str = None, startswith_str = "rrtmg4nn_004")
    #     file_list2 = file_list2 + list(file_mapping_source.keys())

    return file_list1
    # return file_list1 


# def get_filelist_method2():
#     file_list = []


def generate_fullyear_testset(out_filename, time_divide,spatial_divide ,if_mask = True):

    
    # time_divide= 20
    # spatial_divide = 20
    file_list = get_filelist_method3()
    print(file_list)
    # print(file_list)
    
    
    single_height_variable = ["aldif","aldir", "asdif", "asdir", "cosz", "landfrac", "sicefrac", "snow", "solc", "tsfc", "emiss"]
    
    
    multi_height_variable = ["ccl4vmr", "cfc11vmr", "cfc12vmr", "cfc22vmr", "ch4vmr", "cldfrac",   "co2vmr", "n2ovmr", "o2vmr", "o3vmr", "play", "qc", "qg", "qi", "qr", "qs", "qv", "tlay"]
    
    
    label_variable = ['swuflx', 'swdflx', 'lwuflx', 'lwdflx']
    auxiliary_variable = ["plev"]
    
    pad_zero_list = ["aldif", "aldir", "asdif", "asdir", "cosz", "solc", "swuflx", "swdflx"]
    
    
    ncout = Dataset(out_filename, 'w', 'NETCDF4')
    ncout.createDimension('nz1',57)
    ncout.createDimension('nz',56)
    ncout.createDimension('nt',None)
    ncout.createDimension('np',None)
    ncout.createDimension('nbndlw',2)
    
    for variable in single_height_variable:
        print(variable)

        start_time = time.time()
        if(variable == "emiss"):
            f = ncout.createVariable(variable, 'float32',("nt","np","nbndlw"))
        else:
            f = ncout.createVariable(variable, 'float32',("nt","np"))
                
        
        for index,file_name in enumerate(file_list):

            df = nc.Dataset(file_name)
            data = df["swuflx"][0,:][:,0]

            if(if_mask == True):
                data_len = len(np.where(~data.mask)[0])
            else:
                data_len = len(data)
            
            print(f"data_len:{data_len }")

            if(variable == "emiss"):
                if(if_mask == True):
                    temp = df[variable][0:240:time_divide,np.where(~data.mask)[0][0:data_len:spatial_divide],0:2]
                else:
                    temp = df[variable][0:240:time_divide,0:data_len:spatial_divide,0:2]

                if(variable in pad_zero_list):
                    temp[temp < 0.0] = 0.0


            else:
                if(if_mask == True):
                    temp = df[variable][0:240:time_divide,np.where(~data.mask)[0][0:data_len:spatial_divide]]
                else:
                    temp = df[variable][0:240:time_divide,0:data_len:spatial_divide]
                    print(temp.shape)


                if(variable in pad_zero_list):
                    temp[temp < 0.0] = 0.0
            df.close()
            if(index ==0):
                data_concat = temp
            else:
                data_concat = np.concatenate([data_concat, temp], axis = 1)
            
           
        f[:] = data_concat
        print(f"elapse time = {time.time() - start_time}")
            
                
    
    for variable in multi_height_variable:
        print(variable)
        start_time = time.time()
        f = ncout.createVariable(variable, 'float32',("nt","np","nz"))
        
        
        for index,file_name in enumerate(file_list):
            df = nc.Dataset(file_name)
            data = df["swuflx"][0,:][:,0]

            if(if_mask == True):
                data_len = len(np.where(~data.mask)[0])
            else:
                data_len = len(data)
            
            print(f"data_len:{data_len }")
            
            
            if(if_mask == True):
                temp = df[variable][0:240:time_divide,np.where(~data.mask)[0][0:data_len:spatial_divide],:]
            else:
                temp = df[variable][0:240:time_divide,0:data_len:spatial_divide,:]

            df.close()
            if(variable in pad_zero_list):
                temp[temp < 0.0] = 0.0


            if(index ==0):
                data_concat = temp
            else:
                data_concat = np.concatenate([data_concat, temp], axis = 1)
                
        f[:] = data_concat
        print(f"elapse time = {time.time() - start_time}")
    
    
    
    for variable in label_variable:
        print(variable)
        start_time = time.time()
        f = ncout.createVariable(variable, 'float32',("nt","np","nz1"))
        
        for index,file_name in enumerate(file_list):
            
            df = nc.Dataset(file_name)
            data = df["swuflx"][0,:][:,0]

            if(if_mask == True):
                data_len = len(np.where(~data.mask)[0])
            else:
                data_len = len(data)

            
            if(if_mask == True):
                temp = df[variable][0:240:time_divide,np.where(~data.mask)[0]
                [0:data_len:spatial_divide],:]
            else:
                temp = df[variable][0:240:time_divide,0:data_len:spatial_divide,:]

            df.close()
            if(variable in pad_zero_list):
                temp[temp < 0.0] = 0.0

            
            if(index ==0):
                data_concat = temp
            else:
                data_concat = np.concatenate([data_concat, temp], axis = 1)
                    
        f[:] = data_concat 
        print(f"elapse time = {time.time() - start_time}")   
        
    
    
    for variable in auxiliary_variable:
        print(variable)
        start_time = time.time()
        f = ncout.createVariable(variable, 'float32',("nt","np","nz1"))
        
        for index,file_name in enumerate(file_list):
            
            df = nc.Dataset(file_name)
            data = df["swuflx"][0,:][:,0]

            if(if_mask == True):
                data_len = len(np.where(~data.mask)[0])
            else:
                data_len = len(data)
            
            if(if_mask == True):
                temp = df[variable][0:240:time_divide,np.where(~data.mask)[0][0:data_len:spatial_divide],:]
            else:
                temp = df[variable][0:240:time_divide,0:data_len:spatial_divide,:]


            df.close()
            if(variable in pad_zero_list):
                temp[temp < 0.0] = 0.0

            if(index ==0):
                data_concat = temp
            else:
                data_concat = np.concatenate([data_concat, temp], axis = 1)
        f[:] = data_concat    
        print(f"elapse time = {time.time() - start_time}")
    ncout.close()


def generate_1timeslice():
    out_filename = "/home/eason.yyc/data/radiation/fullyear_data_test/snapshot_1.nc"
    time_divide = 1
    spatial_divide = 1
    generate_fullyear_testset(out_filename,time_divide, spatial_divide,if_mask= False)
    print("finish generate")

def generate():
    out_filename = "/home/eason.yyc/data/radiation/fullyear_data_test/fullyear_trainset_20_5_useful.nc"
    time_divide = 20
    spatial_divide = 5
    generate_fullyear_testset(out_filename,time_divide, spatial_divide)
    print("finish generate")


def calculate_shape():
    # file_name = "/Users/yaoyichen/dataset/radiation/MPAS/data_1year/fullyear_testset_60_60_1237.nc"
    file_name = "/home/eason.yyc/data/radiation/fullyear_data/fullyear_trainset_20_5_useful.nc"
    # file_name = "/home/eason.yyc/data/radiation/fullyear_data_test/fullyear_testset_5_5_1237_useful.nc"
    df = nc.Dataset(file_name)
    data = df["swuflx"]
    print(data.shape)


if __name__ == "__main__":
    generate_1timeslice()
    