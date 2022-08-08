import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import netCDF4 as nc
import os
import io
import numpy as np
import time
import xarray as xr
# from .config import norm_mapping


# class RtmMpasDataset(Dataset):
#     def __init__(self, nc_file, root_dir, from_time=0, end_time=36,
#                  from_point=0, end_point=36, norm_mapping={}):
#         self.nc_file = nc_file
#         self.root_dir = root_dir
#         self.from_time = from_time
#         self.end_time = end_time
#         self.from_point = from_point
#         self.end_point = end_point
#         self.total_point = 10242
#         self.single_height_variable = ["aldif",
#                                        "aldir", "asdif", "asdir", "cosz", "landfrac", "sicefrac", "snow", "solc", "tsfc", "emiss"]
#         self.single_feature = np.zeros(
#             [len(self.single_height_variable), 1])

#         self.multi_height_variable = ["ccl4vmr", "cfc11vmr", "cfc12vmr", "cfc22vmr", "ch4vmr", "cldfrac",
#                                       "co2vmr", "n2ovmr", "o2vmr", "o3vmr", "play", "qc", "qg", "qi", "qr", "qs", "qv", "tlay"]

#         self.multi_feature = np.zeros([len(self.multi_height_variable), 56])

#         self.label_variable = ['swuflx', 'swdflx', 'lwuflx', 'lwdflx']

#         self.label_feature = np.zeros(
#             [len(self.label_variable), 57])
#         self.norm_mapping = norm_mapping

#     def __len__(self):
#         return (self.end_time - self.from_time)*(self.end_point - self.from_point)

#     def __getitem__(self, index):
#         full_file_path = os.path.join(self.root_dir, self.nc_file)
#         df = nc.Dataset(full_file_path)
#         time_index = index // (self.end_point -
#                                self.from_point) + self.from_time
#         point_index = index - time_index * \
#             (self.end_point - self.from_point) + self.from_point
#         for variable_index, variable_name in enumerate(self.single_height_variable):
#             if(variable_name == "emiss"):
#                 self.single_feature[:, variable_index, 0] = \
#                     (df.variables[variable_name][:,
#                                                  point_index, 0] - self.norm_mapping[variable_name]["mean"]) / self.norm_mapping[variable_name]["scale"]
#             else:
#                 self.single_feature[:, variable_index, 0] = \
#                     (df.variables[variable_name][:,
#                                                  point_index] - self.norm_mapping[variable_name]["mean"]) / self.norm_mapping[variable_name]["scale"]

#         single_feature_tt = torch.tensor(
#             self.single_feature, dtype=torch.float32)

#         for variable_index, variable_name in enumerate(self.multi_height_variable):
#             self.multi_feature[variable_index,
#                                :] = \
#                 (df.variables[variable_name][time_index, point_index, :] -
#                  self.norm_mapping[variable_name]["mean"])/self.norm_mapping[variable_name]["scale"]

#         multi_feature_tt = torch.tensor(self.multi_feature, dtype=torch.float32)

#         for variable_index, variable_name in enumerate(self.label_variable):
#             self.label_feature[variable_index,
#                                :] = \
#                 (df.variables[variable_name][time_index, point_index, :] -
#                  self.norm_mapping[variable_name]["mean"])/self.norm_mapping[variable_name]["scale"]

#         feature_tf = torch.cat(
#             [torch.tile(single_feature_tt, [56]), multi_feature_tt], dim=0)

#         label_feature_tt = torch.tensor(self.label_feature, dtype=torch.float32)

#         return (feature_tf, label_feature_tt)


class RtmMpasDatasetWholeTime(Dataset):
    def __init__(self, nc_file, root_dir, time_length=72, type="train", norm_mapping={}):
        self.nc_file = nc_file
        self.root_dir = root_dir
        # self.from_point = from_point
        # self.end_point = end_point
        self.total_point = 10242
        self.folds = 10
        self.type = type
        self.single_height_variable = ["aldif",
                                       "aldir", "asdif", "asdir", "cosz", "landfrac", "sicefrac", "snow", "solc", "tsfc", "emiss"]

        self.single_feature = np.zeros(
            [time_length, len(self.single_height_variable), 1])
        # self.single_height_variable = ["aldif",
        #                                "aldir", "asdif", "asdir", "cosz", "landfrac", "sicefrac", "snow", "solc", "tsfc"]

        self.multi_height_variable = ["ccl4vmr", "cfc11vmr", "cfc12vmr", "cfc22vmr", "ch4vmr", "cldfrac",
                                      "co2vmr", "n2ovmr", "o2vmr", "o3vmr", "play", "qc", "qg", "qi", "qr", "qs", "qv", "tlay"]

        self.multi_feature = np.zeros(
            [time_length, len(self.multi_height_variable), 56])

        self.label_variable = ['swuflx', 'swdflx', 'lwuflx', 'lwdflx']

        self.label_feature = np.zeros(
            [time_length, len(self.label_variable), 57])
        self.norm_mapping = norm_mapping

        self.auxiliary_variable = ["plev"]
        self.auxiliary_feature = np.zeros(
            [time_length, len(self.auxiliary_variable), 57])

    def __len__(self):
        if(self.type == "train"):
            return self.total_point - self.total_point // self.folds
        elif (self.type == "test"):
            return self.total_point // self.folds


    def __getitem__(self, index):
        full_file_path = os.path.join(self.root_dir, self.nc_file)

        df = xr.open_dataset(full_file_path, engine="netcdf4")

        if (self.type == "train"):
            point_index = index//(self.folds - 1) * \
                self.folds + (index % (self.folds - 1))
        elif (self.type == "test"):
            point_index = (index + 1) * self.folds - 1

        for variable_index, variable_name in enumerate(self.single_height_variable):
            if(variable_name == "emiss"):
                self.single_feature[:, variable_index, 0] = \
                    (df.variables[variable_name][:,
                                                 point_index, 0] - self.norm_mapping[variable_name]["mean"]) / self.norm_mapping[variable_name]["scale"]
            else:
                self.single_feature[:, variable_index, 0] = \
                    (df.variables[variable_name][:,
                                                 point_index] - self.norm_mapping[variable_name]["mean"]) / self.norm_mapping[variable_name]["scale"]

        single_feature_tt = torch.tensor(
            self.single_feature, dtype=torch.float32)

        for variable_index, variable_name in enumerate(self.multi_height_variable):
            self.multi_feature[:, variable_index,
                               :] = (df.variables[variable_name][:, point_index, :] -
                                     self.norm_mapping[variable_name]["mean"])/self.norm_mapping[variable_name]["scale"]

        multi_feature_tt = torch.tensor(
            self.multi_feature, dtype=torch.float32)

        for variable_index, variable_name in enumerate(self.label_variable):
            self.label_feature[:, variable_index,
                               :] = (df.variables[variable_name][:, point_index, :] -
                                     self.norm_mapping[variable_name]["mean"])/self.norm_mapping[variable_name]["scale"]

        for variable_index, variable_name in enumerate(self.auxiliary_variable):
            self.auxiliary_feature[:, variable_index,
                                   :] = df.variables[variable_name][:, point_index, :]

        feature_tf = torch.cat(
            [torch.tile(single_feature_tt, [1, 56]), multi_feature_tt], dim=(1))

        label_feature_tt = torch.tensor(
            self.label_feature, dtype=torch.float32)

        auxiliary_result_tf = torch.tensor(
            self.auxiliary_feature, dtype=torch.float32)

        return (feature_tf, label_feature_tt, auxiliary_result_tf)


class RtmMpasDatasetWholeTimeLarge(Dataset):
    """
    from_time: 起始的 时间 index 
    end_time:  结束的 时间 index 
    batch_divid_number:  空间划分为 batch_divid_number
    point_folds: point_folds是一个空间块的大小
    time_folds: time_fold 是一个时间块的大小
    """
    def __init__(self, nc_file, root_dir, from_time, end_time,
                 batch_divid_number, point_folds, time_folds, norm_mapping={},
                 vertical_layers=57, point_number = 10242,random_throw = False,
                 only_layer = False):
        """
        batch_divid_number
        point_folds
        """
        self.nc_file = nc_file
        self.root_dir = root_dir

        self.from_time = from_time
        self.end_time = end_time

        self.batch_divid_number = batch_divid_number
        self.point_folds = point_folds
        self.time_folds = time_folds
        self.point_number = point_number
        self.random_throw = random_throw
        self.only_layer = only_layer

        self.norm_mapping = norm_mapping
        self.vertical_layers = vertical_layers
        print( f"{self.vertical_layers}"  )

        self.block_size = self.point_number//self.batch_divid_number

        ### 单层的feature ###
        self.single_height_variable = ["aldif",
                                       "aldir", "asdif", "asdir", "cosz", "landfrac", "sicefrac", "snow", "solc", "tsfc", "emiss"]
        
        """
        为了增加随机数，总长度 -1 
        """
        self.points_in_batch = self.block_size//self.point_folds -1

        self.single_feature = np.zeros(
            [self.points_in_batch, len(self.single_height_variable), 1])

        ### 多层的feature ###
        self.multi_height_variable = ["ccl4vmr", "cfc11vmr", "cfc12vmr", "cfc22vmr", "ch4vmr", "cldfrac",
                                      "co2vmr", "n2ovmr", "o2vmr", "o3vmr", "play", "qc", "qg", "qi", "qr", "qs", "qv", "tlay"]
        # self.multi_feature = np.zeros(
        #     [self.points_in_batch, len(self.multi_height_variable), self.vertical_layers])

        ### 多层需要积分的变量 ###
        self.multi_height_cumsum_variable = {"cldfrac": 0, "qc": 1}
        # self.multi_cumsum_feature = np.zeros(
        #     [self.points_in_batch, 2*len(self.multi_height_cumsum_variable), self.vertical_layers])

        ### 输出变量 ###
        self.label_variable = ['swuflx', 'swdflx', 'lwuflx', 'lwdflx']
        # self.label_feature = np.zeros(
        #     [self.points_in_batch, len(self.label_variable), self.vertical_layers])

        ### 气压层 ###
        self.auxiliary_variable = ["plev"]
        # self.auxiliary_feature = np.zeros(
        #     [self.points_in_batch, len(self.auxiliary_variable), self.vertical_layers])

    def __len__(self):
        return (self.end_time - self.from_time)//self.time_folds * self.batch_divid_number

    def __getitem__(self, index):
        full_file_path = os.path.join(self.root_dir, self.nc_file)
        # df = nc.Dataset(full_file_path)
        df = xr.open_dataset(full_file_path, engine="netcdf4")

        """
        增加了随机性 self.random_throw 的时候生效, 否则就沿用以前的特征生成方式。
        """
        if(self.random_throw == True):
            keep_size  = 55 - np.random.choice(25)
            keep_index = np.concatenate([np.asarray([0]), np.random.choice(np.arange(1,56), size  = keep_size, replace = False), np.asarray([56]) ])
            keep_index.sort()
            keep_index_level = keep_index
            keep_index_layer = keep_index_level[0:-1]
            
        else:
            keep_index_level = np.arange(0,57)
            keep_index_layer = np.arange(0,56)
        """
        """
        self.multi_feature = np.zeros(
            [self.points_in_batch, len(self.multi_height_variable), len(keep_index_level)])

        self.multi_cumsum_feature = np.zeros(
            [self.points_in_batch, 2*len(self.multi_height_cumsum_variable), len(keep_index_level)])

        self.label_feature = np.zeros(
            [self.points_in_batch, len(self.label_variable), len(keep_index_level)])

        self.auxiliary_feature = np.zeros(
            [self.points_in_batch, len(self.auxiliary_variable), len(keep_index_level)])
        

        time_index = (index // self.batch_divid_number) * \
            self.time_folds + self.from_time + \
            np.random.randint(self.time_folds)


        remain_index = index % self.batch_divid_number

        total_folds = self.block_size // self.point_folds

        # global_index_list = np.arange((remain_index * self.block_size),
        #                               (remain_index * self.block_size +
        #                                total_folds * self.point_folds))

        global_index_list = np.arange((remain_index * self.block_size),
                              (remain_index * self.block_size +
                               self.block_size))

        """
        为了增加随机数，总长度 -1 
        """
        inside_start_shift = np.random.randint(self.point_folds)
        # inside_index_list = np.arange(inside_start_shift, inside_start_shift+ (total_folds-1) * self.point_folds -1, self.point_folds)
        inside_index_list = np.arange(inside_start_shift, inside_start_shift + self.points_in_batch*self.point_folds , self.point_folds)

        index_list = np.arange((remain_index * self.block_size),
                               (remain_index * self.block_size +
                                total_folds*self.point_folds), self.point_folds)

        for variable_index, variable_name in enumerate(self.single_height_variable):
            if(variable_name == "emiss"):
                temp = (df.variables[variable_name][time_index, global_index_list, 0] -
                        self.norm_mapping[variable_name]["mean"]) / self.norm_mapping[variable_name]["scale"]
                # print("### 2 ###")
                # print(temp.shape)
                self.single_feature[:, variable_index,
                                    0] = temp[inside_index_list]

            else:
                temp = (df.variables[variable_name][time_index, global_index_list] -
                        self.norm_mapping[variable_name]["mean"]) / self.norm_mapping[variable_name]["scale"]
                self.single_feature[:, variable_index,
                                    0] = temp[inside_index_list]

        single_feature_tt = torch.tensor(
            self.single_feature, dtype=torch.float32)

        for variable_index, variable_name in enumerate(self.multi_height_variable):
            temp = (df.variables[variable_name][time_index, global_index_list, :] -
                    self.norm_mapping[variable_name]["mean"]) / self.norm_mapping[variable_name]["scale"]

            # print("### 3 ###")
            # print(temp.shape)

            temp_value = np.array(temp[inside_index_list, ::]).take(keep_index_layer, axis = 1)

            self.multi_feature[:, variable_index,
                               1:len(keep_index_layer) + 1 ] = temp_value
            self.multi_feature[:, variable_index,
                               0] = self.multi_feature[:, variable_index, 1]

            # print("success")
            if(variable_name in self.multi_height_cumsum_variable):
                # 获取index
                variable_index = self.multi_height_cumsum_variable[variable_name]
                temp_value_cumsum_forward = np.cumsum(
                    temp_value, axis=1)/20.0
                temp_value_cumsum_backward = np.cumsum(
                    temp_value[:, ::-1], axis=1)/20.0

                self.multi_cumsum_feature[:, variable_index,
                                          1:len(keep_index_level)] = temp_value_cumsum_forward
                self.multi_cumsum_feature[:, variable_index,
                                          0] = self.multi_cumsum_feature[:, variable_index, 1]

                self.multi_cumsum_feature[:, len(self.multi_height_cumsum_variable) + variable_index,
                                          1:len(keep_index_level)] = temp_value_cumsum_backward
                self.multi_cumsum_feature[:, len(self.multi_height_cumsum_variable) + variable_index,
                                          0] = self.multi_cumsum_feature[:, variable_index, 1]


        multi_feature_tt = torch.tensor(
            self.multi_feature, dtype=torch.float32)
        multi_cumsum_feature_tt = torch.tensor(
            self.multi_cumsum_feature, dtype=torch.float32)

        for variable_index, variable_name in enumerate(self.label_variable):
            temp = (df.variables[variable_name][time_index, global_index_list, :] -
                    self.norm_mapping[variable_name]["mean"]) / self.norm_mapping[variable_name]["scale"]

            self.label_feature[:, variable_index,
                               :] = np.array(temp[inside_index_list, ::]).take(keep_index_level,  axis = 1)

        for variable_index, variable_name in enumerate(self.auxiliary_variable):
            temp = df.variables[variable_name][time_index,
                                               global_index_list, :]

            self.auxiliary_feature[:, variable_index,
                                   :] = np.array(temp[inside_index_list, ::]).take(keep_index_level,  axis = 1)

        label_feature_tt = torch.tensor(
            self.label_feature, dtype=torch.float32)

        auxiliary_result_tf = torch.tensor(
            self.auxiliary_feature, dtype=torch.float32)

        p_diff = auxiliary_result_tf - torch.roll(auxiliary_result_tf, -1, 2)
        p_diff = torch.cat([p_diff[:, :, 0:1], p_diff[:, :, 0:-1]], dim=2)

        if(self.only_layer):
            # 29个特征
            feature_tf = torch.cat(
                [torch.tile(single_feature_tt, [1, len(keep_index_level)]),
                 multi_feature_tt] , 
                dim=(1))
        else:
            # 34个特征
            feature_tf = torch.cat(
                [torch.tile(single_feature_tt, [1, len(keep_index_level)]),
                 multi_feature_tt,
                  (p_diff - 17.2)/9.8,
                  multi_cumsum_feature_tt
                 ], dim=(1))

        return (feature_tf, label_feature_tt, auxiliary_result_tf)




class RtmMpasDatasetWholeTimeFullyear(Dataset):
    """
    from_time: 起始的 时间 index 
    end_time:  结束的 时间 index 
    batch_divid_number:  空间划分为 batch_divid_number
    point_folds: point_folds是一个空间块的大小
    time_folds: time_fold 是一个时间块的大小
    """
    def __init__(self, nc_file, root_dir, from_time, end_time,
                 batch_divid_number, point_folds, time_folds, norm_mapping={},
                 vertical_layers=57,point_number = 5000, remove_mask = True):
        """
        batch_divid_number
        point_folds
        """
        self.nc_file = nc_file
        self.root_dir = root_dir

        self.from_time = from_time
        self.end_time = end_time

        self.batch_divid_number = batch_divid_number
        self.point_folds = point_folds
        self.time_folds = time_folds
        self.point_number = point_number
        self.norm_mapping = norm_mapping
        self.vertical_layers = vertical_layers

        self.block_size = self.point_number//self.batch_divid_number

        ### 单层的feature ###
        self.single_height_variable = ["aldif",
                                       "aldir", "asdif", "asdir", "cosz", "landfrac", "sicefrac", "snow", "solc", "tsfc", "emiss"]
                                
        
        """
        为了增加随机数，总长度 -1 
        """
        self.points_in_batch = self.block_size//self.point_folds -1

        self.remove_mask = remove_mask

        print(f"point number:{self.point_number},\
                batch_divid_number:{self.batch_divid_number},\
                block_size:{self.block_size},\
                points_in_batch:{self.points_in_batch}")

        self.single_feature = np.zeros(
            [self.points_in_batch, len(self.single_height_variable), 1])

        ### 多层的feature ###
        self.multi_height_variable = ["ccl4vmr", "cfc11vmr", "cfc12vmr", "cfc22vmr", "ch4vmr", "cldfrac",
                                      "co2vmr", "n2ovmr", "o2vmr", "o3vmr", "play", "qc", "qg", "qi", "qr", "qs", "qv", "tlay"]
        self.multi_feature = np.zeros(
            [self.points_in_batch, len(self.multi_height_variable), self.vertical_layers])

        ### 多层需要积分的变量 ###
        self.multi_height_cumsum_variable = {"cldfrac": 0, "qc": 1}
        self.multi_cumsum_feature = np.zeros(
            [self.points_in_batch, 2*len(self.multi_height_cumsum_variable), self.vertical_layers])

        ### 输出变量 ###
        self.label_variable = ['swuflx', 'swdflx', 'lwuflx', 'lwdflx']
        self.label_feature = np.zeros(
            [self.points_in_batch, len(self.label_variable), self.vertical_layers])

        ### 气压层 ###
        self.auxiliary_variable = ["plev"]
        self.auxiliary_feature = np.zeros(
            [self.points_in_batch, len(self.auxiliary_variable), self.vertical_layers])


        self.pad_zero_list = ["aldif", "aldir", "asdif", "asdir", "cosz", "solc", "swuflx", "swdflx"]

    def set_root_foler_name(self,root_dir,nc_file):
        self.root_dir = root_dir
        self.nc_file = nc_file


    def __len__(self):
        return (self.end_time - self.from_time)//self.time_folds * self.batch_divid_number

    def __getitem__(self, index):
        full_file_path = os.path.join(self.root_dir, self.nc_file)
        df = nc.Dataset(full_file_path)

        time_index = (index // self.batch_divid_number) * \
            self.time_folds + self.from_time + \
            np.random.randint(self.time_folds)

        # print(f"time_index:{time_index}")

        remain_index = index % self.batch_divid_number

        total_folds = self.block_size // self.point_folds

        global_index_list = np.arange((remain_index * self.block_size),
                                      (remain_index * self.block_size +
                                       self.block_size))
        # print("global list")
        # print(global_index_list)

        """
        为了增加随机数，总长度 -1 
        """
        inside_start_shift = np.random.randint(self.point_folds)
        # inside_index_list = np.arange(inside_start_shift, inside_start_shift+ (total_folds-1) * self.point_folds -1, self.point_folds)

        inside_index_list = np.arange(inside_start_shift, inside_start_shift + self.points_in_batch*self.point_folds , self.point_folds)
        # print("inside list")
        # print(inside_index_list)

        index_list = np.arange((remain_index * self.block_size),
                               (remain_index * self.block_size +
                                total_folds*self.point_folds), self.point_folds)

        for variable_index, variable_name in enumerate(self.single_height_variable):

            if(variable_name == "emiss"):
                data = df.variables[variable_name][time_index,:,0]

                if(self.remove_mask):
                    data = data[~data.mask]
                    if(data.shape[0] == 1):
                        data = np.squeeze(data,axis = 0)

                data = data[global_index_list]


                temp = (data -
                        self.norm_mapping[variable_name]["mean"]) / self.norm_mapping[variable_name]["scale"]
                self.single_feature[:, variable_index,
                                    0] = temp[inside_index_list]

            else:
                data = df.variables[variable_name][time_index,:]

                if(self.remove_mask):
                    data = data[~data.mask]
                    if(data.shape[0] == 1):
                        data = np.squeeze(data,axis = 0)

                data = data[global_index_list]
                if(variable_name in self.pad_zero_list):
                    data[data < 0.0] = 0.0
                    
                temp = (data - self.norm_mapping[variable_name]["mean"]) / self.norm_mapping[variable_name]["scale"]
                self.single_feature[:, variable_index,
                                    0] = temp[inside_index_list]


        single_feature_tt = torch.tensor(
            self.single_feature, dtype=torch.float32)


        for variable_index, variable_name in enumerate(self.multi_height_variable):
            data = df.variables[variable_name][time_index,:]

            if(self.remove_mask):
                data = data[~data.mask].reshape(-1,self.vertical_layers -1)
            data = data[global_index_list]

            if(variable_name in self.pad_zero_list):
                data[data < 0.0] = 0.0
            temp = (data - self.norm_mapping[variable_name]["mean"]) / self.norm_mapping[variable_name]["scale"]

            temp_value = temp[inside_index_list, ::]
            self.multi_feature[:, variable_index,
                               1:self.vertical_layers] = temp_value
            self.multi_feature[:, variable_index,
                               0] = self.multi_feature[:, variable_index, 1]

            if(variable_name in self.multi_height_cumsum_variable):
                # 获取index
                variable_index = self.multi_height_cumsum_variable[variable_name]
                temp_value_cumsum_forward = np.cumsum(
                    temp_value, axis=1)/20.0
                temp_value_cumsum_backward = np.cumsum(
                    temp_value[:, ::-1], axis=1)/20.0

                self.multi_cumsum_feature[:, variable_index,
                                          1:self.vertical_layers] = temp_value_cumsum_forward
                self.multi_cumsum_feature[:, variable_index,
                                          0] = self.multi_cumsum_feature[:, variable_index, 1]

                self.multi_cumsum_feature[:, len(self.multi_height_cumsum_variable) + variable_index,
                                          1:self.vertical_layers] = temp_value_cumsum_backward
                self.multi_cumsum_feature[:, len(self.multi_height_cumsum_variable) + variable_index,
                                          0] = self.multi_cumsum_feature[:, variable_index, 1]

        multi_feature_tt = torch.tensor(
            self.multi_feature, dtype=torch.float32)
        multi_cumsum_feature_tt = torch.tensor(
            self.multi_cumsum_feature, dtype=torch.float32)

        for variable_index, variable_name in enumerate(self.label_variable):
            
            data = df.variables[variable_name][time_index,:]

            if(self.remove_mask):
                data = data[~data.mask].reshape(-1,self.vertical_layers)
            data = data[global_index_list,:]

            if(variable_name in self.pad_zero_list):
                data[data < 0.0] = 0.0


            temp = (data - self.norm_mapping[variable_name]["mean"]) / self.norm_mapping[variable_name]["scale"]
            self.label_feature[:, variable_index,
                               :] = temp[inside_index_list, ::]

        for variable_index, variable_name in enumerate(self.auxiliary_variable):

            # data = df.variables[variable_name][time_index,
            #                                    global_index_list, :]

            
            data = df.variables[variable_name][time_index,:]

            if(self.remove_mask):
                data = data[~data.mask].reshape(-1, self.vertical_layers)

            data = data[global_index_list,:]

            if(variable_name in self.pad_zero_list):
                data[data < 0.0] = 0.0

            temp = data
            self.auxiliary_feature[:, variable_index,
                                   :] = temp[inside_index_list, ::]

        label_feature_tt = torch.tensor(
            self.label_feature, dtype=torch.float32)

        auxiliary_result_tf = torch.tensor(
            self.auxiliary_feature, dtype=torch.float32)

        p_diff = auxiliary_result_tf - torch.roll(auxiliary_result_tf, -1, 2)
        p_diff = torch.cat([p_diff[:, :, 0:1], p_diff[:, :, 0:-1]], dim=2)

        feature_tf = torch.cat(
            [torch.tile(single_feature_tt, [1, self.vertical_layers]),
             multi_feature_tt,
             (p_diff - 17.2)/9.8,
             multi_cumsum_feature_tt
             ], dim=(1))

        return (feature_tf, label_feature_tt, auxiliary_result_tf)





def test_local():
    nc_file = "rrtmg4nn.nc"
    root_dir = "/Users/yaoyichen/dataset/radiation/MPAS/data_1day/"

    from config import norm_mapping

    train_dataset = RtmMpasDatasetWholeTimeLarge(
        nc_file, root_dir,
        from_time=0, end_time=72,
        batch_divid_number=1, point_folds=5, time_folds=3,
        norm_mapping=norm_mapping, vertical_layers=57)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=1000, shuffle=True, num_workers=2)

    t1 = time.time()
    for batch_idx, (feature, targets, auxis) in enumerate(train_loader):
        t2 = time.time()
        print(batch_idx, feature.shape, targets.shape, auxis.shape, t2 - t1)
        print(torch.mean(feature, dim=(0, 1, 3)))
        print(torch.std(feature, dim=(0, 1, 3)))
        t1 = time.time()




def test_local2():
    """
    采用RtmMpasDatasetWholeTimeFullyear的 dataset, 为了证明数据是OK的
    """
    nc_file = "rrtmg4nn.nc"
    root_dir = "/Users/yaoyichen/dataset/radiation/MPAS/data_1day/"

    from config import norm_mapping,norm_mapping_fullyear

    train_dataset = RtmMpasDatasetWholeTimeFullyear(
        nc_file, root_dir,
        from_time=0, end_time=72,
        batch_divid_number=1, point_folds=5, time_folds=3,
        norm_mapping=norm_mapping_fullyear, vertical_layers=57)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=100, shuffle=True, num_workers=2)

    t1 = time.time()
    for batch_idx, (feature, targets, auxis) in enumerate(train_loader):
        t2 = time.time()
        print(batch_idx, feature.shape, targets.shape, auxis.shape, t2 - t1)
        print(torch.mean(feature, dim=(0, 1, 3)))
        print(torch.std(feature, dim=(0, 1, 3)))
        t1 = time.time()

def test_large():
    nc_file = "rrtmg4nn.nc"
    root_dir = "../../data/radiation/"

    from config import norm_mapping
    train_dataset = RtmMpasDatasetWholeTimeLarge(
        nc_file, root_dir,
        from_time=0, end_time=1152,
        batch_divid_number=20, point_folds=5, time_folds=3,
        norm_mapping=norm_mapping, vertical_layers=57)

    print(f"len dataset:{len(train_dataset)}")

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=7, shuffle=True, num_workers=16)

    t1 = time.time()
    for batch_idx, (feature, targets, auxis) in enumerate(train_loader):
        t2 = time.time()
        print(batch_idx, feature.shape, targets.shape, auxis.shape, t2 - t1)
        t1 = time.time()



def test_fullyear():
    nc_file = "rrtmg4nn_00000.000.nc"
    root_dir = "/Users/yaoyichen/dataset/radiation/MPAS/data_1year/"

    from config import norm_mapping,norm_mapping_fullyear
    train_dataset = RtmMpasDatasetWholeTimeFullyear(
        nc_file, root_dir,
        from_time=0, end_time=240,
        batch_divid_number=20, point_folds=5, time_folds=3,
        norm_mapping=norm_mapping_fullyear, vertical_layers=57)

    print(f"len dataset:{len(train_dataset)}")

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=7, shuffle=True, num_workers=0)

    t1 = time.time()
    print("IN "*3)
    for batch_idx, (feature, targets, auxis) in enumerate(train_loader):
        t2 = time.time()
        # print(batch_idx, feature.shape, targets.shape, auxis.shape, t2 - t1)
        t1 = time.time()
        # print(torch.mean(feature,(0,1,3)))
        # print(torch.mean(targets,(0,1,3)))
        # print(torch.mean(auxis,(0,1,3)))


def test_flux():
    import numpy as np
    nc_file = "rrtmg4nn.nc"
    root_dir = "/Users/yaoyichen"
    full_file_path = os.path.join(root_dir, nc_file)
    df = xr.open_dataset(full_file_path, engine="netcdf4")

    label_variable = ['swuflx', 'swdflx', 'lwuflx', 'lwdflx']
    time_index = np.arange(10, 20, 2)
    global_index_list = np.arange(1300, 1301, 1)
    # global_index_list = 1300

    swuflx = df.variables["swuflx"][time_index, global_index_list, :]
    swdflx = df.variables["swdflx"][time_index, global_index_list, :]
    swhr = df.variables["swhr"][time_index, global_index_list, :]

    lwuflx = df.variables["lwuflx"][time_index, global_index_list, :]
    lwdflx = df.variables["lwdflx"][time_index, global_index_list, :]

    lwhr = df.variables["lwhr"][time_index, global_index_list, :]

    fnet = lwuflx - lwdflx
    plev = df.variables["plev"][time_index, global_index_list, :]

    delta_fnet = fnet - np.roll(fnet, 1, 2)
    delta_p = plev - np.roll(plev, 1, 2)

    g = 9.8066  # m s^-2
    # reference to WRF/share/module_model_constants.F gas constant of dry air
    rgas = 287.0
    cp = 7.*rgas/2.
    heatfac = g*8.64*10**4/(cp*100)

    hr = delta_fnet[:, :, 1::]/delta_p[:, :, 1::] * heatfac

    print(np.mean(hr), np.mean(swhr))
    print(hr[0, 0, :], lwhr[0, 0, :])
    # print(len(hr[0, 10, :]), len(swhr[0, 10, :]))
    ### lw 可以看所有层, sw 只能看 部分层 ###


def countcloudcover():
    nc_file = "rrtmg4nn.nc"
    root_dir = "/Users/yaoyichen"

    full_file_path = os.path.join(root_dir, nc_file)
    df = xr.open_dataset(full_file_path, engine="netcdf4")

    time_index = np.arange(0, 1152, 1)
    swuflx = df.variables["cldfrac"][time_index, :, :]
    swuflx_sum = np.sum(swuflx, axis=2)

    print(swuflx.shape, swuflx_sum.shape)

    no_cloud = len(np.where(swuflx_sum < 0.2)[0])
    single_cloud = len(np.where((swuflx_sum >= 0.2) & (swuflx_sum <= 1.2))[0])
    multi_cloud = len(np.where(swuflx_sum >= 1.2)[0])
    number_sum = no_cloud + single_cloud + multi_cloud
    print(
        f"no_cloud: {no_cloud}, single_cloud:{single_cloud}, multi_cloud:{multi_cloud}")
    print(
        f"no_cloud: {no_cloud/number_sum}, single_cloud:{single_cloud/number_sum}, multi_cloud:{multi_cloud/number_sum}")

    time_index = np.arange(1152, 1440, 1)

    swuflx = df.variables["cldfrac"][time_index, :, :]
    swuflx_sum = np.sum(swuflx, axis=2)

    print(swuflx.shape, swuflx_sum.shape)
    no_cloud = len(np.where(swuflx_sum < 0.2)[0])
    single_cloud = len(np.where((swuflx_sum >= 0.2) & (swuflx_sum <= 1.2))[0])
    multi_cloud = len(np.where(swuflx_sum >= 1.2)[0])
    number_sum = no_cloud + single_cloud + multi_cloud
    print(
        f"no_cloud: {no_cloud}, single_cloud:{single_cloud}, multi_cloud:{multi_cloud}")
    print(
        f"no_cloud: {no_cloud/number_sum}, single_cloud:{single_cloud/number_sum}, multi_cloud:{multi_cloud/number_sum}")


def plotcloud(cldfrac, swuflx, swdflx, lwuflx, lwdflx, plev, play, swhr, lwhr, filename):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    fig = plt.figure(tight_layout=True, figsize=(7, 7),)
    gs = gridspec.GridSpec(3, 1)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(play, cldfrac, label="cloud cover")

    ax.set_title("cloud cover", fontsize=10)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim([0, 1.1])
    ax.set_xticks(np.arange(1000, 0, -100))
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.legend()

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(plev, swuflx, label="short up", color="coral")
    ax.plot(plev, swdflx, label="short down", color="red")
    ax.plot(plev, lwuflx, label="long up", color="royalblue")
    ax.plot(plev, lwdflx, label="long down", color="lightskyblue")
    ax.legend()

    ax.set_title("flux ", fontsize=10)
    ax.set_xticks(np.arange(1000, 0, -100))
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.legend()

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(play, swhr, label="short", color="red")
    ax.plot(play, lwhr, label="long", color="royalblue")
    ax.set_xticks(np.arange(1000, 0, -100))
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_title("heat rate ", fontsize=10)
    # ax.set_yticks(np.arange(-20, 20, 2))
    # ax.set_ylim([-20, 20])
    ax.legend()

    plt.savefig(filename, dpi=500, bbox_inches='tight')
    plt.close(fig)
    plt.clf()
    plt.cla()


def plot_module():
    nc_file = "rrtmg4nn.nc"
    root_dir = "/Users/yaoyichen"

    full_file_path = os.path.join(root_dir, nc_file)
    df = xr.open_dataset(full_file_path, engine="netcdf4")

    for i in range(100):
        time_index = np.random.randint(72)
        point_index = np.random.randint(10242)

        cldfrac = df.variables["cldfrac"][time_index, point_index, :]
        swuflx = df.variables["swuflx"][time_index, point_index, :]
        swdflx = df.variables["swdflx"][time_index, point_index, :]
        lwuflx = df.variables["lwuflx"][time_index, point_index, :]
        lwdflx = df.variables["lwdflx"][time_index, point_index, :]
        plev = df.variables["plev"][time_index, point_index, :]
        play = df.variables["play"][time_index, point_index, :]
        swhr = df.variables["swhr"][time_index, point_index, :]
        lwhr = df.variables["lwhr"][time_index, point_index, :]

        plotcloud(cldfrac, swuflx, swdflx, lwuflx, lwdflx,
                  plev, play, swhr, lwhr, filename="../results/cloud_view/" + str(i) + ".png")


if __name__ == "__main__":
    # test_local2()
    test_fullyear()
