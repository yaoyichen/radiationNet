#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 10:29:55 2021

@author: yaoyichen
"""

import torch
import numpy as np
import sys
import os
from tqdm import tqdm
sys.path.append("..")
from utils.plot_helper import plot_RTM, plot_HeatRate


class RunningMeter(object):
    """
        Computes and stores the average and current value
        for smoothing of the time series
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0.0
        self.count = 0

    def update(self, value, count):
        self.count = self.count + count
        self.value = self.value + value*count

    def getmean(self):
        return self.value/self.count

    def getsqrtmean(self):
        return np.sqrt(self.value/self.count)


def MSELoss_valid(predict, true):
    """
    只计算 flux 较大的值
    """
    valid_index = torch.where(torch.abs(true) > 0.5)
    valid_length = len(valid_index[0])

    if valid_length == 0:
        return valid_length, torch.zeros([1])
    else:
        return valid_length, torch.mean(torch.square(predict[valid_index] - true[valid_index]))


def MSELoss_all(predict, true):

    return predict.shape.numel(), torch.mean(torch.square(predict - true))


def MBE_all(predict, true):
    return predict.shape.numel(), torch.mean(predict - true)

def MAELoss_all(predict, true):

    return predict.shape.numel(), torch.mean(torch.abs(predict - true))


def unnormalized_mpas(predicts, targets, norm_mapping, index_mapping):
    predicts_unnorm = torch.zeros(predicts.shape).to(predicts.device)
    targets_unnorm = torch.zeros(targets.shape).to(targets.device)
    for index, value in index_mapping.items():
        predicts_unnorm[:, index, :] = predicts[:, index, :]*norm_mapping[index_mapping[index]
                                                                          ]["scale"] + norm_mapping[index_mapping[index]]["mean"]

        targets_unnorm[:, index, :] = targets[:, index, :]*norm_mapping[index_mapping[index]
                                                                        ]["scale"] + norm_mapping[index_mapping[index]]["mean"]
    return predicts_unnorm, targets_unnorm


def unnormalized_climart(predicts, targets, target_mean,target_std ):
    target_mean  =target_mean.to(predicts.device)
    target_std = target_std.to(predicts.device)
    predicts_unnorm  = predicts*target_std  + target_mean
    targets_unnorm = targets*target_std + target_mean

    return predicts_unnorm, targets_unnorm


def get_heat_rate(predicts, targets, pressure):
    swhr_predict = calculate_hr(
        predicts[:, 0:1, :], predicts[:, 1:2, :], pressure)
    swhr_target = calculate_hr(
        targets[:, 0:1, :], targets[:, 1:2, :], pressure)
    # the last of the short wave do not need to calculate
    swhr_predict[:, :, -1] = 0.0
    swhr_target[:, :, -1] = 0.0

    lwhr_predict = calculate_hr(
        predicts[:, 2:3, :], predicts[:, 3:4, :], pressure)
    lwhr_target = calculate_hr(
        targets[:, 2:3, :], targets[:, 3:4, :], pressure)
    return swhr_predict, swhr_target, lwhr_predict, lwhr_target



def calculate_hr(up, down, pressure):
    g = 9.8066  # m s^-2
    # reference to WRF/share/module_model_constants.F gas constant of dry air
    rgas = 287.0
    cp = 7.*rgas/2.
    heatfac = g*8.64*10**4/(cp*100)

    net = up - down
    net_delta = net - \
        torch.roll(net, 1, 2)
    p_delta = pressure - torch.roll(pressure, 1, 2)
    return net_delta[:, :, 1::]/p_delta[:, :, 1::] * heatfac


def check_accuracy_evaluate(loader, model, norm_mapping, index_mapping, device, args, if_plot=False,target_norm_info = None):
    model.eval()
    # 长短波底层 rmse
    sw_flux_bottom_rmse = RunningMeter()
    lw_flux_bottom_rmse = RunningMeter()
    sw_flux_bottom_mbe = RunningMeter()
    # lw_flux_bottom_mbe = RunningMeter()


    # 长短波底层 rmse
    sw_flux_top_rmse = RunningMeter()
    # lw_flux_top_rmse = RunningMeter()
    sw_flux_top_mbe = RunningMeter()
    # lw_flux_top_mbe = RunningMeter()

    # 长短波整体 rmse
    sw_flux_rmse = RunningMeter()
    lw_flux_rmse = RunningMeter()
    sw_flux_mbe = RunningMeter()
    # lw_flux_mbe = RunningMeter()



    sw_hr_rmse = RunningMeter()
    lw_hr_rmse = RunningMeter()

    sw_hr_mae = RunningMeter()
    lw_hr_mae = RunningMeter()


    with torch.no_grad():
        # loop = tqdm(loader)
        for batch_idx, (feature, targets, auxis) in enumerate(loader):
            # print(batch_idx, feature.shape)
            
            if((args.dataset_type == "Large") or (args.dataset_type == "Small") or (args.dataset_type == "FullYear") or (args.dataset_type == "WRF")):
                feature_shape = feature.shape
                target_shape = targets.shape
                auxis_shape = auxis.shape

                # print(feature_shape.shape, target_shape.shape, auxis_shape.shape)
                inner_batch_size = feature_shape[0]*feature_shape[1]
                feature = feature.reshape(
                    inner_batch_size, feature_shape[2], feature_shape[3]).to(device=device)
                targets = targets.reshape(
                    inner_batch_size, target_shape[2], target_shape[3]).to(device=device)
                auxis = auxis.reshape(
                    inner_batch_size, auxis_shape[2], auxis_shape[3]).to(device=device)
            else:
                feature = feature.to(device=device)
                targets = targets.to(device=device)
                auxis = auxis.to(device=device)

            feature_shape = feature.shape
            target_shape = targets.shape
            auxis_shape = auxis.shape

            # Get data to cuda if possible
            predicts = model(feature)

            # 此处去归一化
            if(args.dataset_type == "CliMart"):
                """
                由于CliMart没有直接给出 variable name, 所以直接存储了各个变量的 mean 和 std
                """
                target_mean, target_std = target_norm_info
                predicts_unnorm, targets_unnorm = unnormalized_climart(predicts, targets,
                target_mean, target_std)
                
            elif((args.dataset_type == "Large") or (args.dataset_type == "Small")):
                # 此处去归一化, 计算 Heat Rate
                predicts_unnorm, targets_unnorm = unnormalized_mpas(
                    predicts, targets, norm_mapping, index_mapping)

            elif((args.dataset_type == "FullYear") or (args.dataset_type == "WRF")):
                predicts_unnorm, targets_unnorm = unnormalized_mpas(
                    predicts, targets, norm_mapping, index_mapping)


            swhr_predict, swhr_target, lwhr_predict, lwhr_target = get_heat_rate(
                predicts_unnorm, targets_unnorm, auxis)

            # 短波 rmse
            valid_length, valid_value = MSELoss_all(
                predicts_unnorm[:, 0:2, :], targets_unnorm[:, 0:2, :])
            sw_flux_rmse.update(valid_value.item(), valid_length)
            valid_length, valid_value = MSELoss_all(predicts_unnorm[:, 0:2, 0], targets_unnorm[:, 0:2, 0])
            sw_flux_bottom_rmse.update(valid_value.item(), valid_length)


            valid_length, valid_value = MSELoss_all(predicts_unnorm[:, 0:2, -1], targets_unnorm[:, 0:2, -1])
            sw_flux_top_rmse.update(valid_value.item(), valid_length)


            # 短波 mbe
            valid_length, valid_value = MBE_all(
                predicts_unnorm[:, 0:2, :], targets_unnorm[:, 0:2, :])
            sw_flux_mbe.update(valid_value.item(), valid_length)
            valid_length, valid_value = MBE_all(predicts_unnorm[:, 0:2, 0], targets_unnorm[:, 0:2, 0])
            sw_flux_bottom_mbe.update(valid_value.item(), valid_length)

            valid_length, valid_value = MBE_all(predicts_unnorm[:, 0:2, -1], targets_unnorm[:, 0:2, -1])
            sw_flux_top_mbe.update(valid_value.item(), valid_length)


            # 长波 flux
            valid_length, valid_value = MSELoss_all( 
                predicts_unnorm[:, 2:4, :], targets_unnorm[:, 2:4, :])
            lw_flux_rmse.update(valid_value.item(), valid_length)

            valid_length, valid_value = MSELoss_all(predicts_unnorm[:, 2:4, 0], targets_unnorm[:, 2:4, 0])
            lw_flux_bottom_rmse.update(valid_value.item(), valid_length)


            # 短波 heat rate
            valid_length, valid_value = MSELoss_all(swhr_predict, swhr_target)
            sw_hr_rmse.update(valid_value.item(), valid_length)

            valid_length, valid_value = MAELoss_all(swhr_predict, swhr_target)
            sw_hr_mae.update(valid_value.item(), valid_length)
            
            # 长波 heat rate
            valid_length, valid_value = MSELoss_all(lwhr_predict, lwhr_target)
            lw_hr_rmse.update(valid_value.item(), valid_length)

            valid_length, valid_value = MAELoss_all(lwhr_predict, lwhr_target)
            lw_hr_mae.update(valid_value.item(), valid_length)

            if(if_plot):
                if(batch_idx < 50):
                    print("making plot " + str(batch_idx))
                    file_name_RTM = os.path.join(
                        "results", args.main_folder, args.sub_folder, "Flux" +
                        str(batch_idx) + ".png")
                    plot_RTM(predicts_unnorm, targets_unnorm, file_name_RTM, sample_index=0)

                    file_name_HR = os.path.join(
                        "results", args.main_folder, args.sub_folder, "HR" +
                        str(batch_idx) + ".png")
                    plot_HeatRate(swhr_predict, swhr_target,
                                  lwhr_predict, lwhr_target, file_name_HR, sample_index=0)
    """
    sw_flux_rmse, lw_flux_rmse, sw_hr_rmse, lw_hr_rmse, sw_hr_mae, lw_hr_mae,\
        sw_flux_bottom_rmse,lw_flux_bottom_rmse,\
        sw_flux_top_rmse,  \
        sw_flux_mbe,sw_flux_bottom_mbe, sw_flux_top_mbe
    """
    return sw_flux_rmse.getsqrtmean(), lw_flux_rmse.getsqrtmean(), sw_hr_rmse.getsqrtmean(), lw_hr_rmse.getsqrtmean(), sw_hr_mae.getmean(), lw_hr_mae.getmean(),\
        sw_flux_bottom_rmse.getsqrtmean(),lw_flux_bottom_rmse.getsqrtmean(),\
        sw_flux_top_rmse.getsqrtmean(),  \
        sw_flux_mbe.getmean(),sw_flux_bottom_mbe.getmean(), sw_flux_top_mbe.getmean()





def check_accuracy(loader, model, norm_mapping, index_mapping, device, args,target_norm_info = None):
    """
    外面包了一层汇总的
    """
    sw_flux_rmse, lw_flux_rmse, sw_hr_rmse, lw_hr_rmse, sw_hr_mae, lw_hr_mae,\
        sw_flux_bottom_rmse,lw_flux_bottom_rmse,\
        sw_flux_top_rmse,  \
        sw_flux_mbe,sw_flux_bottom_mbe, sw_flux_top_mbe = check_accuracy_evaluate(
        loader, model, norm_mapping, index_mapping, device, args, target_norm_info = target_norm_info)
    return [sw_flux_rmse, lw_flux_rmse, sw_hr_rmse, lw_hr_rmse, sw_hr_mae, lw_hr_mae],\
    [sw_flux_rmse, sw_flux_mbe, sw_flux_bottom_rmse, sw_flux_bottom_mbe, sw_flux_top_rmse, sw_flux_top_mbe],[lw_flux_bottom_rmse]