#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:54:07 2021

@author: yaoyichen
"""

from pyparsing import col
import torch
import torch.nn as nn
import sys
import os
os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
from models.model_prepare import load_model
from utils.model_helper import ModelUtils
from utils.data_helper import RtmMpasDatasetWholeTime, RtmMpasDatasetWholeTimeLarge,RtmMpasDatasetWholeTimeFullyear
from utils.file_helper import FileUtils
from utils.plot_helper import plot_RTM
from utils.config import norm_mapping,norm_mapping_fullyear,norm_mapping_fullyear_new
from utils.config import climart_96_feature_mean, climart_96_feature_std, climart_96_target_mean, climart_96_target_std
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time
import numpy as np
import argparse
import logging
from utils.make_wrfRRTMG_data import WrfRRTMGDataset
from utils.config import norm_mapping_wrf07

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.evaluate_helper import check_accuracy_evaluate, unnormalized_mpas, MSELoss_all,get_heat_rate,check_accuracy
from utils.evaluate_helper import unnormalized_climart
from climart.data_wrangling.h5_dataset import RT_HdF5_SingleDataset,ClimART_HdF5_Dataset
from utils.climart_data_helper import climart_collate_fn
from utils.climart_data_helper import fullyear_collate_fn_random,fullyear_collate_fn_random_half


# 需要修改的东西， summary的名字， plot的名字，测评模块
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train the RTM model')
    parser.add_argument('--nc_file', type=str, default='rrtmg4nn.nc')
    parser.add_argument('--root_dir', type=str, default="")
    parser.add_argument('--train_file', type=str, default="")
    parser.add_argument('--test_file', type=str, default="")
    parser.add_argument('--train_point_number', type=int, default=100)
    parser.add_argument('--test_point_number', type=int, default=100)

    parser.add_argument('--test_time_number', type=int, default=2)
    parser.add_argument('--main_folder', type=str, default='temp')
    parser.add_argument('--sub_folder', type=str, default='temp')
    parser.add_argument('--prefix', type=str, default='temp')

    parser.add_argument('--dataset_type', type=str, default='Large')
    parser.add_argument('--loss_type', type=str, default='v01')
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    parser.add_argument('--batch_size', type=int, default=200,
                        help='batch_size')
    parser.add_argument('--model_name', type=str, default="FC",
                        help='model name')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='batch_size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='num_epochs')

    parser.add_argument('--save_model', choices=('True', 'False'))
    parser.add_argument('--save_checkpoint_name',
                        type=str, default="test.pth.tar")
    parser.add_argument('--save_per_samples', type=int, default=10000)
    parser.add_argument('--load_model', choices=('True', 'False'))
    parser.add_argument('--load_checkpoint_name',
                        type=str, default="test.pth.tar")
    parser.add_argument('--random_throw', choices=('True', 'False'),default = "False")
    parser.add_argument('--only_layer', choices=('True', 'False'),default = "False")
    args = parser.parse_args()
    return args

args = parse_args()
FileUtils.makedir(os.path.join("logs", args.main_folder, args.sub_folder))
FileUtils.makedir(os.path.join("results", args.main_folder, args.sub_folder))
FileUtils.makedir(os.path.join("runs", args.main_folder, args.sub_folder))
FileUtils.makedir(os.path.join(
    "checkpoints", args.main_folder, args.sub_folder))

if(args.random_throw == "True"):
    args.random_throw_boolean = True
else:
    args.random_throw_boolean = False

if(args.only_layer == "True"):
    args.only_layer_boolean = True
else:
    args.only_layer_boolean = False

# set logging
filehandler = logging.FileHandler(os.path.join(
    "logs", args.main_folder, args.sub_folder, args.prefix + "_log.txt"))
streamhandler = logging.StreamHandler()
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logging.getLogger('matplotlib.font_manager').disabled = True
logger.addHandler(filehandler)
logger.addHandler(streamhandler)


# random variable
random_state = 0
np.random.seed(random_state)
torch.manual_seed(random_state)
torch.set_printoptions(precision=5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

writer = SummaryWriter(
    f"runs/{args.main_folder}/{args.sub_folder}/")
step = 0

print(f"args.dataset_type:{args.dataset_type}")
if (args.dataset_type == "Large"):
    train_dataset = RtmMpasDatasetWholeTimeLarge(
        args.nc_file, args.root_dir,
        from_time=0, end_time=1152,
        batch_divid_number=20,  point_folds=3, time_folds=2, norm_mapping=norm_mapping,
        random_throw = args.random_throw_boolean,
        only_layer = args.only_layer_boolean)

    test_dataset = RtmMpasDatasetWholeTimeLarge(
        args.nc_file, args.root_dir,
        from_time=1152, end_time=1440,
        batch_divid_number=20, point_folds=3,  time_folds=2, norm_mapping=norm_mapping,
        only_layer = args.only_layer_boolean)


elif(args.dataset_type == "CliMart"):
    # ,1999,2005
    train_dataset = ClimART_HdF5_Dataset(years=[1990,1999,2005], name='Train',
                                    output_normalization=True,
                                    exp_type='pristine',
                                    target_type = ['shortwave','longwave'],
                                    target_variable = ['rsuc', 'rsdc', 'rluc', 'rldc', 'hrsc','hrlc'],
                                    load_h5_into_mem=False,
                                    log_scaling = False
                                   )
    #                               2007,2008,2009,2010,2011,2012,2013,2014
    # 
    test_dataset = ClimART_HdF5_Dataset(years=[2007,2008,2009,2010,2011,2012,2013,2014], name='Train',
                            output_normalization=True,
                            exp_type='pristine',
                            target_type = ['shortwave','longwave'],
                            target_variable = ['rsuc', 'rsdc', 'rluc', 'rldc', 'hrsc','hrlc'],
                            load_h5_into_mem=False,
                            log_scaling = False
                            )

if (args.dataset_type == "Small"):
    train_dataset = RtmMpasDatasetWholeTimeLarge(
        args.nc_file, args.root_dir,
        from_time=0, end_time=60,
        batch_divid_number=20,  point_folds=5, time_folds=10, norm_mapping=norm_mapping)

    test_dataset = RtmMpasDatasetWholeTimeLarge(
        args.nc_file, args.root_dir,
        from_time=60, end_time=72,
        batch_divid_number=20, point_folds=100,  time_folds=10, norm_mapping=norm_mapping)


if (args.dataset_type == "WRF"):
    train_dataset =  WrfRRTMGDataset(vertical_layers = 57, 
            type = "train", norm_mapping=norm_mapping_wrf07)
    
    test_dataset = WrfRRTMGDataset(vertical_layers = 57, 
            type = "test", norm_mapping=norm_mapping_wrf07)



if (args.dataset_type == "FullYear"):

    train_dataset = RtmMpasDatasetWholeTimeLarge(
        args.train_file, "",
        from_time=0, end_time=1,
        # batch_divid_number=int(960/6)*4, 
        batch_divid_number=int(960/6), 
         point_folds=1, time_folds=1, norm_mapping=norm_mapping_fullyear_new,
         point_number= args.train_point_number,
        #  random_throw = args.random_throw_boolean,
         only_layer = args.only_layer_boolean)
    
    test_dataset = RtmMpasDatasetWholeTimeLarge(
        args.test_file, "",
        from_time=0, end_time=1,
        batch_divid_number=int(960/6), 
        point_folds=1, time_folds=1, norm_mapping=norm_mapping_fullyear_new,
        point_number= args.test_point_number,
        only_layer = args.only_layer_boolean)


if(args.dataset_type == "CliMart"):
    import random
    indices = list(range(len(train_dataset)))
    random.shuffle(indices)
    split = len(train_dataset) 
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])

    train_loader = DataLoader(
        dataset=train_dataset, sampler = train_sampler, batch_size= args.batch_size,
        num_workers= args.num_workers, pin_memory=False, collate_fn = climart_collate_fn,
        shuffle= False)


    indices = list(range(len(test_dataset)))
    random.shuffle(indices)
    split = len(test_dataset) //10
    print(split)
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])

    test_loader = DataLoader(
        dataset=test_dataset, sampler = test_sampler, batch_size= args.batch_size,
        num_workers= args.num_workers, pin_memory=False, collate_fn = climart_collate_fn,
        shuffle= False)


elif((args.dataset_type == "Large") or (args.dataset_type == "Small")):
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False,
        collate_fn = fullyear_collate_fn_random)


    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False)
        

elif(args.dataset_type == "FullYear"):
    if(args.random_throw_boolean == True): 
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=False ,
            collate_fn = fullyear_collate_fn_random_half)
    else:

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=False )

    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False )


if (args.dataset_type == "WRF"):
    train_loader = DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=False )

    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False )



logger.info(
    f"train size:{len(train_dataset)}, test size:{len(test_dataset)}")

if(args.dataset_type == "CliMart"):
    model = load_model(args.model_name, device,
                    feature_channel=96, signal_length=50)
                    
elif((args.dataset_type == "Large") or (args.dataset_type == "Small")):
    model = load_model(args.model_name, device,
                    feature_channel=34, signal_length=57)

elif(args.dataset_type == "FullYear"):
    if(args.only_layer_boolean == True):
        model = load_model(args.model_name, device,
                        feature_channel=29, signal_length=57)
    else:
        model = load_model(args.model_name, device,
                        feature_channel=34, signal_length=57)


elif(args.dataset_type == "WRF"):
    if(args.only_layer_boolean == True):
        model = load_model(args.model_name, device,
                        feature_channel=29, signal_length=57)
    else:
        model = load_model(args.model_name, device,
                        feature_channel=34, signal_length=57)


model_info = ModelUtils.get_parameter_number(model)
logger.info(model_info)


model.to(device)



criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Define Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=5, verbose=True
)

if(args.load_model == "True"):
    ModelUtils.load_checkpoint(torch.load(os.path.join("checkpoints",
                                                       args.main_folder, args.sub_folder, args.load_checkpoint_name)),
                               model, optimizer)
if(args.save_model == "True"):
    save_counter = 0


if torch.cuda.is_available():
    model.cuda()
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)



index_mapping = {0: "swuflx", 1: "swdflx", 2: "lwuflx", 3: "lwdflx"}



logger.info("start training...")


previous_time = time.time()

for epoch in range(args.num_epochs):

    loop = tqdm(train_loader)
    model.train()
    sum_train_mse = 0.0
    sum_train_mae = 0.0
    sum_train_swhr = 0.0
    sum_train_lwhr = 0.0

    sum_train_flux_mse = 0.0 

    num_samples = 0
    schedule_losses = []

    logger.info(f"epoch:{epoch}, elapse time:{time.time() - previous_time}")
    previous_time = time.time()

    for batch_idx, (feature, targets, auxis) in enumerate(train_loader):
        # if(batch_idx > 2):
        #     break
        if(epoch == 0 and batch_idx == 0):
            logger.info(f"feature shape:{feature.shape}, target shape:{targets.shape}, auxis shape:{auxis.shape}" )

        
        feature_shape = feature.shape
        target_shape = targets.shape
        auxis_shape = auxis.shape
        # Get data to cuda if possible

        if((args.dataset_type == "Large") or (args.dataset_type == "Small") or (args.dataset_type == "FullYear") or (args.dataset_type == "WRF") ):
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


        predicts = model(feature)

        # 此处去归一化, 计算 Heat Rate
        if(args.dataset_type == "CliMart"):
            predicts_unnorm, targets_unnorm = unnormalized_climart(predicts, targets,
            climart_96_target_mean, climart_96_target_std)

        elif((args.dataset_type == "Large") or (args.dataset_type == "Small")):
            predicts_unnorm, targets_unnorm = unnormalized_mpas(
                predicts, targets, norm_mapping, index_mapping)

        elif(args.dataset_type == "FullYear"):
            predicts_unnorm, targets_unnorm = unnormalized_mpas(
                predicts, targets, norm_mapping_fullyear_new, index_mapping)

        elif(args.dataset_type == "WRF"):
            predicts_unnorm, targets_unnorm = unnormalized_mpas(
                predicts, targets, norm_mapping_wrf07, index_mapping)

        swhr_predict, swhr_target, lwhr_predict, lwhr_target = get_heat_rate(
            predicts_unnorm, targets_unnorm, auxis)
        

        _, sw_hr_rmse = MSELoss_all(swhr_predict, swhr_target)
        _, lw_hr_rmse = MSELoss_all(lwhr_predict, lwhr_target)


        if(args.loss_type == "flux"):
            """
            只有flux loss, 主要用于只监督flux的场景
            """
            loss_mse = criterion_mse(predicts[:,0:4,:], targets[:,0:4,:])
            loss_mae = criterion_mae(predicts, targets)
            loss_mse_bottom = criterion_mse(predicts[:,:,0], targets[:,:,0])

            total_loss = loss_mse

            """
            train集 带量纲loss的统计, 只统计短波
            """
            flux_mse  = criterion_mse(predicts_unnorm[:,0:2,:], targets_unnorm[:,0:2,:])


        if(args.loss_type == "climart"):
            """
            只有 短波 flux loss, 主要用于 climart数据集
            """
            loss_mse = criterion_mse(predicts[:,0:2,:], targets[:,0:2,:])
            loss_mae = criterion_mae(predicts[:,0:2,:], targets[:,0:2,:])
            loss_mse_bottom = criterion_mse(predicts[:,:,0], targets[:,:,0])
            total_loss = loss_mse


            """
            train集 带量纲loss的统计, 只统计短波
            """
            flux_mse  = criterion_mse(predicts_unnorm[:,0:2,:], targets_unnorm[:,0:2,:])


        if(args.loss_type == "v01"):
            """
            v01版本的loss
            """
            loss_mse = criterion_mse(predicts[:,0:4,:], targets[:,0:4,:])
            loss_mae = criterion_mae(predicts, targets)
            loss_mse_bottom = criterion_mse(predicts[:,:,0], targets[:,:,0])

            total_loss = loss_mse + 0.001*(sw_hr_rmse + lw_hr_rmse)


            """
            train集 带量纲loss的统计, 
            """
            flux_mse  = criterion_mse(predicts_unnorm[:,0:2,:], targets_unnorm[:,0:2,:])


        if(args.loss_type == "hr"):
            """
            v01版本的loss
            """
            loss_mse = criterion_mse(predicts[:,0:4,:], targets[:,0:4,:])
            loss_mae = criterion_mae(predicts, targets)
            loss_mse_bottom = criterion_mse(predicts[:,:,0], targets[:,:,0])

            total_loss =  0.1*(sw_hr_rmse + lw_hr_rmse)

            """
            train集 带量纲loss的统计, 
            """
            flux_mse  = criterion_mse(predicts_unnorm[:,0:2,:], targets_unnorm[:,0:2,:])

        
        if(args.loss_type == "v02"):
            """
            v01版本的loss
            """
            loss_mse = criterion_mse(predicts[:,0:4,:], targets[:,0:4,:])
            loss_mae = criterion_mae(predicts, targets)
            loss_mse_bottom = criterion_mse(predicts[:,:,0], targets[:,:,0])

            total_loss = loss_mse + 0.1*(sw_hr_rmse + lw_hr_rmse) + 2.0*loss_mse_bottom


            """
            train集 带量纲loss的统计
            """
            flux_mse  = criterion_mse(predicts_unnorm[:,0:2,:], targets_unnorm[:,0:2,:])


        num_samples = num_samples + feature_shape[0]
        sum_train_mse = sum_train_mse + feature_shape[0]*loss_mse.item()
        sum_train_mae = sum_train_mae + feature_shape[0]*loss_mae.item()
        sum_train_swhr = sum_train_swhr + feature_shape[0]*sw_hr_rmse.item()
        sum_train_lwhr = sum_train_lwhr + feature_shape[0]*lw_hr_rmse.item()
        sum_train_flux_mse = sum_train_flux_mse + feature_shape[0]*flux_mse.item()

        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        writer.add_scalar("train_mse", loss_mse.item(), global_step=step)
        writer.add_scalar("train_mae", loss_mae.item(), global_step=step)
        step = step + args.batch_size
        

        if (args.save_model == "True"):
            save_counter = save_counter + args.batch_size
            if (save_counter > args.save_per_samples):
                if torch.cuda.device_count() > 1:
                    checkpoint = {"state_dict": model.module.state_dict(
                    ), "optimizer": optimizer.state_dict()}
                else:
                    checkpoint = {"state_dict": model.state_dict(
                    ), "optimizer": optimizer.state_dict()}

                filename = os.path.join("checkpoints", args.main_folder,
                                        args.sub_folder, args.prefix + str(epoch).zfill(4) + args.save_checkpoint_name + ".pth.tar")

                filename_full = os.path.join("checkpoints", args.main_folder,
                                             args.sub_folder, args.prefix + str(epoch).zfill(4) + args.save_checkpoint_name + ".pth")

                ModelUtils.save_checkpoint(
                    checkpoint, filename=filename)
                if torch.cuda.device_count() > 1:
                    torch.save(model.module, filename_full)
                else:
                    torch.save(model, filename_full)

                save_counter = 0

    if(epoch % 1 ==0 ):

        train_mse, train_mae, train_swhr, train_lwhr, train_flux_rmse =\
            sum_train_mse/num_samples, sum_train_mae/num_samples, \
            sum_train_swhr/num_samples, sum_train_lwhr/num_samples,np.sqrt(sum_train_flux_mse/num_samples)

        if(args.dataset_type == "CliMart"):
            target_norm_info = [climart_96_target_mean, climart_96_target_std]
        
        elif((args.dataset_type == "Large") or (args.dataset_type == "Small") or (args.dataset_type == "FullYear") or (args.dataset_type == "WRF")):
            target_norm_info = None

        if (args.dataset_type == "Large") or (args.dataset_type == "Small"):
            sw_flux_rmse, lw_flux_rmse, sw_hr_rmse, lw_hr_rmse, sw_hr_mae, lw_hr_mae = check_accuracy(
                test_loader, model, norm_mapping, index_mapping, device, args, target_norm_info)

        if (args.dataset_type == "WRF"):
            [sw_flux_rmse, lw_flux_rmse, sw_hr_rmse, lw_hr_rmse, sw_hr_mae, lw_hr_mae],[sw_flux_rmse, sw_flux_mbe, sw_flux_bottom_rmse, sw_flux_bottom_mbe, sw_flux_top_rmse, sw_flux_top_mbe],[lw_flux_bottom_rmse] = check_accuracy(
                test_loader, model, norm_mapping_wrf07, index_mapping, device, args, target_norm_info)


        elif(args.dataset_type == "FullYear"):
            """
            采用不一样的无量纲方式
            """
            # sw_flux_rmse, lw_flux_rmse, sw_hr_rmse, lw_hr_rmse, sw_hr_mae, lw_hr_mae = check_accuracy(
            #     test_loader, model, norm_mapping_fullyear, index_mapping, device, args, target_norm_info)

            [sw_flux_rmse, lw_flux_rmse, sw_hr_rmse, lw_hr_rmse, sw_hr_mae, lw_hr_mae],[sw_flux_rmse, sw_flux_mbe, sw_flux_bottom_rmse, sw_flux_bottom_mbe, sw_flux_top_rmse, sw_flux_top_mbe],[lw_flux_bottom_rmse] = check_accuracy(
        test_loader, model, norm_mapping, index_mapping, device, args, target_norm_info)

        schedule_losses.append(sw_hr_rmse+ lw_hr_rmse)

        logger.info(
            f"epoch: {epoch}, train_mse: {train_mse: .3e}, train_mae: {train_mae: .3e},\
            train_swhr: {train_swhr: .3e}, train_lwhr: {train_lwhr: .3e}, \
            train_flux_rmse:{train_flux_rmse: .3e}")

        logger.info(
            f"sw_flux_rmse:{sw_flux_rmse: .3e}, lw_flux_rmse:{lw_flux_rmse: .3e}, \
            sw_hr_rmse:{sw_hr_rmse: .3e}, lw_hr_rmse:{lw_hr_rmse: .3e}\
            sw_hr_mae:{sw_hr_mae: .3e}, lw_hr_mae:{lw_hr_mae: .3e}")
        

        logger.info(
            f"sw_flux_rmse:{sw_flux_rmse: .3e}, sw_flux_mbe:{sw_flux_mbe: .3e}, \
            sw_flux_bottom_rmse:{sw_flux_bottom_rmse: .3e}, sw_flux_bottom_mbe:{sw_flux_bottom_mbe: .3e}\
            sw_flux_top_rmse:{sw_flux_top_rmse: .3e}, sw_flux_top_mbe:{sw_flux_top_mbe: .3e}")

        mean_loss = sum(schedule_losses) / len(schedule_losses)
        scheduler.step(mean_loss)
