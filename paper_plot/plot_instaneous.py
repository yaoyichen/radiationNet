#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 16:10:40 2022

@author: yaoyichen
"""
#%%
import  numpy as np
import torch

import matplotlib.pyplot as plt


def parssing_result(tensor_pt_list):
    predicts_unnorm_list = []
    targets_unnorm_list = []
    swhr_predict_list = []
    swhr_target_list = []
    lwhr_predict_list = []
    lwhr_target_list = []
    
    for tensor_pt_name in tensor_pt_list:
        print(tensor_pt_name)
        input_= torch.load(tensor_pt_name, map_location=torch.device('cpu'))
        predicts_unnorm = input_["predicts_unnorm"]
        targets_unnorm = input_["targets_unnorm"]
        swhr_predict = input_["swhr_predict"]
        swhr_target = input_["swhr_target"]
        lwhr_predict = input_["lwhr_predict"]
        lwhr_target = input_["lwhr_target"]
        
        predicts_unnorm_list.append( predicts_unnorm ) 
        targets_unnorm_list.append( targets_unnorm ) 
        swhr_predict_list.append( swhr_predict) 
        swhr_target_list.append( swhr_target) 
        lwhr_predict_list.append( lwhr_predict ) 
        lwhr_target_list.append( lwhr_target ) 
    
    
    return predicts_unnorm_list, targets_unnorm_list, \
            swhr_predict_list,  swhr_target_list, \
            lwhr_predict_list, lwhr_target_list

# tensor_pt_list = ['tensor_slice_fc.pt', 'tensor_slice_unet.pt', 
#                   'tensor_slice_lstm.pt','tensor_slice_att.pt']

tensor_pt_list = ['tensor_fc.pt','tensor_unet.pt', 
                   'tensor_lstm.pt','tensor_att.pt']
                # 'tensor_test_fc.pt','tensor_test_fc.pt']
# tensor_pt_list = ['tensor_slice_fc.pt','tensor_slice_unet208.pt', 
#                   'tensor_slice_lstm.pt','tensor_slice_att_296.pt']


predicts_unnorm_list, targets_unnorm_list, \
        swhr_predict_list,  swhr_target_list, \
        lwhr_predict_list, lwhr_target_list = parssing_result(tensor_pt_list)


h_level_vector = np.array(
[1010.2, 1004.9, 998.1, 990.0, 980.5,
    969.5, 957.0, 943.0, 927.6, 910.7,
    892.4, 872.7, 851.7, 829.4, 806.0,
    781.4, 755.7, 729.1, 701.5, 673.1,
    644.1, 614.6, 584.6, 554.4, 524.0,
    493.5, 463.2, 433.0, 403.1, 373.5,
    344.4, 315.8, 288.0, 261.1, 235.4,
    210.7, 187.3, 165.3, 144.8, 125.9,
    108.4, 92.6, 78.9, 67.0, 57.0,
    48.7, 41.8, 35.9, 30.9, 26.6,
    23.0, 19.8, 17.2, 14.9, 12.9,
    11.2, 7.2])


h_lay_vector = np.array(
[1007.5, 1001.5, 994.1, 985.3, 975.0, 963.2, 950.0, 935.3, 919.1, 901.5,
    882.5, 862.2, 840.6, 817.7, 793.7, 768.6, 742.4, 715.3, 687.3, 658.6,
    629.3, 599.6, 569.5, 539.2, 508.8, 478.3, 448.1, 418.1, 388.3, 358.9,
    330.1, 301.9, 274.6, 248.3, 223.0, 199.0, 176.3, 155.1, 135.3, 117.1,
    100.5, 85.8, 72.9, 62.0, 52.9,  45.3, 38.9, 33.4, 28.7, 24.8,
    21.4, 18.5, 16.0, 13.9, 12.0, 9.2])

qc_profile = np.load('qc_profile.npy') 


# h_level_vector = h_level_vector/np.max(h_level_vector)
# h_lay_vector = h_lay_vector/np.max(h_lay_vector)
title_list = ["FC","U-Net", "Bi-LSTM","Transformer"]

font1 = {'family'  : 'Times New Roman',
          'weight' : 'normal',
          'size'   :  23 }
#%%
"""
短波的结果
"""
index_number_list = [20000, 10001, 30100 ]
# index_number_list = [20000, 10001, 30100]
# index_number_list = [4000, 9000, 14000]
fig, axs = plt.subplots(3,3, figsize=(25, 16),sharey=False)

for i in range(len(index_number_list)):
    axs[0,i].plot(h_level_vector,    targets_unnorm_list[0][index_number_list[i],0,:], label = "Ground Truth")
    for j in range(4):
        axs[0,i].plot(h_level_vector, predicts_unnorm_list[j][index_number_list[i],0,:], label = title_list[j])
        axs[0,i].set_ylim([0, 500])
        axs[0,i].invert_xaxis()
    
    axs[0,i].grid(linestyle = "--",c = "k", )
    axs[0,i].set_xlim([1000, 0])
    
    
axs[0,0].legend(fontsize = 20, loc = "upper left" )
axs[0,0].set_ylabel( r"SW Upward Flux [$\mathrm{W \cdot m^{-2}}$]", fontsize = 22, font = font1)
# axs[0,0].text(700,  500*1.06,"no liquid cloud", size=28,
#                          verticalalignment='center', rotation=0,
#                          font = font1)
# axs[0,1].text(800,  500*1.06,"single-layer liquid cloud", size=28,
#                          verticalalignment='center', rotation=0,
#                          font = font1)
# axs[0,2].text(800,  500*1.06,"multi-layer liquid cloud", size=28,
#                          verticalalignment='center', rotation=0,
#                          font = font1)




for i in range(len(index_number_list)):
    
    axs[1,i].plot(h_level_vector,    targets_unnorm_list[0][index_number_list[i],1,:], label = "Ground Truth")
    # 
    for j in range(4):
        axs[1,i].plot(h_level_vector, predicts_unnorm_list[j][index_number_list[i],1,:], label = title_list[j])
    axs[1,i].set_ylim([0, 1000])
    axs[1,i].invert_xaxis()
    axs[1,i].grid(linestyle = "--",c = "k", )
    axs[1,i].set_xlim([1000, 0])
    
    # axs[1,i].set_yticks([])
    
axs[1,0].set_ylabel( r"SW Downward Flux [$\mathrm{W \cdot m^{-2}}$]", fontsize = 22, font = font1)


for i in range(len(index_number_list)):
    axs[2,i].plot(h_lay_vector,    swhr_target_list[0][index_number_list[i],0,:], label = "Ground Truth")
    # 
    for j in range(4):
        axs[2,i].plot(h_lay_vector, swhr_predict_list[j][index_number_list[i],0,:], label = title_list[j])
        axs[2,i].invert_xaxis()
        axs[2,i].grid(linestyle = "--",c = "k", )
        axs[2,i].set_xlim([1000, 0])
    
    axs[2,i].set_ylim([0, 15])
axs[2,0].set_ylabel( r"SW Heating Rate $\mathrm{[K/d]}$",fontsize = 24, font = font1)


#%%
# import matplotlib.font_manager as font_manager

    
# index_number_list = [20000, 10001, 30100 ]

fig, axs = plt.subplots(1,3, figsize=(25, 6),sharey=True)
for i in range(len(index_number_list)):
    axs[i].plot(h_level_vector.flatten(), qc_profile[index_number_list[i],:].flatten(), label = "Cloud ", c= "k")
    axs[i].grid(linestyle = "--",c = "k", )
    
    axs[i].invert_xaxis()
    axs[i].set_xlim([1000, 0])
    
axs[0].set_ylabel( r"Cloud water mixing ratio (kg/kg)", fontsize = 24, font = font1)
axs[0].text(700,  5e-4*1.1,"no liquid cloud", size=32,
                         verticalalignment='center', rotation=0,
                         font = font1)
axs[1].text(800,  5e-4*1.1,"single-layer liquid cloud", size=32,
                         verticalalignment='center', rotation=0,
                         font = font1)
axs[2].text(800,  5e-4*1.1,"multi-layer liquid cloud", size=32,
                         verticalalignment='center', rotation=0,
                         font = font1)


# axs[0].set_xlabel("Pressure (hPa)", fontsize = 28,  font = font1)
# axs[1].set_xlabel("Pressure (hPa)", fontsize = 28,  font = font1)
# axs[2].set_xlabel("Pressure (hPa)", fontsize = 28,  font = font1)


#%%
"""
长波的结果
"""
import matplotlib.font_manager as font_manager
fig, axs = plt.subplots(3,3, figsize=(25, 16),sharey=False)


font = font_manager.FontProperties(family='Times New Roman',
                                   weight='normal',
                                   style='normal', size=23)


for i in range(len(index_number_list)):
    axs[0,i].plot(h_level_vector,  targets_unnorm_list[0][index_number_list[i],2,:], label = "Ground Truth", c= "k")
    for j in range(4):
        axs[0,i].plot(h_level_vector, predicts_unnorm_list[j][index_number_list[i],2,:], label = title_list[j])
    
    axs[0,i].set_ylim([200, 500])
    axs[0,i].invert_xaxis()
    axs[0,i].set_xlim([1000, 0])
    
    axs[0,i].grid(linestyle = "--",c = "k", )
    
    
# axs[0,0].legend( prop=font, loc = "upper left" )
axs[0,0].set_ylabel( r"LW Upward Flux [$\mathrm{W \cdot m^{-2}}$]", fontsize = 22, font = font1)
# axs[0,0].text(700,  500*1.06,"no liquid cloud", size=28,
#                          verticalalignment='center', rotation=0,
#                          font = font1)
# axs[0,1].text(800,  500*1.06,"single-layer liquid cloud", size=28,
#                          verticalalignment='center', rotation=0,
#                          font = font1)
# axs[0,2].text(800,  500*1.06,"multi-layer liquid cloud", size=28,
#                          verticalalignment='center', rotation=0,
#                          font = font1)



for i in range(len(index_number_list)):
    axs[1,i].plot(h_level_vector,    targets_unnorm_list[0][index_number_list[i],3,:], label = "Ground Truth" , c= "k")
    # 
    for j in range(4):
        axs[1,i].plot(h_level_vector, predicts_unnorm_list[j][index_number_list[i],3,:], label = title_list[j])
        axs[1,i].set_ylim( [0, 500] )
        axs[1,i].set_xlim( [1000, 0] )
        # axs[1,i].invert_xaxis()
    axs[1,i].grid(linestyle = "--",c = "k", )
    
axs[1,0].set_ylabel( r"LW Downward Flux [$\mathrm{W \cdot m^{-2}}$]", fontsize = 22, font = font1)



for i in range(len(index_number_list)):
    axs[2,i].plot(h_lay_vector,    lwhr_target_list[0][index_number_list[i],0,:], label = "Ground Truth" , c= "k")
    # 
    for j in range(4):
        axs[2,i].plot(h_lay_vector, lwhr_predict_list[j][index_number_list[i],0,:], label = title_list[j])
    axs[2,i].invert_xaxis()

    axs[2,0].set_ylim([-30, 5])
    axs[2,i].set_xlim([1000, 0])
    axs[2,i].invert_xaxis()
    axs[2,i].grid(linestyle = "--",c = "k", )
axs[2,0].set_ylabel( r"LW Heating Rate $\mathrm{[K/d]}$",fontsize = 24, font = font1)


axs[2,0].set_xlabel("Pressure (hPa)", fontsize = 28,  font = font1)
axs[2,1].set_xlabel("Pressure (hPa)", fontsize = 28,  font = font1)
axs[2,2].set_xlabel("Pressure (hPa)", fontsize = 28,  font = font1)
