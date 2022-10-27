#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:53:48 2022

@author: yaoyichen
"""

# 做一个分布比较的图
# FC, CNN, LSTM, ATT
# 短波 Flux
# 长波 Flux
# 短波HR 
# 长波HR

import  numpy as np
import torch
import matplotlib.pyplot as plt


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

# h_level_vector = h_level_vector/np.max(h_level_vector) 
# h_lay_vector   =  h_lay_vector/np.max(h_lay_vector) 
    
    
font1 = {'family'  : 'Times New Roman',
          'weight' : 'normal',
          'size'   :  23 }

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




#%%

tensor_pt_list = ['tensor_fc.pt', 'tensor_unet.pt', 
                  'tensor_lstm.pt','tensor_att.pt']

predicts_unnorm_list, targets_unnorm_list, \
        swhr_predict_list,  swhr_target_list, \
        lwhr_predict_list, lwhr_target_list = parssing_result(tensor_pt_list)

for i in range(7) :
    # result = lwhr_predict_list[i] - lwhr_target_list[i] + swhr_predict_list[i] - swhr_target_list[i]
    result = (predicts_unnorm_list[i][:,1,:] - predicts_unnorm_list[i][:,0,:] + predicts_unnorm_list[i][:,3,:] - predicts_unnorm_list[i][:,2,:]) \
        - (targets_unnorm_list[i][:,1,:] - targets_unnorm_list[i][:,0,:] + targets_unnorm_list[i][:,3,:] - targets_unnorm_list[i][:,2,:])
    
    result_mae = result.mean(axis = 0)
    # result_mae = result.mean()
    print(result_mae[-1])
#%%

# tensor_pt_list = ['tensor_fc.pt', 'tensor_unet208.pt', 
#                   'tensor_lstm.pt','tensor_att_296.pt',
#                   'tensor_gru.pt','tensor_fno.pt']

tensor_pt_list = ['tensor_fc.pt', 'tensor_unet.pt', 
                  'tensor_lstm.pt','tensor_att.pt']

predicts_unnorm_list, targets_unnorm_list, \
        swhr_predict_list,  swhr_target_list, \
        lwhr_predict_list, lwhr_target_list = parssing_result(tensor_pt_list)


fig, axs = plt.subplots(2,4,figsize=(23, 16))

title_list = ["FC","U-Net", "Bi-LSTM","Transformer"]

for i in range(4) :
    
    
    result = swhr_predict_list[i] - swhr_target_list[i]
    result = result.squeeze()
    # result = predicts_unnorm_list[i][:,1,:] - targets_unnorm_list[i][:,1,:]
    
    result_bias = result.mean(axis = 0)
    result_mae = np.abs(result).mean(axis = 0)
    result_std = result.std(axis = 0)

    height_vector = h_lay_vector
    
    axs[0,i].plot(result_mae.flatten(),  height_vector, "-", c = "b",label = "mean abs. error")
    axs[0,i].plot(result_bias.flatten(), height_vector, '--', c = "b",label = "mean error")
    axs[0,i].fill_betweenx(height_vector, result_bias.flatten() - result_std.flatten(), 
                      result_bias.flatten() + result_std.flatten(), 
                      alpha = 0.3, color = "b",label = "mean std")
    
    # axs[i].legend(fontsize = 20)
    axs[0,i].invert_yaxis()
    axs[0,i].set_xlim([-0.5, 0.5])
    axs[0,i].set_ylim([1000,0.0])
    # axs[0,i].set_xlabel( "Heating rate (K d-1)", fontsize = 28, font = font1)
    axs[0,i].set_ylabel("Pressure (hPa)"  , fontsize = 24, font = font1)
    # axs[0,i].grid("--",c = "k", )
    axs[0,i].grid(linestyle = "--",c = "k", )
    axs[0,i].set_title(title_list[i], fontsize = 28, font = font1, y=1.05 )
    
    
    
    
    
    result = lwhr_predict_list[i] - lwhr_target_list[i]
    result = result.squeeze()
    # result = predicts_unnorm_list[i][:,1,:] - targets_unnorm_list[i][:,1,:]
    
    result_bias = result.mean(axis = 0)
    result_mae = np.abs(result).mean(axis = 0)
    result_std = result.std(axis = 0)

    height_vector = h_lay_vector
    
    axs[1,i].plot(result_mae.flatten(),  height_vector, "-", c = "b",label = "mean abs. error")
    axs[1,i].plot(result_bias.flatten(), height_vector, '--', c = "b",label = "mean error")
    axs[1,i].fill_betweenx(height_vector, result_bias.flatten() - result_std.flatten(), 
                      result_bias.flatten() + result_std.flatten(), 
                      alpha = 0.3, color = "b",label = "mean std")
    
    # axs[i].legend(fontsize = 20)
    axs[1,i].invert_yaxis()
    axs[1,i].set_xlim([-1.0, 1.0])
    axs[1,i].set_ylim([1000,0.0])
    axs[1,i].set_xlabel( "Heating rate (K d-1)", fontsize = 28, font = font1)
    axs[1,i].set_ylabel("Pressure (hPa)"  , fontsize = 24, font = font1)
    axs[1,i].grid(linestyle = "--",c = "k", )
    


axs[0,0].text(-0.5*1.8, 1000* 0.5,"SW Heating Rate", size=28,
                         verticalalignment='center', rotation=90,
                         font = font1)

axs[1,0].text(-1*1.8,   1000*0.5,"LW Heating Rate", size=28,
                         verticalalignment='center', rotation=90,
                         font = font1)

axs[0,0].legend(fontsize = 24 , loc = "upper left")
#%%

fig, axs = plt.subplots(2,4,figsize=(23, 16))

title_list = ["FC","U-Net", "Bi-LSTM","Transformer"]

for i in range( 4) :
    

    # result = predicts_unnorm_list[i][:,2,:] - targets_unnorm_list[i][:,2,:] + \
    result = torch.concat( [ predicts_unnorm_list[i][:,0,:] - targets_unnorm_list[i][:,0,:] ,
                             predicts_unnorm_list[i][:,1,:] - targets_unnorm_list[i][:,1,:] ],
                          )
    
    result_bias = result.mean(axis = 0)
    result_mae = np.abs(result).mean(axis = 0)
    result_std = result.std(axis = 0)

    height_vector = h_level_vector
    
    axs[0,i].plot(result_mae.flatten(),  height_vector, "-", c = "b",label = "mean abs. error")
    axs[0,i].plot(result_bias.flatten(), height_vector, '--', c = "b",label = "mean error")
    axs[0,i].fill_betweenx(height_vector, result_bias.flatten() - result_std.flatten(), 
                      result_bias.flatten() + result_std.flatten(), 
                      alpha = 0.3, color = "b",label = "mean std")
    
    # axs[i].legend(fontsize = 20)
    axs[0,i].invert_yaxis()
    axs[0,i].set_xlim([-30.0, 30.0])
    axs[0,i].set_ylim([1000,0.0])
    # axs[0,i].set_xlabel( r"Flux [$\mathrm{W \cdot m^{-2}}$]", fontsize = 28, font = font1)

    
    axs[0,i].grid(linestyle = "--",c = "k", )
    axs[0,i].set_title(title_list[i], fontsize = 28, font = font1, y=1.05 )
    

    
    result = torch.concat( [ predicts_unnorm_list[i][:,2,:] - targets_unnorm_list[i][:,2,:] ,
                             predicts_unnorm_list[i][:,3,:] - targets_unnorm_list[i][:,3,:] ],
                          )
    
    result_bias = result.mean(axis = 0)
    result_mae = np.abs(result).mean(axis = 0)
    result_std = result.std(axis = 0)

    height_vector = h_level_vector
    
    axs[1,i].plot(result_mae.flatten(),  height_vector, "-", c = "b",label = "mean abs. error")
    axs[1,i].plot(result_bias.flatten(), height_vector, '--', c = "b",label = "mean error")
    axs[1,i].fill_betweenx(height_vector, result_bias.flatten() - result_std.flatten(), 
                      result_bias.flatten() + result_std.flatten(), 
                      alpha = 0.3, color = "b",label = "mean std")
    
    # axs[i].legend(fontsize = 20)
    axs[1,i].invert_yaxis()
    axs[1,i].set_xlim([-20, 20])
    axs[1,i].set_ylim([1000.,0.0])
    axs[1,i].set_xlabel( r"Flux [$\mathrm{W \cdot m^{-2}}$]", fontsize = 28, font = font1)
    axs[1,i].set_ylabel("Pressure (hPa)"  , fontsize = 20, font = font1)
    axs[1,i].grid(linestyle = "--",c = "k", )
    


axs[0,0].set_ylabel("Pressure (hPa)"  , fontsize = 24, font = font1)
axs[1,0].set_ylabel("Pressure (hPa)"  , fontsize = 24, font = font1)

axs[0,0].text(-30*1.8,  1000*0.5,"SW Flux", size=28,
                         verticalalignment='center', rotation=90,
                         font = font1)

axs[1,0].text(-20*1.8,  1000*0.5,"LW Flux", size=28,
                         verticalalignment='center', rotation=90,
                         font = font1)
# axs[0,0].legend(fontsize = 24, loc = "upper left")
