#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 18:05:05 2022

@author: yaoyichen
"""

import  numpy as np
import torch


def interpolate_scatter_to_grid(x_grid_vector, y_grid_vector,
                                x_ref, y_ref,v_ref):
    """
    将经纬度的数据映射到 grid网格上
    x_ref,y_ref,v_ref 都是 [N, 1]的数据格式
    经纬度作图
    """
    from scipy.interpolate import griddata
    
    grid_y, grid_x = np.meshgrid(y_grid_vector, x_grid_vector, )


    left_pad_index = x_ref > 300
    left_pad_x  = x_ref[left_pad_index][:,np.newaxis] - 360
    left_pad_y  = y_ref[left_pad_index][:,np.newaxis]
    left_pad_v = v_ref[left_pad_index][:,np.newaxis]
    
    right_pad_index = x_ref < 60
    right_pad_x  = x_ref[right_pad_index][:,np.newaxis] + 360
    right_pad_y  = y_ref[right_pad_index][:,np.newaxis]
    right_pad_v = v_ref[right_pad_index][:,np.newaxis]
    
    scatter_x_full = np.concatenate([x_ref, left_pad_x, right_pad_x], axis =0 )
    scatter_y_full = np.concatenate([y_ref, left_pad_y, right_pad_y], axis =0 )
    scatter_v_full = np.concatenate([v_ref, left_pad_v, right_pad_v], axis =0 )
    
    points = np.concatenate( [scatter_x_full, scatter_y_full], axis = 1)
    values = scatter_v_full
    grid_v = griddata(points, values, (grid_x, grid_y), method='cubic')
    
    return grid_x, grid_y, grid_v




def generate_valid_lonlat(spacing = 1, 
                          test_point_number = 16392,
                          batch_divid_number = 10):
    
    block_len_cumsum = np.asarray([ 10220,  20426,  30714,  40976,  51259,  61470,  71746,  81951,
            92175, 102424, 112625, 122880, 133151, 143377, 153587, 163842])
    
    
    with open('lonlat_all.npy', 'rb') as f:
        lon_np_org = np.load(f)
        lat_np_org = np.load(f)
        
        
    start_index = 0
    lon_np_valid = []
    lat_np_valid = []
    for block_end in block_len_cumsum:
        lon_temp = lon_np_org[start_index:block_end:spacing]
        lat_temp = lat_np_org[start_index:block_end:spacing]
        lon_np_valid += list(lon_temp)
        lat_np_valid += list(lat_temp)
        start_index = block_end
        
    lon_np_valid = np.asarray(lon_np_valid)
    lat_np_valid = np.asarray(lat_np_valid)
    
    print(len(lon_np_valid), test_point_number)
    assert len(lon_np_valid) == test_point_number
    
    each_size = test_point_number//batch_divid_number
    
    result_lon_list = []
    result_lat_list = []
    
    for i in range(batch_divid_number):
        result_lon_list = result_lon_list + list(lon_np_valid[i*each_size : (i+1)*each_size -1])
        result_lat_list = result_lat_list + list(lat_np_valid[i*each_size : (i+1)*each_size-1])
        
    
    return np.asarray(result_lon_list), np.asarray(result_lat_list)

lon_np, lat_np = generate_valid_lonlat(spacing = 1, 
                          test_point_number = 163842,
                          batch_divid_number = 10)

lon_np, lat_np = generate_valid_lonlat(spacing = 1, 
                          test_point_number = 163842,
                          batch_divid_number = 10)


x_ref = lon_np[:,np.newaxis]
y_ref = lat_np[:,np.newaxis]



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

tensor_pt_list = ['tensor_slice_fc.pt', 'tensor_slice_unet208.pt', 
                  'tensor_slice_lstm.pt','tensor_slice_att_296.pt']



predicts_unnorm_list, targets_unnorm_list, \
        swhr_predict_list,  swhr_target_list, \
        lwhr_predict_list, lwhr_target_list = parssing_result(tensor_pt_list)



v_ref_true = np.asarray(targets_unnorm_list[3][:,1,0])[:,np.newaxis] - np.asarray(targets_unnorm_list[3][:,0,0])[:,np.newaxis]



x_grid_vector = np.arange( 0, 360, 0.5)
y_grid_vector = np.arange(-87, 87, 0.5)


grid_x, grid_y, grid_v_true = interpolate_scatter_to_grid(x_grid_vector, y_grid_vector, x_ref,y_ref,v_ref_true)
# grid_x, grid_y, grid_v_predict = interpolate_scatter_to_grid(x_grid_vector, y_grid_vector, x_ref,y_ref,v_ref_predict)
# grid_x, grid_y, grid_v_diff = interpolate_scatter_to_grid(x_grid_vector, y_grid_vector, x_ref,y_ref,v_ref_diff)
grid_v_true = grid_v_true.squeeze()
# grid_v_predict = grid_v_predict.squeeze()
# grid_v_diff = grid_v_diff.squeeze()

#%%
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from numpy import linspace
from numpy import meshgrid


fig, axs = plt.subplots(5,2, figsize=(16, 25),sharey=True)
axs[0,0].axis('off')
axs[0,1].axis('off')
axs[1,0].axis('off')
axs[1,1].axis('off')
axs[2,0].axis('off')
axs[2,1].axis('off')
axs[3,0].axis('off')
axs[3,1].axis('off')
axs[4,0].axis('off')
axs[4,1].axis('off')

ax = fig.add_subplot(5, 2, 1)

m = Basemap(projection='merc', \
            llcrnrlat=-75, urcrnrlat=75, \
            llcrnrlon=0, urcrnrlon=360, \
            lat_ts=20, \
            resolution='i' )
    
font1 = {'family'  : 'Times New Roman',
          'weight' : 'normal',
          'size'   :  23 }


basemap_x, basemap_y = m(np.transpose(grid_x,[1,0]),
                           np.transpose(grid_y,[1,0]))


# v_min = min(grid_v_true.min() ,grid_v_predict.min())
# v_max = max(grid_v_true.max() ,grid_v_predict.max())

v_min = grid_v_true.min() -50
v_max = grid_v_true.max() 

clev = np.linspace(v_min, v_max,100) #Adjust the .001 to get finer gradient

cs1 = m.contourf(basemap_x, 
           basemap_y, 
           np.transpose(grid_v_true,[1,0]), clev,vmax = v_max, vmin = v_min,
            cmap= "viridis"
           ) 

m.drawcoastlines(color='black', linewidth= 0.5)  # add coastlines


clb = fig.colorbar(cs1, orientation='vertical', 
                    location = "right",
                    label = "Flux [W/m2]", 
                    ticklocation = "top"    ,
                    ticks = [   0,  100,  200,  300,  400,  500,  600,  700,  800,  900, 1000],
                    fraction = 0.023
                )

# axs[0,0].text(0.02 ,0.88,"(a)",font = font1,fontsize = 28)
# clb.set_ticklabels([r'$0$', r'$100$', r'$200$',r'$300$', r'$400$', r'$500$', r'$600$', r'$700$',r'$800$',r'$900$'],fontsize = 13)
clb.set_label(r"Flux [$\mathrm{W \cdot m^{-2}}$]", fontsize = 14, labelpad=-10, 
                                      y=1.1, rotation=0, font = font1)
                  

# plt.xlabel("longitude" , fontsize = 16, font = font1)
plt.ylabel("latitude" , fontsize = 20, font = font1)


for i in range(4):
    v_ref_predict = np.asarray(predicts_unnorm_list[i][:,1,0])[:,np.newaxis] - np.asarray(predicts_unnorm_list[i][:,0,0])[:,np.newaxis]

    grid_x, grid_y, grid_v_predict = interpolate_scatter_to_grid(x_grid_vector, y_grid_vector, x_ref,y_ref,v_ref_predict)
    grid_v_predict = grid_v_predict.squeeze()
    
    ax = fig.add_subplot(5,2, 2*(i+1)+ 1)
    
    m = Basemap(projection='merc', \
                llcrnrlat = -75, urcrnrlat = 75, \
                llcrnrlon = 0, urcrnrlon = 360, \
                lat_ts = 20, \
                resolution='i')
        
    cs1 = m.contourf(basemap_x, 
               basemap_y, 
               np.transpose(grid_v_predict, [1,0]),
               clev, vmax = v_max, vmin = v_min,
               cmap= "viridis"
               ) 
    
    m.drawcoastlines(color='black', linewidth= 0.5)  # add coastlines
    
    clb = fig.colorbar(cs1, orientation='vertical', 
                        location = "right",
                        label = "Flux [W/m2]", 
                        ticklocation = "top"    ,
                        ticks =[   0,  100,  200,  300,  400,  500,  600,  700,  800,  900, 1000],
                        fraction = 0.023
                        )
    
    # axs[0,0].text(0.02 ,0.88,"(a)",font = font1,fontsize = 28)
    # clb.set_ticklabels([r'$0$', r'$100$', r'$200$',r'$300$', r'$400$', r'$500$', r'$600$', r'$700$',r'$800$',r'$900$'],fontsize = 13)
    clb.set_label(r"Flux [$\mathrm{W \cdot m^{-2}}$]", fontsize = 14, labelpad=-10, 
                                          y=1.1, rotation=0, font = font1)
                      
    
    # plt.xlabel("longitude" , fontsize = 16, font = font1)
    plt.ylabel("latitude" , fontsize = 20, font = font1)

plt.xlabel("longitude" , fontsize = 20, font = font1)
    # axs[2,0].text(0.02,0.88,"(e)",font = font1,fontsize = 28)


for i in range(4):
    v_ref_predict = np.asarray(predicts_unnorm_list[i][:,1,0])[:,np.newaxis] - np.asarray(predicts_unnorm_list[i][:,0,0])[:,np.newaxis]
    v_ref_true  =    np.asarray(targets_unnorm_list[i][:,1,0])[:,np.newaxis] - np.asarray(targets_unnorm_list[i][:,0,0])[:,np.newaxis] 
    v_ref_diff = v_ref_predict - v_ref_true

    grid_x, grid_y, grid_ref_diff = interpolate_scatter_to_grid(x_grid_vector, y_grid_vector, x_ref,y_ref,v_ref_diff)
    grid_ref_diff = grid_ref_diff.squeeze()
    
    ax = fig.add_subplot(5,2, 2*(i+1)+ 2)
    
    m = Basemap(projection='merc', \
                llcrnrlat = -75, urcrnrlat = 75, \
                llcrnrlon = 0, urcrnrlon = 360, \
                lat_ts = 20, \
                resolution='i')
    
    
    v_max = 12
    v_min = -12
    grid_ref_diff[grid_ref_diff > v_max ] = v_max
    grid_ref_diff[grid_ref_diff < v_min ] = v_min
    
    clev = np.linspace(v_min, v_max,100) 
    cs1 = m.contourf(basemap_x, 
               basemap_y, 
               np.transpose(grid_ref_diff, [1,0]),
               clev, vmax = v_max, vmin = v_min,
               cmap= "viridis"
               ) 
    
    m.drawcoastlines(color='black', linewidth= 0.5)  # add coastlines
    
    clb = fig.colorbar(cs1, orientation='vertical', 
                        location = "right",
                        label = "Flux [W/m2]", 
                        ticklocation = "top"    ,
                        ticks = [-12,-9,-6,-3,0,3,6, 9,12],
                        fraction = 0.023
                        )
    

    # axs[0,0].text(0.02 ,0.88,"(a)",font = font1,fontsize = 28)
    # clb.set_ticklabels([r'$0$', r'$100$', r'$200$',r'$300$', r'$400$', r'$500$', r'$600$', r'$700$',r'$800$',r'$900$'],fontsize = 13)
    clb.set_ticklabels([r'$-12$', r'$-9$', r'$-6$',r'$-3$', r'$0$', r'$3$', r'$6$', r'$9$',r'$12$'],fontsize = 13)
    clb.set_label(r"Flux [$\mathrm{W \cdot m^{-2}}$]", fontsize = 14, labelpad=-10, 
                                          y=1.1, rotation=0, font = font1)
                      
    
    # plt.xlabel("longitude" , fontsize = 16, font = font1)
    plt.ylabel("latitude" , fontsize = 20, font = font1)
    
plt.xlabel("longitude" , fontsize = 20, font = font1)



axs[0,0].text(-0.1,  0.6,"Ground Truth", size=28,
                         verticalalignment='center', rotation=90,
                         font = font1)
axs[1,0].text(-0.1,  0.6,"FC", size=28,
                         verticalalignment='center', rotation=90,
                         font = font1)
axs[2,0].text(-0.1,  0.6,"U-Net", size=28,
                         verticalalignment='center', rotation=90,
                         font = font1)
axs[3,0].text(-0.1,  0.6,"Bi-LSTM", size=28,
                         verticalalignment='center', rotation=90,
                         font = font1)
axs[4,0].text(-0.1, 0.6,"Transformer", size=28,
                         verticalalignment='center', rotation=90,
                         font = font1)


axs[1,0].text(0.4,  1.06,"Prediction", size=28,
                         font = font1)
axs[1,1].text(0.4,  1.06,"Error", size=28,
                         font = font1)

    
