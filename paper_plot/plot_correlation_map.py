#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 18:53:48 2022

@author: yaoyichen
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import meshgrid
import time
import torch

def generate_correlation_contourmap(x_val, y_val, min_value, max_value, grid_number = 300):
    print(min_value, max_value, grid_number)
    x_vector = np.linspace(min_value, max_value, grid_number)
    spacing = x_vector[1] - x_vector[0]

    result_np = np.zeros( (len(x_vector) , len(x_vector)) )
    
    x_index = np.array( (x_val - min_value)//spacing, dtype = np.int32 )
    y_index = np.array( (y_val - min_value)//spacing, dtype = np.int32 )
    
    x_index_vector = x_index[(x_index >0) & (x_index< grid_number) & (y_index >0) & (y_index< grid_number)]
    y_index_vector = y_index[(x_index >0) & (x_index< grid_number) & (y_index >0) & (y_index< grid_number)]
    

    for x_index_, y_index_ in zip(x_index_vector,y_index_vector):
        result_np[x_index_, y_index_] += 1
    
    result_np = result_np/np.sum(result_np)
    result_np_plot = result_np[:]
    return x_vector, result_np_plot


"""
swfl   0,1300
lwfl   0,700
swhr  0,10
lwhr  -15,15
"""



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

tensor_pt_list = ['tensor_test_fc.pt', 'tensor_test_unet.pt', 
                  'tensor_test_lstm.pt','tensor_test_att.pt']


predicts_unnorm_list, targets_unnorm_list, \
        swhr_predict_list,  swhr_target_list, \
        lwhr_predict_list, lwhr_target_list = parssing_result(tensor_pt_list)


#%%
fig, axs = plt.subplots(4, 4, figsize=(30, 30))

for i in range(4):

    
    targets_unnorm = targets_unnorm_list[i]
    predicts_unnorm = predicts_unnorm_list[i]
    swhr_target = swhr_target_list[i]
    swhr_predict = swhr_predict_list[i]
    lwhr_target =   lwhr_target_list[i]
    lwhr_predict =  lwhr_predict_list[i]
    
    
    x_vector_swflux, result_np_plot_swflux = generate_correlation_contourmap( x_val = targets_unnorm[0:-1:1 , 0:2, 0:57:1].flatten(),
                                            y_val = predicts_unnorm[0:-1:1, 0:2, 0:57:1].flatten(),
                                            min_value = 0, max_value = 1300, grid_number = 300)  
    
    x_vector_lwflux, result_np_plot_lwflux = generate_correlation_contourmap(x_val = targets_unnorm[0:-1:1 , 2:4, 0:57:1].flatten(),
                                            y_val = predicts_unnorm[0:-1:1, 2:4, 0:57:1].flatten(),
                                            min_value = 0, max_value = 700, grid_number = 300) 
    
    x_vector_swhr, result_np_plot_swhr = generate_correlation_contourmap( x_val = swhr_target[0:-1:1 , 0, 0:56:1].flatten(), 
                                            y_val = swhr_predict[0:-1:1 , 0, 0:56:1].flatten(), 
                                            min_value = 0, max_value = 10, grid_number = 100)    
    
    start_time = time.time()
    x_vector_lwhr, result_np_plot_lwhr = generate_correlation_contourmap( x_val = lwhr_target[0:-1:1 , 0, 0:56:1].flatten(), 
                                                                        y_val = lwhr_predict[0:-1:1 , 0, 0:56:1].flatten(), 
                                                                        min_value = -15, max_value = 15, grid_number = 100) 
    
    
    result_np_plot_swhr[result_np_plot_swhr< 1.0*np.e**-12.5] = 0.0
    result_np_plot_lwhr[result_np_plot_lwhr< 1.0*np.e**-13.5] = 0.0
    
    
    
    
    def calcuate_correlation_coefficient(x,y):
        bias = np.mean(x - y)
        correlation_coefficient = np.corrcoef(x, y)[0,1]
        rmse = np.std(x - y)
        print(f"bias:{bias}, correlation:{correlation_coefficient},rmse:{rmse} ")
        return bias, correlation_coefficient, rmse
    
    
    swfl_bias,swfl_correlation, swfl_rmse =  calcuate_correlation_coefficient(np.array(targets_unnorm[ 0:-1:1 , 0:2, 0:57:1].flatten()),
                                     np.array(predicts_unnorm[0:-1:1, 0:2, 0:57:1].flatten()) )
    
    lwfl_bias,lwfl_correlation, lwfl_rmse =  calcuate_correlation_coefficient(np.array(targets_unnorm[0:-1:1 , 2:4, 0:57:1].flatten()),
                                     np.array(predicts_unnorm[0:-1:1, 2:4, 0:57:1].flatten()) )
    
    swhr_bias,swhr_correlation, swhr_rmse =  calcuate_correlation_coefficient(np.array(swhr_target[ 0:-1:1 , 0, 0:56:1].flatten()), 
                                     np.array(swhr_predict[0:-1:1 , 0, 0:56:1].flatten()))
    
    lwhr_bias,lwhr_correlation, lwhr_rmse=  calcuate_correlation_coefficient(np.array(lwhr_target[ 0:-1:1, 0, 0:57:1].flatten()),
                                     np.array(lwhr_predict[0:-1:1, 0, 0:57:1].flatten()))
    
    
    
    
    font1 = {'family'  : 'Times New Roman',
              'weight' : 'normal',
              'size'   :  23 }
    
    result_np_plot_log = np.log(result_np_plot_swflux)
    x_mesh, y_mesh = meshgrid(x_vector_swflux,x_vector_swflux) 
    fig0_contourf = axs[i,0].contourf(x_mesh,y_mesh, result_np_plot_log)
    # axs[i,0].text(-0.2*(x_vector_swflux.max() - x_vector_swflux.min()) + x_vector_swflux.min()
    #               ,0.92*(x_vector_swflux.max() - x_vector_swflux.min()) + x_vector_swflux.min(), 
    #               "(a)",font = font1,fontsize = 28)
    
    
    
    axs[i,0].text(0.05*(x_vector_swflux.max() - x_vector_swflux.min()) + x_vector_swflux.min()
                  ,0.9*(x_vector_swflux.max() - x_vector_swflux.min()) + x_vector_swflux.min(), 
                  f"R2:{swfl_correlation:.5f}",font = font1,fontsize = 24)
    
    axs[i,0].grid()
    axs[i,0].axis('equal')
    axs[i,0].set_aspect('equal', 'box')
    axs[i,0].set_xticks([0,200,400,600,800,1000,1200],fontsize=20)
    axs[3,0].set_title(r"SW Flux $\mathrm{[W/m^{-2}]}$",fontsize = 24, y=-0.15,font = font1)
    axs[i,0].plot([x_vector_swflux.min(), x_vector_swflux.max()],[x_vector_swflux.min(), x_vector_swflux.max()] ) 
    clb = fig.colorbar(fig0_contourf, ax=axs[i,0], ticks = [-3,-5,-7,-9,-11,-13,-15],
                       fraction = 0.04)
    clb.ax.set_title('Log(#)' ,font = font1,fontsize = 16)
    clb.set_ticklabels([r'$-3$', r'$-5$', r'$-7$',r'$-9$', r'$-11$', r'$-13$', r'$-15$'],fontsize = 16)
    
    
    result_np_plot_log = np.log(result_np_plot_lwflux)
    x_mesh, y_mesh = meshgrid(x_vector_lwflux,x_vector_lwflux) 
    fig1_contourf = axs[i,1].contourf(x_mesh,y_mesh, result_np_plot_log)
    # axs[i,1].text(-0.2*(x_vector_lwflux.max() - x_vector_lwflux.min()) + x_vector_lwflux.min()
    #               ,0.92*(x_vector_lwflux.max() - x_vector_lwflux.min()) + x_vector_lwflux.min(), 
    #               "(b)",font = font1,fontsize = 28)
    
    
    axs[i,1].text(0.05*(x_vector_lwflux.max() - x_vector_lwflux.min()) + x_vector_lwflux.min()
                  ,0.9*(x_vector_lwflux.max() - x_vector_lwflux.min()) + x_vector_lwflux.min(), 
                  f"$R^2$:{lwfl_correlation:.5f}",font = font1,fontsize = 24)
    
    axs[i,1].grid()
    axs[i,1].axis('equal')
    axs[i,1].set_aspect('equal', 'box')
    axs[i,1].set_xticks([0,100,200,300,400,500,600,700],fontsize=20)
    axs[3,1].set_title(r"LW Flux $\mathrm{[W/m^{-2}]}$",fontsize = 24, y= -0.15 ,font = font1)
    axs[i,1].plot([x_vector_lwflux.min(), x_vector_lwflux.max()],[x_vector_lwflux.min(), x_vector_lwflux.max()] ) 
    
    clb = fig.colorbar(fig1_contourf, ax=axs[i,1],ticks = [-2,-5,-8,-11,-14,-17],
                       fraction = 0.04)
    clb.ax.set_title('Log(#)' ,font = font1,fontsize = 16)
    clb.set_ticklabels([r'$-2$', r'$-5$', r'$-8$',r'$-11$', r'$-14$', r'$-17$'],fontsize = 13)
    
    
    result_np_plot_log = np.log(result_np_plot_swhr)
    x_mesh, y_mesh = meshgrid(x_vector_swhr,x_vector_swhr) 
    fig2_contourf = axs[i,2].contourf(x_mesh,y_mesh, result_np_plot_log,
                                     # cmap = "magma"
                                     )
    # axs[i,2].text(-0.2*(x_vector_swhr.max() - x_vector_swhr.min()) + x_vector_swhr.min()
    #               ,0.92*(x_vector_swhr.max() - x_vector_swhr.min()) + x_vector_swhr.min(), 
    #               "(c)",font = font1,fontsize = 28)
    
    
    axs[i,2].text(0.05*(x_vector_swhr.max() - x_vector_swhr.min()) + x_vector_swhr.min()
                  ,0.9*(x_vector_swhr.max() - x_vector_swhr.min()) + x_vector_swhr.min(), 
                  f"$R^2$:{swhr_correlation:.5f}",font = font1,fontsize = 24)
    
    axs[i,2].grid()
    axs[i,2].axis('equal')
    axs[i,2].set_aspect('equal', 'box')
    axs[i,2].set_xticks([0,2,4,6,8,10],fontsize=20)
    axs[3,2].set_title(r"SW Heating Rate $\mathrm{[K/d]}$", fontsize = 24, y=-0.15,font = font1)
    axs[i,2].plot([x_vector_swhr.min(), x_vector_swhr.max()],[x_vector_swhr.min(), x_vector_swhr.max()] ) 
    clb = fig.colorbar(fig2_contourf, ax=axs[i,2],ticks = [-3,-5,-7,-9,-11,-13],
                       fraction = 0.04)
    clb.ax.set_title('Log(#)',font = font1,fontsize = 16)
    clb.set_ticklabels([r'$-3$', r'$-5$', r'$-7$',r'$-9$', r'$-11$', r'$-13$'],fontsize = 16)
    
    
    result_np_plot_log = np.log(result_np_plot_lwhr)
    x_mesh, y_mesh = meshgrid(x_vector_lwhr,x_vector_lwhr) 
    fig3_contourf = axs[i,3].contourf(x_mesh,y_mesh, result_np_plot_log,
                                    #  cmap = "gist_heat"
                                    )
                                    
    # axs[i,3].text(-0.2*(x_vector_lwhr.max() - x_vector_lwhr.min()) + x_vector_lwhr.min()
    #               ,0.92*(x_vector_lwhr.max() - x_vector_lwhr.min()) + x_vector_lwhr.min(), 
    #               "(d)",font = font1,fontsize = 28)
    
    axs[i,3].text(0.05*(x_vector_lwhr.max() - x_vector_lwhr.min()) + x_vector_lwhr.min()
                  ,0.9*(x_vector_lwhr.max() - x_vector_lwhr.min()) + x_vector_lwhr.min(), 
                  f"$R^2$:{lwhr_correlation:.5f}",font = font1,fontsize = 24)
    
    axs[i,3].grid()
    axs[i,3].axis('equal')
    axs[i,3].set_aspect('equal', 'box')
    axs[i,3].set_xticks([-15,-10,-5,0,5,10,15],fontsize=20)
    axs[3,3].set_title(r"LW Heating Rate $\mathrm{[K/d]}$", fontsize = 24, y=-0.15,font = font1)
    axs[i,3].plot([x_vector_lwhr.min(), x_vector_lwhr.max()],[x_vector_lwhr.min(), x_vector_lwhr.max()] ) 
    
    clb = fig.colorbar(fig3_contourf, ax=axs[i,3],ticks = [-2,-4,-6,-8,-10,-12],
                       fraction = 0.04)
    clb.ax.set_title('Log(#)',font = font1,fontsize = 16)
    clb.set_ticklabels([r'$-2$', r'$-4$', r'$-6$',r'$-8$', r'$-10$', r'$-12$'],fontsize = 16)


axs[0,0].text(1300*0.4, 1300*1.1, "SW Flux", font = font1,fontsize = 28, rotation = "0")
axs[0,1].text(700*0.4 ,  700*1.1, "LW Flux", font = font1,fontsize = 28, rotation = "0")
axs[0,2].text(2,   10*1.1, "SW Heating Rate", font = font1,fontsize = 28, rotation = "0")
axs[0,3].text(-8,        30*1.1 - 15 , "LW Heating Rate", font = font1,fontsize = 28, rotation = "0")


axs[0,0].text(-300, 500, "FC", font = font1,fontsize = 28, rotation = "90")
axs[1,0].text(-300, 400, "U-Net", font = font1,fontsize = 28, rotation = "90")
axs[2,0].text(-300, 400, "Bi-LSTM", font = font1,fontsize = 28, rotation = "90")
axs[3,0].text(-300, 400, "Transformer", font = font1,fontsize = 28, rotation = "90")    
