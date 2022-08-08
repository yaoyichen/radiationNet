
import torch
from utils.config  import  climart_96_feature_mean, climart_96_feature_std
from utils.config import climart_96_target_mean, climart_96_target_std
import numpy as np

def climart_collate_fn(batch):
    feature_list = list()
    label_feature_list = list()
    auxiliary_result_list =  list()
    heatrate_label_list = list()

    single_feature = torch.zeros([82, 50])
    multiple_feature = torch.zeros([14, 50])

    for d in batch:
        single_feature = torch.tile(d[0]["globals"].unsqueeze(-1),[1,50])
        multiple_feature[:,1::] = torch.flip(d[0]["layers"].permute([1,0]),dims=(1,))
        multiple_feature[:,0] = multiple_feature[:,1]

        label_feature_tf = d[1][0:200].reshape( [4, 50] )
        label_feature_list.append(torch.flip(label_feature_tf,dims = (1,)))

        """
        压力单位不一样, 除以100改到 百帕。 
        """
        auxiliary_result_tf = torch.flip(d[0]["levels"][:,2:3].permute([1,0]),dims=(1,))/100.
        auxiliary_result_list.append(auxiliary_result_tf) 

        heat_rate_feature_tf = d[1][200:298].reshape( [2, 49] )
        heatrate_label_list.append(torch.flip(heat_rate_feature_tf,dims = (1,))) 

        feature_tf = torch.cat(
            [single_feature,
             multiple_feature], dim = (0))
        feature_list.append(feature_tf) 

    # feature_mean = torch.zeros([96]).unsqueeze(-1)
    # feature_std = torch.ones([96]).unsqueeze(-1)

    feature_list = (torch.stack(feature_list, dim = 0) - climart_96_feature_mean)/climart_96_feature_std
    label_feature_list = (torch.stack(label_feature_list, dim = 0) - climart_96_target_mean)/climart_96_target_std
    auxiliary_result_list = torch.stack(auxiliary_result_list, dim = 0)
    heatrate_label_list = torch.stack(heatrate_label_list, dim = 0)

    # return (feature_list, label_feature_list, auxiliary_result_list,heatrate_label_list)
    return (feature_list, label_feature_list, auxiliary_result_list)




# def fullyear_collate_fn(batch, keep_index_level_torch ):
    #  = torch.concat([torch.tensor([0]), keep_index_level_torch[0:-1]+1])
    # print(keep_index_level_torch.shape)
    # print(len(batch))
    # print(len(batch[0]))
    # print(batch[0].shape)
    # print("#"*20)
def fullyear_collate_fn_random(batch, ):
    keep_size  = 55 - np.random.choice(25)
    keep_index = np.concatenate([np.asarray([0]), np.random.choice(np.arange(1,56), size  = keep_size, replace = False), np.asarray([56]) ])
    keep_index.sort()
    keep_index_level = keep_index
    keep_index_level_torch = torch.tensor(keep_index_level) 


    keep_index_lay_torch = torch.concat((torch.tensor([0]), keep_index_level_torch[0:-1]+1 ))   

    new_feature_list, new_target_list, new_auxis_list= [], [], []
  
    for inner_batch in batch:
        # label_list.append(_label)
        # processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        # text_list.append(processed_text)
        (feature, targets, auxis) = inner_batch
        new_feature =  torch.index_select(feature,  -1,  keep_index_lay_torch)
        new_targets = torch.index_select( targets, -1, keep_index_level_torch)
        new_auxis = torch.index_select(auxis, -1, keep_index_level_torch) 

        # print(new_feature.shape,new_targets.shape,  new_auxis.shape ) 
        new_feature_list.append(new_feature.numpy()) 
        new_target_list.append(new_targets.numpy()) 
        new_auxis_list.append(new_auxis.numpy()) 

    # print(len(new_feature_list))
    new_feature_list_torch = torch.tensor(new_feature_list, dtype=torch.float32)
    new_target_list_torch = torch.tensor(new_target_list, dtype=torch.float32)
    new_auxis_list_torch = torch.tensor(new_auxis_list, dtype=torch.float32)
    
    
    return (new_feature_list_torch, new_target_list_torch, new_auxis_list_torch) 



def fullyear_collate_fn_random_half(batch ):
    if(np.random.rand() > 0.5):
        keep_size  = 55 - np.random.choice(25)
    else:
        keep_size = 55
    keep_index = np.concatenate([np.asarray([0]), np.random.choice(np.arange(1,56), size  = keep_size, replace = False), np.asarray([56]) ])
    keep_index.sort()
    keep_index_level = keep_index
    keep_index_level_torch = torch.tensor(keep_index_level) 


    keep_index_lay_torch = torch.concat((torch.tensor([0]), keep_index_level_torch[0:-1] + 1 ))   

    new_feature_list, new_target_list, new_auxis_list= [], [], []
  
    for inner_batch in batch:
        # label_list.append(_label)
        # processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        # text_list.append(processed_text)
        (feature, targets, auxis) = inner_batch
        new_feature =  torch.index_select(feature,  -1,  keep_index_lay_torch)
        new_targets = torch.index_select( targets, -1, keep_index_level_torch)
        new_auxis = torch.index_select(auxis, -1, keep_index_level_torch) 

        # print(new_feature.shape,new_targets.shape,  new_auxis.shape ) 
        new_feature_list.append(new_feature.numpy()) 
        new_target_list.append(new_targets.numpy()) 
        new_auxis_list.append(new_auxis.numpy()) 

    # print(len(new_feature_list))
    new_feature_list_torch = torch.tensor(new_feature_list, dtype=torch.float32)
    new_target_list_torch = torch.tensor(new_target_list, dtype=torch.float32)
    new_auxis_list_torch = torch.tensor(new_auxis_list, dtype=torch.float32)
    
    
    return (new_feature_list_torch, new_target_list_torch, new_auxis_list_torch) 


def fullyear_collate_fn_47(batch ):
    keep_index_level_torch = torch.tensor([ 0,  1,  2,  3,  4,   6,  7,  8,  9,  11, 12, 13, 14,  16,17, 18, 19,  21, 22, 23, 24, 26, 27, 28, 29,  31, 32, 33,
       34,  36, 37, 38, 39,  41, 42, 43, 44,  46, 47, 48, 49,
       51, 52, 53, 54, 55, 56])

    keep_index_lay_torch = torch.concat((torch.tensor([0]), keep_index_level_torch[0:-1]+1 ))   
    new_feature_list, new_target_list, new_auxis_list= [], [], []
  
    for inner_batch in batch:
        (feature, targets, auxis) = inner_batch
        new_feature =  torch.index_select(feature,  -1,  keep_index_lay_torch)
        new_targets = torch.index_select( targets, -1, keep_index_level_torch)
        new_auxis = torch.index_select(auxis, -1, keep_index_level_torch) 

        new_feature_list.append(new_feature.numpy()) 
        new_target_list.append(new_targets.numpy()) 
        new_auxis_list.append(new_auxis.numpy()) 

    new_feature_list_torch = torch.tensor(new_feature_list, dtype=torch.float32)
    new_target_list_torch = torch.tensor(new_target_list, dtype=torch.float32)
    new_auxis_list_torch = torch.tensor(new_auxis_list, dtype=torch.float32)
    
    return (new_feature_list_torch, new_target_list_torch, new_auxis_list_torch) 
