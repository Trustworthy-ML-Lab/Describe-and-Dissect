import torch
import clip
import data_utils
import utils

mode_list = ['topk-sq-mean', 'reg', 'mean', 'median', 'sq-mean', 'compare_images+topk_sq_mean', 'compare_images+mean']

def find_by_last(top_avg, comp_key):
    for i, pair in enumerate(top_avg):
        if pair[1] == comp_key:
            return i
    raise Exception("Invalid label id")

# mean of top-k values squared
def topk_sq_mean(ranks, k = 5):
    top_vals = []
    for label_id in ranks:
        sq_sum = 0
        for i in range(min(k, len(ranks[label_id]))):
            sq_sum += (ranks[label_id][i] ** 2)
        if len(ranks[label_id]) == 0:
            top_vals.append((len(ranks) + 1, label_id))
        else: 
            top_vals.append((sq_sum / min(k, len(ranks[label_id])), label_id))
    top_vals.sort()
    return top_vals

def mean(ranks):
    top_vals = []
    for label_id in ranks:
        if len(ranks[label_id]) == 0:
            top_vals.append((len(ranks) + 1, label_id))
        else: 
            top_vals.append((sum(ranks[label_id])/len(ranks[label_id]), label_id))
    top_vals.sort()
    return top_vals

def median(ranks):
    top_vals = []
    for label_id in ranks:
        top_vals.append((stats.median(ranks[label_id]), label_id))
    top_vals.sort()
    return top_vals

# mean of squared values
def sq_mean(ranks):
    top_vals = []
    for label_id in ranks:
        top_vals.append((sum([val**2 for val in ranks[label_id]])/len(ranks[label_id]), label_id))
    top_vals.sort()
    return top_vals

def compare_images(target_images, all_generated_images, clip_name, device, target_name, num_images = 5, model=None, preprocess=None):
    top_vals = []
    
    clip_model, clip_preprocess = clip.load(clip_name, device=device)
    if target_name == 'custom':
        target_model, target_preprocess = data_utils.get_target_model(target_name, device, model, preprocess)
    else:
        target_model, target_preprocess = data_utils.get_target_model(target_name, device)
        
    target_features = utils.get_clip_image_features(clip_model, clip_preprocess, target_images, device = device).float()
    target_features /= target_features.norm(dim=-1, keepdim=True)
    
    for label_id in all_generated_images:
        
        generated_images = all_generated_images[label_id]

        if len(generated_images) == 0:
            top_vals.append((-1, label_id))
            continue
            
        generated_features = utils.get_clip_image_features(clip_model, clip_preprocess, generated_images, device = device).float()
        generated_features /= generated_features.norm(dim=-1, keepdim=True)

        inner = (target_features @ generated_features.T)

        sim_idx = torch.mean(inner)
        
        top_vals.append((sim_idx, label_id))
    
    top_vals.sort(reverse=True)
    return top_vals
    
# get score of label
def get_score(ranks, mode = 'topk-sq-mean', hyp_param = None):
    if mode not in mode_list:
        raise Exception("Invalid score mode '{}'".format(mode))
    
    if mode == 'topk-sq-mean' or mode == 'compare_images+topk_sq_mean':
        return topk_sq_mean(ranks, hyp_param)
    if mode == 'mean' or mode == 'compare_images+mean':
        return mean(ranks)
    if mode == 'median':
        return median(ranks)
    if mode == 'sq-mean':
        return sq_mean(ranks)
