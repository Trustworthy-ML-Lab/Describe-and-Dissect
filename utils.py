import sys
import os
import math
import numpy as np
import torch
import clip
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import data_utils
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image

PM_SUFFIX = {"max":"_max", "avg":""}

# adjust layers as necessary depending on target model
layer_dic = {'layer1':1, 'layer2':2, 'layer3':3, 'layer4':4, 'fc':'fc', 'encoder':'encoder'}

def get_activation(outputs, mode):
    '''
    mode: how to pool activations: one of avg, max
    for fc or ViT neurons does no pooling
    '''
    if mode=='avg':
        def hook(model, input, output):
            if len(output.shape)==4: #CNN layers
                outputs.append(output.mean(dim=[2,3]).detach())
            elif len(output.shape)==3: #ViT
                outputs.append(output[:, 0].clone())
            elif len(output.shape)==2: #FC layers
                outputs.append(output.detach())
    elif mode=='max':
        def hook(model, input, output):
            if len(output.shape)==4: #CNN layers
                outputs.append(output.amax(dim=[2,3]).detach())
            elif len(output.shape)==3: #ViT
                outputs.append(output[:, 0].clone())
            elif len(output.shape)==2: #FC layers
                outputs.append(output.detach())
    return hook

def get_mean_activation(outputs):
    def hook(model, input, output):
        if len(output.shape)==4: #CNN layers
            outputs.append(output.detach())
        elif len(output.shape)==3: #ViT
            outputs.append(output[:, 0].clone())
        elif len(output.shape)==2: #FC layers
            outputs.append(output.detach())
    return hook
    
def get_save_names(target_name, target_layer, d_probe, pool_mode, base_dir, saved_acts_dir):

    saved_acts_dir = os.path.join(base_dir, saved_acts_dir)
    target_save_name = "{}/{}_{}_{}{}.pt".format(saved_acts_dir, d_probe, target_name, target_layer,
                                                 PM_SUFFIX[pool_mode])
    
    return target_save_name

def save_target_activations(target_model, dataset, save_name, target_layers = ["layer4"], batch_size = 1000,
                            device = "cuda", pool_mode='avg'):
    """
    save_name: save_file path, should include {} which will be formatted by layer names
    """
    _make_save_dir(save_name)
    save_names = {}    
    for target_layer in target_layers:
        save_names[target_layer] = save_name.format(target_layer)

    if _all_saved(save_names):
        return
    
    all_features = {target_layer:[] for target_layer in target_layers}
    
    hooks = {}
        
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(target_layer)
        hooks[target_layer] = eval(command)
    
    with torch.no_grad():
        if "tile2vec" not in save_name:
            for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
                features = target_model(images.to(device))
        else:
            for pix_data in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
                pix_data = torch.squeeze(pix_data, 1)
                features = target_model.encode(pix_data.to(device))
    
    for target_layer in target_layers:
        torch.save(torch.cat(all_features[target_layer]), save_names[target_layer])
        hooks[target_layer].remove()
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def IoU(box, check_box):
    x1,y1,x2,y2 = box;
    
    box_area = abs(x2-x1) * abs(y2-y1)
    
    cx1, cy1, cx2, cy2 = check_box
    check_area = abs(cx2-cx1) * abs(cy2-cy1)

    overlap_x = max(0,min(cx2, x2) - max(cx1, x1))
    overlap_y = max(0,min(cy2, y2) - max(cy1, y1))

    overlap = overlap_x * overlap_y
    union = box_area + check_area - overlap
    
    return overlap / union
        
def compare(box1, box2):
    x1,y1,x2,y2 = box1;
    box1_area = abs(x2-x1) * abs(y2-y1)
    cx1, cy1, cx2, cy2 = box2
    box2_area = abs(cx2-cx1) * abs(cy2-cy1)
    
    if box1_area < box2_area:
        return 1
    elif box1_area > box2_area:
        return -1
    else:
        return 0

def get_target_activations(target_name, images, target_layers = ["layer4"], batch_size = 1000,
                            device = "cuda", pool_mode='avg', model=None, preprocess=None):
    """
    gets target model activations without saving 
    """
    
    if target_name == 'custom':
        target_model, preprocess = data_utils.get_target_model(target_name, device, model, preprocess)
    else:
        target_model, preprocess = data_utils.get_target_model(target_name, device)
    
    all_features = {target_layer:[] for target_layer in target_layers}
    
    hooks = {}
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(target_layer)
        hooks[target_layer] = eval(command)
    
    with torch.no_grad():
        for image in images:
            if "tile2vec" not in target_name:
                features = target_model(preprocess(image).unsqueeze(0).to(device))
            elif "custom" in target_name:
                features = target_model(preprocess(np.array(image)).unsqueeze(0).to(device))
            else:
                features = target_model.encode(image.unsqueeze(0).to(device))
    
    for target_layer in target_layers:
        all_features[target_layer] = torch.cat(all_features[target_layer])
        hooks[target_layer].remove()
    return all_features[target_layers[0]]

def get_clip_image_features(model, preprocess, images, batch_size=1000, device = "cuda"):
    
    all_features = []
    with torch.no_grad():
        for image in images:
            features = model.encode_image(preprocess(image).unsqueeze(0).to(device))
            all_features.append(features)
    all_features = torch.cat(all_features)
    return all_features
    

def get_clip_text_features(model, text, batch_size=1000):
    """
    gets text features without saving, useful with dynamic concept sets
    """
    text_features = []
    with torch.no_grad():
        for i in range(math.ceil(len(text)/batch_size)):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    return text_features

def save_activations(target_name, target_layers, d_probe, 
                    batch_size, device, pool_mode, base_dir, saved_acts_dir, model=None, preprocess=None):
    if target_name == 'custom':
        target_model, target_preprocess = data_utils.get_target_model(target_name, device, model, preprocess)
    else:
        target_model, target_preprocess = data_utils.get_target_model(target_name, device)
    #setup data
    data_t = data_utils.get_data(d_probe, target_preprocess)
    
    target_save_name = get_save_names(target_name = target_name,
                                target_layer = '{}', d_probe = d_probe,
                                pool_mode=pool_mode, base_dir = base_dir, saved_acts_dir = saved_acts_dir)

    save_target_activations(target_model, data_t, target_save_name, target_layers,
                            batch_size, device, pool_mode)
    return

def get_cos_similarity(preds, gt, clip_model, mpnet_model, device="cuda", batch_size=200):
    """
    preds: predicted concepts, list of strings
    gt: correct concepts, list of strings
    """
    pred_tokens = clip.tokenize(preds).to(device)
    gt_tokens = clip.tokenize(gt).to(device)
    pred_embeds = []
    gt_embeds = []

    #print(preds)
    with torch.no_grad():
        for i in range(math.ceil(len(pred_tokens)/batch_size)):
            pred_embeds.append(clip_model.encode_text(pred_tokens[batch_size*i:batch_size*(i+1)]))
            gt_embeds.append(clip_model.encode_text(gt_tokens[batch_size*i:batch_size*(i+1)]))

        pred_embeds = torch.cat(pred_embeds, dim=0)
        pred_embeds /= pred_embeds.norm(dim=-1, keepdim=True)
        gt_embeds = torch.cat(gt_embeds, dim=0)
        gt_embeds /= gt_embeds.norm(dim=-1, keepdim=True)

    #l2_norm_pred = torch.norm(pred_embeds-gt_embeds, dim=1)
    cos_sim_clip = torch.sum(pred_embeds*gt_embeds, dim=1)

    gt_embeds = mpnet_model.encode([gt_x for gt_x in gt])
    pred_embeds = mpnet_model.encode(preds)
    cos_sim_mpnet = np.sum(pred_embeds*gt_embeds, axis=1)

    return float(torch.mean(cos_sim_clip)), float(np.mean(cos_sim_mpnet))

def find_by_last(top_avg, comp_key):
    for i, pair in enumerate(top_avg):
        if pair[1] == comp_key:
            return i
    raise Exception("Invalid label id")
    
def _all_saved(save_names):
    """
    save_names: {layer_name:save_path} dict
    Returns True if there is a file corresponding to each one of the values in save_names,
    else Returns False
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True

def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return

def get_top_images(orig_id, top_ids, top_images, all_imgs, all_img_ids, 
                   num_images_to_check = 10, blip_batch_size = 10, convert_from_np = False):
    
    fig = plt.figure(figsize=(15, 7))
    images = []
    
    for top_id in top_ids[:, orig_id]:
        # Save top num_images_to_check activating images
        if len(images) == min(blip_batch_size, num_images_to_check): 
            break
        # Check if image ID is valid (due to single model pass)
        if top_id not in all_img_ids[orig_id]:
            continue
        
        im = all_imgs[top_id]
        if convert_from_np: im = Image.fromarray(np.uint8(im.permute(1,2,0).numpy() * 255))
        images.append(im)
        top_images[orig_id].append(im)
        
        # Add image to plot
        fig.add_subplot(2, 5, len(images))
        im = im.resize([375,375])
        plt.imshow(im)
        plt.axis('off')
    
    plt.show()
    plt.close()

    return fig, images, top_images

def rank_images(target_feats, orig_id, labels_to_check, add_im_id, add_im, top_K_param):
    
    # Sort images based on activation
    top_vals, top_ids = torch.sort(target_feats, dim=0, descending = True)
    top_image_id = top_ids[:,orig_id]

    # Ranks: label_id -> (indicies of corresponding images in sorted target_feats)
    ranks = {label_id:[] for label_id in range(labels_to_check)}
    highest_activating = {label_id:[] for label_id in range(labels_to_check)}

    # Insert indices of image activations into ranks
    for label_id in range(labels_to_check):
        for i, img_id in enumerate(top_image_id):
            if img_id.item() in add_im_id[label_id]:
                ranks[label_id].append(i)
                if i < top_K_param:
                    highest_activating[label_id].append(add_im[img_id.item()])
        ranks[label_id].sort()
    
    return ranks, highest_activating
    
    
def image_embedding_mean(img_set, clip_name, device):
    clip_model, clip_preprocess = clip.load(clip_name, device=device)
    
    image_features = get_clip_image_features(clip_model, clip_preprocess, img_set, device = device)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    embedding_matrix = image_features @ image_features.T
    
    return torch.mean(embedding_matrix)

def text_embedding_mean(labels, clip_name, batch_size, device):
    clip_model, clip_preprocess = clip.load(clip_name, device=device)
    
    text_features = []
    labels = clip.tokenize(["{}".format(word) for word in labels]).to(device)
    with torch.no_grad():
        for i in range(math.ceil(len(labels)/batch_size)):
            text_features.append(clip_model.encode_text(labels[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    embedding_matrix = text_features @ text_features.T
    
    return torch.mean(embedding_matrix)
    
# Define Dataloader for BLIP
class CustomDataset(Dataset):
    def __init__(self, images):
        self.transform = transforms.ToTensor()
        self.data = [self.transform(image) for image in images]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        return image
    
    
def create_layer_folder(results_dir, base_dir, target_name, d_probe, layer, tag):

    sys.path.append(base_dir)
    save_folder_name = '{}_{}_layer{}_results_{}'.format(target_name, d_probe, layer_dic[layer], tag)
    file_path = os.path.join(base_dir, results_dir, save_folder_name)
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        os.makedirs(os.path.join(file_path, 'activating_image_figures'))
        os.makedirs(os.path.join(file_path, 'all_potential_concepts'))
        os.makedirs(os.path.join(file_path, 'all_result_concepts'))
        
    return file_path


def save_result_concepts_path(save_dir, target_name, d_probe, layer, tag):
    return save_dir + "/{}_{}_layer{}_results_{}/all_result_concepts/".format(target_name, d_probe, layer_dic[layer], tag)
    
def save_threshold_path(save_dir, target_name, d_probe, layer, tag):
    return save_dir + "/{}_{}_layer{}_results_{}/thresholding_results/".format(target_name, d_probe, layer_dic[layer], tag)

# Set up file/directory paths for saving results

def save_activating_fig(fig, results_path, neuron_id):
    fig.savefig(os.path.join(results_path, "activating_image_figures", 'neuron_{}_fig.png'.format(neuron_id)))
    
def save_potential_concepts(all_concepts, results_path):
    save_path = os.path.join(results_path, "all_potential_concepts", "all_potential_concepts.csv")
    all_concepts.to_csv(save_path, index=False)
    
def save_final_results(final_concepts, results_path):
    save_path = os.path.join(results_path, "all_result_concepts", "final_results.csv")
    final_concepts.to_csv(save_path, index=False)
