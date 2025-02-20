import os
import argparse
import sys
sys.path.append("..")

import torch
import utils
import data_utils
import DnD_models
import scoring_function

import pandas as pd
import random
from tqdm import tqdm

from diffusers import StableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler

from transformers import BlipProcessor, BlipForConditionalGeneration

import warnings
warnings.filterwarnings('ignore')

BLIP_PATH = """BLIP_PATH"""
OPENAI_KEY = """OPENAI_KEY"""

parser = argparse.ArgumentParser(description='Describe-and-Dissect')

parser.add_argument("--clip_model", type=str, default="ViT-B/16", 
                    choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
                   help="Which CLIP-model to use")
parser.add_argument("--target_model", type=str, default="resnet50", 
                   help=""""Which model to dissect, supported options are pretrained imagenet models from
                        torchvision and resnet18_places.""")
parser.add_argument("--target_layer", type=str, default="layer4",
                    help="""Which layer neurons to describe. Follows the naming scheme of the Pytorch module used (eg.layer1)""")
parser.add_argument("--ids_to_check", type=str, default=None, help="Neuron ids to check, comma seperated (ex. '1,2,3')")
parser.add_argument("--d_probe", type=str, default="imagenet_broden", 
                    choices = ["imagenet_broden", "cifar100_val", "imagenet_val", "broden"])
parser.add_argument("--scoring_function", type=str, default="topk-compare", 
                    choices = ["mean","topk-sq-mean","topk-compare","image-products"], 
                    help="Scoring function")
parser.add_argument("--num_images_to_check", type=int, default=10, help="Number of images to check during attention cropping")
parser.add_argument("--num_crops_per_image", type=int, default=4, help="Max crops per activating image")
parser.add_argument("--batch_size", type=int, default=200, help="Batch size when running model")
parser.add_argument("--blip_batch_size", type=int, default=10, help="Batch size when running blip")
parser.add_argument("--BLIP_PATH", type=str, default=None, help="Path to pretrained BLIP")
parser.add_argument("--OPENAI_KEY", type=str, default=None, help="OpenAI key")
parser.add_argument("--device", type=str, default="cuda", help="Whether to use gpu/which gpu")
parser.add_argument("--results_dir", type=str, default="./experiments/exp_results", help="Folder to save results")
parser.add_argument("--saved_acts_dir", type=str, default="./experiments/saved_activations", help="Where to save layer activations")
parser.add_argument("--pool_mode", type=str, default="avg", help="Aggregation function for channels, max or avg")
parser.add_argument("--tag", type=str, default="", help="Clarification tag to distinguish between saved results")
args = parser.parse_args()

clip_name = args.clip_model
target_name = args.target_model
target_layer = args.target_layer
ids_to_check = args.ids_to_check
d_probe = args.d_probe
scoring_func = args.scoring_function
num_images_to_check = args.num_images_to_check
num_crops_per_image = args.num_crops_per_image
batch_size = args.batch_size
blip_batch_size = args.blip_batch_size
device = args.device
pool_mode = args.pool_mode
tag = args.tag
results_dir = args.results_dir
saved_acts_dir = args.saved_acts_dir

if __name__ == '__main__':
    if(ids_to_check == None):ids_to_check = [i for i in range(20)]
    else: 
        ids_to_check = ids_to_check.split(',')
        ids_to_check = [int(id) for id in ids_to_check]
        
    if BLIP_PATH is None: raise "Please provide path to BLIP model"
    if OPENAI_KEY is None: raise "Please provide OPENAI KEY, APIs can be created at https://platform.openai.com/"
    
    #### Setup ####
    print("Loading Models...")
    
    ## Load BLIP Model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device) 
    pretrained_dict = torch.load(BLIP_PATH)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()

    ## Load Stable Diffusion
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    generator = torch.Generator(device=device).manual_seed(0)
    pipe = pipe.to(device)
    
    # Setting up file/directory paths for saving results
    pot_column_names = ['Neuron ID'] + ['Concept {}'.format(i) for i in range(5)]
    all_concepts = pd.DataFrame(columns=pot_column_names)
    result_column_names = ['Neuron ID', 'Label 1', 'Label 2', 'Label 3']
    final_concepts = pd.DataFrame(columns=result_column_names)
    
    # Create results folder
    results_path = utils.create_layer_folder(results_dir = results_dir, base_dir = ".", target_name = target_name, 
                          d_probe = d_probe, layer = target_layer, tag = tag)

    ## Setup GPT
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + OPENAI_KEY,
    }   

    print("Loading Target Model Activations...",end="")
    ## Save activations / Load data
    target_save_name = utils.get_save_names(target_name = target_name,
                                      target_layer = target_layer, d_probe = d_probe,
                                      pool_mode=pool_mode, base_dir = '.', saved_acts_dir = saved_acts_dir)

    utils.save_activations(target_name = target_name, target_layers = [target_layer],
                           d_probe = d_probe, batch_size = batch_size, device = device,
                           pool_mode=pool_mode, base_dir = '.', saved_acts_dir = saved_acts_dir)

    target_feats = torch.load(target_save_name, map_location='cpu')

    pil_data = data_utils.get_data(d_probe)

    print("Done")

    #### Step 1 - Generate Attention Crops ####
    print("\nStep 1: Cropping Activating Images")
    top_vals, top_ids = torch.topk(target_feats, k=num_images_to_check, dim=0)

    all_imgs = []
    all_img_ids = {neuron_id:[] for neuron_id in ids_to_check}

    for t, orig_id in enumerate(tqdm(ids_to_check)):
        activating_images = []
        for i, top_id in enumerate(top_ids[:, orig_id]):
            im, label = pil_data[top_id]
            im = im.resize([375,375])
            all_img_ids[orig_id].append(len(all_imgs))
            all_imgs.append(im)
            activating_images.append(im)

        cropped_images = []
        if(target_layer != 'fc'):
            cropped_images = DnD_models.get_attention_crops(target_name, activating_images, orig_id, num_crops_per_image = 4, target_layers = [target_layer], device = device)

        for img in cropped_images:
            all_img_ids[orig_id].append(len(all_imgs))
            all_imgs.append(img)

            
    #### Step 2 - Generate Candidate Concepts ####
    print("\nStep 2: Candidate Concept Generation")
    
    target_feats = utils.get_target_activations(target_name, all_imgs, [target_layer])
    top_vals, top_ids = torch.sort(target_feats, dim=0, descending = True)
    comp_words = {orig_id : [] for orig_id in ids_to_check}
    top_images = {orig_id:[] for orig_id in ids_to_check}

    for neuron_num, orig_id in enumerate(tqdm(ids_to_check)):

        fig, images, top_images = utils.get_top_images(orig_id, top_ids, top_images, 
                                                       all_imgs, all_img_ids, num_images_to_check, 
                                                       blip_batch_size)
        utils.save_activating_fig(fig, results_path, orig_id)

        descriptions = DnD_models.blip_caption(model, processor, images, blip_batch_size, device, print_labels = False)
        for i, description in enumerate(descriptions):
            descriptions[i] = DnD_models.GPT_simplify(description, headers = headers)
            
        for i in range(5):
            cand_concept = DnD_models.GPT_model_single(descriptions, headers = headers)
            comp_words[orig_id].append(cand_concept)
            random.shuffle(descriptions)
        all_concepts.loc[len(all_concepts)] = [orig_id] + comp_words[orig_id]

    utils.save_potential_concepts(all_concepts, results_path)

    
    #### Step 3 – Fine-tune for Stable Diffusion ####
    print("\nStep 3: Best Concept Selection")

    """
    We adjust concepts with certain vague words to help SD generation
    """

    replace_set = ['design','designs','graphic','graphics']
    for orig_id in ids_to_check:
        comp_words[orig_id] = [concept.lower() for concept in comp_words[orig_id]]
        for i, word in enumerate(comp_words[orig_id]):
            if word[-1] == '.':
                comp_words[orig_id][i] = word[:-1]
            if word.split()[-1] in replace_set:
                new_concept = word + ' background'
                comp_words[orig_id].append(new_concept)
        comp_words[orig_id] = list(set(comp_words[orig_id]))
                
    pil_data = data_utils.get_data(d_probe)
    d_probe_len = len(pil_data)
    all_final_results = {neuron_id : [] for neuron_id in ids_to_check}

    num_images_per_prompt = 10
    top_K_param = 10
    beta_images_param = 5
    scoring_func = 'topk-sq-mean'

    sd_prompt = 'One realistic image of {}'
    num_inference_steps = 50

    for list_id, orig_id in enumerate(ids_to_check):
        print("Neuron {} ({}/{})".format(orig_id, list_id + 1, len(ids_to_check)))

        word_list = comp_words[orig_id]
        labels_to_check = len(word_list)

        add_im = {}
        add_im_id = {}
        all_sd_imgs = []

        for label_id in range(labels_to_check):
            pred_label = sd_prompt.format(word_list[label_id])
            add_im_id[label_id] = []

            add_im, add_im_id, all_sd_imgs = DnD_models.generate_sd_images(add_im, add_im_id, all_sd_imgs, 
                                                                      pred_label, label_id, pipe, generator,
                                                                      num_images_per_prompt, num_inference_steps)
            
        # Concept Scoring
        target_feats = utils.get_target_activations(target_name, all_sd_imgs, [target_layer])
        ranks, highest_activating = utils.rank_images(target_feats, orig_id, labels_to_check,
                                                     add_im_id, add_im, top_K_param)
        clip_weight = scoring_function.compare_images(top_images[orig_id], highest_activating, clip_name, 
                                                      device, target_name, top_K_param)
        top_avg_topk = scoring_function.get_score(ranks, mode = scoring_func, hyp_param = beta_images_param)

        top_avg_comb = []
        for i in range(len(clip_weight)):
            concept_rank = len(top_avg_topk) - scoring_function.find_by_last(top_avg_topk, clip_weight[i][1])
            weight = clip_weight[i][0]
            concept_score = concept_rank * weight
            top_avg_comb.append((concept_score, clip_weight[i][1]))
        top_avg_comb.sort(reverse = True)

        for label_num in range(3):
            if(label_num < len(top_avg_comb)):
                all_final_results[orig_id] += [word_list[top_avg_comb[label_num][1]]]
            else:
                all_final_results[orig_id] += [' ']
        final_concepts.loc[len(final_concepts)] = [orig_id] + all_final_results[orig_id]

    utils.save_final_results(final_concepts, results_path)
