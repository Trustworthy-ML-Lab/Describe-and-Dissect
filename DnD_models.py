import sys
import os
sys.path.append(".")

import data_utils
import utils

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
from torchvision import transforms
from io import BytesIO
import base64
import cv2
import functools
import json
import requests

OPENAI_KEY = ''

set_headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + OPENAI_KEY,
}

def get_attention_crops(target_name, images, neuron_id, num_crops_per_image = 4, target_layers = ["layer4"], batch_size = 1000,
                            device = "cuda", pool_mode='avg', return_bounding_box = False, model=None, preprocess=None):
    
    if target_name == 'custom':
        target_model, preprocess = data_utils.get_target_model(target_name, device, model, preprocess)
    else:
        target_model, preprocess = data_utils.get_target_model(target_name, device)
    all_features = {target_layer:[] for target_layer in target_layers}
    hooks = {}
    
    transform = transforms.ToPILImage()
    
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(utils.get_mean_activation(all_features[target_layer]))".format(target_layer)
        hooks[target_layer] = eval(command)
        
    with torch.no_grad():
        for image in images:
            if "tile2vec" not in target_name:
                features = target_model(preprocess(image).unsqueeze(0).to(device))
            elif "custom" in target_name:
                features = target_model(preprocess(np.array(image)).unsqueeze(0).to(device))
            else:
                features = target_model.encode(image.unsqueeze(0).to(device))
    
    all_heatmaps = {target_layer:[] for target_layer in target_layers}
    for target_layer in target_layers:
        all_features[target_layer] = torch.cat(all_features[target_layer])
        hooks[target_layer].remove()
        
        for i in range(len(all_features[target_layer])):
            if "tile2vec" in target_name:
                images[i] = Image.fromarray(np.uint8(images[i].permute(1,2,0).numpy() * 255))
                
            if target_layer != 'fc' and target_layer != 'encoder':
                heatmap = transform(all_features[target_layer][i][neuron_id])
            elif target_layer == 'encoder':
                unflattend_img = torch.unflatten(all_features[target_layer][i],0,(16,16,3))
                unflattend_img = torch.permute(unflattend_img, (2,0,1))
                heatmap = ImageOps.grayscale(transform(unflattend_img))
            else:
                heatmap = transform(all_features[target_layer][i])
            heatmap = heatmap.resize([images[i].size[0],images[i].size[1]])
            heatmap = np.array(heatmap)
            all_heatmaps[target_layer].append(heatmap)
        if(return_bounding_box == True):
            utils.show_binarized_heatmap(all_heatmaps[target_layer])
    
    all_image_crops = [];
    all_bb_box = {layer : {i:[] for i in range(len(all_heatmaps[target_layer]))} for layer in target_layers}
    thresholded_feature_maps = []
    thresholds = []
    for target_layer in target_layers:
        for i, heatmap in enumerate(all_heatmaps[target_layer]): 
            thresh_val, thresh = cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            thresholded_feature_maps.append(thresh)
            thresholds.append(thresh_val)
           
            bb_cor = []
            # Find contours
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                box = (x, y, x + w, y + h)
                bb_cor.append(box)
                
            bb_cor = sorted(bb_cor, key=functools.cmp_to_key(utils.compare))
            
            cropped_bb = []
            for box in bb_cor:
                if len(cropped_bb) == num_crops_per_image:
                    break
                p = 0
                good_to_add = True
                while p < len(cropped_bb):
                    if utils.IoU(box, cropped_bb[p]) <= 0.5: 
                        p += 1
                    else:
                        good_to_add = False
                        break
                if good_to_add and utils.IoU(box,(0,0,heatmap.shape[0],heatmap.shape[1])) < 0.8:
                    cropped_img = images[i].crop(box)
                    cropped_img = cropped_img.resize([heatmap.shape[0],heatmap.shape[1]])
                    if "tile2vec" in target_name:
                        cropped_img = (torch.from_numpy(np.float64(np.array(cropped_img)) / 255).permute(2, 0, 1)).type(torch.FloatTensor)
                    all_image_crops.append(cropped_img)
                    cropped_bb.append(box)
            all_bb_box[target_layer][i] = cropped_bb
        if(return_bounding_box == True):
            utils.show_otsu_threshold(thresholded_feature_maps, thresholds)
            utils.show_bbox_on_heatmap(all_heatmaps[target_layer], all_bb_box[target_layer])  
    if return_bounding_box == True:
        return all_bb_box[target_layers[0]], all_image_crops
    else:
        del all_bb_box
        return all_image_crops

def blip_caption(model, processor, images, blip_batch_size, device, print_labels = False):
    gen_kwargs = {"max_new_tokens": 32, "min_length": 5}
    custom_dataset = utils.CustomDataset(images)
    dataloader = DataLoader(custom_dataset, batch_size=blip_batch_size, shuffle=True)
    
    # Generating BLIP captions
    descriptions = []
    
    for batch in tqdm(dataloader):
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
        out = model.generate(pixel_values=pixel_values, **gen_kwargs)
        captions = processor.batch_decode(out, skip_special_tokens=True)
        descriptions.extend(captions)
    
    for cap_count, cap in enumerate(descriptions):
        cap = cap.replace("output : ", "")
        cap = cap.replace("< | eos | >", "")
        cap = cap.replace("<", "")
        cap = cap.replace("|", "")
        cap = cap.replace("eos", "")
        cap = cap.replace(">", "")
        cap = cap.replace("'", "")
        cap = cap.replace('"', '')
        if print_labels: print("Image {} caption: {}".format(cap_count + 1, cap))
    
    return descriptions

def GPT_model_single(descriptions, headers = None, model='gpt-4-0125-preview', temperature=0.3):

    if headers is None: headers = set_headers
    
    content_user_1 = (
        "State one coherent and concise concept label that is 1-5 words long that can semantically summarize and represent most, not necessarily all, of the conceptual similarities in the following descriptions: "
        + "a purple background with a very soft texture."
        + ", "
        + "a brown background with a diagonal pattern of lines and lines."
        + ", "
        + "a white windmill with a red door and a red door in the middle of the picture."
        + ", "
        + "a beige background with a rough texture of linen."
        + ", "
        + "a beige background with a rough texture and a very soft texture."
    )
    content_assist_1 = (
        "multicolored textiles"
    )
    content_user_2 = (
        "State one coherent and concise concept label that is 1-5 words long that can semantically summarize and represent most, not necessarily all, of the conceptual similarities in the following descriptions: "
        + "a little girl is sitting in a red tractor with the word sofy on the front."
        + ", "
        + "a toy car sits on a red ottoman in a play room."
        + ", "
        + "a red dress with silver studs and a silver belt."
        + ", "
        + "a red chevrolet camaro is on display at a car show."
        + ", "
        + "a red spool of a cable with the word red on it."
    )
    content_assist_2 = (
        "red-themed scenes"
    )
    
    content_user = "Only state your answer without a period and quotation marks and do not simply repeat the descriptions. State one coherent and concise concept label that is 1-5 words long and can semantically summarize and represent most, not necessarily all, of the conceptual similarities in the following descriptions: "
    for i in range(len(descriptions)):
        content_user = content_user + descriptions[i]
        if descriptions[i] != descriptions[-1]:
            content_user = content_user + ', '
            
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content_user_1
            },
            {
                "role": "assistant",
                "content": content_assist_1
            },
            {
                "role": "user",
                "content": content_user_2
            },
            {
                "role": "assistant",
                "content": content_assist_2
            },
            {
                "role": "user",
                "content": content_user
            }
        ],
        "temperature": temperature
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_data = response.json()
    blip_pred = response_data['choices'][0]['message']['content']
    if ':' in blip_pred:
        pred_split = blip_pred.split(':')
        blip_pred = pred_split[1]

    refeed_content = "Only state your answer without a period and do not simply repeat the descriptions. State one coherent and concise concept label that is 1-5 words long that can semantically summarize and represent most, not necessarily all, of the conceptual similarities in the following descriptions: "

    if ',' in blip_pred:
        gpt_list = blip_pred.split(',')
        for i in range(len(gpt_list)):
            refeed_content = refeed_content + gpt_list[i]
            if gpt_list[i] != gpt_list[-1]:
                refeed_content = refeed_content + ', '
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": refeed_content
                }
            ],
            "temperature": temperature
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_data = response.json()
        blip_pred = response_data['choices'][0]['message']['content']
        if ':' in blip_pred:
            pred_split = blip_pred.split(':')
            blip_pred = pred_split[1]

    return blip_pred

def GPT_simplify(description, headers = None, model='gpt-4-0125-preview', temperature=0.3):

    if headers is None: headers = set_headers
    
    content_user_1 = (
        "State one coherent and concise concept label that is 1-5 words long that simplifies the following description: "
        + "a red background with a red background and a red background with a red background."
    )
    content_assist_1 = (
        "A red background"
    )
    
    content_user = "Only state your answer without a period and quotation marks. Do not number your answer. State one coherent and concise concept label that simplifies the following description and deletes any unnecessary details: "
    content_user += description
            
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content_user_1
            },
            {
                "role": "assistant",
                "content": content_assist_1
            },
            {
                "role": "user",
                "content": content_user
            }
        ],
        "temperature": temperature
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_data = response.json()
    blip_pred = response_data['choices'][0]['message']['content']
        
    return blip_pred

def GPT_caption(activating_imgs, headers = None, model="gpt-4o-mini", temperature=0.3):

    if headers is None: headers = set_headers
        
    base64_imgs = []
    for img in activating_imgs:
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        base64_imgs.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))


    img_content = []
    for img in base64_imgs:
        img_content.append({
                            "type": "image_url",
                            "image_url": 
                                {
                                    "url": f"data:image/jpeg;base64,{img}"
                                }
                        })
    payload = {
        "model": model,
        "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text",
                        "text": "State one coherent and concise concept label that is 1-5 words long that can semantically summarize and represent most, not necessarily all, of the conceptual similarities in the following images."
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type":"text",
                            "text": "State one coherent and concise concept label that is 1-5 words long that can semantically summarize and represent most, not necessarily all, of the conceptual similarities in the following images."
                        }
                    ] + img_content
                }
            ],
        "max_tokens": 300,
        "temperature": temperature
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_data = response.json()
    description = response_data['choices'][0]['message']['content']

    return description 

def generate_sd_images(add_im, add_im_id, all_sd_imgs, pred_label, label_id, pipe, 
                       generator, num_images_per_prompt, num_inference_steps = 50):
    
    
    image_set = pipe(pred_label, generator = generator, 
                     num_images_per_prompt = num_images_per_prompt, num_inference_steps = num_inference_steps)

    for i in range(num_images_per_prompt):
        # Rescale image
        image = image_set.images[i]
        image = image.resize([32,32])

        all_sd_imgs.append(image)
        new_idx = len(add_im)
        add_im[new_idx] = image # Add image to list
        add_im_id[label_id].append(new_idx) # map new image indices to corresponding label_id
        
    del image_set
    torch.cuda.empty_cache()
    return add_im, add_im_id, all_sd_imgs


# GPT Concept Generation for NAIP Classification

gpt_one_shot = {
    "layer1" : ("State one coherent and concise concept label 1-5 words long related to landscapes/satellite imagery that semantically summarizes and represents most, not necessarily all, of the conceptual similarities in the following descriptions. Focus on colors, textures, and patterns. After, print one most likely natural landscape described by the satellite imagery captions, in parenthesis, on the same line. Be confident and do not be vague: ",
                "a white background with a white background and a white background" + ","
                "a white and blue background with a pattern" + ","
                "a white and gray room with a white wall" + ","
                "a white and black cat sitting on a white surface" + ","
                "a white and black wallpaper with a white border",
                "The color white (clouds)"), 
    "layer2" : ("State one coherent and concise concept label 1-5 words long related to landscapes/satellite imagery that semantically summarizes and represents most, not necessarily all, of the conceptual similarities in the following descriptions. Focus on colors, textures, and patterns. After, print one most likely natural landscape described by the satellite imagery captions, in parenthesis, on the same line. Be confident and do not be vague: ",
                "a green field with a few small trees" + ","
                "a green and black background with a small pattern" + ","
                "a green and black background with a small amount of light" + ","
                "a green and white striped pattern" + ","
                "a field with a lot of green grass",
                "striped green field (farmland/pastures)"), 
    "layer3" : ("State one coherent and concise concept label 1-5 words long related to landscapes/satellite imagery that semantically summarizes and represents most, not necessarily all, of the conceptual similarities in the following descriptions. Focus on colors, textures, and patterns. After, print one most likely natural landscape described by the satellite imagery captions, in parenthesis, on the same line. Be confident and do not be vague: ",
                "a green and white striped pattern" + "," 
                "a green background with a white border" + ","
                "a black and white striped background" + ","
                "a green and white pattern with a small white dot" + "," 
                "a green and white striped pattern with a small white dots",
                "Green and white patterns/stripes (farmland with clouds)"),
    "layer4" : ("State one coherent and concise concept label 1-5 words long related to landscapes/satellite imagery that semantically summarizes and represents most, not necessarily all, of the conceptual similarities in the following descriptions. Focus on colors, textures, and patterns. After, print one most likely natural landscape described by the satellite imagery captions, in parenthesis, on the same line. Be confident and do not be vague: ",
                "a line of blue dots on a beige background" + ", "
                "a beige background with a small square shape" + ", "
                "a beige background with a small square shape" + ", "
                "a beige background with a small square pattern" + ", "
                "a brown paper background",
                "beige backgrounds (deserts)"),
    "layer5" : ("State one coherent and concise concept label 1-5 words long related to landscapes/satellite imagery that semantically summarizes and represents most, not necessarily all, of the conceptual similarities in the following descriptions. Focus on colors, textures, and patterns. After, print one most likely natural landscape described by the satellite imagery captions, in parenthesis, on the same line. Be confident and do not be vague: ",
                "a line of blue dots on a beige background" + ", "
                "a beige background with a small square shape" + ", "
                "a beige background with a small square shape" + ", "
                "a beige background with a small square pattern" + ", "
                "a brown paper background",
                "beige backgrounds (deserts)")
}

def GPT_model_naip(descriptions, headers = None, model='gpt-4-0125-preview', temperature=0.3, layer = "layer4"):

    if headers is None: headers = set_headers
        
    content_user_1 = (
       gpt_one_shot[layer][0] + gpt_one_shot[layer][1]
    )
    content_assist_1 = (
        gpt_one_shot[layer][2]
    )
    
    content_user = gpt_one_shot[layer][0]
    for i in range(len(descriptions)):
        content_user = content_user + descriptions[i]
        if descriptions[i] != descriptions[-1]:
            content_user = content_user + ', '
            
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content_user_1
            },
            {
                "role": "assistant",
                "content": content_assist_1
            },
            {
                "role": "user",
                "content": content_user
            }
        ],
        "temperature": temperature
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_data = response.json()
    blip_pred = response_data['choices'][0]['message']['content']
    if ':' in blip_pred:
        pred_split = blip_pred.split(':')
        blip_pred = pred_split[1]

    refeed_content = "Only state your answer without a period and do not simply repeat the descriptions. Only print on one line. " + gpt_one_shot[layer][0]

    if ',' in blip_pred:
        gpt_list = blip_pred.split(',')
        for i in range(len(gpt_list)):
            refeed_content = refeed_content + gpt_list[i]
            if gpt_list[i] != gpt_list[-1]:
                refeed_content = refeed_content + ', '
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": refeed_content
                }
            ],
            "temperature": temperature
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_data = response.json()
        blip_pred = response_data['choices'][0]['message']['content']
        if ':' in blip_pred:
            pred_split = blip_pred.split(':')
            blip_pred = pred_split[1]

    return blip_pred
