import requests
import PIL.Image
from io import BytesIO
import time
import os
import numpy as np
from math import floor, ceil, sqrt
from tqdm.notebook import tqdm

# Returns an image from an URL as a PIL image
def img_from_url(url, max_tries=10):
    tries = 0
    while tries < max_tries:
        try:
            response = requests.get(url, timeout=30)
            img_bytes = BytesIO(response.content)
            img = PIL.Image.open(img_bytes).convert("RGB")
            return img
        except:
            tries += 1
        time.sleep(1)

def load_img(path):
    return PIL.Image.open(path).convert("RGB")

def new_dir(folder):
    os.makedirs(folder, exist_ok=True)
    return folder

def make_palette(colors, height=50, width=300):
    assert colors.shape[1] == 3 # Assume RGB
    palette = np.zeros((height, width, 3), np.uint8)
    steps = width/len(colors)
    for i, color in enumerate(colors): 
        palette[:, int(i*steps):(int((i+1)*steps)), :] = color
    return palette

def plot_imgs(files, s):
    d = floor(sqrt(len(files)))
    dx = dy = d
    if d**2 < len(files): # If there are images that do not fit into the square matrix...
        dy+=ceil((len(files)-d**2)/dx) # ...extend rows to accomodate these images

    # Set up canvas
    canvas = np.ones((dy*s, dx*s, 3), dtype=np.uint8) * 255

    # Plot images
    with tqdm(total=d**2) as pbar:
        for y in range(dy):            
            for x in range(dx):
                #print(y,x)
                pos = y*dx+x
                if pos==(len(files)):
                    break
                img = PIL.Image.open(files[pos]).convert("RGB")
                img.thumbnail((s, s))
                npimg = np.array(img)
                canvas[y*s:y*s+npimg.shape[0], x*s:x*s+npimg.shape[1], :] = npimg
                pbar.update(1)
                
    return canvas

def get_all_files(folder, ext=None):
    all = []
    for r, ds, fs in os.walk(folder):
        for f in fs:
            if ext is None or f.endswith(ext):
                all.append(os.path.join(r, f))
    return all

def sort_dict(d):
    return dict(sorted(d.items(), key=lambda x: x[1]))

def from_device(tensor):
    return tensor.detach().cpu().numpy()

def plot_imgs_features(files, s, features):
    d = ceil(sqrt(len(files)))
    
    # Set up canvas
    # As images are anchored in the upper left corner increase size by 1
    canvas = np.ones(((d+1)*s, (d+1)*s, 3), dtype=np.uint8) * 255

    for i, file_ in enumerate(tqdm(files)):
        img = PIL.Image.open(file_).convert("RGB")
        img.thumbnail((s, s))
        npimg = np.array(img)
        y = int(features[i,0] * (d*s))
        x = int(features[i,1] * (d*s))  
        canvas[y:y+npimg.shape[0], x:x+npimg.shape[1], :] = npimg
                
    return canvas