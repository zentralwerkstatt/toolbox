import requests
import PIL.Image
from io import BytesIO
import time
import os
import numpy as np
from math import floor, ceil, sqrt
from tqdm.notebook import tqdm
import numpy as np
import seaborn as sns
import clip
import torch as t
from umap import UMAP
from IPython.display import display

def set_cuda():
    device = "cuda" if t.cuda.is_available() else "cpu"
    return device

device = set_cuda()
model, transforms = clip.load("ViT-B/32", device=device)

def from_device(tensor):
    return tensor.detach().cpu().numpy()

def CLIP_img(img):
    with t.no_grad():
        input_ = transforms(img).unsqueeze(0).to(device)
        output = model.encode_image(input_)
        output /= output.norm(dim=-1, keepdim=True)
        return from_device(output).astype(np.float32).flatten()

def flatten_img(img, s):
    img = img.resize((s,s))
    img_np = np.array(img)
    return img_np.flatten()

def reduce_features(features):
    reducer = UMAP()
    reduced_features = reducer.fit_transform(features)
    reduced_features -= reduced_features.min(axis=0)
    reduced_features /= reduced_features.max(axis=0)
    return reduced_features

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

def wget(url, path):
    data = requests.get(url).content
    with open(path, "wb") as f:
        f.write(data)

def load_img(path):
    return PIL.Image.open(path).convert("RGB")

def show_img(img):
    display(img)

def new_dir(folder):
    os.makedirs(folder, exist_ok=True)
    return folder

def wgets(urls, folder):
    new_dir(folder)
    for i, url in enumerate(tqdm(urls)):
        ext = url[-3]
        wget(url, os.path.join(folder, f"{i:010d}.{ext}"))

def make_palette(colors, height=50, width=300):
    palette = np.zeros((height, width, 3), np.uint8)
    steps = width/len(colors)
    for i, color in enumerate(colors): 
        palette[:, int(i*steps):(int((i+1)*steps)), :] = color
    return PIL.Image.fromarray(palette)

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

def color_img(s, c):
    img = np.zeros((s, s, 3), dtype=np.uint8)
    img[:,:,0] = c[0]
    img[:,:,1] = c[1]
    img[:,:,2] = c[2]
    return PIL.Image.fromarray(img)

def random_palette(n):
    rgb_palette = {}
    for i, color_rgb in enumerate(sns.color_palette(None, n)):
        rgb_palette[i] = (int(color_rgb[0]*255), int(color_rgb[1]*255), int(color_rgb[2]*255))
    return rgb_palette

def plot_imgs_grid(paths, s):
    d = floor(sqrt(len(paths)))
    dx = dy = d
    if d**2 < len(paths): # If there are images that do not fit into the square matrix...
        dy+=ceil((len(paths)-d**2)/dx) # ...extend rows to accomodate these images

    # Set up canvas
    canvas = np.ones((dy*s, dx*s, 3), dtype=np.uint8) * 255

    # Plot images
    for y in range(dy):            
        for x in range(dx):
            #print(y,x)
            pos = y*dx+x
            if pos==(len(paths)):
                break
            img = PIL.Image.open(paths[pos]).convert("RGB")
            img.thumbnail((s, s))
            npimg = np.array(img)
            canvas[y*s:y*s+npimg.shape[0], x*s:x*s+npimg.shape[1], :] = npimg
                
    return PIL.Image.fromarray(canvas)

def plot_imgs_features(paths, s, features, borders=None):
    
    def border(npimg, d, color_rgb):
        assert d*2 < max(npimg.shape)
        npimg[0:d,:,:] = color_rgb
        npimg[:,0:d,:] = color_rgb
        npimg[npimg.shape[0]-d:npimg.shape[0],:,:] = color_rgb
        npimg[:,npimg.shape[1]-d:npimg.shape[1],:] = color_rgb
        return npimg

    d = ceil(sqrt(len(paths)))
    
    # Set up canvas
    # As images are anchored in the upper left corner increase size by 1
    canvas = np.ones(((d+1)*s, (d+1)*s, 3), dtype=np.uint8) * 255

    # Plot images
    for i, path in enumerate(tqdm(paths)):
        img = PIL.Image.open(path).convert("RGB")
        img.thumbnail((s, s))
        npimg = np.array(img)
        if borders:
            npimg = border(npimg, 3, borders[i])
        y = int(features[i,0] * (d*s))
        x = int(features[i,1] * (d*s))  
        canvas[y:y+npimg.shape[0], x:x+npimg.shape[1], :] = npimg
                
    return PIL.Image.fromarray(canvas)