import numpy as np
from skimage import filters
from skimage.util import random_noise, img_as_ubyte, img_as_float
from scipy.ndimage.interpolation import zoom as npzoom
from skimage.transform import rescale
import PIL

# Normalizes arr
def downscale(arr):
    dtype = arr.dtype
    arr = arr.astype(np.float32)
    og_min, og_max = float(arr.min()), float(arr.max())
    # og_min, og_max = -212., 2047.
    # og_min, og_max = 33.315, 1258.7339
    
    return (arr - og_min) / (og_max - og_min), (og_min, og_max)
    
# Changes normalized array to its original range
def upscale(arr, extrema):
    og_min, og_max = extrema
    return (arr * float(og_max - og_min) + og_min).astype(np.float32)
    
# Assumes image data is normalized
def recenter(img_new, img_og):
    out = img_new.copy()
    mean_diff = np.mean(img_new) - np.mean(img_og)
    if mean_diff > 0:
        out[out < mean_diff] = 0
    elif mean_diff < 0:
        out[out > (1 - mean_diff)] = 0
    out -= mean_diff
    return out

def zero_crap(img, scale=4, upsample=False):
    from skimage.transform import rescale
    x = np.array(img).astype(np.float32)
    x, extrema = downscale(x)
    x = upscale(x, extrema)
    
    channel_axis = len(x.shape) if len(x.shape) > 2 else None
    x = rescale(x, scale=1/scale, order=0, channel_axis=channel_axis)
    return PIL.Image.fromarray(x)

# def fluo_G_D(x, scale=4, upsample=False):
#     xn = np.array(x)
#     xorig_max = xn.max()
#     xn = xn.astype(np.float32)
#     xn /= float(np.iinfo(np.uint8).max)

#     x = np.array(x)
#     mu, sigma = 0, 5
#     noise = np.random.normal(mu, sigma*0.05, x.shape)
#     x = np.clip(x + noise, 0, 1)
#     x_down = npzoom(x, 1/scale, order=1)
#     #x_up = npzoom(x_down, scale, order=1)
#     return PIL.Image.fromarray(x_down.astype(np.uint8))
# 
# def fluo_AG_D(x, scale=4, upsample=False):
#     xn = np.array(x)
#     xorig_max = xn.max()
#     xn = xn.astype(np.float32)
#     xn /= float(np.iinfo(np.uint8).max)

#     lvar = filters.gaussian(xn, sigma=5) + 1e-10
#     xn = random_noise(xn, mode='localvar', local_vars=lvar*0.5)
#     new_max = xn.max()
#     x = xn
#     if new_max > 0:
#         xn /= new_max
#     xn *= xorig_max
#     x_down = npzoom(x, 1/scale, order=1)
#     #x_up = npzoom(x_down, scale, order=1)
#     return PIL.Image.fromarray(x_down.astype(np.uint8))

# def fluo_downsampleonly(x, scale=4, upsample=False):
#     xn = np.array(x)
#     xorig_max = xn.max()
#     xn = xn.astype(np.float32)
#     xn /= float(np.iinfo(np.uint8).max)
#     new_max = xn.max()
#     x = xn
#     if new_max > 0:
#         xn /= new_max
#     xn *= xorig_max
#     x_down = npzoom(x, 1/scale, order=1)
#     #x_up = npzoom(x_down, scale, order=1)
#     return PIL.Image.fromarray(x_down.astype(np.uint8))

# def fluo_SP_D(x, scale=4, upsample=False):
#     xn = np.array(x)
#     xorig_max = xn.max()
#     xn = xn.astype(np.float32)
#     xn /= float(np.iinfo(np.uint8).max)
#     xn = random_noise(xn, mode='salt', amount=0.005)
#     xn = random_noise(xn, mode='pepper', amount=0.005)
#     new_max = xn.max()
#     x = xn
#     if new_max > 0:
#         xn /= new_max
#     xn *= xorig_max
#     x_down = npzoom(x, 1/scale, order=1)
#     #x_up = npzoom(x_down, scale, order=1)
#     return PIL.Image.fromarray(x_down.astype(np.uint8))

# def fluo_SP_AG_D_sameas_preprint(x, scale=4, upsample=False):
#     xn = np.array(x)
#     xorig_max = xn.max()
#     xn = xn.astype(np.float32)
#     xn /= float(np.iinfo(np.uint8).max)
#     xn = random_noise(xn, mode='salt', amount=0.005)
#     xn = random_noise(xn, mode='pepper', amount=0.005)
#     lvar = filters.gaussian(xn, sigma=5) + 1e-10
#     xn = random_noise(xn, mode='localvar', local_vars=lvar*0.5)
#     new_max = xn.max()
#     x = xn
#     if new_max > 0:
#         xn /= new_max
#     xn *= xorig_max
#     x_down = npzoom(x, 1/scale, order=1)
#     return PIL.Image.fromarray(x_down.astype(np.uint8))

# def fluo_SP_AG_D_sameas_preprint_rescale(x, scale=4, upsample=False):
#     xn = np.array(x)
#     xorig_max = xn.max()
#     xn = xn.astype(np.float32)
#     xn /= float(np.iinfo(np.uint8).max)
#     xn = random_noise(xn, mode='salt', amount=0.005)
#     xn = random_noise(xn, mode='pepper', amount=0.005)
#     lvar = filters.gaussian(xn, sigma=5) + 1e-10
#     xn = random_noise(xn, mode='localvar', local_vars=lvar*0.5)
#     new_max = xn.max()
#     x = xn
#     if new_max > 0:
#         xn /= new_max
#     xn *= xorig_max
#     multichannel = len(x.shape) > 2
#     x_down = rescale(x, scale=1/scale, order=1, multichannel=multichannel)
#     return PIL.Image.fromarray(x_down.astype(np.uint8))

# def em_AG_D_sameas_preprint(x, scale=4, upsample=False):
#     lvar = filters.gaussian(x, sigma=3)
#     x = random_noise(x, mode='localvar', local_vars=lvar*0.05)
#     x_down = npzoom(x, 1/scale, order=1)
#     x_up = npzoom(x_down, scale, order=1)
#     return x_down, x_up

# def em_downsampleonly(x, scale=4, upsample=False):
#     x_down = npzoom(x, 1/scale, order=1)
#     x_up = npzoom(x_down, scale, order=1)
#     return x_down, x_up

# def em_G_D_001(x, scale=4, upsample=False):
#     noise = np.random.normal(0, 3, x.shape)
#     x = x + noise
#     x = x - x.min()
#     x = x/x.max()
#     x_down = npzoom(x, 1/scale, order=1)
#     x_up = npzoom(x_down, scale, order=1)
#     return x_down, x_up

# def em_G_D_002(x, scale=4, upsample=False):
#     x = img_as_float(x)
#     mu, sigma = 0, 3
#     noise = np.random.normal(mu, sigma*0.05, x.shape)
#     x = np.clip(x + noise, 0, 1)
#     x_down = npzoom(x, 1/scale, order=1)
#     x_up = npzoom(x_down, scale, order=1)
#     return x_down, x_up

# def em_P_D_001(x, scale=4, upsample=False):
#     x = random_noise(x, mode='poisson', seed=1)
#     x_down = npzoom(x, 1/scale, order=1)
#     x_up = npzoom(x_down, scale, order=1)
#     return x_down, x_up

def new_crap_AG_SP(x, scale=4, upsample=False):
    xn = np.array(x).astype(np.float32)
    xn, extrema = downscale(xn)
    xn_og = xn.copy()
    
    xn = random_noise(xn, mode='salt', amount=0.001)
    xn = random_noise(xn, mode='pepper', amount=0.001)

    lvar = filters.gaussian(xn, sigma=3) + 1e-10
    xn = random_noise(xn, mode='localvar', local_vars=lvar*0.1)

    xn = recenter(xn, xn_og)

    xn = upscale(xn, extrema)
    
    channel_axis = len(xn.shape) if len(xn.shape) > 2 else None
    xn = rescale(xn, scale=1/scale, order=0, channel_axis=channel_axis)
    return PIL.Image.fromarray(xn.astype(np.float32))
    
def new_crap_AG_blur_SP(img, scale=4, upsample=False):
    def sift_vals(arr, low=0.35, high=0.65):
        return arr[np.logical_and(arr > np.quantile(arr, low), arr < np.quantile(arr, high))]
    
    sigma = 3
    xn = np.array(img).astype(np.float32)
    xn, extrema = downscale(xn)
    
    lvar = filters.gaussian(xn, sigma=1) + 1e-10
    xn = random_noise(xn, mode='localvar', local_vars=lvar*0.001)
    
    og_mean = np.mean(sift_vals(xn))
    
    xn = random_noise(xn, mode='salt', amount=0.005)
    xn = random_noise(xn, mode='pepper', amount=0.005)
    
    channel_axis = len(xn.shape) if len(xn.shape) > 2 else None
    xn = filters.gaussian(xn, sigma=sigma, truncate=2, channel_axis=channel_axis)
    xn = xn + (og_mean - np.mean(sift_vals(xn)))

    xn = upscale(xn, extrema)

    xn = rescale(xn, scale=1/scale, order=0, channel_axis=channel_axis)
    return PIL.Image.fromarray(xn.astype(np.float32))

# def new_crap(x, scale=4, upsample=False):
#     xn = np.array(x)
#     xorig_max = xn.max()
#     xn = xn.astype(np.float32)
#     xn /= float(np.iinfo(np.uint8).max)

#     xn = random_noise(xn, mode='salt', amount=0.005)
#     xn = random_noise(xn, mode='pepper', amount=0.005)
#     lvar = filters.gaussian(xn, sigma=5) + 1e-10
#     xn = random_noise(xn, mode='localvar', local_vars=lvar*0.5)
#     new_max = xn.max()
#     x = xn
#     if new_max > 0:
#         xn /= new_max
#     xn *= xorig_max
#     multichannel = len(x.shape) > 2
#     x = rescale(x, scale=1/scale, order=1, multichannel=multichannel)
#     return PIL.Image.fromarray(x.astype(np.uint8))
    
def new_crap_blur(img, scale=4, upsample=False):
    def sift_vals(arr, low=0.35, high=0.65):
        return arr[np.logical_and(arr > np.quantile(arr, low), arr < np.quantile(arr, high))]
    
    sigma = 5
    xn = np.array(img).astype(np.float32)
    xn, extrema = downscale(xn)
    
    og_mean = np.mean(sift_vals(xn))
    
    channel_axis = len(xn.shape) if len(xn.shape) > 2 else None
    xn = filters.gaussian(xn, sigma=sigma, truncate=5, channel_axis=channel_axis)
    xn = xn + (og_mean - np.mean(sift_vals(xn)))

    xn = upscale(xn, extrema)

    xn = rescale(xn, scale=1/scale, order=0, channel_axis=channel_axis)
    return PIL.Image.fromarray(xn)

# ###not sure about this one
# def em_AG_P_D_001(x, scale=4, upsample=False):
#     poisson_noisemap = np.random.poisson(x, size=None)
#     set_trace()
#     lvar = filters.gaussian(x, sigma=3)
#     x = random_noise(x, mode='localvar', local_vars=lvar*0.05)
#     x = x + poisson_noisemap
#     #x = x - x.min()
#     #x = x/x.max()
#     x_down = npzoom(x, 1/scale, order=1)
#     x_up = npzoom(x_down, scale, order=1)
#     return x_down, x_up

# def new_crap_raw(x, scale=4, upsample=False):
#     xn = np.array(x)
#     xorig_max = xn.max()
#     xn = xn.astype(np.float32)
#     xn /= float(np.iinfo(np.uint8).max)

#     # lvar = filters.gaussian(xn, sigma=5) + 1e-10
#     # xn = random_noise(xn, mode='localvar', local_vars=lvar*0.5)

#     # xn = random_noise(xn, mode='salt', amount=0.005)
#     # xn = random_noise(xn, mode='pepper', amount=0.005)

#     new_max = xn.max()
#     x = xn
#     if new_max > 0:
#         xn /= new_max
#     xn *= xorig_max
#     multichannel = len(x.shape) > 2

#     xn = rescale(xn, scale=1/scale, order=1, multichannel=multichannel)
#     return PIL.Image.fromarray(xn.astype(np.uint8))

# def new_crap_SP(x, scale=4, upsample=False):
#     xn = np.array(x)
#     xorig_max = xn.max()
#     xn = xn.astype(np.float32)
#     xn /= float(np.iinfo(np.uint8).max)

#     # lvar = filters.gaussian(xn, sigma=5) + 1e-10
#     # xn = random_noise(xn, mode='localvar', local_vars=lvar*0.5)

#     xn = random_noise(xn, mode='salt', amount=0.005)
#     xn = random_noise(xn, mode='pepper', amount=0.005)

#     new_max = xn.max()
#     x = xn
#     if new_max > 0:
#         xn /= new_max
#     xn *= xorig_max
#     multichannel = len(x.shape) > 2

#     xn = rescale(xn, scale=1/scale, order=1, multichannel=multichannel)
#     return PIL.Image.fromarray(xn.astype(np.uint8))
    
# def new_crap_AG(x, scale=4, upsample=False):
#     xn = np.array(x)
#     xorig_max = xn.max()
#     xn = xn.astype(np.float32)
#     xn /= float(np.iinfo(np.uint8).max)

#     lvar = filters.gaussian(xn, sigma=5) + 1e-10
#     xn = random_noise(xn, mode='localvar', local_vars=lvar*0.5)

#     # xn = random_noise(xn, mode='salt', amount=0.005)
#     # xn = random_noise(xn, mode='pepper', amount=0.005)

#     new_max = xn.max()
#     x = xn
#     if new_max > 0:
#         xn /= new_max
#     xn *= xorig_max
#     multichannel = len(x.shape) > 2

#     xn = rescale(xn, scale=1/scale, order=1, multichannel=multichannel)
#     return PIL.Image.fromarray(xn.astype(np.uint8))

# def new_crap_AG_SP_8(x, scale=4, upsample=False):
#     xn = np.array(x)
#     xorig_max = xn.max()
#     xn = xn.astype(np.float32)
#     xn /= float(np.iinfo(np.int8).max)

#     lvar = filters.gaussian(xn, sigma=5) + 1e-10
#     xn = random_noise(xn, mode='localvar', local_vars=lvar*0.5)

#     xn = random_noise(xn, mode='salt', amount=0.005)
#     xn = random_noise(xn, mode='pepper', amount=0.005)

#     new_max = xn.max()
#     x = xn
#     if new_max > 0:
#         xn /= new_max
#     xn *= xorig_max
#     multichannel = len(x.shape) > 2

#     xn = rescale(xn, scale=1/scale, order=1, multichannel=multichannel)
#     return PIL.Image.fromarray(xn.astype(np.uint8))