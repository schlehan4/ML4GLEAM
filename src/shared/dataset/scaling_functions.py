import numpy as np
from astropy.visualization import ZScaleInterval, ImageNormalize

def hep(gamma, image_data):
    b = 1 #oder 0 wenn hep zwischen 0 und 1
    c = np.max(image_data)
    # Sicherstellen, dass image_data nicht negativ ist
    if np.min(image_data) < 0:
        image_data -= np.min(image_data)
        c -= np.min(image_data)
    # HEP-Normalisierung anwenden
    return ((b + 1) * (image_data / c) ** (1.0 / gamma)) - b
            
def hep_0(gamma, min, max, image_data):
    b = 0 #oder 0 wenn hep zwischen 0 und 1
    c = max
    local_min = image_data.min()
    if local_min < 0:
        c = max - min
        image_data -= local_min
    """if min < 0:
        c = max - min
        image_data -= min"""
    # HEP-Normalisierung anwenden
    # image_data = ((b + 1) * (image_data / c) ** (1.0 / gamma)) - b
    return ((b + 1) * (image_data / c) ** (1.0 / gamma)) - b
           
    
def z_score(image_data):
    #between 0 and 1
    mean = np.mean(image_data)
    std_dv=np.std(image_data)
    return (image_data / mean) - std_dv

def min_max(image_data):
    return (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data) + 1e-8)

def zscale_min_max(image_data):
    norm = ImageNormalize(image_data, interval=ZScaleInterval())
    norm_img = norm(image_data)
    return norm_img
    