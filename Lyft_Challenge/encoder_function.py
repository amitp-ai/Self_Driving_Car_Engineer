import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
import scipy.misc

#BOTH OF THESE FUNCTIONS ARE PROVIDED BY UDACITY#

# Define encoder function
def encode(array):
    pil_img = Image.fromarray(array)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

# Define Decoder function (used for debug)
def decode(packet):
    img = base64.b64decode(packet)
    filename = './data/Test_Data/decode_image.png'
    with open(filename, 'wb') as f:
            f.write(img)
    result = scipy.misc.imread(filename)
    return result


def encode_inference_output_image(rgb_frame):
    # To encode inference output

    # Udacity Implementation (only using R channel, and R channel = 255 for car and R channel = 0 for road and other values between 0 and 255 for everything else)
    # # Grab red channel  
    # red = rgb_frame[:,:,0]
    # # Look for cars
    # binary_car_result = np.where(red>250,1,0).astype('uint8')
    # # Look for road
    # #binary_road_result = binary_car_result = np.where(red<20,1,0).astype('uint8') #I think this is wrong. corrected below.
    # binary_road_result = np.where(red<20,1,0).astype('uint8') #this is correct, i think.

    # I have Modified Udacity's implementation because in my case Green = road and Blue = car
    # Look for Road
    green_ch = rgb_frame[:,:,1]
    binary_road_result = np.where(green_ch == 255, 1, 0).astype('uint8')
    #Look for cars
    blue_ch = rgb_frame[:,:,2]
    binary_car_result = np.where(blue_ch == 255, 1, 0).astype('uint8')

    return [encode(binary_car_result), encode(binary_road_result)]