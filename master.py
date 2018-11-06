import os
import re
import time
from glob import glob
import numpy as np
import torch
from PIL import Image
import extract_img
import get_color_stats
from test import test

start_time = time.time()
# Set parameters for model
cwd_working = os.getcwd()
nnClassCount = 2
trBatchSize = 16
pathModel = 'm-epoch-49-21062018-033536.pth.tar'

image_name = input('Image name  : ')
system_command = './rice_poly_circle ' + str(image_name) + '   > output.txt'
os.system(system_command)

cur_path = os.getcwd()
print("Reading Images ...")
print(cur_path)
if not os.path.exists(cur_path + '\\' + 'output'):
    os.makedirs(cur_path + '\\' + 'output')

col = Image.open(image_name + '.jpg')  # read image
gray = col.convert('L')  # convert image to monochrome
bw = gray.point(lambda x: 0 if x < 128 else 255, '1')  # Binarization
bw.save(cur_path + '\\' + 'output' + '\\' + image_name + "result.png")  # save it

m = re.search('[\w]+', image_name)
image_name_without_ext = m.group(0)
# result_image_name = image_name_without_ext + 'result.png'

os.chdir(cur_path + '\\' + 'output')
print("Segmenting ...")
print(os.getcwd())
extract_img.crop_grains(image_name + "result.png")
name_of_folder = image_name_without_ext + 'result'

# os.chdir(cur_path + '\\' + 'output')
print("Segmentation Over ...")
rslt_fldr_path = os.getcwd()
print(rslt_fldr_path)
images = glob('*.png')
imagetensor = torch.ByteTensor(len(images), 3, 224, 224)
print("Image Cropping ..... Wait !!!")
for i, image in enumerate(images):
    m_ = re.search('[\w]+', image)
    image_title = m_.group(0)
    img = Image.open(image)
    np_img = np.array(img)
    np_img = np.transpose(np_img, (2, 1, 0))
    img_tensor = torch.from_numpy(np_img)
    imagetensor[i] = img_tensor
os.chdir(cur_path)  # under rice///
size_stats = test(pathModel, nnClassCount, imagetensor, trBatchSize)
os.chdir(rslt_fldr_path)
color_stats = get_color_stats.get_color_statistics(name_of_folder, 40)
data = [size_stats, color_stats]
# print(os.getcwd()) ////rice2eesult
# os.chdir(cwd_working) ////rice
print("\nExecution time = %s seconds" % (time.time() - start_time))
