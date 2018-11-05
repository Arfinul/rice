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

col = Image.open(image_name+'.jpg')  # read image
gray = col.convert('L')  # convert image to monochrome
bw = gray.point(lambda x: 0 if x < 128 else 255, '1')  # Binarization
bw.save(image_name+"result.png")  # save it

# cur_path = os.getcwd()
# print(cur_path)
# if not os.path.exists(cur_path+'\\'+'output'):
#     os.makedirs(cur_path+'\\'+'output')

# os.chdir(path + '/' + name_of_folder)

m = re.search('[\w]+', image_name)
image_name_without_ext = m.group(0)
result_image_name = image_name_without_ext + 'result.png'
print(result_image_name)
extract_img.crop_grains(result_image_name)
name_of_folder = image_name_without_ext + 'result'
print(name_of_folder)
path = os.getcwd()
os.chdir(path + '/' + name_of_folder)
end = time.time()
#	print('time taken ')
#	print(end - start)
images = glob('*.png')
imagetensor = torch.ByteTensor(len(images), 3, 224, 224)
for i, image in enumerate(images):
    print(image)
    m_ = re.search('[\w]+', image)
    image_title = m_.group(0)
    img = Image.open(image)
    np_img = np.array(img)
    np_img = np.transpose(np_img, (2, 1, 0))
    img_tensor = torch.from_numpy(np_img)
    imagetensor[i] = img_tensor
    # print(image_title)
    # print(image)
    # your code goes in here
    # later return to path
os.chdir(path)
size_stats = test(pathModel, nnClassCount, imagetensor, trBatchSize)
color_stats = get_color_stats.get_color_statistics(name_of_folder, 40)
data = [size_stats, color_stats]
print(os.getcwd())
os.chdir(cwd_working)
print("\nExecution time = %s seconds" % (time.time() - start_time))
#	end_final= time.time()
#	print('time takne now')
#	print(end_final - end)
