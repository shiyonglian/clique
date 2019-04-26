import os
import glob
import shutil
import cv2

data_dir = '/root/workspace/shiyonglian/RDN-TensorFlow-master/Eval/DIV2K_valid_HR'
data = glob.glob(os.path.join(data_dir, "*.bmp"))
data += glob.glob(os.path.join(data_dir, "*.jpg"))
data += glob.glob(os.path.join(data_dir, "*.png"))

save_bad_dir = '/root/workspace/shiyonglian/RDN-TensorFlow-master/Eval/bad_imgs'
if not os.path.exists(save_bad_dir):
	os.makedirs(save_bad_dir)

for i in range(len(data)):
	path = data[i]
	try:
		img = cv2.imread(path)
		shape = img.shape
	except:
		print('bad image:', path)
		shutil.move(path, save_bad_dir)

