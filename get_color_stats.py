import grain_channel_avg_var
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

def get_color_statistics(image_folder,n_bins):
	cwd = os.getcwd()
	
	os.chdir(cwd+'/' + image_folder)
	images_listed = glob('*.png')
	red_channel_avg = []
	green_channel_avg = []
	blue_channel_avg = []
	red_channel_variance = []
	green_channel_variance = []
	blue_channel_variance = []
	for image in images_listed:
		[rc_mean,gc_mean,bc_mean,rc_var,gc_var,bc_var]=grain_channel_avg_var.get_avg_var_col(image)
		red_channel_avg.append(rc_mean)
		green_channel_avg.append(gc_mean)
		blue_channel_avg.append(bc_mean)
		red_channel_variance.append(rc_var)
		green_channel_variance.append(gc_var)
		blue_channel_variance.append(bc_var)
	data = [red_channel_avg , green_channel_avg , blue_channel_avg, [np.mean(red_channel_avg)],[np.var(red_channel_avg)] , [np.mean(green_channel_avg)],[np.var(green_channel_avg)], [np.mean(blue_channel_avg)],[np.var(blue_channel_avg)]  ]
	return data
#	_,axs = plt.subplots(1,3,sharey=True,tight_layout=True )
#	print(len(red_channel_avg))
#	axs[0].hist(red_channel_avg,color = "red")
#	axs[1].hist(green_channel_avg, color = "green")
#	axs[2].hist(blue_channel_avg,color = "blue")
#	print('Red Channel Avg  : '+str(np.mean(red_channel_avg)))
#	print('Green Channel Avg  : '+str(np.mean(green_channel_avg)))
#	print('Blue Channel Avg  : '+str(np.mean(blue_channel_avg)))

#	print('Red Channel Var : '+str(np.var(red_channel_avg)))
#	print('Green Channel Var : '+str(np.var(green_channel_avg)))
#	print('Blue Channel Var : '+str(np.var(blue_channel_avg)))

#	axs[0].plot(red_channel_avg )
#	axs[0].set_title('Red Channel Mean ')
#	axs[1].plot(green_channel_avg )
#	axs[1].set_title('Green Channel Mean ')
#	axs[2].plot(blue_channel_avg )
#	axs[2].set_title('Blue Channel Mean ')
#	plt.show()
#	_,axs = plt.subplots(1,3,sharey=True,tight_layout=True )
#	axs[0].plot(red_channel_variance )
#	axs[0].set_title('Red Channel Variance')
#	axs[1].plot(green_channel_variance )
#	axs[1].set_title('Green Channel Variance')
#	axs[2].plot(blue_channel_variance )
#	axs[2].set_title('Blue Of Variance')
#	plt.show()
	os.chdir(cwd)
