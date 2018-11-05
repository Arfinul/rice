from skimage import io
import matplotlib.pyplot as plt
import numpy as np

def get_avg_var_col(image_name):
  image_SK = io.imread(image_name)
  image_SK_red = image_SK[:,:,0]
  image_SK_green = image_SK[:,:,1]
  image_SK_blue = image_SK[:,:,2]
  RC_ = image_SK_red.flatten()
  GC_ = image_SK_green.flatten()
  BC_ = image_SK_blue.flatten()
  
  rc_ = RC_[RC_!=0]
  gc_ = GC_[GC_!=0]
  bc_ = BC_[BC_!=0]

  #x = np.trim_zeros(X_)
  #y = np.trim_zeros(Y_)
  #z = np.trim_zeros(Z_)

  rc = np.trim_zeros(rc_)
  gc = np.trim_zeros(gc_)
  bc = np.trim_zeros(bc_)
#calculate stats per grain
  rc_mean = np.mean(rc)
  gc_mean = np.mean(gc)
  bc_mean = np.mean(bc)
  rc_var = np.var(rc)
  gc_var = np.var(gc)
  bc_var = np.var(bc)
  return rc_mean,gc_mean,bc_mean,rc_var,gc_var,bc_var
