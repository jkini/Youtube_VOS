import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
devices = ['/gpu:0']
multi_gpu = False

img_height = 384
img_width = 384
batch_size = 2
n_epochs = 80
n_frames = 4
learning_rate = 0.00001

use_resnet_weights = True # to be used at training time (first training)
use_trained_weights = False  # to be used at training time (second training onwards)
train = True 

model_num = 1
epoch_save = 0  # to be used at inference time
save_every_n_epochs = 10

rand_frame_skip = 1
use_all_frames= False

max_vids = 100
wait_for_data = 5  # in seconds
batches_until_print = 1

#data_dir = '/home/jkini/Datasets/Youtube_VOS_Pretrain/'
#data_dir = '/home/jyoti/git/youtube_vos/youtube-vos/'
#data_dir = '/groups/mshah/data/youtube-vos/'
data_dir = '/home/jkini/Datasets/Youtube_VOS/'
resnet_file_name = '../resnet_v1_50.ckpt'
tf_logs_dir = 'logs/'
output_inference_dir = './Annotations/'
output_file_name = 'models/output%d.txt' % model_num
save_file_name = ('models/model%d' % model_num) + '_%d.ckpt'
save_file_best_name = ('models/model%d' % model_num) + '_%d.ckpt'
