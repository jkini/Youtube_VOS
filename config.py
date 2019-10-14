import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

data_dir='../Youtube_VOS_Pretrain/'

batch_size = 3
n_epochs = 35
n_epochs_til_full = 5
learning_rate, beta1, epsilon = 0.00001, 0.5, 1e-6

n_frames = inner_frames = tot_n_frames = n_lstm_frames = 8

frame_size = (112, 112)  # (224, 224) (112, 112)
hr_frame_size = (256, 448)
rand_frame_skip = lstm_frame_skip = 1  # Max frame skip if using all frames
use_all_frames = False  # If False, only use every 5th frame. If True, use every frame but only calculate loss on 5th.
use_trained_weights = False
train = True

stop_grad_for_hr = False

multi_gpu = False

model_num = 1
epoch_save = 0  # to be used at inference time
output_inference_file = './Annotations/'

save_every_n_epochs = 10
output_file_name = 'models/output%d.txt' % model_num
save_file_name = ('models/model%d' % model_num) + '_%d.ckpt'
save_file_best_name = ('models/best_model%d' % model_num) + '_%d.ckpt'

lr_segment_coef = 0.02
hr_segment_coef = 0.02

dataset = 'YoutubeVOS'
wait_for_data = 5  # in seconds
batches_until_print = 1

inv_temp = 0.5
inv_temp_delta = 0.1
pose_dimension = 4

print_layers = True
