import tensorflow as tf
import config
from lstm_network import FullNetwork
from load_youtube_data3 import YoutubeTrainDataGen as TrainDataGen
import sys
import time
import numpy as np

def get_num_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('Num of parameters:', total_parameters)
    sys.stdout.flush()

def train_one_epoch(sess, capsnet, data_gen, writer, loss_summary, prev_batch_num):
    start_time = time.time()
    # continues until no more training data is generated
    batch, s_losses = 0.0, 0
    while data_gen.has_data(config.batch_size):
        batch_data = data_gen.get_batch(config.batch_size)
        x_batch, bbox_batch, mask_batch = batch_data
        x_seg_batch = [seg[0] for seg in bbox_batch]

        #n_frames = np.random.randint(6, 12)
        n_frames = config.inner_frames
        x_batch2, bbox_batch2, mask_batch2 = [], [], []
        for i in range(len(x_batch)):
            x_batch2.append(x_batch[i][:n_frames])
            bbox_batch2.append(bbox_batch[i][:n_frames])
            mask_batch2.append(mask_batch[i][:n_frames])

        if config.multi_gpu and len(x_batch) == 1:
            print('Batch size of one, not running')
            continue
        _, outputs = sess.run([capsnet.train_op, loss_summary],
                           feed_dict={capsnet.x_vid_input: x_batch2, capsnet.y_input: bbox_batch2,
                                      capsnet.x_seg_input: x_seg_batch, capsnet.y_input_mask: mask_batch2})

        # s_losses += s_loss
        # seg_acc += s_acc

        batch += 1
      
        writer.add_summary(outputs, prev_batch_num+batch)
        
        if batch % config.batches_until_print == 0:
            print('Finished %d batches. %d(s) since start. Avg Segmentation Loss is %.4f.'
                  % (batch, time.time() - start_time), end='\r')
            sys.stdout.flush()

    return [outputs / batch, prev_batch_num+batch]

def train_network(gpu_config):
    lstm_network = FullNetwork(input_shape=config.hr_frame_size)

    with tf.Session(graph=lstm_network.graph, config=gpu_config) as sess:
        tf.global_variables_initializer().run()
        
        loss_summary = tf.summary.scalar('loss', lstm_network.segmentation_loss)

        writer = tf.summary.FileWriter('./logs/summary', sess.graph)
        prev_batch_num=0
        get_num_params()

        # if config.use_trained_weights:
            # lstm_network.load(sess, config.save_file_best_name % config.epoch_save)  # Uncomment to train from saved weights
            # print('Loaded in old weights')
        # else:
            # config.clear_output()


        n_eps_after_acc, best_loss = -1, 1000000
        print('Training on %s' % config.dataset)
        for ep in range(1, config.n_epochs + 1):
            print(20 * '*', 'epoch', ep, 20 * '*')
            sys.stdout.flush()

            # Trains network for 1 epoch
            data_gen = TrainDataGen(config.wait_for_data, crop_size=config.hr_frame_size, n_frames=config.n_lstm_frames,
                                    rand_frame_skip=config.rand_frame_skip, use_all=config.use_all_frames)
            seg_loss, prev_batch_num = train_one_epoch(sess, lstm_network, data_gen, writer, loss_summary, prev_batch_num)

            # config.write_output('Epoch%d: SL: %.4f.\n' % (ep, seg_loss))

            # saves every 10 epochs
            if ep % config.save_every_n_epochs == 0:
                try:
                    lstm_network.save(sess, config.save_file_name % 1)
                    # config.write_output('Saved Network\n')
                except:
                    print('Failed to save network!!!')
                    sys.stdout.flush()

            # saves when validation loss becomes smaller (after 50 epochs to save space)
            t_loss = seg_loss

            if t_loss < best_loss:
                best_loss = t_loss
                try:
                    lstm_network.save(sess, config.save_file_best_name % 0)
                    # config.write_output('Saved Network - Minimum val\n')
                except:
                    print('Failed to save network!!!')
                    sys.stdout.flush()

        writer.close()

    tf.reset_default_graph()

def main():
    gpu_config = tf.ConfigProto(allow_soft_placement=True)
    gpu_config.gpu_options.allow_growth = True
    train_network(gpu_config)

main()

