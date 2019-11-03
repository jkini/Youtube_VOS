import tensorflow as tf
import config
from network import FullNetwork
from data_loader import YoutubeTrainDataGen as TrainDataGen
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

def train_one_epoch(sess, net, data_gen, writer, prev_batch_num):
    start_time = time.time()
    # continues until no more training data is generated
    batch, s_losses = 0.0, 0
    while data_gen.has_data(config.batch_size):
        batch_data = data_gen.get_batch(config.batch_size)
        x_batch, bbox_batch, mask_batch = batch_data
        x_seg_batch = [seg[:-1] for seg in bbox_batch]

        #n_frames = np.random.randint(6, 12)
        x_batch2, bbox_batch2, mask_batch2 = [], [], []
        for i in range(len(x_batch)):
            x_batch2.append(x_batch[i][:config.n_frames])
            bbox_batch2.append(bbox_batch[i][:config.n_frames])
            mask_batch2.append(mask_batch[i][:config.n_frames])

        if config.multi_gpu and len(x_batch) == 1:
            print('Batch size of one, not running')
            continue
        _, summary, s_loss = sess.run([net.train_op, net.summary, net.segmentation_loss],
                           feed_dict={net.x_input: x_batch2, net.y_input: bbox_batch2,
                                      net.x_seg_input: x_seg_batch, net.y_input_mask: mask_batch2})

        s_losses += s_loss
        batch += 1
          
        writer.add_summary(summary, prev_batch_num+batch)
        
        if batch % config.batches_until_print == 0:
            print('Finished %d batches. %d(s) since start.'
                  % (batch, time.time() - start_time), end='\r')
            sys.stdout.flush()
    return [s_losses/batch, prev_batch_num+batch]

def train_network(gpu_config):
    net = FullNetwork(input_shape=(config.img_height, config.img_width))

    with tf.Session(graph=net.graph, config=gpu_config) as sess:
        tf.global_variables_initializer().run()
       
        if config.use_resnet_weights:
            old_model_scope = ''
            mem_model_scope = 'mem_encoder/'
            curr_model_scope = 'curr_encoder/'
            
            mem_map = {variable.name[len(mem_model_scope):]: variable
                            for variable in net.variables_to_restore
                            if variable.name.startswith(mem_model_scope)}
            mem_map = {name.split(":")[0]: variable
                            for name, variable in mem_map.items()
                            if name.startswith(old_model_scope)}
            mem_saver = tf.train.Saver(mem_map)
            mem_saver.restore(sess, config.resnet_file_name)

            curr_map = {variable.name[len(curr_model_scope):]: variable
                                for variable in net.variables_to_restore
                                if variable.name.startswith(curr_model_scope)}
            curr_map = {name.split(":")[0]: variable
                                for name, variable in curr_map.items()
                                if name.startswith(old_model_scope)}
            curr_saver = tf.train.Saver(curr_map)
            curr_saver.restore(sess, config.resnet_file_name)

        writer = tf.summary.FileWriter('{0}model_{1}'.format(config.tf_logs_dir,config.model_num), sess.graph)
        prev_batch_num=0
        get_num_params()

        if config.use_trained_weights:
            net.load(sess, config.save_file_best_name % config.epoch_save)  # Uncomment to train from saved weights
            print('Loaded in old weights')
        # else:
            # config.clear_output()

        n_eps_after_acc, best_loss = -1, 1000000
        print('Training on %s' % config.data_dir)
        for ep in range(1, config.n_epochs + 1):
            print(20 * '*', 'epoch', ep, 20 * '*')
            sys.stdout.flush()

            # Trains network for 1 epoch
            data_gen = TrainDataGen(config.wait_for_data, crop_size=(config.img_height, config.img_width), n_frames=config.n_frames,
                                    rand_frame_skip=config.rand_frame_skip, use_all=config.use_all_frames)
            seg_loss, prev_batch_num = train_one_epoch(sess, net, data_gen, writer, prev_batch_num)

            # config.write_output('Epoch%d: SL: %.4f.\n' % (ep, seg_loss))

            # saves every 10 epochs
            if ep % config.save_every_n_epochs == 0:
                try:
                    net.save(sess, config.save_file_name % 1)
                    # config.write_output('Saved Network\n')
                except:
                    print('Failed to save network!!!')
                    sys.stdout.flush()

            # saves when validation loss becomes smaller (after 50 epochs to save space)
            t_loss = seg_loss

            if t_loss < best_loss:
                best_loss = t_loss
                try:
                    net.save(sess, config.save_file_best_name % 0)
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

