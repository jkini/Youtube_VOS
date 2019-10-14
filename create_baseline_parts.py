import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
import convlstmcell
from config import train, inner_frames

VGG_MEAN = [103.939, 116.779, 123.68]
weight_dict = np.load('vgg16.npy', encoding='latin1', allow_pickle = True).item()
fc6_np = np.load('fc6_1_np.npy', allow_pickle = True)
print("npy file loaded")

def get_fc_t_filter(name, inc_seg):
    fc6_t = tf.constant(fc6_np)
    return tf.Variable(fc6_t, name="fc_filter")

def get_fc_bias(name):
    return tf.Variable(weight_dict[name][1], name="fc_biases")

def fc_t_layer(inputs, name, inc_seg=False):
    with tf.variable_scope(name):
        filt = get_fc_t_filter(name, inc_seg)
        conv = tf.nn.conv2d(inputs, filt, [1, 1, 1, 1], padding='SAME')
        conv_biases = get_fc_bias(name)
        bias = tf.nn.bias_add(conv, conv_biases)
        return bias

#####################################

def get_conv_filter(name, inc_seg):
    if inc_seg:
        loaded_weights = tf.constant(weight_dict[name][0])
        initializer = tf.contrib.layers.xavier_initializer()
        out = initializer((3, 3, 1, 64))
        return tf.Variable(tf.concat((loaded_weights, out), axis=2) , name="filter")
    else:
        return tf.Variable(weight_dict[name][0], name="filter")

def get_bias(name):
    return tf.Variable(weight_dict[name][1], name="biases")

def conv_layer(inputs, name, inc_seg=False, firstConv= False):
    with tf.variable_scope(name):
        filt = get_conv_filter(name, inc_seg)
        #filt = tf.stop_gradient(filt)
        if (name=='conv1_1' and firstConv):
            extra_channel = tf.get_variable('extraChannel', dtype= tf.float32, initializer= tf.contrib.layers.xavier_initializer(),shape= tf.reduce_mean(filt, 2, True).get_shape())
            filt =  tf.concat([filt, extra_channel], 2)
            pass
        conv = tf.nn.conv2d(inputs, filt, [1, 1, 1, 1], padding='SAME')

        conv_biases = get_bias(name)
        bias = tf.nn.bias_add(conv, conv_biases)

        relu = tf.nn.relu(bias)
        # relu = tf.stop_gradient(relu)
        return relu

def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def create_vgg_encoder(input_img):
    #tsize =  input_img.get_shape()[1]

    input_img = tf.reshape(input_img,[-1]+list(input_img.get_shape())[2:])
    # input_seg = tf.reshape(input_seg,[-1]+input_seg.get_shape()[2:])
    with tf.variable_scope('vgg16_encoder'):
        rgb_scaled = input_img * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        #assert red.get_shape().as_list()[1:] == [224, 224, 1]
        #assert green.get_shape().as_list()[1:] == [224, 224, 1]
        #assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        #assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        conv1_1 = conv_layer(bgr, "conv1_1")
        conv1_2 = conv_layer(conv1_1, "conv1_2")
        pool1 = max_pool(conv1_2, 'pool1')

        conv2_1 = conv_layer(pool1, "conv2_1")
        conv2_2 = conv_layer(conv2_1, "conv2_2")
        pool2 = max_pool(conv2_2, 'pool2')

        conv3_1 = conv_layer(pool2, "conv3_1")
        conv3_2 = conv_layer(conv3_1, "conv3_2")
        conv3_3 = conv_layer(conv3_2, "conv3_3")
        pool3 = max_pool(conv3_3, 'pool3')

        conv4_1 = conv_layer(pool3, "conv4_1")
        conv4_2 = conv_layer(conv4_1, "conv4_2")
        conv4_3 = conv_layer(conv4_2, "conv4_3")
        pool4 = max_pool(conv4_3, 'pool4')

        conv5_1 = conv_layer(pool4, "conv5_1")
        conv5_2 = conv_layer(conv5_1, "conv5_2")
        conv5_3 = conv_layer(conv5_2, "conv5_3")
        pool5 = max_pool(conv5_3, 'pool5') 

        # convfc1 = tf.layers.conv2d(pool5, 4096, (1, 1), name='fc1',
        #                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                            bias_initializer=tf.contrib.layers.xavier_initializer())

        convfc1 = fc_t_layer(pool5, "fc6") 
        #convfc1 = tf.layers.batch_normalization(convfc1, training=train, renorm=True)
        convfc1 = tf.nn.relu(convfc1)
        
        enc_fc2 = tf.layers.conv2d(convfc1, 512, (1, 1), name='enc_fc2',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                   bias_initializer=tf.contrib.layers.xavier_initializer())

        #enc_fc2 = tf.layers.batch_normalization(enc_fc2, training=train, renorm=True)       
        enc_fc2 = tf.nn.relu(enc_fc2) 

        #enc_fc2 = tf.reshape(enc_fc2, [-1, tsize]+list(enc_fc2.get_shape())[1:])
        enc_fc2 = tf.reshape(enc_fc2, [-1, inner_frames-1]+list(enc_fc2.get_shape())[1:])
        # enc_fc3 = tf.reshape(enc_fc3, [-1, tsize]+enc_fc3.get_shape()[1:]) 

        return enc_fc2


def create_vgg_initializer(input_img, input_seg):
    with tf.variable_scope('vgg16_initializer'):
        # input_seg =  tf.concat([input_seg, input_seg, input_seg], axis=3)        
        input_img = tf.concat([input_img, input_seg], axis = -1)  
        rgb_scaled = input_img * 255.0

        # Convert RGB to BGR
        red, green, blue, seg = tf.split(axis=3, num_or_size_splits=4, value=rgb_scaled)
        seg = seg - 127.5
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2]
        ])
        #assert bgr.get_shape().as_list()[1:] == [224, 224, 4]
        conv1_1 = conv_layer(bgr, "conv1_1")
        conv1_1_seg = tf.layers.conv2d(seg, filters=32, kernel_size=(3, 3), activation='relu', padding='SAME')

        conv1_1 = tf.concat([conv1_1, conv1_1_seg], axis=3)

        conv1_1 = tf.layers.conv2d(conv1_1, filters=64, kernel_size=(1, 1), activation='relu', padding='SAME')

        conv1_2 = conv_layer(conv1_1, "conv1_2")
        pool1 = max_pool(conv1_2, 'pool1')

        conv2_1 = conv_layer(pool1, "conv2_1")
        conv2_2 = conv_layer(conv2_1, "conv2_2")
        pool2 = max_pool(conv2_2, 'pool2')

        conv3_1 = conv_layer(pool2, "conv3_1")
        conv3_2 = conv_layer(conv3_1, "conv3_2")
        conv3_3 = conv_layer(conv3_2, "conv3_3")
        pool3 = max_pool(conv3_3, 'pool3')

        conv4_1 = conv_layer(pool3, "conv4_1")
        conv4_2 = conv_layer(conv4_1, "conv4_2")
        conv4_3 = conv_layer(conv4_2, "conv4_3")
        pool4 = max_pool(conv4_3, 'pool4')

        conv5_1 = conv_layer(pool4, "conv5_1")
        conv5_2 = conv_layer(conv5_1, "conv5_2")
        conv5_3 = conv_layer(conv5_2, "conv5_3")
        pool5 = max_pool(conv5_3, 'pool5')

        # convfc1 = tf.layers.conv2d(pool5, 4096, (1, 1), name='fc1',
        #                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                            bias_initializer=tf.contrib.layers.xavier_initializer())
        

        convfc1 = fc_t_layer(pool5, "fc6") 
        #convfc1 = tf.layers.batch_normalization(convfc1, training=train, renorm=True)
        convfc1 = tf.nn.relu(convfc1)

        enc_fc2 = tf.layers.conv2d(convfc1, 512, (1, 1), name='init_fc2',
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.contrib.layers.xavier_initializer())
        #enc_fc2 = tf.layers.batch_normalization(enc_fc2, training=train, renorm=True)
        enc_fc2 = tf.nn.relu(enc_fc2)

        enc_fc3 = tf.layers.conv2d(convfc1, 512, (1, 1), name='init_fc3',
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.contrib.layers.xavier_initializer())
        #enc_fc3 = tf.layers.batch_normalization(enc_fc3, training=train, renorm=True)
        enc_fc3 = tf.nn.relu(enc_fc3)
        return enc_fc2, enc_fc3

'''
def create_convlstm(c_0, h_0, input_sequence):
    _, _, h, w, ch = input_sequence.get_shape()
    h, w, ch = int(h), int(w), int(ch)
    with tf.variable_scope('lstm'): # TODO change to be their LSTM
        conv_lstm_cell = tf.contrib.rnn.ConvLSTMCell(2, [h, w, ch], 512, [3, 3], name='conv_lstm_cell')
        init_state = tf.nn.rnn_cell.LSTMStateTuple(c_0, h_0)
        return tf.nn.dynamic_rnn(conv_lstm_cell, input_sequence, initial_state=init_state, time_major=False,
                                 dtype=tf.float32)
                                 #outputs, state = create_convlstm
'''

def create_convlstm(c_0, h_0, input_sequence):
    _, _, h, w, ch = input_sequence.get_shape()
    h, w, ch = int(h), int(w), int(ch)
    with tf.variable_scope('lstm'):
        conv_lstm_cell = convlstmcell.ConvLSTMCell([h,w], 512, [3, 3], activation=tf.nn.relu)
        init_state = tf.nn.rnn_cell.LSTMStateTuple(c_0, h_0)
        return tf.nn.dynamic_rnn(conv_lstm_cell, input_sequence, initial_state=init_state, time_major=False,
                                 dtype=tf.float32)


def create_decoder(h_input):
    #tsize = h_input.get_shape()[1]
    h_input = tf.reshape(h_input,[-1]+list(h_input.get_shape())[2:])
    with tf.variable_scope('decoder'):
        deconv1 = tf.layers.conv2d_transpose(h_input, 512, (5, 5), (2, 2), activation=tf.nn.relu, padding='SAME',
                                             name='deconv1', kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             bias_initializer=tf.contrib.layers.xavier_initializer())
        #deconv1 = tf.layers.batch_normalization(deconv1, training=train, renorm=True)
        #deconv1 = tf.nn.relu(deconv1)

        deconv2 = tf.layers.conv2d_transpose(deconv1, 256, (5, 5), (2, 2), activation=tf.nn.relu, padding='SAME',
                                             name='deconv2', kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             bias_initializer=tf.contrib.layers.xavier_initializer())
        #deconv2 = tf.layers.batch_normalization(deconv2, training=train, renorm=True)
        #deconv2 = tf.nn.relu(deconv2)

        deconv3 = tf.layers.conv2d_transpose(deconv2, 128, (5, 5), (2, 2), activation=tf.nn.relu, padding='SAME',
                                             name='deconv3', kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             bias_initializer=tf.contrib.layers.xavier_initializer())
        #deconv3 = tf.layers.batch_normalization(deconv3, training=train, renorm=True)
        #deconv3 = tf.nn.relu(deconv3)

        deconv4 = tf.layers.conv2d_transpose(deconv3, 64, (5, 5), (2, 2), activation=tf.nn.relu, padding='SAME',
                                             name='deconv4', kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             bias_initializer=tf.contrib.layers.xavier_initializer())
        #deconv4 = tf.layers.batch_normalization(deconv4, training=train, renorm=True)
        #deconv4 = tf.nn.relu(deconv4)

        deconv5 = tf.layers.conv2d_transpose(deconv4, 64, (5, 5), (2, 2), activation=tf.nn.relu, padding='SAME',
                                             name='deconv5', kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             bias_initializer=tf.contrib.layers.xavier_initializer())
        output_seg = tf.layers.conv2d(deconv5, 1, (5, 5), activation=None, padding='SAME', name='output_seg',
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      bias_initializer=tf.contrib.layers.xavier_initializer())
        
        #output_seg = tf.reshape(output_seg, [-1, tsize]+list(output_seg.get_shape())[1:])
        output_seg = tf.reshape(output_seg, [-1, inner_frames-1]+list(output_seg.get_shape())[1:])

        return output_seg


