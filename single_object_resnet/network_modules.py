import config
from tensorflow.contrib.slim.python.slim.nets.resnet_utils import conv2d_same
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import bottleneck, resnet_arg_scope
import tensorflow as tf
slim = tf.contrib.slim

_RGB_MEAN = [123.68, 116.78, 103.94]

def conv2d_seg(inp, num_outputs, kernel_size, stride, scope):
  if stride>1:
    padding = 'VALID'
  else:
    padding = 'SAME'
  extra_channel = tf.random_uniform_initializer(seed=0)((7, 7, 1, 64))
  conv_weights = tf.get_variable('conv1/weights', (7, 7, 3, 64), dtype=tf.float32)
  filt = tf.Variable(tf.concat((conv_weights, extra_channel), axis=2), name=scope)
  out = tf.nn.conv2d(inp, filt, [1, stride, stride, 1], padding)
  return out

def mem_encoder(img, seg, is_training):
    image = tf.reshape(img, [-1] + list(img.get_shape())[2:])
    seg = tf.reshape(seg, [-1] + list(seg.get_shape())[2:])
    image = image - tf.constant(_RGB_MEAN, dtype=tf.float32, shape=(1, 1, 1, 3))
    seg = seg - 127.5
    image_seg = tf.concat([image, seg], axis=-1)

    with tf.contrib.slim.arg_scope(resnet_arg_scope(batch_norm_decay=0.9, weight_decay=0.0)):
        with tf.variable_scope('mem_encoder'):
            with tf.variable_scope('resnet_v1_50', values=[image]) as sc:
                end_points_collection = sc.name + '_end_points'
                with slim.arg_scope([slim.conv2d, bottleneck],
                                    outputs_collections=end_points_collection):
                    with slim.arg_scope([slim.batch_norm], is_training=is_training):
                        net = image_seg
                        net = conv2d_seg(net, 64, 7, stride=2, scope='mem_conv1_seg')
                        net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')

                        with tf.variable_scope('block1', values=[net]) as sc_block:
                            with tf.variable_scope('unit_1', values=[net]):
                                net = bottleneck(net, depth=4 * 64, depth_bottleneck=64, stride=1)
                            with tf.variable_scope('unit_2', values=[net]):
                                net = bottleneck(net, depth=4 * 64, depth_bottleneck=64, stride=1)
                            with tf.variable_scope('unit_3', values=[net]):
                                net = bottleneck(net, depth=4 * 64, depth_bottleneck=64, stride=2)

                        with tf.variable_scope('block2', values=[net]) as sc_block:
                            with tf.variable_scope('unit_1', values=[net]):
                                net = bottleneck(net, depth=4 * 128, depth_bottleneck=128, stride=1)
                            with tf.variable_scope('unit_2', values=[net]):
                                net = bottleneck(net, depth=4 * 128, depth_bottleneck=128, stride=1)
                            with tf.variable_scope('unit_3', values=[net]):
                                net = bottleneck(net, depth=4 * 128, depth_bottleneck=128, stride=1)
                            with tf.variable_scope('unit_4', values=[net]):
                                net = bottleneck(net, depth=4 * 128, depth_bottleneck=128, stride=2)

                        with tf.variable_scope('block3', values=[net]) as sc_block:
                            with tf.variable_scope('unit_1', values=[net]):
                                net = bottleneck(net, depth=4 * 256, depth_bottleneck=256, stride=1)
                            with tf.variable_scope('unit_2', values=[net]):
                                net = bottleneck(net, depth=4 * 256, depth_bottleneck=256, stride=1)
                            with tf.variable_scope('unit_3', values=[net]):
                                net = bottleneck(net, depth=4 * 256, depth_bottleneck=256, stride=1)
                            with tf.variable_scope('unit_4', values=[net]):
                                net = bottleneck(net, depth=4 * 256, depth_bottleneck=256, stride=1)
                            with tf.variable_scope('unit_5', values=[net]):
                                net = bottleneck(net, depth=4 * 256, depth_bottleneck=256, stride=1)
                            with tf.variable_scope('unit_6', values=[net]):
                                net = bottleneck(net, depth=4 * 256, depth_bottleneck=256, stride=2)

                        key = tf.layers.conv2d(net, filters=int(net.get_shape()[-1]) / 8, kernel_size=(1, 1),
                                               activation=None, padding='SAME', name='mem_key')
                        value = tf.layers.conv2d(net, filters=int(net.get_shape()[-1]) / 2, kernel_size=(1, 1),
                                                 activation=None, padding='SAME', name='mem_value')

        net = tf.reshape(net, [config.batch_size, -1] + list(net.get_shape())[1:])
        key = tf.reshape(key, [config.batch_size, -1] + list(key.get_shape())[1:])
        value = tf.reshape(value, [config.batch_size, -1] + list(value.get_shape())[1:])
        return key, value, net

def curr_encoder(img, is_training):
    image = tf.reshape(img, [-1] + list(img.get_shape())[2:])
    image = image - tf.constant(_RGB_MEAN, dtype=tf.float32, shape=(1, 1, 1, 3))

    with tf.contrib.slim.arg_scope(resnet_arg_scope(batch_norm_decay=0.9, weight_decay=0.0)):
        with tf.variable_scope('curr_encoder'):
            with tf.variable_scope('resnet_v1_50', values=[image]) as sc:
                end_points_collection = sc.name + '_end_points'
                with slim.arg_scope([slim.conv2d, bottleneck],
                                    outputs_collections=end_points_collection):
                    with slim.arg_scope([slim.batch_norm], is_training=is_training):
                        net = image
                        net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
                        net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')

                        with tf.variable_scope('block1', values=[net]) as sc_block:
                            with tf.variable_scope('unit_1', values=[net]):
                                net = bottleneck(net, depth=4 * 64, depth_bottleneck=64, stride=1)
                            with tf.variable_scope('unit_2', values=[net]):
                                block1 = net = bottleneck(net, depth=4 * 64, depth_bottleneck=64, stride=1)
                            with tf.variable_scope('unit_3', values=[net]):
                                net = bottleneck(net, depth=4 * 64, depth_bottleneck=64, stride=2)

                        with tf.variable_scope('block2', values=[net]) as sc_block:
                            with tf.variable_scope('unit_1', values=[net]):
                                net = bottleneck(net, depth=4 * 128, depth_bottleneck=128, stride=1)
                            with tf.variable_scope('unit_2', values=[net]):
                                net = bottleneck(net, depth=4 * 128, depth_bottleneck=128, stride=1)
                            with tf.variable_scope('unit_3', values=[net]):
                                block2 = net = bottleneck(net, depth=4 * 128, depth_bottleneck=128, stride=1)
                            with tf.variable_scope('unit_4', values=[net]):
                                net = bottleneck(net, depth=4 * 128, depth_bottleneck=128, stride=2)

                        with tf.variable_scope('block3', values=[net]) as sc_block:
                            with tf.variable_scope('unit_1', values=[net]):
                                net = bottleneck(net, depth=4 * 256, depth_bottleneck=256, stride=1)
                            with tf.variable_scope('unit_2', values=[net]):
                                net = bottleneck(net, depth=4 * 256, depth_bottleneck=256, stride=1)
                            with tf.variable_scope('unit_3', values=[net]):
                                net = bottleneck(net, depth=4 * 256, depth_bottleneck=256, stride=1)
                            with tf.variable_scope('unit_4', values=[net]):
                                net = bottleneck(net, depth=4 * 256, depth_bottleneck=256, stride=1)
                            with tf.variable_scope('unit_5', values=[net]):
                                block3 = net = bottleneck(net, depth=4 * 256, depth_bottleneck=256, stride=1)
                            with tf.variable_scope('unit_6', values=[net]):
                                net = bottleneck(net, depth=4 * 256, depth_bottleneck=256, stride=2)

                        key = tf.layers.conv2d(net, filters=int(net.get_shape()[-1]) / 8, kernel_size=(1, 1),
                                               activation=None, padding='SAME', name='curr_key')
                        value = tf.layers.conv2d(net, filters=int(net.get_shape()[-1]) / 2, kernel_size=(1, 1),
                                                 activation=None, padding='SAME', name='curr_value')

        net = tf.reshape(net, [config.batch_size, -1] + list(net.get_shape())[1:])
        key = tf.reshape(key, [config.batch_size, -1] + list(key.get_shape())[1:])
        value = tf.reshape(value, [config.batch_size, -1] + list(value.get_shape())[1:])
        return key, value, net, block1, block2, block3

def attention(mem_key, mem_value, curr_key, curr_value):
  with tf.variable_scope('attention'):
    # Flatten from (B,H,W,C) to (B,HW,C)
    curr_key_flat = tf.reshape(curr_key, [tf.shape(curr_key)[0], -1, tf.shape(curr_key)[-1]])
    # Flatten from (B,T,H,W,C) to (B,THW,C)
    mem_key_flat = tf.reshape(mem_key, [tf.shape(mem_key)[0], -1, tf.shape(mem_key)[-1]])
    mem_value_flat = tf.reshape(mem_value, [tf.shape(mem_value)[0], -1, tf.shape(mem_value)[-1]])
    # Compute f(a, b) -> (B,HW,THW)
    f = tf.matmul(curr_key_flat, tf.transpose(mem_key_flat, [0, 2, 1]))
    f = tf.nn.softmax(f)
    # Compute f * g ("self-attention") -> (B,HW,C)
    fg = tf.matmul(f, mem_value_flat)
    fg = tf.reshape(fg, tf.shape(curr_value))
    net = tf.concat((curr_value, fg), axis=-1)
    return net

def res_block(inp):
  res = tf.layers.conv2d(inp, 256, (3, 3), activation=tf.nn.relu, padding='SAME')
  res = tf.layers.conv2d(res, 256, (3, 3), activation=tf.nn.relu, padding='SAME')
  out = inp + res
  return out

def refine_module(encoder_inp, decoder_inp):
  ref = tf.layers.conv2d(encoder_inp, 256, (3, 3), activation=None, padding='SAME')
  ref = res_block(ref)
  up = ref + tf.image.resize_bilinear(decoder_inp, size=(decoder_inp.get_shape()[1]*2,
                                                                      decoder_inp.get_shape()[2]*2))
  out = res_block(up)
  return out

def curr_decoder(inp, block1, block2, block3):
    dec_input = tf.reshape(inp, [-1] + list(inp.get_shape())[2:])

    with tf.variable_scope('curr_decoder'):
        deconv1 = tf.layers.conv2d(dec_input, 256, (3, 3), activation=None, padding='SAME')
        deconv_res_1 = res_block(deconv1)

        with tf.variable_scope('refine_module'):
            with tf.variable_scope('refine_module_1'):
                deconv_ref_1 = refine_module(block3, deconv_res_1)
            with tf.variable_scope('refine_module_2'):
                deconv_ref_2 = refine_module(block2, deconv_ref_1)
            with tf.variable_scope('refine_module_3'):
                deconv_ref_3 = refine_module(block1, deconv_ref_2)

            out_seg = tf.layers.conv2d(deconv_ref_3, 2, (3, 3), activation=None, padding='SAME', name='output_seg')
            out_seg = tf.image.resize_bilinear(out_seg, size=(out_seg.get_shape()[1] * 4,
                                                                    out_seg.get_shape()[2] * 4))
            out_seg = tf.reshape(out_seg, [config.batch_size, -1] + list(out_seg.get_shape())[1:])
        return out_seg
