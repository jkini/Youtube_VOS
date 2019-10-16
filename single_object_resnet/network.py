import config
import tensorflow as tf
import sys
from network_modules import create_vgg_initializer, create_vgg_encoder, create_decoder, create_convlstm


class FullNetwork(object):
    def __init__(self, input_shape=(256, 448)):
        self.graph = tf.Graph()
        self.input_shape = input_shape

        with self.graph.as_default():
            with tf.variable_scope("global_part", reuse = tf.AUTO_REUSE):
                self.x_vid_input = tf.placeholder(dtype=tf.float32, shape=(None, None, input_shape[0], input_shape[1], 3),
                                                name='x_vid_input')
                self.x_seg_input = tf.placeholder(dtype=tf.float32, shape=(None, input_shape[0], input_shape[1], 1),
                                                name='x_seg_input')
                self.y_input = tf.placeholder(dtype=tf.float32, shape=(None, None, input_shape[0], input_shape[1], 1),
                                            name='y_input')

                self.y_input_mask = tf.placeholder(dtype=tf.float32, shape=(None, None))

                self.init_network()

                self.init_seg_loss()
                self.init_optimizer()

                self.saver = tf.train.Saver()

    def init_network(self):
        c_0, h_0 = create_vgg_initializer(self.x_vid_input[:, 0, :], self.x_seg_input)
        # b_size = tf.shape(self.x_vid_input)[0]
        # n_frames = tf.shape(self.x_vid_input)[1]
        #x_vid_input_res = tf.reshape(self.x_vid_input,
        #                             (b_size*n_frames, self.input_shape[0], self.input_shape[1], 3))
        x_t = create_vgg_encoder(self.x_vid_input[:, 1:, :])
        _, t, h, w, ch = x_t.get_shape()
        h, w, ch = int(h), int(w), int(ch)
        #x_t = tf.reshape(x_t, (b_size, n_frames, h, w, ch))
        outputs, state = create_convlstm(c_0, h_0, x_t)
        #outputs = tf.concat((outputs,sc_op), axis=-1)
        _, _, h, w, ch = outputs.get_shape()
        h, w, ch = int(h), int(w), int(ch)
        #outputs_res = tf.reshape(outputs, (b_size*n_frames, h, w, ch))
        dec_out = create_decoder(outputs)
        self.segment_layer = dec_out
        #self.segment_layer = tf.reshape(dec_out, (b_size, n_frames, self.input_shape[0], self.input_shape[1], 1))
        if(config.train==False):
            #self.segment_layer=tf.nn.sigmoid(self.segment_layer)
            self.segment_layer_sig = tf.nn.sigmoid(self.segment_layer)
        print('Network initialized')

    def init_seg_loss(self):
        segment = self.segment_layer
        #y_bbox = self.y_input
        y_input_mask = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.y_input_mask, axis=-1), axis=-1), axis=-1)
        segmentation_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_input[:, 1:, :], logits=segment)
        segmentation_loss = segmentation_loss * y_input_mask[:, 1:, :]
        segmentation_loss = tf.reduce_mean(tf.reduce_sum(segmentation_loss, axis=[1, 2, 3, 4]))
        #pred_seg = tf.cast(tf.greater(segment, 0.0), tf.float32)
        #self.seg_acc = tf.reduce_mean(tf.cast(tf.equal(pred_seg, y_bbox), tf.float32))
        self.segmentation_loss = segmentation_loss #* config.hr_segment_coef
        print('Loss Initialized')

    def init_optimizer(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate, name='Adam')
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=config.learning_rate, name='SGD')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(loss=self.segmentation_loss, colocate_gradients_with_ops=True)

    def save(self, sess, file_name):
        save_path = self.saver.save(sess, file_name)
        print("Model saved in file: %s" % save_path)
        sys.stdout.flush()

    def load(self, sess, file_name):
        self.saver.restore(sess, file_name)
        print('Model restored.')
        sys.stdout.flush()

#a = FullNetwork()
