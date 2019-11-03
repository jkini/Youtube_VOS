import config
import tensorflow as tf
import sys
from network_modules import mem_encoder, curr_encoder, attention, curr_decoder

class FullNetwork(object):
    def __init__(self, input_shape=(256, 448)):
        self.graph = tf.Graph()
        self.input_shape = input_shape
        with self.graph.as_default():
            self.x_input = tf.placeholder(dtype=tf.float32, shape=(None, None, input_shape[0], input_shape[1], 3),
                                            name='x_input')
            self.x_seg_input = tf.placeholder(dtype=tf.float32, shape=(None, None, input_shape[0], input_shape[1], 1),
                                            name='x_seg_input')
            self.y_input = tf.placeholder(dtype=tf.float32, shape=(None, None, input_shape[0], input_shape[1], 1),
                                        name='y_input')
            self.y_input_mask = tf.placeholder(dtype=tf.float32, shape=(None, None))
            self.init_network()
            self._variables_to_restore()
            self.init_seg_loss()
            self._summary()
            self.init_optimizer()               
            self.saver = tf.train.Saver()

    def init_network(self):
#         c_0, h_0 = create_vgg_initializer(self.x_vid_input[:, 0], self.x_seg_input)
#         x_vid_input_res = self.x_vid_input[:, 1:]
#         x_vid_input_res = tf.reshape(x_vid_input_res,
#                                     (-1, self.input_shape[0], self.input_shape[1], 3))
#         x_t = create_vgg_encoder(x_vid_input_res)
#         _, h, w, ch = x_t.get_shape()
#         h, w, ch = int(h), int(w), int(ch)
#         x_t = tf.reshape(x_t, (config.batch_size, -1, h, w, ch))
#         outputs, state = create_convlstm(c_0, h_0, x_t)
#         _, _, h, w, ch = outputs.get_shape()
#         h, w, ch = int(h), int(w), int(ch)
#         outputs_res = tf.reshape(outputs, (-1, h, w, ch))
#         dec_out = create_decoder(outputs_res)
#         self.segment_layer = tf.reshape(dec_out, (config.batch_size, -1, self.input_shape[0], self.input_shape[1], 1))
#         self.segment_layer_sig = tf.nn.sigmoid(self.segment_layer)
#         print('Network initialized')
        
#         c_0, h_0 = create_vgg_initializer(self.x_vid_input[:, 0, :], self.x_seg_input)
        
        # x_t, block1, block2, block3 = create_resnet_curr_encoder(self.x_vid_input[:, 1:, :])
        # _, t, h, w, ch = x_t.get_shape()
        # h, w, ch = int(h), int(w), int(ch)
        # dec_out = create_curr_decoder(x_t, block1, block2, block3)

        mem_key, mem_value, mem_net = mem_encoder(self.x_input[:, :-1, :], self.x_seg_input, is_training=False)
        curr_key, curr_value, curr_net, block1, block2, block3 = curr_encoder(self.x_input[:, -1:, :], is_training=False)
        attn_net = attention(mem_key, mem_value, curr_key, curr_value)
        out_seg = curr_decoder(attn_net, block1, block2, block3)

        self.segment_layer = out_seg
        self.segment_layer_soft = tf.nn.softmax(self.segment_layer)
        print('Network initialized')
    
    def _variables_to_restore(self):
        exclude = ['mem_encoder/resnet_v1_50/mem', 'curr_encoder/resnet_v1_50/curr', 'attention', 'decoder']
        self.variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=exclude)

    def init_seg_loss(self):
        segment = self.segment_layer
        y_input_mask = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.y_input_mask, axis=-1), axis=-1), axis=-1)
        segmentation_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_input[:, -1:, :], logits=segment)
        segmentation_loss = segmentation_loss * y_input_mask[:, -1:, :]
        segmentation_loss = tf.reduce_mean(tf.reduce_sum(segmentation_loss, axis=[1, 2, 3, 4]))
        self.segmentation_loss = segmentation_loss
        print('Loss Initialized')

    def init_optimizer(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate, name='Adam')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(loss=self.segmentation_loss, colocate_gradients_with_ops=True)
            
    def _summary(self):
        summary = []
        summary.append(tf.summary.scalar('seg_loss', self.segmentation_loss))
        inp_img = tf.reshape(self.x_input,(-1, self.input_shape[0], self.input_shape[1], 3))
        summary.append(tf.summary.image('input image', inp_img))
        pred_mask = tf.reshape(self.segment_layer_soft,(-1, self.input_shape[0], self.input_shape[1], 1))
        summary.append(tf.summary.image('predicted mask', pred_mask))
        self.summary = tf.summary.merge(summary)

    def save(self, sess, file_name):
        save_path = self.saver.save(sess, file_name)
        print("Model saved in file: %s" % save_path)
        sys.stdout.flush()

    def load(self, sess, file_name):
        self.saver.restore(sess, file_name)
        print('Model restored.')
        sys.stdout.flush()
