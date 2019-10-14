import config
import os
import numpy as np
import time
from threading import Thread
from scipy.misc import imread, imresize
import json
from lstm_network import FullNetwork
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


def mkdir(dl_path):
    if not os.path.exists(dl_path):
        print("path doesn't exist. trying to make %s" % dl_path)
        os.mkdir(dl_path)
    else:
        print('%s exists, cannot make directory' % dl_path)
        #exit()

data_loc = '/groups/mshah/data/youtube-vos/'#'/home/kevin/HD2TB/Datasets/YoutubeVOS/'
#data_loc = '/home/jyoti/git/youtube_vos/youtube-vos/'#'/home/kevin/HD2TB/Datasets/YoutubeVOS/'
#data_loc ='/home/c2-1/yogesh/datasets/youtube-vos/'

def get_split_names(tr_or_val):
    split_file = data_loc + '%s_all_frames/meta.json' % tr_or_val

    with open(split_file) as f:
        data = json.load(f)
        files = list(data['videos'].keys())

    return sorted(files), data


def load_video_and_seg(file_name, tr_or_val='valid'):
    video_dir = data_loc + ('%s/JPEGImages/%s/' % (tr_or_val, file_name))
    #video_dir = data_loc + ('%s_all_frames/JPEGImages/%s/' % (tr_or_val, file_name))

    frame_names = sorted(os.listdir(video_dir))
    frames = []
    for fname in frame_names:
        frames.append(imread(video_dir + fname))

    video = np.stack(frames, axis=0)

    segment_dir = data_loc + ('%s/Annotations/%s/' % (tr_or_val, file_name))
    #segment_dir = data_loc + ('%s_all_frames/Annotations/%s/' % (tr_or_val, file_name))

    seg_frame_names = sorted(os.listdir(segment_dir))

    segmentations = []
    for seg_frame in seg_frame_names:
        i = frame_names.index(seg_frame[:-4] + '.jpg')
        image = Image.open(segment_dir + seg_frame)
        image_np = np.array(image)
        segmentations.append((i, image_np))

    if tr_or_val == 'valid':
        video_dir = data_loc + ('%s/JPEGImages/%s/' % (tr_or_val, file_name))
        save_frame_names = sorted(os.listdir(video_dir))
    else:
        save_frame_names = frame_names

    original_size = (video.shape[1], video.shape[2])

    return original_size, frame_names, video, segmentations, save_frame_names


def resize_frame(frame, target_size=(120, 120), interp='bilinear'):
    t_h, t_w = target_size

    return imresize(frame, (t_h, t_w), interp=interp)


def process_segmentations(segmentations, colors):
    # input is segmentations in for (i, color_seg) for each frame and the color values
    # output should is of the form (i, color, fb_seg) for each instance
    fin_segmentations = []
    for i, seg in segmentations:
        for color in colors:
            gt_seg = np.where(seg == color, 1, 0)
            if np.sum(gt_seg) == 0:
                continue
            gt_seg = np.expand_dims(gt_seg, axis=-1)
            #gt_seg = np.where(np.sum(gt_seg, axis=-1, keepdims=True) == 3, 1, 0)
            fin_segmentations.append((i, color, gt_seg))

    return fin_segmentations


class YoutubeValidDataGen(object):
    def __init__(self, valid_or_test='valid', sec_to_wait=5, frame_skip=1, n_threads=1, target_size=(120, 120), hr_target_size=(256, 448)):
        self.test_files, self.json_dict = get_split_names(valid_or_test)
        self.test_files = self.test_files
        self.sec_to_wait = sec_to_wait
        self.frame_skip = frame_skip

        self.valid_or_test = valid_or_test

        self.target_size = target_size
        self.hr_target_size = hr_target_size

        self.data_queue = []

        self.thread_list = []
        for i in range(n_threads):
            load_thread = Thread(target=self.__load_and_process_data)
            load_thread.start()
            self.thread_list.append(load_thread)

        print('Waiting %d (s) to load data' % sec_to_wait)
        time.sleep(self.sec_to_wait)

    def __load_and_process_data(self):
        while self.test_files:
            while len(self.data_queue) >= 5:
                time.sleep(1)

            vid_name = self.test_files.pop(0)

            orig_size, frame_names, video, segmentations, save_frame_names = load_video_and_seg(vid_name, tr_or_val=self.valid_or_test)

            colors = sorted([int(i) for i in self.json_dict['videos'][vid_name]['objects'].keys()])
            segmentations = process_segmentations(segmentations, colors)

            res_video = []
            for f in range(video.shape[0]):
                res_video.append(resize_frame(video[f], config.hr_frame_size, interp='bilinear'))
            res_video = np.stack(res_video, axis=0)

            '''
            vid_name contains the name
            original_size has (H, W) of original frame
            frame_names has the names of the frames [00000.jpg, 00005.jpg, ...]
            video contains the video scaled [0.0,1.0]
            segmentations contains a list (index of frame, color of segmentation, fb segmentation shape: H, W, 1)
            palette is the PIL palette used
            '''

            self.data_queue.append((vid_name, orig_size, frame_names, video/255., segmentations, save_frame_names, res_video/255.))
        print('Loading data thread finished')

    def get_next_video(self):
        while len(self.data_queue) == 0:
            print('Waiting on data')
            time.sleep(self.sec_to_wait)

        return self.data_queue.pop(0)

    def has_data(self):
        return self.data_queue != [] or self.test_files != []


def disp_video(vid_hr, vid_lr, seg_hr, seg_lr):
    fig = plt.figure(figsize=(8, 8))
    for i in range(8):
        fig.add_subplot(8, 8, i+1)
        plt.imshow(vid_hr[i, :, :, :])
    for i in range(8):
        fig.add_subplot(8, 8, 8+i+1)
        plt.imshow(vid_lr[i, :, :, :])
    for i in range(8):
        fig.add_subplot(8, 8, 16+i+1)
        plt.imshow(seg_hr[i, :, :, 0])
    for i in range(8):
        fig.add_subplot(8, 8, 24+i+1)
        plt.imshow(seg_lr[i, :, :, 0])
    plt.show()


def get_bounds1(img):
    h_sum = np.sum(img, axis=1)
    w_sum = np.sum(img, axis=0)

    hs = np.where(h_sum > 0)
    ws = np.where(w_sum > 0)

    try:
        h0 = hs[0][0]
        h1 = hs[0][-1]
        w0 = ws[0][0]
        w1 = ws[0][-1]
    except:
        return 0, img.shape[0], 0, img.shape[1]

    return h0, h1, w0, w1


def gen_sub(sess, lstm_network):
    # makes the file which things come in
    fin_dir = config.output_inference_file
    mkdir(fin_dir)

    datagen = YoutubeValidDataGen(valid_or_test='valid', target_size=config.frame_size, hr_target_size=config.hr_frame_size)

    seg_ph = tf.placeholder(dtype=tf.float32, shape=(None, config.hr_frame_size[0], config.hr_frame_size[1], 1))
    size_ph = tf.placeholder(dtype=tf.int32, shape=(2,))
    resize_op = tf.image.resize_images(seg_ph, size_ph, tf.image.ResizeMethod.BILINEAR)

    test_img = Image.open('/groups/mshah/data/youtube-vos/valid/Annotations/0a49f5265b/00000.png')
    #test_img = Image.open('/home/jyoti/git/youtube_vos/youtube-vos/valid/Annotations/0a49f5265b/00000.png')
    vid_counter = 0
    start_time = time.time()
    while datagen.has_data():
        vid_name, original_size, frame_names, video, segmentations, save_frame_names, res_video = datagen.get_next_video()

        fb_segs, colors = [], []

        f, h, w, _ = video.shape
        n_frames = 10

        for frame_start, color, start_segmentation in sorted(segmentations, key=lambda k: k[1]):
            fb_segmentation = np.zeros((f, h, w, 1))
            colors.append(color)

            i = frame_start
            fb_segmentation[i] = start_segmentation

            first_frame_seg = resize_frame(start_segmentation[:, :, 0], (config.hr_frame_size[0], config.hr_frame_size[1]), 'nearest')
            first_frame_seg = np.expand_dims(first_frame_seg, axis=-1) / 255.

            while i < f:
                if(f-i<n_frames):
                    new_video_in = np.concatenate((res_video[i:i+n_frames], np.repeat(np.zeros(res_video[0: 1].shape), n_frames-(f-i) , axis=0)), axis=0)
                else:
                    new_video_in = res_video[i:i+n_frames]

                # print('f:', f)
                # print('i:', i)
                # print('f-i:', f-i)
                # print('new_video_in shape:', new_video_in.shape)
                # print('first_frame_seg shape:', first_frame_seg.shape)
                seg_pred = sess.run(lstm_network.segment_layer_sig, feed_dict={lstm_network.x_vid_input: [new_video_in],
                                                                               lstm_network.x_seg_input: [first_frame_seg]})

                print('Newtork output extracted')

                s =  np.expand_dims(first_frame_seg, 0)

                s = np.expand_dims(s, 0)

                seg_pred = np.concatenate((s, seg_pred), 1)

                first_frame_seg = np.round(seg_pred[0][-1])

                res_seg_pred = sess.run(resize_op, feed_dict={seg_ph: seg_pred[0], size_ph: (h, w)}) 

                # print('res_seg_pred shape:', res_seg_pred.shape)    
                # print('i+n_frames', i+n_frames)  

                if(f-i<n_frames):
                    fb_segmentation[i:f] = res_seg_pred[:f-i]
                else:
                    fb_segmentation[i:i+n_frames] = res_seg_pred

                i += n_frames-1

            fb_segs.append(fb_segmentation)

        colors.insert(0, 0)
        #print(colors)
        if colors != sorted(colors):
            print('Error Colors')
            exit()

        fb_segs.insert(0, np.ones((f, h, w, 1)) * .499999999)
        fb_segs_concat = np.concatenate(fb_segs, axis=-1)

        fb_segs_argmax = np.argmax(fb_segs_concat, axis=-1).astype(dtype=np.uint8)

        # creates the directory for the video
        vid_dir = fin_dir + vid_name + '/'
        mkdir(vid_dir)

        # saves predicted segmentation
        for f in range(fb_segs_argmax.shape[0]):
            frame_name = vid_dir + frame_names[f][:-4] + '.png'
            if (frame_names[f][:-4] + '.jpg') in save_frame_names:
                c = Image.fromarray(fb_segs_argmax[f], mode='P')
                c.putpalette(test_img.getpalette())
                c.save(frame_name, "PNG", mode='P')

        vid_counter += 1

        if vid_counter % 50 == 0:
            print('Finished %d videos in %d seconds.' % (vid_counter, time.time() - start_time))


def main():
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    lstm_network = FullNetwork(input_shape=config.hr_frame_size)
    with tf.Session(graph=lstm_network.graph, config=gpu_config) as sess:
        tf.global_variables_initializer().run()
        lstm_network.load(sess, config.save_file_best_name % config.epoch_save)
        #print('a', tf.Variable(lstm_network[name][1], name="biases"))
        #print('a', tf.Variable(lstm_network[name][1], name="biases"))
        gen_sub(sess, lstm_network)


main()