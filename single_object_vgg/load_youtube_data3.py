import json
import os
from scipy.misc import imread, imresize
import numpy as np
import random
from threading import Thread
import time
from PIL import Image
import sys
import config

data_loc = config.data_dir
max_vids = 100

def get_split_names(tr_or_val):
    split_file = data_loc + '%s/meta.json' % tr_or_val

    all_files = []
    with open(split_file) as f:
        data = json.load(f)
        files = sorted(list(data['videos'].keys()))
        for file_name in files:
            all_files.append((file_name, data['videos'][file_name]['objects']))
    return all_files


# This does not use interpolated frames
def load_video(file_name, tr_or_val='train', shuffle=True, n_frames=8, frame_skip=1, use_all=True):
    video_dir_orig = data_loc + ('%s/JPEGImages/%s/' % (tr_or_val, file_name))
    if use_all:
        video_dir = data_loc + ('%s_all_frames/JPEGImages/%s/' % (tr_or_val, file_name))
        segment_dir = data_loc + ('%s_all_frames/Annotations/%s/' % (tr_or_val, file_name))
    else:
        video_dir = data_loc + ('%s/JPEGImages/%s/' % (tr_or_val, file_name))
        segment_dir = data_loc + ('%s/Annotations/%s/' % (tr_or_val, file_name))
        frame_skip = 1

    orig_frame_names = sorted(os.listdir(video_dir_orig))
    frame_names = sorted(os.listdir(video_dir))
    seg_frame_names = sorted(os.listdir(segment_dir))
    frame_names = sorted([x for x in frame_names if x[:-4] + '.png' in seg_frame_names])

    try:
        if shuffle:
            start_frame = np.random.randint(0, len(frame_names) - n_frames*frame_skip)
        else:
            start_frame = 0
    except:
        # print('%s does not have 8 or more frames.' % file_name)
        start_frame = 0

    while frame_names[start_frame] not in orig_frame_names:  # ensures the first frame is not interpolated
        start_frame -= 1

    # loads video
    frames = []
    for f in range(start_frame, start_frame + n_frames*frame_skip, frame_skip):
        try:
            frames.append(imread(video_dir + frame_names[f], mode='RGB'))
        except Exception as ex:
            print('Exception:', ex)
            sys.stdout.flush()        
            print('File name:', file_name)
            sys.stdout.flush()     
            frames.append(frames[-1])

    video = np.stack(frames, axis=0)

    # loads segmentations
    frames, y_mask = [], []
    for f in range(start_frame, start_frame + n_frames*frame_skip, frame_skip):
        try:
            frames.append(np.array(Image.open(segment_dir + seg_frame_names[f])))
            y_mask.append(1.0)
        except:
            frames.append(frames[-1])
            y_mask.append(0.0)

        # try:
            # if (seg_frame_names[f][:-4] + '.jpg') not in orig_frame_names:
                # frames.append(frames[-1])
                # y_mask.append(0.0)
            # else:
                # frames.append(np.array(Image.open(segment_dir + seg_frame_names[f])))
                # y_mask.append(1.0)
        # except:
            # frames.append(frames[-1])
            # y_mask.append(0.0)
        
    segmentation = np.stack(frames, axis=0)

    return np.asarray(video), np.asarray(segmentation), np.asarray(y_mask)


def resize_video(video, segmentation, target_size=(120, 120)):
    frames, h, w, _ = video.shape

    t_h, t_w = target_size

    video_res = np.zeros((frames, t_h, t_w, 3), np.uint8)
    segment_res = np.zeros((frames, t_h, t_w), np.uint8)
    for frame in range(frames):
        video_res[frame] = imresize(video[frame], (t_h, t_w))
        segment_res[frame] = imresize(segmentation[frame], (t_h, t_w), interp='nearest')

    return np.asarray(video_res/255.), np.asarray(segment_res/255.)


class YoutubeTrainDataGen(object):
    def __init__(self, sec_to_wait=5, n_threads=10, crop_size=(256, 448), augment_data=True, n_frames=8,
                 rand_frame_skip=4, use_all=True):
        self.train_files = get_split_names('train')

        self.sec_to_wait = sec_to_wait

        self.augment = augment_data
        self.rand_frame_skip = rand_frame_skip
        self.use_all = use_all

        self.crop_size = crop_size

        self.n_frames = n_frames

        np.random.seed(None)
        random.shuffle(self.train_files)

        self.data_queue = []

        self.thread_list = []
        for i in range(n_threads):
            load_thread = Thread(target=self.__load_and_process_data)
            load_thread.start()
            self.thread_list.append(load_thread)

        print('Waiting %d (s) to load data' % sec_to_wait)
        sys.stdout.flush()
        time.sleep(self.sec_to_wait)

    def __load_and_process_data(self):
        while self.train_files:
            while len(self.data_queue) >= max_vids:
                time.sleep(1)

            try:
                vid_name, fdict = self.train_files.pop()
            except:
                continue  # Thread issue

            frame_skip = np.random.randint(self.rand_frame_skip) + 1

            video, segmentation, y_mask = load_video(vid_name, tr_or_val='train', n_frames=self.n_frames, frame_skip=frame_skip, use_all=self.use_all)

            # find objects in the first frame
            allowable_colors = []
            for obj_id in fdict.keys():
                color = int(obj_id)
                if np.any(segmentation[0] == color):
                    allowable_colors.append(color)

            # no objects in the first frame
            if len(allowable_colors) == 0:
                continue

            # selects the object and gets the objects segmentation for the clip
            color = random.choice(allowable_colors)
            gt_seg = np.where(segmentation == color, 1, 0)

            #cropped_video, cropped_segmentation = perform_window_crop(video, gt_seg)

            # if self.augment:
            #     cropped_video, cropped_segmentation = flip_clip(cropped_video, cropped_segmentation)

            cropped_video, cropped_segmentation = video, gt_seg

            video_hr, seg_hr = resize_video(cropped_video, cropped_segmentation, self.crop_size)

            seg_hr = np.expand_dims(seg_hr, axis=-1)

            self.data_queue.append((video_hr, seg_hr, y_mask))
        print('Loading data thread finished')
        sys.stdout.flush()

    def get_batch(self, batch_size=5):
        while len(self.data_queue) < batch_size and self.train_files:
            print('Waiting on data. # Already Loaded = %s' % str(len(self.data_queue)))
            sys.stdout.flush()
            time.sleep(self.sec_to_wait)

        batch_size = min(batch_size, len(self.data_queue))
        batch_x_hr, batch_seg_hr, batch_y_mask = [], [], []
        for i in range(batch_size):
            vid_hr, bbox_hr, y_mask = self.data_queue.pop(0)
            batch_x_hr.append(vid_hr)
            batch_seg_hr.append(bbox_hr)
            batch_y_mask.append(y_mask)

        return np.asarray(batch_x_hr), np.asarray(batch_seg_hr), np.asarray(batch_y_mask)

    def has_data(self, batch_size=5):
        return len(self.data_queue) >= batch_size or len(self.train_files) >= batch_size
