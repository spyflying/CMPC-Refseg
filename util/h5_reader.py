from __future__ import print_function

import numpy as np
import os
import threading
import queue as queue
import h5py

def run_prefetch(prefetch_queue, h5_file, h5_img, num_batch, shuffle):
    n_batch_prefetch = 0
    fetch_order = np.arange(num_batch)
    img_size=(320, 320)
    while True:
        # Shuffle the batch order for every epoch
        if n_batch_prefetch == 0 and shuffle:
            fetch_order = np.random.permutation(num_batch)

        # Load batch from file
        batch_id = fetch_order[n_batch_prefetch]
        mask = h5_file['answers'][batch_id]   # [320, 320]
        image_id = h5_file['image_idxs'][batch_id]   # int
        refexp = h5_file['refexps'][batch_id]     # [60]
        sent = h5_file['sentence'][batch_id]

        # read images
        img = h5_img['images'][image_id]  # [320, 320, 3]
        batch = {'mask_batch': mask,
                 'text_batch': refexp,
                 'im_batch': img}

        # add loaded batch to fetchqing queue
        prefetch_queue.put(batch, block=True)

        # Move to next batch
        n_batch_prefetch = (n_batch_prefetch + 1) % num_batch

class DataReader:
    def __init__(self, h5_file_name, h5_image_name, shuffle=True, prefetch_num=8):
        # self.img_folder = img_folder
        self.h5_file_name = h5_file_name
        self.h5_image = h5_image_name
        self.shuffle = shuffle
        self.prefetch_num = prefetch_num

        self.n_batch = 0
        self.n_epoch = 0

        # Search the folder to see the number of num_batch
        self.h5_file = h5py.File(h5_file_name, 'r')
        self.h5_image = h5py.File(h5_image_name, 'r')
        num_batch = self.h5_file['image_idxs'].shape[0]   # n?
        if num_batch > 0:
            print('found %d batches within %s' % (num_batch, h5_file_name))
        else:
            raise RuntimeError('no batches within %s' % (h5_file_name))
        self.num_batch = num_batch  # 一共有多少个batch

        # Start prefetching thread
        self.prefetch_queue = queue.Queue(maxsize=prefetch_num)
        # 读数据的线程，只有一个？
        self.prefetch_thread = threading.Thread(target=run_prefetch,
            args=(self.prefetch_queue, self.h5_file,
                  self.h5_image, self.num_batch, self.shuffle))
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def read_batch(self, is_log = True):
        if is_log:
            print('data reader: epoch = %d, batch = %d / %d' % (self.n_epoch, self.n_batch, self.num_batch))

        # Get a batch from the prefetching queue
        if self.prefetch_queue.empty():
            print('data reader: waiting for file input (IO is slow)...')
        batch = self.prefetch_queue.get(block=True)
        self.n_batch = (self.n_batch + 1) % self.num_batch
        self.n_epoch += (self.n_batch == 0)
        return batch
