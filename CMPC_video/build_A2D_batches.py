import os
import numpy as np
import skimage
import skimage.io
import csv
import glob
import h5py
import re

from tqdm import tqdm
from util import im_processing, text_processing

debug = False
# root directory
root_dir = os.getcwd()
# data directory
a2d_dir = '/mnt/lustre/share/huitianrui/DATASET/A2D-Sentences'


def build_a2d_batches(T, input_H, input_W, video=False):
    """
    Build data batches of A2D Sentence dataset

    Args:
         T: limit of number of words
         input_H: height of input frame of I3D backbone
         input_W: width of input frame of I3D backbone
         video: select consecutive frames or standalone frame
    """

    query_file = os.path.join(a2d_dir, 'a2d_annotation.txt')
    frame_dir = os.path.join(a2d_dir, 'Release/frames')
    vocab_file = os.path.join(root_dir, 'data/vocabulary_Gref.txt')

    dataset_name = 'a2d_sent_new'
    out_dataset_dir = os.path.join(root_dir, dataset_name)
    if not os.path.exists(out_dataset_dir):
        os.mkdir(out_dataset_dir)
    test_batch = os.path.join(out_dataset_dir, 'test_batch')
    train_batch = os.path.join(out_dataset_dir, 'train_batch')
    if not os.path.exists(test_batch): 
        os.mkdir(test_batch)
    if not os.path.exists(train_batch):
        os.mkdir(train_batch)

    vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)
    test_prefix_list = list()
    train_prefix_list = list()
    split_dict = gen_split_dict()
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')

    with open(query_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        total_count = 0
        test_count = 0
        train_count = 0
        all_zero_mask_count = 0
        for row in tqdm(reader):
            # each video belongs to test or train
            video_id = row[0]
            data_prefix = video_id
            if split_dict[data_prefix] == 1:
                save_dir = test_batch
                test_prefix_list.append(data_prefix)
                test = True
            else:
                save_dir = train_batch
                train_prefix_list.append(data_prefix)
                test = False
            # load sentence
            instance_id = int(row[1])
            sent = row[2].lower()
            words = SENTENCE_SPLIT_REGEX.split(sent.strip())
            words = [w for w in words if len(w.strip()) > 0]
            # remove punctuation and restrict sentence within 20 words
            if words[-1] == '.':
                words = words[:-1]
            if len(words) > T:
                words = words[:T]
            n_sent = ""
            for w in words:
                n_sent = n_sent + w + ' '
            n_sent = n_sent.strip()
            n_sent = n_sent.encode('utf-8').decode("utf-8")
            text = text_processing.preprocess_sentence(n_sent, vocab_dict, T)

            image_paths = list()
            # for each video, get all the gt masks of a certain instance
            masks, frame_ids = get_masks(video_id, instance_id)

            for frame_id in frame_ids:
                image_path = os.path.join(frame_dir, video_id, '{:0>5d}.png'.format(frame_id))
                image_paths.append(image_path)

            for frame_id, image_path, mask in zip(frame_ids, image_paths, masks):
                # abandon all zero mask batch
                if np.sum(mask) == 0:
                    print("all zeros mask caught")
                    all_zero_mask_count += 1
                    continue
                if video:
                    # obtain 16 consecutive frames centered at the gt frame
                    frame_paths = frame_range(frame_id=frame_id, frame_dir=os.path.join(frame_dir, video_id))
                else:
                    # only use the gt frame
                    frame_paths = list()
                frames = list()
                if test:
                    count = test_count
                    test_count = test_count + 1
                    prefix = 'test_'
                    image = skimage.io.imread(image_path)
                    for frame_path in frame_paths:
                        frames.append(skimage.io.imread(frame_path))
                else:
                    prefix = 'train_'
                    count = train_count
                    train_count = train_count + 1
                    image = skimage.io.imread(image_path)
                    image = skimage.img_as_ubyte(im_processing.resize_and_pad(image, input_H, input_W))
                    mask = im_processing.resize_and_pad(mask, input_H, input_W)
                    for frame_path in frame_paths:
                        frame = skimage.io.imread(frame_path)
                        frame = skimage.img_as_ubyte(im_processing.resize_and_pad(frame, input_H, input_W))
                        frames.append(frame)

                if debug:
                    m0 = mask[:, :, np.newaxis]
                    m0 = (m0 > 0).astype(np.uint8)
                    m0 = np.concatenate([m0, m0, m0], axis=2)
                    debug_image = image * m0
                    skimage.io.imsave('./debug/{}_{}_{}.png'.format(data_prefix, frame_id,
                                                                    sent.replace(' ', '_')), debug_image)

                # save batches
                np.savez(file=os.path.join(save_dir, dataset_name + '_' + prefix + str(count)),
                         text_batch=text,
                         mask_batch=(mask > 0),
                         sent_batch=[sent],
                         im_batch=image,
                         frame_id=frame_id,
                         frames=frames)
                total_count = total_count + 1

        print()
        print("num of all zeros masks is: {}".format(all_zero_mask_count))


def frame_range(frame_id, frame_dir):
    frame_paths = os.listdir(frame_dir)
    frame_paths.sort()
    biggest = frame_paths[-1]
    frame_num = int(biggest[:-4])
    start = frame_id - 8
    end = frame_id + 8
    result = list()
    for i in range(start, end):
        if i < 1:
            frame_id = 1
        elif i > frame_num:
            frame_id = frame_num
        else:
            frame_id = i
        result.append(os.path.join(frame_dir, '{:0>5d}.png'.format(frame_id)))
    assert len(result) == 16
    return result


def gen_split_dict():
    split_file = os.path.join(a2d_dir, 'Release/videoset.csv')
    result = dict()
    result.setdefault(0)
    with open(split_file, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            video_id = line[0]
            split_code = line[-1]
            result[video_id] = int(split_code)
    return result


def get_masks(video_id, instance_id):
    anno_dir = os.path.join(a2d_dir, 'a2d_annotation_with_instances')
    masks_path = os.path.join(anno_dir, video_id, '*')
    mask_files = glob.glob(masks_path)
    mask_files.sort()
    masks = list()
    frame_ids = list()

    for mask_file in mask_files:
        f = h5py.File(mask_file, 'r')
        instance_ids = f['instance'][:]
        if instance_ids.shape[0] == 1:
            mask = f['reMask'][:].T
        else:
            index = np.argwhere(instance_ids == instance_id)
            index = np.squeeze(index)
            mask = f['reMask'][index].T
            mask = np.squeeze(mask)
            if index.size != 1:
                mask = np.sum(mask, axis=2)

        masks.append(mask)
        base_name = os.path.basename(mask_file)
        frame_id = int(base_name[:-3])
        frame_ids.append(frame_id)
        f.close()
    return masks, frame_ids


if __name__ == "__main__":
    T = 20
    input_H = 320
    input_W = 320
    build_a2d_batches(T=T, input_H=input_H, input_W=input_W, video=True)
