# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:59:07 2017

@author: tomhope
"""

from __future__ import print_function
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

tf.compat.v1.disable_eager_execution()

save_dir = "../data/mnist"

# Download data to save_dir
data_sets = tfds.load(name='mnist', data_dir=save_dir)

data_splits = ["train", "test", "validation"]
for d in range(len(data_splits)):
    print("saving " + data_splits[d])
    data_set = data_sets[d]

    filename = os.path.join(save_dir, data_splits[d] + '.tfrecords')
    writer = tf.compat.v1.python_io.TFRecordWriter(filename)
    for index in range(data_set.images.shape[0]):
        image = data_set.images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'height': tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[data_set.images.shape[1]])),
                'width': tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[data_set.images.shape[2]])),
                'depth': tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[data_set.images.shape[3]])),
                'label': tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[int(data_set.labels[index])])),
                'image_raw': tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[image]))}))
        writer.write(example.SerializeToString())
    writer.close()


filename = os.path.join(save_dir, 'train.tfrecords')
record_iterator = tf.compat.v1.python_io.tf_record_iterator(filename)
seralized_img_example = next(record_iterator)

example = tf.train.Example()
example.ParseFromString(seralized_img_example)
image = example.features.feature['image_raw'].bytes_list.value
label = example.features.feature['label'].int64_list.value[0]
width = example.features.feature['width'].int64_list.value[0]
height = example.features.feature['height'].int64_list.value[0]

img_flat = np.fromstring(image[0], dtype=np.uint8)
img_reshaped = img_flat.reshape((height, width, -1))
