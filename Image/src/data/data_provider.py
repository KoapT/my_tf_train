#! /usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   Editor      : PyCharm
#   File name   : data_provider.py
#   Author      : Koap
#   Created date: 2020/6/24 下午5:03
#   Description :
#
#================================================================

import tensorflow as tf

feature_dict = {
      # Base content.
      'image/encoded':
          tf.FixedLenFeature((), tf.string, default_value=''),
      'image/filename':
          tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format':
          tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'image/height':
          tf.FixedLenFeature((), tf.int64, default_value=0),
      'image/width':
          tf.FixedLenFeature((), tf.int64, default_value=0),
      # Image-level labels.
      'image/class/label':
          tf.VarLenFeature(tf.int64),
      'image/class/text':
          tf.VarLenFeature(tf.string),
      # Object boxes and classes.
      'image/object/bbox/xmin':
          tf.VarLenFeature(tf.float32),
      'image/object/bbox/xmax':
          tf.VarLenFeature(tf.float32),
      'image/object/bbox/ymin':
          tf.VarLenFeature(tf.float32),
      'image/object/bbox/ymax':
          tf.VarLenFeature(tf.float32),
      'image/object/class/label':
          tf.VarLenFeature(tf.int64),
      'image/object/class/text':
          tf.VarLenFeature(tf.string),
      'image/object/area':
          tf.VarLenFeature(tf.float32),
      'image/object/is_crowd':
          tf.VarLenFeature(tf.int64),
      'image/object/difficult':
          tf.VarLenFeature(tf.int64),
      'image/object/group_of':
          tf.VarLenFeature(tf.int64),
      'image/object/weight':
          tf.VarLenFeature(tf.float32),
      'image/object/mask':
          tf.VarLenFeature(tf.string),
      # Segmentation mask.
      'image/segmentation/class/encoded':
          tf.FixedLenFeature((), tf.string, default_value=''),
      'image/segmentation/class/format':
          tf.FixedLenFeature((), tf.string, default_value='png'),
      # Depth.
      'image/depth/encoded':
          tf.FixedLenFeature((), tf.string, default_value=''),
      'image/depth/format':
          tf.FixedLenFeature((), tf.string, default_value='png'),
  }


def get_input(data_path,batch_size=16):
    dataset = tf.data.TFRecordDataset([data_path])

    def parser(record):
        features = tf.parse_single_example(
            record,
            features=feature_dict)

        # 解析图片和标签信息。
        decoded_image = tf.decode_raw(features['image_raw'], tf.uint8)
        reshaped_image = tf.reshape(decoded_image, [784])
        retyped_image = tf.cast(reshaped_image, tf.float32)
        label = tf.cast(features['label'], tf.int32)

        return retyped_image, label

    def preprocess():
        pass

    # 定义输入队列。
    dataset = dataset.map(parser, num_parallel_calls=8)
    dataset = dataset.map(preprocess, num_parallel_calls=8)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(10)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size*10)
    iterator = dataset.make_one_shot_iterator()  # 保证每个iterator只读取一次

    features, labels = iterator.get_next()
    return features, labels