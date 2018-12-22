# -*- coding: utf-8 -*-
from fsrcnnutils import (
  read_data, 
  thread_train_setup,
  train_input_setup,
  test_input_setup,
  save_params,
  merge,
  array_image_save
)
from PIL import Image
from cliqueblock import *
from newmodel import *

import time
import os
import numpy as np
import tensorflow as tf
import pdb

#===========================================12-8 model2 =ba shanchu de cancha tianjia l jia l yige conv layer========

# Based on http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html
class FSRCNN(object):
  
  def __init__(self, sess, config):
    self.sess = sess
    self.fast = config.fast
    self.train = config.train
    self.c_dim = config.c_dim
    self.is_grayscale = (self.c_dim == 1)
    self.epoch = config.epoch
    self.scale = config.scale
    self.stride = config.stride
    self.batch_size = config.batch_size
    self.learning_rate = config.learning_rate
    self.momentum = config.momentum
    self.threads = config.threads
    self.params = config.params

    # Different image/label sub-sizes for different scaling factors x2, x3, x4
    scale_factors = [[14, 20], [11, 21], [10, 24]]
    self.image_size, self.label_size = scale_factors[self.scale - 2]
    # Testing uses different strides to ensure sub-images line up correctly
    if not self.train:
      self.stride = [10, 7, 6][self.scale - 2]

    # Different model layer counts and filter sizes for FSRCNN vs FSRCNN-s (fast), (s, d, m) in paper
    model_params = [[56, 12, 4], [32, 5, 1]]
    self.model_params = model_params[self.fast]
    
    self.checkpoint_dir = config.checkpoint_dir
    self.output_dir = config.output_dir
    self.data_dir = config.data_dir
    self.build_model()

#############========================================================================================================
  def build_model(self):
    self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
    self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')
    # Batch size differs in training vs testing
    self.batch = tf.placeholder(tf.int32, shape=[], name='batch')

    # # FSCRNN-s (fast) has smaller filters and less layers but can achieve faster performance
    s, d, m = self.model_params
    #
    expand_weight, deconv_weight = 'w{}'.format(m + 3), 'w{}'.format(m + 4)
    self.weights = {
      'w1': tf.Variable(tf.random_normal([5, 5, 1, s], stddev=0.0378, dtype=tf.float32), name='w1'),
      'w2': tf.Variable(tf.random_normal([1, 1, s, d], stddev=0.3536, dtype=tf.float32), name='w2'),
      expand_weight: tf.Variable(tf.random_normal([1, 1, d, s], stddev=0.189, dtype=tf.float32), name=expand_weight),
      deconv_weight: tf.Variable(tf.random_normal([9, 9, 1, s], stddev=0.0001, dtype=tf.float32), name=deconv_weight)
    }

    expand_bias, deconv_bias = 'b{}'.format(m + 3), 'b{}'.format(m + 4)
    self.biases = {
      'b1': tf.Variable(tf.zeros([s]), name='b1'),
      'b2': tf.Variable(tf.zeros([d]), name='b2'),
      expand_bias: tf.Variable(tf.zeros([s]), name=expand_bias),
      deconv_bias: tf.Variable(tf.zeros([1]), name=deconv_bias)
    }

    # Create the m mapping layers weights/biases
    for i in range(3, m + 3):
      weight_name, bias_name = 'w{}'.format(i), 'b{}'.format(i)
      self.weights[weight_name] = tf.Variable(tf.random_normal([3, 3, d, d], stddev=0.1179, dtype=tf.float32), name=weight_name)
      self.biases[bias_name] = tf.Variable(tf.zeros([d]), name=bias_name)
    # tensor=ourmodel(self.images)
    self.pred = self.model()
    # self.pred = tensor

#=======================================================================================================================

    # Loss function (MSE)
    self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.labels - self.pred), reduction_indices=0))
    print(self.labels)
    print(self.pred)

    self.saver = tf.train.Saver()

  def run(self):
    # SGD with momentum ##adam
    self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    tf.initialize_all_variables().run()

    # if self.load(self.checkpoint_dir):
    #   print(" [*] Load SUCCESS")
    # else:
    #   print(" [!] Load failed...")

    if self.params:
      save_params(self.sess, self.weights, self.biases)
    elif self.train:
      self.run_train()
    else:
      self.run_test()

  def run_train(self):
    start_time = time.time()
    print("Beginning training setup...")
    if self.threads == 1:
      train_input_setup(self)
    else:
      thread_train_setup(self)
    print("Training setup took {} seconds with {} threads".format(time.time() - start_time, self.threads))

    data_dir = os.path.join('./{}'.format(self.checkpoint_dir), "train.h5")
    train_data, train_label = read_data(data_dir)
    print("Total setup time took {} seconds with {} threads".format(time.time() - start_time, self.threads))

    print("Training...")
    start_time = time.time()
    start_average, end_average, counter = 0, 0, 0

    for ep in xrange(self.epoch):
      # Run by batch images
      batch_idxs = len(train_data) // self.batch_size
      batch_average = 0
      for idx in xrange(0, batch_idxs):
        batch_images = train_data[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_labels = train_label[idx * self.batch_size : (idx + 1) * self.batch_size]

        counter += 1
        _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels, self.batch: self.batch_size})
        batch_average += err

        if counter % 10 == 0:
          print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
            % ((ep+1), counter, time.time() - start_time, err))

        # Save every 500 steps
        if counter % 500 == 0:
          self.save(self.checkpoint_dir, counter)
      print(batch_idxs)
      batch_average = float(batch_average) / batch_idxs

      if ep < (self.epoch * 0.2):
        start_average += batch_average
      elif ep >= (self.epoch * 0.8):
        end_average += batch_average

    # Compare loss of the first 20% and the last 20% epochs
    start_average = float(start_average) / (self.epoch * 0.2)
    end_average = float(end_average) / (self.epoch * 0.2)
    print("Start Average: [%.6f], End Average: [%.6f], Improved: [%.2f%%]" \
      % (start_average, end_average, 100 - (100*end_average/start_average)))

    # Linux desktop notification when training has been completed
    # title = "Training complete - FSRCNN"
    # notification = "{}-{}-{} done training after {} epochs".format(self.image_size, self.label_size, self.stride, self.epoch);
    # notify_command = 'notify-send "{}" "{}"'.format(title, notification)
    # os.system(notify_command)

  def run_test(self):
    nx, ny = test_input_setup(self)
    data_dir = os.path.join('./{}'.format(self.checkpoint_dir), "test.h5")
    test_data, test_label = read_data(data_dir)

    print("Testing...")

    start_time = time.time()
    result = self.pred.eval({self.images: test_data, self.labels: test_label, self.batch: nx * ny})
    print("Took %.3f seconds" % (time.time() - start_time))

    result = merge(result, [nx, ny])
    result = result.squeeze()
    image_path = os.path.join(os.getcwd(), self.output_dir)
    image_path = os.path.join(image_path, "test_image.png")

    array_image_save(result * 255, image_path)

#=====================================================================================================================##
  def model1(self):
    # ===============================Feature Extraction===============================================================##
    conv_feature = self.prelu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='VALID') + self.biases['b1'], 1)
    print(conv_feature)
    # Shrinking
    conv_shrink = self.prelu(tf.nn.conv2d(conv_feature, self.weights['w2'], strides=[1,1,1,1], padding='SAME') + self.biases['b2'], 2)

    # ===========================Mapping (# mapping layers = m)====================================================#####
    prev_layer, m = conv_shrink, self.model_params[2]
    for i in range(3, m + 3):
      weights, biases = self.weights['w{}'.format(i)], self.biases['b{}'.format(i)]
      prev_layer = self.prelu(tf.nn.conv2d(prev_layer, weights, strides=[1,1,1,1], padding='SAME') + biases, i)

    s, d, m = self.model_params
    conv_s_w2 = tf.get_variable("conv_s_w2", [1, 1, d, s],
                                initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
    conv_s_b2 = tf.get_variable("conv_s_b2", [s], initializer=tf.constant_initializer(0))

    dconv_s_w2 = tf.get_variable("dconv_s_w2", [9, 9, 1, s],
                                 initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
    dconv_s_b2 = tf.get_variable("dconv_s_b2", [1], initializer=tf.constant_initializer(0))

    # ======================================Expanding================================================================###
    expand_weights, expand_biases = self.weights['w{}'.format(m + 3)], self.biases['b{}'.format(m + 3)]
    conv_expand = self.prelu(tf.nn.conv2d(prev_layer, conv_s_w2, strides=[1,1,1,1], padding='SAME') + conv_s_b2, 7)

    # ====================================Deconvolution=============================================================####
    deconv_output = [self.batch, self.label_size, self.label_size, self.c_dim]
    deconv_stride = [1,  self.scale, self.scale, 1]
    deconv_weights, deconv_biases = self.weights['w{}'.format(m + 4)], self.biases['b{}'.format(m + 4)]
    conv_deconv = tf.nn.conv2d_transpose(conv_expand, dconv_s_w2, output_shape=deconv_output, strides=deconv_stride, padding='SAME') + dconv_s_b2

    return conv_deconv






########################################################################################################################

  def model(self):
    #############################    shallow Network  ##################################################################

    conv_s_w1 = tf.get_variable("conv_s_w1", [5, 5, 1, 4],initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9)))
    conv_s_b1 = tf.get_variable("conv_s_b1", [4], initializer=tf.constant_initializer(0))
    conv_feature = self.prelu(tf.nn.conv2d(self.images, conv_s_w1, strides=[1, 1, 1, 1], padding='VALID') + conv_s_b1, 1)

    conv_s_w2 = tf.get_variable("conv_s_w2", [3, 3, 4, 32],initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9)))
    conv_s_b2 = tf.get_variable("conv_s_b2", [32], initializer=tf.constant_initializer(0))
    conv_feature = self.prelu(tf.nn.conv2d(conv_feature, conv_s_w2, strides=[1, 1, 1, 1], padding='SAME') + conv_s_b2, 2)

    conv_s_w3 = tf.get_variable("conv_s_w3", [5, 5, 1, 32],initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
    conv_s_b3 = tf.get_variable("conv_s_b3", [1], initializer=tf.constant_initializer(0))
    deconv_stride = [1,  self.scale, self.scale, 1]
    deconv_output = [self.batch, self.label_size, self.label_size, self.c_dim]
    tensor = tf.nn.conv2d_transpose(conv_feature, conv_s_w3, output_shape=deconv_output, strides=deconv_stride,
                                    padding='SAME') + conv_s_b3
##=======================================================================================================================
#############################     Deep Network  ########################################################################
    conv_D_Fw1 = tf.get_variable("conv_D_Fw1", [5, 5, 1, 64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9)))
    conv_D_Fb1 = tf.get_variable("conv_D_Fb1", [64], initializer=tf.constant_initializer(0))
    tensorD = tf.nn.bias_add(tf.nn.conv2d(self.images, conv_D_Fw1, strides=[1, 1, 1, 1], padding='VALID'), conv_D_Fb1)
    tensorD = tf.nn.relu(tensorD)

    conv_D_Fw11 = tf.get_variable("conv_D_Fw11", [5, 5, 64, 64],initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9)))
    conv_D_Fb11 = tf.get_variable("conv_D_Fb11", [64], initializer=tf.constant_initializer(0))
    tensorD = tf.nn.bias_add(tf.nn.conv2d(tensorD, conv_D_Fw11, strides=[1, 1, 1, 1], padding='SAME'), conv_D_Fb11)
    tensorD1 = tf.nn.relu(tensorD)

    # =============  clique block ==================================== # =============  dense block =================

    tensorB=build_model1(tensorD1, 64, 15, 1, 1, if_a=True, if_b=True, if_c=True)

 # -----------------------------------------------------------------------------------------------------------------

    tensorD = tf.concat((tensorD1, tensorB),axis=3)

  # ==========================================MAPPING ================================================================
    # conv_w = tf.get_variable("conv_19_w", [3,3,64,1], initializer=tf.contrib.layers.xavier_initializer())
    conv_D_Mw1 = tf.get_variable("conv_D_Mw1", [1, 1, 416, 12],initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
    conv_D_Mb1 = tf.get_variable("conv_D_Mb1", [12], initializer=tf.constant_initializer(0))
    tensorD = tf.nn.bias_add(tf.nn.conv2d(tensorD, conv_D_Mw1, strides=[1, 1, 1, 1], padding='SAME'), conv_D_Mb1)
    tensorD = tf.nn.relu(tensorD)

    conv_D_Mw2 = tf.get_variable("conv_D_Mw2", [3, 3, 12, 12],initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
    conv_D_Mb2 = tf.get_variable("conv_D_Mb2", [12], initializer=tf.constant_initializer(0))
    tensorD_2 = tf.nn.bias_add(tf.nn.conv2d(tensorD, conv_D_Mw2, strides=[1, 1, 1, 1], padding='SAME'), conv_D_Mb2)
    tensorD_2 = tf.nn.relu(tensorD_2)

    conv_D_Mw3 = tf.get_variable("conv_D_Mw3", [3, 3, 12, 12], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
    conv_D_Mb3 = tf.get_variable("conv_D_Mb3", [12], initializer=tf.constant_initializer(0))
    tensorD = tf.nn.bias_add(tf.nn.conv2d(tensorD_2, conv_D_Mw3, strides=[1, 1, 1, 1], padding='SAME'), conv_D_Mb3)
    tensorD = tf.nn.relu(tensorD)

 #======================== resnet ======================  #======================== resnet =========================
    tensorD = tf.concat((tensorD1,tensorD,tensorD_2), axis=3)

 # ======================== resnet ======================  #======================== resnet ========================

    conv_D_Mw4 = tf.get_variable("conv_D_Mw4", [3, 3, 88, 12], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
    conv_D_Mb4 = tf.get_variable("conv_D_Mb4", [12], initializer=tf.constant_initializer(0))
    tensorD_4 = tf.nn.bias_add(tf.nn.conv2d(tensorD, conv_D_Mw4, strides=[1, 1, 1, 1], padding='SAME'), conv_D_Mb4)
    tensorD_4 = tf.nn.relu(tensorD_4)

    conv_D_Mw5 = tf.get_variable("conv_D_Mw5", [3, 3, 12, 12],initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
    conv_D_Mb5 = tf.get_variable("conv_D_Mb5", [12], initializer=tf.constant_initializer(0))
    tensorD = tf.nn.bias_add(tf.nn.conv2d(tensorD_4, conv_D_Mw5, strides=[1, 1, 1, 1], padding='SAME'), conv_D_Mb5)
    tensorD = tf.nn.relu(tensorD)

#======================== eltwise 2  =========== #==================================================================
    tensorD = tf.add(tensorD_4, tensorD)

# ======================== eltwise ======== #=======================================================================
##===================== UPsampling =================================================================================
    conv_D_Uw1 = tf.get_variable("conv_D_Uw1", [1, 1, 12, 64],
                                 initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
    conv_D_Ub1 = tf.get_variable("conv_D_Ub1", [64], initializer=tf.constant_initializer(0))
    tensorD = tf.nn.bias_add(tf.nn.conv2d(tensorD, conv_D_Uw1, strides=[1, 1, 1, 1], padding='SAME'), conv_D_Ub1)
    tensorD = tf.nn.relu(tensorD)

    conv_D_Uw3 = tf.get_variable("conv_D_Uw3", [5, 5, 1, 64],
                                initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
    conv_D_Ub3 = tf.get_variable("conv_D_Ub3", [1], initializer=tf.constant_initializer(0))
    deconv_stride = [1, self.scale, self.scale, 1]
    deconv_output = [self.batch, self.label_size, self.label_size, self.c_dim]
    tensorD = tf.nn.conv2d_transpose(tensorD, conv_D_Uw3, output_shape=deconv_output, strides=deconv_stride,
                                    padding='SAME') + conv_D_Ub3

    ##================================= Multi ==========================================================================

    conv_D_MUw1 = tf.get_variable("conv_D_MUw1", [3, 3, 1, 64],initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
    conv_D_MUb1 = tf.get_variable("conv_D_MUb1", [64], initializer=tf.constant_initializer(0))
    tensorD_1 = tf.nn.bias_add(tf.nn.conv2d(tensorD, conv_D_MUw1, strides=[1, 1, 1, 1], padding='SAME'), conv_D_MUb1)
    tensorD_1 = tf.nn.relu(tensorD_1)

    conv_D_MUw2 = tf.get_variable("conv_D_MUw2", [3, 3, 64, 64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
    conv_D_MUb2 = tf.get_variable("conv_D_MUb2", [64], initializer=tf.constant_initializer(0))
    tensorD = tf.nn.bias_add(tf.nn.conv2d(tensorD_1, conv_D_MUw2, strides=[1, 1, 1, 1], padding='SAME'), conv_D_MUb2)
    tensorD = tf.nn.relu(tensorD)

# ======================== resnet ======================  #======================== resnet =========================
    tensorD = tf.concat((tensorD_1, tensorD), axis=3)
# ======================== resnet ======================  #======================== resnet =========================

    conv_D_MUw3 = tf.get_variable("conv_D_MUw3", [3, 3, 128, 64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
    conv_D_MUb3 = tf.get_variable("conv_D_MUb3", [64], initializer=tf.constant_initializer(0))
    tensorD_3 = tf.nn.bias_add(tf.nn.conv2d(tensorD, conv_D_MUw3, strides=[1, 1, 1, 1], padding='SAME'), conv_D_MUb3)
    tensorD_3 = tf.nn.relu(tensorD_3)

    conv_D_MUw4 = tf.get_variable("conv_D_MUw4", [3, 3, 64, 64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
    conv_D_MUb4 = tf.get_variable("conv_D_MUb4", [64], initializer=tf.constant_initializer(0))
    tensorD = tf.nn.bias_add(tf.nn.conv2d(tensorD_3, conv_D_MUw4, strides=[1, 1, 1, 1], padding='SAME'), conv_D_MUb4)
    tensorD = tf.nn.relu(tensorD)

# ======================== resnet ======================  #======================== resnet =========================
    tensorD = tf.concat((tensorD_3, tensorD), axis=3)
# ======================== resnet ======================  #======================== resnet =========================

    conv_D_MUw5 = tf.get_variable("conv_D_MUw5", [1, 1, 128, 16], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
    conv_D_MUb5 = tf.get_variable("conv_D_MUb5", [16], initializer=tf.constant_initializer(0))
    tensorD5 = tf.nn.bias_add(tf.nn.conv2d(tensorD, conv_D_MUw5, strides=[1, 1, 1, 1], padding='SAME'), conv_D_MUb5)
    tensorD5 = tf.nn.relu(tensorD5)

    conv_D_MUw5_1 = tf.get_variable("conv_D_MUw5_1", [1, 1, 16, 4], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
    conv_D_MUb5_1 = tf.get_variable("conv_D_MUb5_1", [4], initializer=tf.constant_initializer(0))
    tensorD5_1 = tf.nn.bias_add(tf.nn.conv2d(tensorD5, conv_D_MUw5_1, strides=[1, 1, 1, 1], padding='SAME'), conv_D_MUb5_1)
    tensorD5_1 = tf.nn.relu(tensorD5_1)

    conv_D_MUw5_2 = tf.get_variable("conv_D_MUw5_2", [3, 3, 16, 4], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
    conv_D_MUb5_2 = tf.get_variable("conv_D_MUb5_2", [4], initializer=tf.constant_initializer(0))
    tensorD5_2 = tf.nn.bias_add(tf.nn.conv2d(tensorD5, conv_D_MUw5_2, strides=[1, 1, 1, 1], padding='SAME'), conv_D_MUb5_2)
    tensorD5_2 = tf.nn.relu(tensorD5_2)

    conv_D_MUw5_3 = tf.get_variable("conv_D_MUw5_3", [5, 5, 16, 4], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
    conv_D_MUb5_3 = tf.get_variable("conv_D_MUb5_3", [4], initializer=tf.constant_initializer(0))
    tensorD5_3 = tf.nn.bias_add(tf.nn.conv2d(tensorD5, conv_D_MUw5_3, strides=[1, 1, 1, 1], padding='SAME'),conv_D_MUb5_3)
    tensorD5_3 = tf.nn.relu(tensorD5_3)

    conv_D_MUw5_4 = tf.get_variable("conv_D_MUw5_4", [7, 7, 16, 4],initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
    conv_D_MUb5_4 = tf.get_variable("conv_D_MUb5_4", [4], initializer=tf.constant_initializer(0))
    tensorD5_4 = tf.nn.bias_add(tf.nn.conv2d(tensorD5, conv_D_MUw5_4, strides=[1, 1, 1, 1], padding='SAME'), conv_D_MUb5_4)
    tensorD5_4 = tf.nn.relu(tensorD5_4)

# ======================== eltwise3  =========== #==================================================================
    tensorD22 = tf.add(tensorD5_1, tensorD5_2)
    tensorD23 = tf.add(tensorD5_3, tensorD5_4)

    tensorD = tf.add(tensorD22, tensorD23)

# ======================== eltwise  =========== #===================================================================

    conv_D_MUw6 = tf.get_variable("conv_D_MUw6", [1, 1, 4, 1], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
    conv_D_MUb6 = tf.get_variable("conv_D_MUb6", [1], initializer=tf.constant_initializer(0))
    tensorD = tf.nn.bias_add(tf.nn.conv2d(tensorD, conv_D_MUw6, strides=[1, 1, 1, 1], padding='SAME'), conv_D_MUb6)

# ======================== eltwise4  =========== #==================================================================
    tensor = tf.add(tensor, tensorD)
# ======================== eltwise  =========== #===================================================================

    conv_Lastw = tf.get_variable("conv_Lastw", [1, 1, 1, 1],initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
    conv_Lastb = tf.get_variable("conv_Lastb", [1], initializer=tf.constant_initializer(0))
    tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_Lastw, strides=[1, 1, 1, 1], padding='SAME'), conv_Lastb)

    return tensor


  def prelu(self, _x, i):
    """
    PreLU tensorflow implementation
    """
    alphas = tf.get_variable('alpha{}'.format(i), _x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg

#=======================================================================================================================

  def save(self, checkpoint_dir, step):
    model_name = "FSRCNN.model"
    model_dir = "%s_%s" % ("fsrcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("fsrcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False