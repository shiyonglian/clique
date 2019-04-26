# -*- coding: UTF-8 -*-
import numpy as np
import time
import os
import glob
from cliqueblock import *
import pdb
import tensorflow as tf
from cbam_block import*
import tensorflow.contrib.layers as layers
import nonlocal_resnet_utils as local




from utils import (
    input_setup,
    get_data_dir,
    get_data_num,
    get_batch,
    get_image,
    imsave,
    imread,
    prepare_data,
    PSNR,
    compute_ssim
)

class RDN(object):

    def __init__(self,
                 sess,
                 is_train,
                 is_test,
                 image_size,
                 c_dim,
                 scale,
                 batch_size,
                 D,
                 C,
                 G,
                 G0,
                 kernel_size
                 ):

        self.sess = sess
        self.is_train = is_train
        self.is_test = is_test
        self.image_size = image_size
        self.c_dim = c_dim
        self.scale = scale
        self.batch_size = batch_size
        self.D = D
        self.C = C
        self.G = G
        self.G0 = G0
        self.kernel_size = kernel_size



    def SFEParams(self):
        G = self.G
        G0 = self.G0
        ks = self.kernel_size
        weightsS = {
            'w_S_1': tf.Variable(tf.random_normal([ks, ks, self.c_dim, G0], stddev=0.01), name='w_S_1'),
            'w_S_2': tf.Variable(tf.random_normal([ks, ks, G0, G], stddev=0.01), name='w_S_2')
        }
        biasesS = {
            'b_S_1': tf.Variable(tf.zeros([G0], name='b_S_1')),
            'b_S_2': tf.Variable(tf.zeros([G], name='b_S_2'))
        }
        return weightsS, biasesS
    

    def UPNParams(self):
        G0 = self.G0
        weightsU = {
            'w_U_1': tf.Variable(tf.random_normal([5, 5, G0, 32], stddev=0.01), name='w_U_1'),
            'w_U_2': tf.Variable(tf.random_normal([3, 3, 32, self.c_dim * self.scale * self.scale ], stddev=np.sqrt(2.0/9/32)), name='w_U_2')
        }
        biasesU = {
            'b_U_1': tf.Variable(tf.zeros([32], name='b_U_1')),
            'b_U_2': tf.Variable(tf.zeros([self.c_dim * self.scale * self.scale ], name='b_U_2'))
        }
        return weightsU, biasesU
    
    def build_model(self, images_shape, labels_shape):
        self.images = tf.placeholder(tf.float32, images_shape, name='images')
        self.labels = tf.placeholder(tf.float32, labels_shape, name='labels')
        self.weightsS, self.biasesS = self.SFEParams()
        self.weightsU, self.biasesU = self.UPNParams()
        self.weight_final = tf.Variable(tf.random_normal([self.kernel_size, self.kernel_size, self.c_dim, self.c_dim], stddev=np.sqrt(2.0/9/3)), name='w_f')
        self.bias_final = tf.Variable(tf.zeros([self.c_dim], name='b_f')),
        self.pred = self.model()


        # self.regularizer = tf.contrib.layers.l2_regularizer(0.1)
        # self.reg_term = tf.contrib.layers.apply_regularization(self.regularizer)
        # self.loss = tf.reduce_mean(tf.square(self.labels - self.pred) + self.reg_term)
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))


#         self.summary = tf.summary.scalar('loss', self.loss)
        self.saver = tf.train.Saver()

    def UPN(self, input_layer):
        x = tf.nn.conv2d(input_layer, self.weightsU['w_U_1'], strides=[1,1,1,1], padding='SAME') + self.biasesU['b_U_1']
        x = tf.nn.relu(x)
        x = tf.nn.conv2d(x, self.weightsU['w_U_2'], strides=[1,1,1,1], padding='SAME') + self.biasesU['b_U_2']
        x = self.PS(x, self.scale)
        return x

#     def relu(x, alpha=0., max_value=None):
#         '''
#         ReLU.

#     alpha: slope of negative section.
#     '''
#         negative_part = tf.nn.relu(-x)
#         x = tf.nn.relu(x)
#         if max_value is not None: 
#             x = tf.clip_by_value(x, tf.cast(0., dtype=_FLOATX), tf.cast(max_value, dtype=_FLOATX))
#         x -= tf.constant(alpha, dtype=_FLOATX) * negative_part
#         return x


    
    def add_layer(self,name, l):
        shape = l.get_shape().as_list()
        in_channel = shape[3]
        with tf.variable_scope(name) as scope:
            #c = BatchNorm('bn1', l)
            c = tf.nn.relu(l)
            #conv_D_Fw1 = tf.get_variable('%s_w'% name, [3, 3, in_channel, in_channel],initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / in_channel)))
            conv_D_Fw1 = tf.get_variable('%s_w'% name, [3, 3, in_channel, in_channel])
            conv_D_Fb1 = tf.get_variable('%s_b' % name, [in_channel], initializer=tf.constant_initializer(0))
            c = tf.nn.bias_add(tf.nn.conv2d(l, conv_D_Fw1, strides=[1, 1, 1, 1], padding='SAME'),conv_D_Fb1)

            #c = self.conv('conv1', c, self.growthRate, 1)
            l = tf.concat([c, l], 3)
        return l
       


    
    def model(self):
        conv_feature1 = tf.nn.conv2d(self.images, self.weightsS['w_S_1'], strides=[1,1,1,1], padding='SAME') + self.biasesS['b_S_1']
        conv_feature2 = tf.nn.conv2d(conv_feature1, self.weightsS['w_S_2'], strides=[1,1,1,1], padding='SAME') + self.biasesS['b_S_2']
        #SHR=self.UPN(conv_feature2)
        
        conv_sw2 = tf.get_variable("conv_sw2", [1, 1,3, 64],initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
        conv_sb2 = tf.get_variable("conv_sb2", [3], initializer=tf.constant_initializer(0))
        deconv_stride = [1,  self.scale, self.scale, 1]
        deconv_output = [self.batch_size, self.image_size*self.scale, self.image_size*self.scale, self.c_dim]
       
        tensorD = tf.nn.conv2d_transpose(conv_feature2, conv_sw2, output_shape=deconv_output, strides=deconv_stride,padding='SAME') + conv_sb2
        
        SHR = tf.nn.relu(tensorD)

        ############################# Deep Network  ########################################################################

        conv_D_Fw1 = tf.get_variable("conv_D_Fw1", [3, 3, 3, 64],initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9)))
        conv_D_Fb1 = tf.get_variable("conv_D_Fb1", [64], initializer=tf.constant_initializer(0))
        tensorD = tf.nn.bias_add(tf.nn.conv2d(self.images, conv_D_Fw1, strides=[1, 1, 1, 1], padding='SAME'),conv_D_Fb1)
        tensorD = tf.nn.relu(tensorD)

        conv_D_Fw11 = tf.get_variable("conv_D_Fw11", [5, 5, 64, 64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9)))
        conv_D_Fb11 = tf.get_variable("conv_D_Fb11", [64], initializer=tf.constant_initializer(0))
        tensorD = tf.nn.bias_add(tf.nn.conv2d(tensorD, conv_D_Fw11, strides=[1, 1, 1, 1], padding='SAME'), conv_D_Fb11)
        tensorD1 = tf.nn.relu(tensorD)
        tensorD1 = cbam_block(tensorD1, 'cbam_block_1', 4)

        # =============  clique block ==================================== # =============  dense block =================

        #tensorB = build_model1(tensorD1, 64, 5, 1, 1, if_a=False, if_b=True, if_c=True)# T=5 represent conv number
        #tensorD = tf.concat((tensorD1, tensorB), axis=3) # 换成dense 可以省略
               
        conv_dense_w = tf.get_variable("conv_dense_w", [1, 1, 64, 64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9)))
        conv_dense_b = tf.get_variable("conv_dense_b", [64], initializer=tf.constant_initializer(0))
        tensorD = tf.nn.bias_add(tf.nn.conv2d(tensorD1, conv_dense_w, strides=[1, 1, 1, 1], padding='SAME'), conv_dense_b)
        tensorD1 = tf.nn.relu(tensorD)
        
        with tf.variable_scope('block1') as scope:
            for i in range(5):
                l = self.add_layer('dense_layer.{}'.format(i),tensorD1 )
        
        with tf.variable_scope('block2') as scope:
            for i in range(5):
                l = self.add_layer('dense_layer.{}'.format(i), l)
        
        conv_dense_w2 = tf.get_variable("conv_dense_w2", [1, 1, 64, 32], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9)))
        conv_dense_b2 = tf.get_variable("conv_dense_b2", [32], initializer=tf.constant_initializer(0))
        tensorD = tf.nn.bias_add(tf.nn.conv2d(tensorD1, conv_dense_w2, strides=[1, 1, 1, 1], padding='SAME'), conv_dense_b2)
        tensorD1 = tf.nn.relu(tensorD)
        
         # ==========================================MAPPING ================================================================

        conv_D_Mw1 = tf.get_variable("conv_D_Mw1", [1, 1, 32, 64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
        conv_D_Mb1 = tf.get_variable("conv_D_Mb1", [64], initializer=tf.constant_initializer(0))
        tensorD = tf.nn.bias_add(tf.nn.conv2d(tensorD, conv_D_Mw1, strides=[1, 1, 1, 1], padding='SAME'), conv_D_Mb1)
        tensorD = tf.nn.relu(tensorD)

        conv_D_Mw2 = tf.get_variable("conv_D_Mw2", [3, 3, 64, 64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
        conv_D_Mb2 = tf.get_variable("conv_D_Mb2", [64], initializer=tf.constant_initializer(0))
        tensorD_2 = tf.nn.bias_add(tf.nn.conv2d(tensorD, conv_D_Mw2, strides=[1, 1, 1, 1], padding='SAME'), conv_D_Mb2)
        tensorD_2 = tf.nn.relu(tensorD_2)
        tensorD_2 = cbam_block(tensorD_2, 'cbam_block_2', 2)
        ##===================== UPsampling =================================================================================

        tensorD_2 = tf.concat((tensorD1, tensorD_2), axis=3)
        conv_D_Uw1 = tf.get_variable("conv_D_Uw1", [1, 1, 96, 64],
                                             initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
        conv_D_Ub1 = tf.get_variable("conv_D_Ub1", [64], initializer=tf.constant_initializer(0))
        tensorD = tf.nn.bias_add(tf.nn.conv2d(tensorD_2, conv_D_Uw1, strides=[1, 1, 1, 1], padding='SAME'), conv_D_Ub1)
        tensorD = tf.nn.relu(tensorD)
        #tensorD = local.nonlocal_dot(tensorD, 32, embed=True, softmax=False, maxpool=False, scope='nonlocal')
        tensorD = local.NonLocalBlock(tensorD, 64, self.batch_size, sub_sample=False, is_bn=True, scope='NonLocalBlock')
       #tensorD = self.UPN(tensorD) #espcn

        conv_D_Uw2 = tf.get_variable("conv_D_Uw2", [1, 1,3, 64],initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
        conv_D_Ub2 = tf.get_variable("conv_D_Ub2", [3], initializer=tf.constant_initializer(0))
        deconv_stride = [1,  self.scale, self.scale, 1]
        deconv_output = [self.batch_size, self.image_size*self.scale, self.image_size*self.scale, self.c_dim]
        tensorD = tf.nn.conv2d_transpose(tensorD, conv_D_Uw2, output_shape=deconv_output, strides=deconv_stride,padding='SAME') + conv_D_Ub2
        tensorD = tf.nn.relu(tensorD)
        ##================================= Multi ==========================================================================
        conv_D_MUw5 = tf.get_variable("conv_D_MUw5", [1, 1, 3, 64],
                                              initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
        conv_D_MUb5 = tf.get_variable("conv_D_MUb5", [64], initializer=tf.constant_initializer(0))
        tensorD5 = tf.nn.bias_add(tf.nn.conv2d(tensorD, conv_D_MUw5, strides=[1, 1, 1, 1], padding='SAME'), conv_D_MUb5)
        tensorD5 = tf.nn.relu(tensorD5)
        tensorD5 = cbam_block(tensorD5, 'cbam_block_D5', 2)
        

        conv_D_MUw5_1 = tf.get_variable("conv_D_MUw5_1", [1, 1, 64, 4],
                                                initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
        conv_D_MUb5_1 = tf.get_variable("conv_D_MUb5_1", [4], initializer=tf.constant_initializer(0))
        tensorD5_1 = tf.nn.bias_add(tf.nn.conv2d(tensorD5, conv_D_MUw5_1, strides=[1, 1, 1, 1], padding='SAME'),
                                            conv_D_MUb5_1)
        tensorD5_1 = tf.nn.relu(tensorD5_1)
        tensorD5_1 = cbam_block(tensorD5_1, 'cbam_block_D5_1', 2)

        conv_D_MUw5_2 = tf.get_variable("conv_D_MUw5_2", [3, 3, 64, 4],
                                                initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
        conv_D_MUb5_2 = tf.get_variable("conv_D_MUb5_2", [4], initializer=tf.constant_initializer(0))
        tensorD5_2 = tf.nn.bias_add(tf.nn.conv2d(tensorD5, conv_D_MUw5_2, strides=[1, 1, 1, 1], padding='SAME'),
                                            conv_D_MUb5_2)
        tensorD5_2 = tf.nn.relu(tensorD5_2)
        tensorD5_2 = cbam_block(tensorD5_2, 'cbam_block_D5_2', 2)
        

        conv_D_MUw5_3 = tf.get_variable("conv_D_MUw5_3", [5, 5, 64, 4],
                                                initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
        conv_D_MUb5_3 = tf.get_variable("conv_D_MUb5_3", [4], initializer=tf.constant_initializer(0))
        tensorD5_3 = tf.nn.bias_add(tf.nn.conv2d(tensorD5, conv_D_MUw5_3, strides=[1, 1, 1, 1], padding='SAME'),
                                            conv_D_MUb5_3)
        tensorD5_3 = tf.nn.relu(tensorD5_3)
        tensorD5_3 = cbam_block(tensorD5_3, 'cbam_block_D5_3', 2)
        

        conv_D_MUw5_4 = tf.get_variable("conv_D_MUw5_4", [7, 7, 64, 4],
                                                initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
        conv_D_MUb5_4 = tf.get_variable("conv_D_MUb5_4", [4], initializer=tf.constant_initializer(0))
        tensorD5_4 = tf.nn.bias_add(tf.nn.conv2d(tensorD5, conv_D_MUw5_4, strides=[1, 1, 1, 1], padding='SAME'),
                                            conv_D_MUb5_4)
        tensorD5_4 = tf.nn.relu(tensorD5_4)
        tensorD5_4 = cbam_block(tensorD5_4, 'cbam_block_D5_4', 2)

        # ======================== eltwise3  =========== #==================================================================
        tensorD22 = tf.add(tensorD5_1, tensorD5_2)
        tensorD23 = tf.add(tensorD5_3, tensorD5_4)

        tensorD = tf.add(tensorD22, tensorD23)

        # ======================== eltwise  =========== #===================================================================

        conv_D_MUw6 = tf.get_variable("conv_D_MUw6", [1, 1, 4, 1],
                                              initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
        conv_D_MUb6 = tf.get_variable("conv_D_MUb6", [1], initializer=tf.constant_initializer(0))
        DHR = tf.nn.bias_add(tf.nn.conv2d(tensorD, conv_D_MUw6, strides=[1, 1, 1, 1], padding='SAME'), conv_D_MUb6)

        # ======================== eltwise4  =========== #==================================================================

        tensor = tf.add(DHR, SHR)
        # ======================== eltwise  =========== #===================================================================

        IHR = tf.nn.conv2d(tensor, self.weight_final, strides=[1, 1, 1, 1], padding='SAME') + self.bias_final

        return IHR

    # NOTE: train with batch size 
    def _phase_shift(self, I, r):
        # Helper function with main phase shift operation
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (self.batch_size, a, b, r, r))
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
        X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r
        return tf.reshape(X, (self.batch_size, a*r, b*r, 1))

    # NOTE: test without batchsize
    def _phase_shift_test(self, I ,r):
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (1, a, b, r, r))
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, b, a*r, r
        X = tf.split(X, b, 0)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, a*r, b*r
        return tf.reshape(X, (1, a*r, b*r, 1))

    def PS(self, X, r):
        # Main OP that you can arbitrarily use in you tensorflow code
        Xc = tf.split(X, 3, 3)
        if self.is_train:
            X = tf.concat([self._phase_shift(x, r) for x in Xc], 3) # Do the concat RGB
        else:
            X = tf.concat([self._phase_shift_test(x, r) for x in Xc], 3) # Do the concat RGB
        return X

    # evaluate stage is included in the training stage
    def train(self, config):
        print("\nPrepare Data...\n")
        min_average_loss = np.inf

        #  generate train.h5
        input_setup(config)
        data_dir = get_data_dir(config.checkpoint_dir)
       # print(data_dir)
        data_num = get_data_num(data_dir)

        images_shape = [None, self.image_size, self.image_size, self.c_dim]
        labels_shape = [None, self.image_size * self.scale, self.image_size * self.scale, self.c_dim]
        self.build_model(images_shape, labels_shape)
        self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
        tf.global_variables_initializer().run(session=self.sess)

        #merged_summary_op = tf.summary.merge_all()
        #summary_writer = tf.summary.FileWriter(config.checkpoint_dir, self.sess.graph)

        # generate evaluate.h5
        data_dir_v ="/root/workspace/shiyonglian/RDN-TensorFlow-master/Test/set5.h5" # "/media/li-547/Liwen/A-final/clique_attention_2_5/Validate/div_2.h5"

        data_num_v = get_data_num(data_dir_v)
        batch_idxs_v = data_num_v // config.batch_size

        counter = self.load(config.checkpoint_dir)
        time_ = time.time()
        print("\nNow Start Training...\n")

        train_loss = open('deconv_nonlocal_cbam.csv', 'a+')
        for ep in range(config.epoch):
            # Run by batch images
            batch_idxs = data_num // config.batch_size
            for idx in range(0, batch_idxs):
                batch_images, batch_labels = get_batch(data_dir, data_num, config.batch_size)
                counter += 1
                _, err = self.sess.run([self.train_op, self.loss],feed_dict={self.images: batch_images, self.labels: batch_labels})
                if counter % 10 == 0:
                    print("Epoch: [%2d], batch: [%2d/%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" % ((ep + 1), idx, batch_idxs, counter, time.time() - time_, err))
                train_loss.write( 'Epoch: [%2d], batch: [%2d/%2d], step: [%2d], time: [%4.4f], loss: [%.8f]'% ((ep+1), idx, batch_idxs, counter, time.time()-time_, err)+'\n')

                if counter % 200 == 0:
                    evaluate_loss = open('evaluate_loss.csv', 'a+')
                    sum_loss = 0

                # evaluate stage
                    for idx_v in range(0, batch_idxs_v):
                        batch_images, batch_labels = get_batch(data_dir_v, data_num_v, config.batch_size)
                        err = self.sess.run(self.loss,feed_dict={self.images: batch_images, self.labels: batch_labels})
                        sum_loss += err
                    local_average_loss = sum_loss / batch_idxs_v
                    print('validatation loss average:[%0.8f]' % (local_average_loss))
                    evaluate_loss.write('step: [%2d],local_average_loss: [%.8f]' % (counter,local_average_loss) + '\n')
                    evaluate_loss.close()

                    if local_average_loss < min_average_loss :    # when local_average_loss < min_average_loss save the model , else not save
                        min_average_loss = local_average_loss
                        self.save(config.checkpoint_dir, counter)
                if counter > 0 and counter == batch_idxs * config.epoch:
                    return
               # summary_str = self.sess.run(merged_summary_op)
               # summary_writer.add_summary(summary_str, counter)
        train_loss.close()


    #  test stage
    def test(self, config):
        print("\nPrepare Data...\n")
        paths = prepare_data(config)
        data_num = len(paths)
        avg_time = 0
        avg_pasn = 0
        avg_ssim = 0
        print("\nNow Start Testing...\n")
        for idx in range(data_num):
            input_, label_ = get_image(paths[idx], config.scale, config.matlab_bicubic)
            images_shape = input_.shape
            labels_shape = label_.shape
            self.build_model(images_shape, labels_shape)
            tf.global_variables_initializer().run(session=self.sess) 
            self.load(config.checkpoint_dir)
            time_ = time.time()
            result = self.sess.run([self.pred], feed_dict={self.images: input_ / 255.0})
            avg_time += time.time() - time_
            self.sess.close()
            tf.reset_default_graph()
            self.sess = tf.Session()
            x = np.squeeze(result) * 255.0
            x = np.clip(x, 0, 255)
            x = x.astype(np.uint8)
            psnr = PSNR(x, label_[0], config.scale)
            ssim = compute_ssim(x, label_[0])
            avg_pasn += psnr
            avg_ssim += ssim
            print("image: %d/%d, time: %.4f, psnr: %.4f" % (idx, data_num, time.time() - time_ , psnr))
            print("image: %d/%d, time: %.4f, ssim: %.4f" % (idx, data_num, time.time() - time_, ssim))
            if not os.path.isdir(os.path.join(os.getcwd(),config.result_dir)):
                os.makedirs(os.path.join(os.getcwd(),config.result_dir))
            imsave(x[:, :, ::-1], config.result_dir+'/%d.png' % idx)
        print("Avg. Time:", avg_time / data_num)
        print("Avg. PSNR:", avg_pasn / data_num)
        print("Avg. SSIM:", avg_ssim / data_num)

    def load(self, checkpoint_dir):
        print("\nReading Checkpoints.....\n")
        model_dir = "%s_%s_%s_%s_x%s" % ("deconv_nonlocal_cbam", self.D, self.C, self.G, self.scale)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        #pdb.set_trace() 
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
            step = int(os.path.basename(ckpt_path).split('-')[1])
            print("\nCheckpoint Loading Success! %s\n" % ckpt_path)
        else:
            step = 0
            print("\nCheckpoint Loading Failed! \n")
        return step

    def save(self, checkpoint_dir, step):
        model_name = "clique.model"
        model_dir = "%s_%s_%s_%s_x%s" % ("deconv_nonlocal_cbam", self.D, self.C, self.G, self.scale)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
             os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load_one_model(self, model_file):
        self.saver.restore(self.sess, os.path.join(os.getcwd(), model_file))
        step = int(os.path.basename(model_file).split('-')[1])
        print("\nCheckpoint Loading Success! %s\n" % model_file)

    def test_all_models(self, config):
        print("\nPrepare Data...\n")
        paths = prepare_data(config)
        data_num = len(paths)
        print("\nNow Start Testing...\n")

        checkpoint_dir = config.checkpoint_dir
        model_dir = "%s_%s_%s_%s_x%s" % ("deconv_nonlocal_cbam", self.D, self.C, self.G, self.scale)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        all_files = glob.glob(os.path.join(checkpoint_dir, '*.meta'))
        model_files = [x[:-5] for x in all_files]

        highest_ssim = 0
        for model_file in model_files:
            avg_time = 0
            avg_pasn = 0
            avg_ssim = 0

            
            for idx in range(data_num):
                input_, label_ = get_image(paths[idx], config.scale, config.matlab_bicubic)
                images_shape = input_.shape
                labels_shape = label_.shape
                self.build_model(images_shape, labels_shape)
                tf.global_variables_initializer().run(session=self.sess)
                #self.load(config.checkpoint_dir)
                self.load_one_model(model_file)
                time_ = time.time()
                result = self.sess.run([self.pred], feed_dict={self.images: input_ / 255.0})
                avg_time += time.time() - time_
                self.sess.close()
                tf.reset_default_graph()
                self.sess = tf.Session()
                x = np.squeeze(result) * 255.0
                x = np.clip(x, 0, 255)
                x = x.astype(np.uint8)
                psnr = PSNR(x, label_[0], config.scale)
                ssim = compute_ssim(x, label_[0])
                avg_pasn += psnr
                avg_ssim += ssim
                print("image: %d/%d, time: %.4f, psnr: %.4f" % (idx, data_num, time.time() - time_, psnr))
                print("image: %d/%d, time: %.4f, ssim: %.4f" % (idx, data_num, time.time() - time_, ssim))

                if not os.path.isdir(os.path.join(os.getcwd(), config.result_dir)):
                    os.makedirs(os.path.join(os.getcwd(), config.result_dir))
                imsave(x[:, :, ::-1], config.result_dir + '/%d.png' % idx)
            print("Avg. Time:", avg_time / data_num)
            print("Avg. PSNR:", avg_pasn / data_num)
            print("Avg. SSIM:", avg_ssim / data_num)

            if highest_ssim<(avg_ssim / data_num):
                highest_ssim = avg_ssim / data_num
                highest_psnr = avg_pasn / data_num
                highest_model = model_file

        print('highest ssim: ',highest_ssim)
        print('highest psnr: ',highest_psnr)
        print('highest model: ',highest_model)


