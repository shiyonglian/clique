# -*- coding: utf-8 -*-

####======================================my original model=============================================================
import tensorflow as tf
import numpy as np
from cliqueblock import *
size=24
def ourmodel(input_tensor):
    with tf.device("/gpu:0"):

    ####### Shallow Network ############################################################################################
        # conv_00_w = tf.get_variable("conv_00_w", [3,3,1,64], initializer=tf.contrib.layers.xavier_initializer())
        conv_s_w1 = tf.get_variable("conv_s_w1", [3, 3, 1, 4],  initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9)))
        conv_s_b1 = tf.get_variable("conv_s_b1", [4], initializer=tf.constant_initializer(0))
        tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_s_w1, strides=[1, 1, 1, 1], padding='VALID'), conv_s_b1))
        print(tensor)
        # conv_w = tf.get_variable("conv_19_w", [3,3,64,1], initializer=tf.contrib.layers.xavier_initializer())
        conv_s_w2 = tf.get_variable("conv_s_w2", [5, 5, 64, 4],  initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
        conv_s_b2 = tf.get_variable("conv_s_b2", [64], initializer=tf.constant_initializer(0))
        # tensor =tf.nn.relu( tf.nn.bias_add(tf.nn.conv2d_transpose(tensor, conv_s_w2, [64, size, size, 1], strides=[1, 4, 4, 1], padding='SAME'), conv_s_b2))
        tensor = tf.nn.conv2d_transpose(tensor, conv_s_w2, output_shape=[64, size, size, 64], strides=[1, 4, 4, 1],
                                         padding='SAME') + conv_s_b2
        conv_s_w3 = tf.get_variable("conv_s_w3", [5, 5, 64, 1], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
        conv_s_b3 = tf.get_variable("conv_s_b3", [1], initializer=tf.constant_initializer(0))
        tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_s_w3, strides=[1, 1, 1, 1], padding='SAME'), conv_s_b3)
        #tensor = tf.nn.relu(tensor)
    # ############################# Deep Network  ########################################################################
        conv_D_Fw1 = tf.get_variable("conv_D_Fw1", [3, 3, 1, 64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9)))
        conv_D_Fb1 = tf.get_variable("conv_D_Fb1", [64], initializer=tf.constant_initializer(0))
        tensorD = tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_D_Fw1, strides=[1, 1, 1, 1], padding='SAME'), conv_D_Fb1)
        tensorD1 = tf.nn.relu(tensorD)

        # =============  clique block ==================================== # =============  dense block =================
        #tensorB=denseblock(tensorD)

        tensorB=build_model(tensorD, 64, 15, 1, 1, if_a=True, if_b=True, if_c=True)

     # -----------------------------------------------------------------------------------------------------------------

        tensorD = tf.concat((tensorD1, tensorB),axis=3)

    ## MAPPING =========================================================================================================
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
        tensorD = tf.concat((input_tensor,tensorD,tensorD_2), axis=3)

     # ======================== resnet ======================  #======================== resnet ========================

        conv_D_Mw4 = tf.get_variable("conv_D_Mw4", [3, 3, 25, 12], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
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
        conv_D_Uw1 = tf.get_variable("conv_D_Uw1", [1, 1, 12, 64],initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
        conv_D_Ub1 = tf.get_variable("conv_D_Ub1", [64], initializer=tf.constant_initializer(0))
        tensorD = tf.nn.bias_add(tf.nn.conv2d(tensorD, conv_D_Uw1, strides=[1, 1, 1, 1], padding='SAME'), conv_D_Ub1)
        tensorD = tf.nn.relu(tensorD)

        conv_D_Uw2 = tf.get_variable("conv_D_Uw2", [15, 15, 64, 64],initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
        conv_D_Ub2 = tf.get_variable("conv_D_Ub2", [64], initializer=tf.constant_initializer(0))
        tensorD= tf.nn.bias_add(tf.nn.conv2d_transpose( tensorD, conv_D_Uw2, [64, size, size,64], strides=[1, 4, 4, 1], padding='SAME'),conv_D_Ub2)
        tensorD = tf.nn.relu(tensorD)

        conv_D_Uw3 = tf.get_variable("conv_D_Uw3", [1, 1, 64, 64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
        conv_D_Ub3 = tf.get_variable("conv_D_Ub3", [64], initializer=tf.constant_initializer(0))
        tensorD = tf.nn.bias_add(tf.nn.conv2d(tensorD, conv_D_Uw3, strides=[1, 1, 1, 1], padding='SAME'), conv_D_Ub3)
        tensorD = tf.nn.relu(tensorD)
    ##================================= Multi ==========================================================================

        conv_D_MUw1 = tf.get_variable("conv_D_MUw1", [3, 3, 64, 64],initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
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
