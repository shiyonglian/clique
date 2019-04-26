import tensorflow as tf
from model_espcn_cbam_clique1_5 import RDN

# from model_clique2_5_attention import RDN

# from model_deconv_clique3_4_cbam import RDN

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean("is_train", True, "if the train")
flags.DEFINE_string("train_set", "/root/workspace/shiyonglian/RDN-TensorFlow-master/Test/Set14", "name of the train set")
flags.DEFINE_boolean("is_test",False, "if the test")
flags.DEFINE_string("test_set", "/root/workspace/shiyonglian/RDN-TensorFlow-master/Test/Set5", "name of the test set")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "name of the checkpoint directory")
flags.DEFINE_string("result_dir", "result", "name of the result directory")
flags.DEFINE_integer("image_size", 48, "the size of image input")
flags.DEFINE_integer("c_dim", 3, "the size of channel")
flags.DEFINE_integer("scale", 2, "the size of scale factor for preprocessing input image")
flags.DEFINE_integer("stride", 48, "the size of stride")
flags.DEFINE_integer("kernel_size", 3, "the size of kernel")
flags.DEFINE_integer("epoch", 100, "number of epoch")
flags.DEFINE_integer("batch_size", 128, "the size of batch")
flags.DEFINE_float("learning_rate", 1e-5 , "the learning rate")
flags.DEFINE_boolean("matlab_bicubic", False, "using bicubic interpolation in matlab")
flags.DEFINE_integer("D", 5, "D")
flags.DEFINE_integer("C", 3, "C")
flags.DEFINE_integer("G", 64, "G")
flags.DEFINE_integer("G0", 64, "G0")



def main(_):
    tf.reset_default_graph()
    config_gpu = tf.ConfigProto()  
    config_gpu.gpu_options.allow_growth=True   
    rdn = RDN(tf.Session(config=config_gpu),
              is_train = FLAGS.is_train,
              is_test = FLAGS.is_test,
              image_size = FLAGS.image_size,
              c_dim = FLAGS.c_dim,
              scale = FLAGS.scale,
              batch_size = FLAGS.batch_size,
              D = FLAGS.D,
              C = FLAGS.C,
              G = FLAGS.G,
              G0 = FLAGS.G0,
              kernel_size = FLAGS.kernel_size
              )

    if rdn.is_train:
        rdn.train(FLAGS)
    else:
        rdn.test_all_models(FLAGS)


if __name__=='__main__':
    tf.app.run()
