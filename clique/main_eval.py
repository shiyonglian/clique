import tensorflow as tf
from model import RDN
# from remodel1 import FSRCNN
import os
import pdb
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean("is_train", True, "if the train")
flags.DEFINE_boolean("matlab_bicubic", False, "using bicubic interpolation in matlab")
flags.DEFINE_integer("image_size", 48, "the size of image input")
flags.DEFINE_integer("c_dim", 3, "the size of channel")
flags.DEFINE_integer("scale", 2, "the size of scale factor for preprocessing input image")
flags.DEFINE_integer("stride", 21, "the size of stride")
flags.DEFINE_integer("epoch", 1, "number of epoch")
flags.DEFINE_integer("batch_size",128 , "the size of batch")
flags.DEFINE_float("learning_rate", 1e-3 , "the learning rate")
flags.DEFINE_boolean("is_eval", True, "if the evaluation")
flags.DEFINE_boolean("is_test", True, "if the evaluation")
#flags.DEFINE_string("test_img", "", "test_img")
flags.DEFINE_string("test_img", "Set5", "test_img")
flags.DEFINE_string("val_img", "div2k_valid", "val_img")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "name of the checkpoint directory")
flags.DEFINE_string("result_dir", "result", "name of the result directory")
# flags.DEFINE_string("train_set", "div2k", "name of the train set")
flags.DEFINE_string("train_set", "91", "name of the train set")#"DIV2K_train_HR"
flags.DEFINE_string("test_set", "/media/li-547/Liwen/RDN-new/RDN-TensorFlow-master/Test/Set14", "name of the test set")
flags.DEFINE_integer("D", 5, "D")
flags.DEFINE_integer("C", 3, "C")
flags.DEFINE_integer("G", 64, "G")
flags.DEFINE_integer("G0", 64, "G0")
flags.DEFINE_integer("kernel_size", 3, "the size of kernel")


def main(_):
    rdn = RDN(tf.Session(),
              is_train = FLAGS.is_train,
              is_eval = FLAGS.is_eval,
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

    max_PSNR = 0
    best_model = './'
    max_epoch = 50
    for epoch in range(max_epoch):
        rdn.train(FLAGS)
        psnr = rdn.eval(FLAGS)
        if max_PSNR<psnr:
            max_PSNR = psnr
            rdn.save(checkpoint_dir=best_model, step=epoch)

    rdn.load(checkpoint_dir=best_model)
    rdn.test(FLAGS)

    '''
    if rdn.is_train:
        rdn.train(FLAGS)
    else:
        if rdn.is_eval:
            rdn.eval(FLAGS)
        else:
            rdn.test(FLAGS)
    '''

if __name__=='__main__':
#    pdb.set_trace()
    tf.app.run()
