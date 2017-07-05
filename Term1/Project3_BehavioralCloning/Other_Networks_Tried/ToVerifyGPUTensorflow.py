#run this file to verify if tensorflow is using gpu
import tensorflow as tf

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

