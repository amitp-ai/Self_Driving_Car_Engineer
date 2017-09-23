import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import math
from datetime import timedelta
import time
import numpy as np


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights    
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path) #loads the graph into sess.graph
    def_graph = tf.get_default_graph()
	#
    image_input = def_graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = def_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = def_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = def_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = def_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    
#    print(layer7_out.name,": ", layer7_out.get_shape())
#    print(layer4_out.name,": ", layer4_out.get_shape())    
#    print(layer3_out.name,": ", layer3_out.get_shape())
#    print(keep_prob,": ", keep_prob.get_shape())
#    print(image_input.name,": ", image_input.get_shape())
#    #
#    #to print all the layers in vgg net
#    operations = sess.graph.get_operations()
#    for operation in operations:
#        print("Operations: ", operation.name)
#        for k in operation.inputs:
#            print(operation.name, "Input: ", k.name, k.get_shape())
#        for k in operation.outputs:
#            print(operation.name, "Output: ", k.name, k.get_shape())
#        print()
#    #end print
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, reg_scale = 1e-14):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    #reg_scale of 0.0 disables it #regularization factor
    layer7_score = tf.layers.conv2d(vgg_layer7_out, 
                                    filters=num_classes, 
                                    kernel_size=[1,1],
                                    strides=[1,1],
                                    padding = 'same',
									kernel_initializer = tf.truncated_normal_initializer(stddev=0.01),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_scale))
    
    layer_7_score_2x_upsample = tf.layers.conv2d_transpose(layer7_score, 
                                                           filters=num_classes,
                                                           kernel_size=[4,4], 
                                                           strides=[2,2], 
                                                           padding = 'same',
					   									   kernel_initializer = tf.truncated_normal_initializer(stddev=0.01),
                                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_scale))
    
    #print(layer_7_score_2x_upsample.get_shape())
    ##
    layer4_score = tf.layers.conv2d(vgg_layer4_out,
                                    filters=num_classes,
                                    kernel_size=[1,1],
                                    strides=[1,1],
                                    padding = 'same',
					   				kernel_initializer = tf.truncated_normal_initializer(stddev=0.01),                                 
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_scale))
    
    #print(layer4_score.get_shape())
    fuse_layers_7_4_score = tf.add(layer_7_score_2x_upsample, layer4_score)
    
    fuse_layers_7_4_score_2x_upsample = tf.layers.conv2d_transpose(fuse_layers_7_4_score,
                                                                   filters=num_classes,
                                                                   kernel_size=[4,4],
                                                                   strides=[2,2],
                                                                   padding = 'same',
     					   									       kernel_initializer = tf.truncated_normal_initializer(stddev=0.01),
                                                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_scale))
   
    #print(fuse_layers_7_4_score_2x_upsample.get_shape()
    ##
    layer3_score = tf.layers.conv2d(vgg_layer3_out,
                                    filters=num_classes,
                                    kernel_size=[1,1],
                                    strides=[1,1],
                                    padding = 'same',
					   				kernel_initializer = tf.truncated_normal_initializer(stddev=0.01),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_scale))
   
    fuse_layers_7_4_3 = tf.add(fuse_layers_7_4_score_2x_upsample, layer3_score)
    
    fuse_layers_7_4_3_8x = tf.layers.conv2d_transpose(fuse_layers_7_4_3,
                                                      filters=num_classes,
                                                      kernel_size=[16,16],
                                                      strides=[8,8],
                                                      padding = 'same',
					   								  kernel_initializer = tf.truncated_normal_initializer(stddev=0.01),
                                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_scale))
    
    #print(fuse_layers_7_4_3_8x.get_shape())
    ##
    return fuse_layers_7_4_3_8x
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    #classification
    logits_non_softmax = tf.reshape(nn_last_layer, (-1,num_classes))
    #logits = tf.nn.softmax(logits_non_softmax) #NO DONT DO THIS AS THE CROSS ENTRPY LOSS FUNCTION RECALCULATES SOFTMAX
    
    
    #print(logits.get_shape())
    #
    #cross entropy loss
    correct_label = tf.reshape(correct_label, (-1,num_classes))
    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits_non_softmax, labels=correct_label)
    unreg_loss = tf.reduce_mean(cross_entropy_loss)
    #
    ##Regularization loss. Note: lambda is taken care of in tf.layers.conv2d() function
	#Actually regularization is already taken care of in conv2d(). So don't do the below.
    #reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    #reg_loss = tf.reduce_sum(reg_loss)
    #
    loss = unreg_loss #tf.add(unreg_loss, reg_loss)
    #
    #training operation
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_optimizer = optimizer.minimize(loss)
    return logits_non_softmax, train_optimizer, loss
tests.test_optimize(optimize)

def accuracy(sess, ground_truth, predictions):
#    #for debug
#    #print(ground_truth.shape)
#    #print(predictions.shape)
#    ground_truth = ground_truth[0:6,:]
#    predictions = predictions[0:6,:]
#    print(ground_truth)
#    print()
#    print(predictions)
    
    ground_truth = tf.convert_to_tensor(ground_truth, dtype=tf.float32)
    predictions = tf.convert_to_tensor(predictions, dtype=tf.float32)
    #
    y_pred_cls = tf.argmax(predictions, axis=1)
    y_true_cls = tf.argmax(ground_truth, axis=1)
    is_prediction_correct = tf.equal(y_pred_cls, y_true_cls) #boolean vector
    accuracy = tf.reduce_mean(tf.cast(is_prediction_correct, tf.float32))
    accuracy = sess.run(accuracy)
    return accuracy

def mean_iou(sess, ground_truth, predictions, num_classes):
#    #for debug
#    #print(ground_truth.shape)
#    #print(predictions.shape)
#    ground_truth = ground_truth[0:6,:]
#    predictions = predictions[0:6,:]
#    print(ground_truth)
#    print()
#    print(predictions)
    
    ground_truth = tf.convert_to_tensor(ground_truth) #datatype inferred from input
    predictions = tf.convert_to_tensor(predictions) #datatype inferred from input
    num_classes = tf.constant(num_classes) #datatype inferred from input
    iou, iou_op = tf.metrics.mean_iou(ground_truth, predictions, num_classes)
    #
    sess.run(tf.local_variables_initializer()) #need to do this for the IOU function. Don't initialize global variables though!
    sess.run(iou_op) #need to run this first
    iou = sess.run(iou)
    return iou

def check_Data_IOU_Accuracy(sess, logits, input_image, keep_prob, num_classes, batch_generator):
 
    batch_input_image, batch_correct_label = next(batch_generator)
    batch_predictions = sess.run(logits, feed_dict={input_image:batch_input_image, keep_prob:1.0}) #keep prob = 1 for predicitons        
    #
    pred_max = np.argmax(batch_predictions, axis=1)
    pred_bool = np.zeros_like(batch_predictions, dtype=np.bool)
    shape0 = pred_bool.shape[0]
    pred_bool[np.arange(0,shape0,1), pred_max] = 1
    #
    batch_correct_label_bool = np.reshape(batch_correct_label, (-1,num_classes))
    #
    #IOU
    check_iou_acc = mean_iou(sess, batch_correct_label_bool, pred_bool, num_classes)
    #
    #Accuracy
    #check_iou_acc = accuracy(sess, batch_correct_label_bool, pred_bool)
    #
    return check_iou_acc
    
    
def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, num_classes=2, logits=None):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    loss = 1e20 #initialize to a large number
    
    start_time = time.time()

    for epoch in range(epochs):
        #do the full batch generator for each epoch
        generator_batch = get_batches_fn(batch_size)
        for batch_input_image, batch_correct_label in generator_batch:
            _,loss = sess.run([train_op, cross_entropy_loss],
                                    feed_dict={input_image:batch_input_image, correct_label:batch_correct_label, keep_prob:0.5})

#        Altearnate Implementation
#        try:
#            while True:
#                batch_input_image, batch_correct_label = next(generator_batch)
#                #do more processing
#                _,loss = sess.run([train_op, cross_entropy_loss],
#                         feed_dict={input_image:batch_input_image, correct_label:batch_correct_label, keep_prob:0.5})
#        except StopIteration:
#            pass #i.e. go to next epoch (i.e. next step in for loop)
#
            
        #print status every epoch
        if(logits != None):
            data_dir = './data'
            image_shape = (160, 576)
            batch_size = 10
            #        
            temp_get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
            temp_batch_generator = temp_get_batches_fn(batch_size)
            check_iou_acc = check_Data_IOU_Accuracy(sess, logits, input_image, keep_prob, num_classes, temp_batch_generator)
            print("Epoch " + str(epoch+1) + "/" + str(epochs) + ": Training IOU is " + str(check_iou_acc))
            #
            temp_get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/validation'), image_shape)   
            temp_batch_generator = temp_get_batches_fn(batch_size)
            check_iou_acc = check_Data_IOU_Accuracy(sess, logits, input_image, keep_prob, num_classes, temp_batch_generator)
            print("Epoch " + str(epoch+1) + "/" + str(epochs) + ": Validation IOU is " + str(check_iou_acc))
            #
            print("Epoch " + str(epoch+1) + "/" + str(epochs) + ": Final Batch's Loss is " + str(loss))
            print()

    stop_time = time.time()
    time_diff = stop_time-start_time
    print("Total Training Time: " + str(timedelta(seconds=int(round(time_diff)))))

tests.test_train_nn(train_nn)

#    #To prevent certain layers from training
#    vgg_layer3_out = tf.stop_gradient(vgg_layer3_out)
#    vgg_layer4_out = tf.stop_gradient(vgg_layer4_out)
#    vgg_layer7_out = tf.stop_gradient(vgg_layer7_out)

def run():
    retrain = False
    predict_from_saved_model = False
    tf.reset_default_graph()
    
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    #Hyperparameters
    reg_lambda = 2e-3 #1e-3
    learning_rate = 1e-3 #1e-4 is suggested in the Berkeley FCN paper
    batch_size = 16 #32 #8 #128
    num_epochs = 25
    
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        #Load VGG Net
        vgg_image_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        
        #Add skip conenctions, 1x1 convolutions, and upsampling for segmentation to the vgg net
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, reg_scale=reg_lambda)
        #Optimize
        #generator_batch = get_batches_fn(batch_size)
        #batch_input_image, batch_correct_label = next(generator_batch)
        #print(batch_input_image.shape)
        #print(batch_correct_label.shape)
        #tmp_batch = batch_correct_label.reshape([-1,num_classes])*1
        #tmp1 = tmp_batch[:,1]==1
        #print(tmp_batch[tmp1][1:100,:])
        
        correct_label = tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1], num_classes])
        logits, train_optimizer, loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
        
        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer()) #initialize all global variables
        saver = tf.train.Saver()
        if(retrain == True):
            saver.restore(sess,tf.train.latest_checkpoint('./saved_model'))
		#Train the network
        if(predict_from_saved_model == False):
            train_nn(sess, num_epochs, batch_size, get_batches_fn, 
                     train_optimizer, loss, vgg_image_input,
                     correct_label, vgg_keep_prob, learning_rate, num_classes, logits)
    		#Save the model		
            saver.save(sess, './saved_model/segmentation_model')

        # TODO: Save inference data using helper.save_inference_samples
        if(predict_from_saved_model == True):
            saver.restore(sess,tf.train.latest_checkpoint('./saved_model'))
		#Do prediction
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, vgg_image_input)

        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()

