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

#THIS CODE IMPLEMENTS UCB'S FCN NETWORK FOR SEMANTIC SEGMENTATION#

# # Check TensorFlow Version
# assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
# print('TensorFlow Version: {}'.format(tf.__version__))

# # Check for a GPU
# if not tf.test.gpu_device_name():
#     warnings.warn('No GPU found. Please use a GPU to train your neural network.')
# else:
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


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
    
    # print(layer7_out.name,": ", layer7_out.get_shape())
    # print(layer4_out.name,": ", layer4_out.get_shape())    
    # print(layer3_out.name,": ", layer3_out.get_shape())
    # print(keep_prob,": ", keep_prob.get_shape())
    # print(image_input.name,": ", image_input.get_shape())
    # #
    # #to print all the layers in vgg net
    # operations = sess.graph.get_operations()
    # for operation in operations:
    #    print("Operations: ", operation.name)
    #    for k in operation.inputs:
    #        print(operation.name, "Input: ", k.name, k.get_shape())
    #    for k in operation.outputs:
    #        print(operation.name, "Output: ", k.name, k.get_shape())
    #    print()
    # #end print
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
# print("Running Test 1 of 4")
# tests.test_load_vgg(load_vgg, tf)


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
    #layer_7_score is a fully conv version of vgg_layer7_out (using 1x1 convolutions)
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
    #As in the end-to-end tranable version of FCN-8, need to first scale vgg_layer4_out by 0.1 (0.01) before doing 1x1 convolution
    #see https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/12
    scaled_vgg_layer4_out = tf.multiply(vgg_layer4_out, 0.1, name='scaled_vgglayer4')
    layer4_score = tf.layers.conv2d(scaled_vgg_layer4_out,
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
    #As in the end-to-end tranable version of FCN-8, need to first scale vgg_layer3_out by 0.01 (0.0001) before doing 1x1 convolution
    #see https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/12
    scaled_vgg_layer3_out = tf.multiply(vgg_layer3_out, 0.01, name='scaled_vgglayer3')
    layer3_score = tf.layers.conv2d(scaled_vgg_layer3_out,
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
# print("Running Test 2 of 4")
# tests.test_layers(layers)


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
    
    #cross entropy loss   
    correct_label = tf.reshape(correct_label, (-1,num_classes)) #shape: batch x num_classes

    #weight the loss function to focus on cars as there are fewer instaces of them and its scored heavily
    cls_weights = tf.constant([[1.0,2.0,4.0]], dtype=tf.float32) #shape: 1x3 (rest,road,car)
    cls_weights = tf.multiply(cls_weights,correct_label) #supports broadcasting (batch x 3)
    cls_weights = tf.reduce_sum(cls_weights,axis=1) #shape:batch x 1
    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits_non_softmax, labels=correct_label) #shape:batch x 1
    cross_entropy_loss = tf.multiply(cls_weights,cross_entropy_loss) #shape:batch x 1
    unreg_loss = tf.reduce_mean(cross_entropy_loss) #scalar
    #
    ##Regularization loss. Note: lambda is taken care of in tf.layers.conv2d() function
	#Actually regularization is already taken care of in conv2d(). So don't do the below.
    #Maybe not. see this https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_loss = tf.reduce_sum(reg_loss)
    
    #loss = unreg_loss
    loss = tf.add(unreg_loss, reg_loss)
    #
    #training operation
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_optimizer = optimizer.minimize(loss)
    return logits_non_softmax, train_optimizer, loss
# print("Running Test 3 of 4")
# tests.test_optimize(optimize)

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
    
    
def check_FScore(sess, logits, input_image, keep_prob, num_classes, batch_generator):
    batch_input_image, batch_correct_label = next(batch_generator)
    batch_predictions = sess.run(logits, feed_dict={input_image:batch_input_image, keep_prob:1.0}) #keep prob = 1 for predicitons        
    #
    pred_max = np.argmax(batch_predictions, axis=1)
    pred_bool = np.zeros_like(batch_predictions, dtype=np.bool) #(num_images x height x width) x num_classes
    shape0 = pred_bool.shape[0]
    pred_bool[np.arange(0,shape0,1), pred_max] = 1
    #
    batch_correct_label_bool = np.reshape(batch_correct_label, (-1,num_classes)) #(num_images x height x width) x num_classes
    #
    #Calculate FScore of car and road classes. No need to calculate for 'everything else' class
    pred_road = pred_bool[:,1]
    pred_car = pred_bool[:,2]
    gt_road = batch_correct_label_bool[:,1]
    gt_car = batch_correct_label_bool[:,2]

    road_TP = np.sum(np.logical_and(gt_road==1, pred_road==1))
    road_FP = np.sum(np.logical_and(gt_road==0, pred_road==1))
    #road_TN = np.sum(np.logical_and(gt_road==0, pred_road==0)) #not needed
    road_FN = np.sum(np.logical_and(gt_road==1, pred_road==0))

    car_TP = np.sum(np.logical_and(gt_car==1, pred_car==1))
    car_FP = np.sum(np.logical_and(gt_car==0, pred_car==1))
    #car_TN = np.sum(np.logical_and(gt_car==0, pred_car==0)) #not needed
    car_FN = np.sum(np.logical_and(gt_car==1, pred_car==0))

    # Generate results
    car_precision = car_TP/(car_TP+car_FP)/1.0
    car_recall = car_TP/(car_TP+car_FN)/1.0
    car_beta = 2
    car_F = (1+car_beta**2) * ((car_precision*car_recall)/(car_beta**2 * car_precision + car_recall))
    
    road_precision = road_TP/(road_TP+road_FP)/1.0
    road_recall = road_TP/(road_TP+road_FN)/1.0
    road_beta = 0.5
    road_F = (1+road_beta**2) * ((road_precision*road_recall)/(road_beta**2 * road_precision + road_recall))

    car_IOU = (car_TP)/(car_TP+car_FP+car_FN)/1.0
    road_IOU = (road_TP)/(road_TP+road_FP+road_FN)/1.0
    IOU = (car_IOU+road_IOU)/2.0
    return_string = ("Car F score: %05.3f  | Car Precision: %05.3f  | Car Recall: %05.3f  |\n\
    Road F score: %05.3f | Road Precision: %05.3f | Road Recall: %05.3f | \n\
    Averaged F score: %05.3f | IOU: %05.3f" %(car_F,car_precision,car_recall,road_F,road_precision,road_recall,((car_F+road_F)/2.0),IOU))

    return return_string

def train_nn(sess, saver, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
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

#        Alternate Implementation
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
            #batch_size = 2 #this is only to print accuracy while running. Use same batch size as when training.
            mini_train_data_yaml = './data/mini_train_val.yaml'
            val_data_yaml = './data/val.yaml'
            temp_batch_size = 16
            # Don't run check_Data_IOU_Accuracy() as it grows the TF graph with number of epochs (use check_FScore instead)
            # Note the below results are for a single batch only!
            temp_get_batches_fn = helper.gen_batch_function(mini_train_data_yaml, image_shape)
            temp_batch_generator = temp_get_batches_fn(temp_batch_size)
            #check_iou_acc = check_Data_IOU_Accuracy(sess, logits, input_image, keep_prob, num_classes, temp_batch_generator)
            #print("Epoch " + str(epoch+1) + "/" + str(epochs) + ": Training IOU is " + str(check_iou_acc))
            Fscore = check_FScore(sess, logits, input_image, keep_prob, num_classes, temp_batch_generator) #returns a string
            print("Epoch " + str(epoch+1) + "/" + str(epochs) + ": Training FScore:\n" + Fscore)
            #
            temp_get_batches_fn = helper.gen_batch_function(val_data_yaml, image_shape)
            temp_batch_generator = temp_get_batches_fn(temp_batch_size)
            #check_iou_acc = check_Data_IOU_Accuracy(sess, logits, input_image, keep_prob, num_classes, temp_batch_generator)
            #print("Epoch " + str(epoch+1) + "/" + str(epochs) + ": Validation IOU is " + str(check_iou_acc))
            Fscore = check_FScore(sess, logits, input_image, keep_prob, num_classes, temp_batch_generator) #returns a string
            print("Epoch " + str(epoch+1) + "/" + str(epochs) + ": Validation FScore:\n" + Fscore)
            #
            print("Epoch " + str(epoch+1) + "/" + str(epochs) + ": Final Batch's Loss is " + str(loss))
            print()

        #save model every 5 epochs
        if((epoch+1) % 5 == 0):
            saver.save(sess, './saved_model/segmentation_model')

    stop_time = time.time()
    time_diff = stop_time-start_time
    print("Total Training Time: " + str(timedelta(seconds=int(round(time_diff)))))

# print("Running Test 4 of 4")
# tests.test_train_nn(train_nn)

#    #To prevent certain layers from training
#    vgg_layer3_out = tf.stop_gradient(vgg_layer3_out)
#    vgg_layer4_out = tf.stop_gradient(vgg_layer4_out)
#    vgg_layer7_out = tf.stop_gradient(vgg_layer7_out)

def run():
    retrain = True
    predict_from_saved_model = False
    tf.reset_default_graph()
    
    num_classes = 3
    image_shape = (160,576) #height,width (numpy)
    data_dir = './data'
    runs_dir = './runs'
    train_data_yaml = './data/train.yaml'

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    #Hyperparameters
    reg_lambda = 1.15e-4 #2e-2 #5e-3 #1e-3
    learning_rate = 1e-4 #4e-4 #1e-3 #1e-4 is suggested in the Berkeley FCN paper
    batch_size = 32 #16 #8
    num_epochs = 20 #10 #25
    
    with tf.Session(graph=tf.Graph()) as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(train_data_yaml, image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        #Load VGG Net
        vgg_image_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        #print(tf.trainable_variables()) #for debug only
        #print(vgg_image_input.dtype) #for debug only
        #exit() #for debug only

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
        ##saver = tf.train.Saver()
        saver = tf.train.Saver(tf.trainable_variables()) #smaller in size
        if(retrain == True):
            saver.restore(sess,tf.train.latest_checkpoint('./saved_model'))
		#Train the network
        if(predict_from_saved_model == False):
            train_nn(sess, saver, num_epochs, batch_size, get_batches_fn, 
                     train_optimizer, loss, vgg_image_input,
                     correct_label, vgg_keep_prob, learning_rate, num_classes, logits)
    		#Save the model		
            saver.save(sess, './saved_model/segmentation_model')

        # TODO: Save inference data using helper.save_inference_samples
        if(predict_from_saved_model == True):
            saver.restore(sess,tf.train.latest_checkpoint('./saved_model'))
		#Do prediction
        helper.save_inference_samples(runs_dir, sess, image_shape, logits, vgg_keep_prob, vgg_image_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
