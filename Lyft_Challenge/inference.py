#This is run when testing the FCN netwrok for the Lyft Challenge#
import main
import glob
import tensorflow as tf
import os
import shutil
import helper
import numpy as np
import skvideo.io
import time
import sys
import encoder_function
import json
import freeze_model
import scipy.misc #only used to save images for everything else use cv2
import cv2 #used  for everything else

def run_Full_Inference(test_video_file_name, print_json_output):
    ################Inference with frozen model#################
    #if print_json_output is True then it will print a json format output and not an mp4 video file
    #and viceversa if print_json_output is False
    tf.reset_default_graph()

    # Define needed variables
    model_dir = './saved_model'
    output_node_names = 'Reshape' #'Reshape,Mul' #the logits variable is named 'reshape' in the tensroflow graph
    image_shape = (160, 576)
    test_data_dir = './data/Test_Data'
    #optimized_frozen_model_filename = './saved_model/graph_frozen.pb'
    optimized_frozen_model_filename = './saved_model/graph_optimized.pb'

    # We use our "load_graph" function to load the frozen graph
    inference_graph = freeze_model.using_the_optimized_graph(optimized_frozen_model_filename)

    # # Verify that we can access the list of operations in the inference_graph graph
    # for op in inference_graph.get_operations():
    #     print(op.name)  # import/image_input, import/keep_prob, import/Reshape

    # Access the input and output nodes (all names are preceded by 'import/')
    vgg_image_input = inference_graph.get_tensor_by_name('import/image_input:0')
    vgg_keep_prob = inference_graph.get_tensor_by_name('import/keep_prob:0')
    non_softmax_logits = inference_graph.get_tensor_by_name('import/Reshape:0')
    logits = tf.nn.softmax(non_softmax_logits, name="softmax_logits")
    # logits = inference_graph.get_tensor_by_name('import/Processing/concat:0')
    #print(tf.Tensor.get_shape(vgg_image_input))
    #print(tf.Tensor.get_shape(logits))

    # Launch a Session
    with tf.Session(graph=inference_graph) as sess:
        # Note: we don't need to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants

        # #For Tensorboard visualization of the graph
        # writer = tf.summary.FileWriter('./logs', sess.graph)
        # writer.close()
        # #tensorboard --logdir="./logs" #to launch tensorboard (from bash terminal) after running the above code
        # print(sess.graph_def.ByteSize())

        # FOR DEBUG ONLY
        # output_image = sess.run(logits, feed_dict={vgg_image_input: np.zeros((1,160,576,3),dtype=np.uint8), vgg_keep_prob: 1.0})
        # output_image = inference.run_Inference(image, sess, image_shape, logits, vgg_keep_prob, vgg_image_input)
        # plt.imshow(output_image) #something is messed up with plt.imshow()
        # plt.show()
        # scipy.misc.imsave('./data/Test_Data/delete_output_image_frozen_model.png', output_image)
        # print(output_image.shape)

        #Do prediction on test images
        # test_Image(test_data_dir, sess, image_shape, logits, vgg_keep_prob, vgg_image_input)
        test_Video(test_video_file_name, print_json_output, sess, image_shape, logits, vgg_keep_prob, vgg_image_input)
        #devices = sess.list_devices() #put this in side the for loop for each video frame to check for memory leak

    ################################################################


def run_Full_Inference_Unfrozen_Model_delete(test_video_file_name, print_json_output):
    #if print_json_output is True then it will print a json format output and not an mp4 video file
    #and viceversa if print_json_output is False
    tf.reset_default_graph()
    
    #Hyperparameters
    reg_lambda = 2e-2 #5e-3 #1e-3
    learning_rate = 1e-4 #4e-4 #1e-3 #1e-4 is suggested in the Berkeley FCN paper
    batch_size = 32 #16 #8
    num_epochs = 20 #10 #25

    num_classes = 3
    image_shape = (160,576)
    data_dir = './data'
    test_data_dir = './data/Test_Data'

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir) #put this in the preinstall linux script
    
    with tf.Session(graph=tf.Graph()) as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # TODO: Build NN using load_vgg, layers, and optimize function
        #Load VGG Net
        vgg_image_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = main.load_vgg(sess, vgg_path)
        
        #Add skip conenctions, 1x1 convolutions, and upsampling for segmentation to the vgg net
        nn_last_layer = main.layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, reg_scale=reg_lambda)
        
        correct_label = tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1], num_classes])
        non_softmax_logits, train_optimizer, loss = main.optimize(nn_last_layer, correct_label, learning_rate, num_classes)
        logits = tf.nn.softmax(non_softmax_logits, name="softmax_logits")

        # TODO: Train NN using the train_nn function
        #print("Initializing TF Global Variables")
        sess.run(tf.global_variables_initializer()) #initialize all global variables (takes time to finish)
        #print("Restoring Saved Model")
        #saver = tf.train.Saver()
        saver = tf.train.Saver(tf.trainable_variables()) #smaller in size
        saver.restore(sess,tf.train.latest_checkpoint('./saved_model')) #takes time to finish

        #Do prediction on test images
        #test_Image(test_data_dir, sess, image_shape, logits, vgg_keep_prob, vgg_image_input)
        test_Video(test_video_file_name, print_json_output, sess, image_shape, logits, vgg_keep_prob, vgg_image_input)


#############For Images###################
def test_Image(test_data_dir, sess, image_shape, logits, keep_prob, image_pl):
    # Make folder for test output
    output_dir = os.path.join(test_data_dir, 'test_results')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Running Inference. Saving test images to: {}'.format(output_dir))

    image_paths = glob.glob(os.path.join(test_data_dir,'*.png'))
    for image_path in image_paths:
        #image = scipy.misc.imread(image_path)
        image = cv2.imread(image_path)[:,:,(2,1,0)] #bgr to rgb
        output_image = run_Inference(image, sess, image_shape, logits, keep_prob, image_pl)
        base_path_output_image = os.path.basename(image_path)
        scipy.misc.imsave(os.path.join(output_dir, base_path_output_image), output_image)


def run_Inference(image, sess, image_shape, logits, keep_prob, image_pl):
    #original image size = 600x800
    orig_image_shape = (600,800) #600x800x3
    #target image size for vgg16 input= 160x576x3 (hence need to crop out unnecessary info. first)
    image = image[160:460,:,:] #crop size = 300x800x3
    #image = scipy.misc.imresize(image, image_shape) #resize
    image = cv2.resize(image, image_shape[::-1]) #cv2 uses width,heigth vs height,width for image shape

    output_image = np.copy(image)
    
    image_prep = image #unnormalized image
    #image_prep = 2.0*(image/255.0-0.5) #nomalized between -1 and 1
    #im_softmax = sess.run([tf.nn.softmax(logits)],feed_dict={keep_prob: 1.0, image_pl: [image_prep]}) #im_softmax is list of length 1
    im_softmax = sess.run([logits],feed_dict={keep_prob: 1.0, image_pl: [image_prep]}) #im_softmax is list of length 1
    #print(len(im_softmax)) #tuple of length 1
    #print(im_softmax[0].shape) #total_num_pixels X num_classes

    #Green color for road and Blue color for Car
    num_classes = im_softmax[0].shape[1]
    im_softmax = im_softmax[0].reshape(image_shape[0], image_shape[1], num_classes) #im_softmax is height X width X num_classes
    # Note: im_softmax is of size = 160x576x3

    #debug only
    # im_softmax = sess.run([logits],feed_dict={keep_prob: 1.0, image_pl: [image_prep]}) #im_softmax is list of length 1
    # im_softmax = im_softmax[0][0,:,:,:]
    # im_softmax = im_softmax.reshape(image_shape[0], image_shape[1], 3) 

    pixel_class_labels = np.argmax(im_softmax,axis=2) #class_label will be of shape image_shape[0] X image_shape[1]
    class_labels = [1,2] #[0,1,2] #everything_else=0 (Red), lane/road=1 (Green), vehicles=2 (Blue). Only show road/lane and vehicles.
    for label in class_labels:
        label_markings = (pixel_class_labels == label).nonzero() #find label predictions (shape image_shape[0] X image_shape[1])
        output_image[label_markings[0],label_markings[1],label] = 255 #make label markings to the corresponding color

    #Resize output_image to original size i.e. 600x800x3
    #But first resize to 300x800x3
    #output_image = scipy.misc.imresize(output_image, (300,800))
    output_image = cv2.resize(output_image, (800,300)) #cv2 uses width,heigth vs height,width for image shape
    final_output_image = np.zeros((600,800,3))
    final_output_image[160:460,:,:] = output_image

    return final_output_image #maybe try to yield if its faster instead of return

#############End For Images###################

#############For VIDEO###################
def test_Video(test_video_file_name, print_json_output, sess, image_shape, logits, keep_prob, image_pl):
    input_test_video = skvideo.io.vread(test_video_file_name) #returns as a numpy array of shape (num_frames x height x width x depth)
    num_frames = input_test_video.shape[0]
    strt_time = time.time() #for debug only
    ###
    if(print_json_output == True): #print json output in the Udacity required format
        #print('Here is the json output:')
        answer_key = {}
        frame = 1 #frame numbering starts at 1
        for image_frame in input_test_video: #iterate through the first dimension i.e. frames
            output_image = run_Inference(image_frame, sess, image_shape, logits, keep_prob, image_pl)
            answer_key[frame] = encoder_function.encode_inference_output_image(output_image)
            frame += 1 #increment frame
        #print(json.dumps(answer_key)) #Print output in proper (Udacity required) json format
        #To save json output to file #
        with open('./data/Test_Data/encode_output_test_video.json', 'w') as out_file:
            json.dump(answer_key, out_file, ensure_ascii=False) #for utf-8 formatting, ensure_ascii=false (actually doesn't matter)
            out_file.close()

    else: #make an output video
        frame_num = 0
        testvideo_base_filename = os.path.basename(test_video_file_name)
        output_file = test_video_file_name.replace(testvideo_base_filename, 'output_' + testvideo_base_filename)
        # Run NN on test video and save the result to another video
        print('Running Inference. Saving Test Video to: {}'.format(output_file))
        output_video = []
        for image_frame in input_test_video: #iterate through the first dimension i.e. frames
            frame_strt_time = time.time()
            output_image = run_Inference(image_frame, sess, image_shape, logits, keep_prob, image_pl)

            #debug only
            #For Tensorboard visualization of the graph
            # if frame_num == 3:
            #     writer = tf.summary.FileWriter('./logs', sess.graph)
            #     writer.close()
            #     exit()
            #tensorboard --logdir="./logs" #to launch tensorboard (from bash terminal) after running the above code
            #print(sess.graph_def.ByteSize())

            #devices = sess.list_devices() #debug only
            output_video.append(output_image)
            frame_num += 1
            print(frame_num, time.time()-frame_strt_time)

        output_video = np.array(output_video)
        skvideo.io.vwrite(output_file,output_video)
    ###
    stp_time = time.time()
    print('Total time: ', stp_time-strt_time) #for debug only
    print('Number of frames is: ', num_frames) #for debug only
    print('FPS: ', num_frames/(stp_time-strt_time))
    return None

#############End For VIDEO###################


if __name__ == '__main__':
    #./preinstall_script #first run this script in bash/command prompt to install required dependencies(for Udacity Workspace)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #to mute tensorflow output printings
    video_file = sys.argv[-1] #returns the full path
    base_video_file_name = os.path.basename(video_file) #only need to compare the base path
    #print(video_file)
    if base_video_file_name == 'inference.py':
        print('Provide Video File Name!')
        #exit() #exit the program
        #run_Full_Inference(test_video_file_name = './data/Test_Data/test_video.mp4', print_json_output=True) #for debug only
        run_Full_Inference_Unfrozen_Model_delete(test_video_file_name = './data/Test_Data/test_video.mp4', print_json_output=False) #for debug only
        print('done')

    else:
        run_Full_Inference(test_video_file_name = video_file, print_json_output=True)
