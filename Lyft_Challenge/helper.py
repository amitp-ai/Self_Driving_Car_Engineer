import re
import random
import numpy as np
import os.path
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import scipy.misc #only used to save images for everything else use cv2
import cv2 #used  for everything else
import yaml


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def image_augmentation(image, gt_image):
    #use cv2 to resize images and read in images as its faster than scipy
    a = np.random.random()

    if a < 0.25: #flip horizontally
        image = cv2.flip(image, 1)
        gt_image = cv2.flip(gt_image, 1)
    elif a < 0.5: #rotate (random angle) image while keeping shape
        max_angle = 20
        angle = int((np.random.random()-0.5)*2*max_angle)
        #input image
        rows,cols,_ = image.shape
        M = cv2.getRotationMatrix2D(center=(cols/2,rows/2),angle=angle,scale=1)
        image = cv2.warpAffine(image,M,(cols,rows))
        gt_image = cv2.warpAffine(gt_image,M,(cols,rows))
    elif a < 0.75: #translate (random amount) image while keeping the shape
        rows,cols,_ = image.shape
        max_tx, max_ty = cols*0.15, rows*0.15 #15% of each dimension
        tx = int((np.random.random()-0.5)*2*max_tx)
        ty = int((np.random.random()-0.5)*2*max_ty)
        M = np.float32([[1,0,tx],[0,1,ty]])
        image = cv2.warpAffine(image,M,(cols,rows))
        gt_image = cv2.warpAffine(gt_image,M,(cols,rows))
    else: #add noise to the image
        max_noise = 50
        noise = np.random.random(image.shape)*max_noise
        image = image + noise
        image = image.astype(np.uint8) #need to do this, otherwise messes up the image as noise is float32
        #noise doesn't affect the gt_image

    return image, gt_image



def gen_batch_function(train_data_yaml, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = yaml.load(open(train_data_yaml, 'rb').read())        
        random.shuffle(image_paths) #to get different ordering of training images every epoch

        #for Debug Only
        #image_paths = image_paths[0:20]

        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = image_file.replace('CameraRGB','PreProcessedCameraSeg')


                #original image size = 600x800
                #target image size for vgg16 input= 160x576 (hence need to crop out unnecessary info. first)
                #image = scipy.misc.imresize(scipy.misc.imread(image_file)[160:460,:,:], image_shape) #crop and resize
                image = cv2.resize(cv2.imread(image_file)[160:460,:,(2,1,0)], image_shape[::-1]) #crop and resize and bgr to rgb and cv2 uses width,heigth vs height,width for image shape
                image_prep = image #unnormalized image
                #image_prep = 2.0*(image/255.0-0.5) #nomalized between -1 and 1
                #image_gt = scipy.misc.imresize(scipy.misc.imread(gt_image_file)[160:460], image_shape) #crop and resize
                image_gt = cv2.resize(cv2.imread(gt_image_file)[160:460,:,(2,1,0)], image_shape[::-1]) #crop and resize and bgr to rgb and cv2 uses width,heigth vs height,width for image shape

                num_classes=3
                multi_class_gt_image = np.zeros((image_gt.shape[0],image_gt.shape[1],num_classes))
                for class_num in range(0,num_classes,1):
                    seg_class = (image_gt[:,:,0] == class_num).nonzero()
                    multi_class_gt_image[seg_class[0],seg_class[1],class_num] = 1
                    #print(seg_class[0].shape, seg_class[1].shape)
                    #plt.imshow(multi_class_gt_image[:,:,class_num])
                    #plt.show()

                #image augmentation
                a = 0.5 #percent of images to augment
                if np.random.random() < a:
                    image_prep, multi_class_gt_image = image_augmentation(image_prep, multi_class_gt_image)
                #else don't augment i.e. use original image
                
                images.append(image_prep)
                gt_images.append(multi_class_gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, test_data_yaml, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    test_data_list = yaml.load(open(test_data_yaml, 'rb').read())
    test_data_list = test_data_list[0:10] #only save 10 test images

    for image_file in test_data_list:
        #original image size = 600x800
        #target image size for vgg16 input= 160x576 (hence need to crop out unnecessary info. first)
        #image = scipy.misc.imresize(scipy.misc.imread(image_file)[160:460,:,:], image_shape) #crop and resize
        image = cv2.resize(cv2.imread(image_file)[160:460,:,(2,1,0)], image_shape[::-1]) #crop and resize and bgr to rgb and cv2 uses width,heigth vs height,width for image shape
        output_image = np.copy(image)
        image_prep = image #unnormalized image
        #image_prep = 2.0*(image/255.0-0.5) #nomalized between -1 and 1

        im_softmax = sess.run([tf.nn.softmax(logits)],{keep_prob: 1.0, image_pl: [image_prep]}) #im_softmax is list of length 1
        #print(len(im_softmax)) #tuple of length 1
        #print(im_softmax[0].shape) #total_num_pixels X num_classes

        num_classes = im_softmax[0].shape[1]
        im_softmax = im_softmax[0].reshape(image_shape[0], image_shape[1], num_classes)

        pixel_class_labels = np.argmax(im_softmax,axis=2) #class_label will be of shape image_shape[0] X image_shape[1]
        class_labels = [1,2] #[0,1,2] #everything_else=0 (Red), lane/road=1 (Green), vehicles=2 (Blue). Only show road/lane and vehicles.
        for label in class_labels:
            label_markings = (pixel_class_labels == label).nonzero() #find label predictions (shape image_shape[0] X image_shape[1])
            output_image[label_markings[0],label_markings[1],label] = 255 #make label markings to the corresponding color

        #Resize output_image to original size i.e. 600x800x3
        #But first resize to 300x800x3
        #output_image = scipy.misc.imresize(output_image, (300,800))
        output_image = cv2.resize(output_image, (800, 300))
        final_output_image = np.zeros((600,800,3))
        final_output_image[160:460,:,:] = output_image
        
        yield os.path.basename(image_file), final_output_image


def save_inference_samples(runs_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    test_data_yaml = './data/test.yaml'
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, test_data_yaml, image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

