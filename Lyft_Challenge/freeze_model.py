import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import inference
import helper
import os
import main

#This is based upon the below stackexchange comment
#https://stackoverflow.com/questions/45382917/how-to-optimize-for-inference-a-simple-saved-tensorflow-1-0-1-graph

def save_tensorflow_graph_pb():
    #This is to save the tensroflow graph as a .pb file in binary
    tf.reset_default_graph()
    
    #Hyperparameters
    reg_lambda = 2e-2 #5e-3 #1e-3
    learning_rate = 1e-4 #4e-4 #1e-3 #1e-4 is suggested in the Berkeley FCN paper
    batch_size = 32 #16 #8
    num_epochs = 20 #10 #25

    num_classes = 3
    image_shape = (160,576)
    data_dir = './data'

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
        logits, train_optimizer, loss = main.optimize(nn_last_layer, correct_label, learning_rate, num_classes)


        sess.run(tf.global_variables_initializer()) #initialize all global variables (takes time to finish)
        #saver = tf.train.Saver()
        saver = tf.train.Saver(tf.trainable_variables()) #smaller in size
        saver.restore(sess,tf.train.latest_checkpoint('./saved_model')) #takes time to finish
        # Save GraphDef
        print('write graphdef')
        tf.train.write_graph(sess.graph_def,'.','./saved_model/graph.pb', as_text=False) #as_text=False means saves in binary
        #with tf.gfile.GFile('./saved_model/graph.pb', 'rb') as f:
        #    f.write(sess.graph_def.SerializeToString())
        
        # save the checkpoints (this is redundant. don't do it actually)
        #saver.save(sess=sess, save_path="./saved_model/segmentation_model_resaved")
    
#step 1:
#save_tensorflow_graph_pb()

#step 2:
# Freeze the tensorflow graph (run the below command in linux bash). This converts variables into constants.
#python -m tensorflow.python.tools.freeze_graph --input_graph ./saved_model/graph.pb --input_binary=true --input_checkpoint ./saved_model/segmentation_model --output_graph ./saved_model/graph_frozen.pb --output_node_names='Reshape'

#step 3:
# Optimize for inference. This extra variables used during training that are not needed for inference e.g. gradients etc
#python -m tensorflow.python.tools.optimize_for_inference --input ./saved_model/graph_frozen.pb --output ./saved_model/graph_optimized.pb --input_names='image_input,keep_prob' --output_names='Reshape'

# Side note for saving multiple outputs
#If there are multiple output nodes, then specify : 
#output_node_names = 'boxes, scores, classes' and import graph by,
#boxes,scores,classes, = tf.import_graph_def(graph_def_optimized, return_elements=['boxes:0', 'scores:0', 'classes:0'])

#step 4:
# Using the optimized graph
def using_the_optimized_graph(optimized_frozen_model_filename = './saved_model/graph_optimized.pb'):
    with tf.gfile.GFile(optimized_frozen_model_filename, 'rb') as f:
       graph_def_optimized = tf.GraphDef()
       graph_def_optimized.ParseFromString(f.read())

    # Then, we import the graph_def_optimized into a new Graph and return it 
    with tf.Graph().as_default() as inference_graph:
        # The name var will be prefixed by 'import/' for every op/nodes in the graph
        # Since we load everything in a new graph, this is not necessary in this case. But good practice in general.
        tf.import_graph_def(graph_def_optimized, name="import")
        #a = tf.import_graph_def(graph_def, return_elements=['image_input'], name='prefix')
        #print(a)

    # #debug only
    # print('Operations in Optimized Graph:')
    # print([op.name for op in inference_graph.get_operations()])
    # print('starting tf session')

    #for debug only
    # with tf.Session(graph=inference_graph) as sess:
    #     # Note: we don't need to initialize/restore anything
    #     # There is no Variables in this graph, only hardcoded constants
    #     logits, = tf.import_graph_def(graph_def_optimized, return_elements=['Reshape:0'], name="import")
    #     vgg_image_input = inference_graph.get_tensor_by_name('import/image_input:0')
    #     vgg_keep_prob = inference_graph.get_tensor_by_name('import/keep_prob:0')
    #     #logits = inference_graph.get_tensor_by_name('import/Reshape:0') #already have from above

    #     # Testing
    #     print('Testing:')
    #     output_image = sess.run(logits, feed_dict={vgg_image_input: np.zeros((1,160,576,3),dtype=np.uint8), vgg_keep_prob: 1.0})
    #     print(output_image.shape, 'should be 160x576 by 3 i.e.', 160*576, 'by 3')

    return inference_graph


#using_the_optimized_graph()
