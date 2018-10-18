#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import argparse
from moviepy.editor import VideoFileClip
import scipy.misc
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
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return input_image, keep_prob, layer3, layer4, layer7

print('\nTesting load_vgg')
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    #1x1 convolution of layer vgg_layer7_out
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                               kernel_initializer = tf.random_normal_initializer(stddev=0.01),
                               kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
                               name="conv_1x1")
    # upsample
    upsample1 = tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, strides = (2, 2) , padding='same',
                                           kernel_initializer = tf.random_normal_initializer(stddev=0.01),
                                           kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
                                           name="upsample1")

    # Apply 1x1 convolution to rescaled VGG layer 4 to reduce # of classes
    vgg_layer4_out_scaled = tf.multiply(vgg_layer4_out, 0.01, name='vgg_layer4_out_scaled')
    
    vgg_layer4_1x1 = tf.layers.conv2d(vgg_layer4_out_scaled, num_classes,
                                      kernel_size=1, strides=1, padding='same',
                                      kernel_initializer = tf.random_normal_initializer(stddev=0.01),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                      name='vgg_layer4_1x1')

    layer4_out = tf.add(upsample1, vgg_layer4_1x1)
    
    upsample2 = tf.layers.conv2d_transpose(layer4_out, num_classes, 4, strides = (2, 2) , padding='same',
                                           kernel_initializer = tf.random_normal_initializer(stddev=0.01),
                                           kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
                                           name='upsample2')
    
    # Apply 1x1 convolution to rescaled VGG layer 3 to reduce # of classes
    vgg_layer3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001, name='vgg_layer3_out_scaled')
    
    vgg_layer3_1x1 = tf.layers.conv2d(vgg_layer3_out_scaled, num_classes,
                                      kernel_size=1, strides=1, padding='same',
                                      kernel_initializer = tf.random_normal_initializer(stddev=0.01),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                      name='vgg_layer3_1x1')

    layer3_out = tf.add(upsample2, vgg_layer3_1x1)
    
    upsample3 = tf.layers.conv2d_transpose(layer3_out, num_classes, 16, strides = (8, 8) , padding='same',
                                             kernel_initializer = tf.random_normal_initializer(stddev=0.01),
                                             kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
                                             name='upsample3')

    # Add a final identity layer
    final_layer = tf.identity(upsample3, name='final_layer')
    
    return final_layer

print('\nTesting layers')
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
    # make logits and labels are 2D tensor 
    #where each row represents a pixel and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1,num_classes))
    
    # the loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= correct_label))

    l2_reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regularization_loss = tf.reduce_sum(l2_reg_loss)

    total_loss = cross_entropy_loss + regularization_loss
    
    # Add cross entropy loss to TensorBoard summary logging
    tf.summary.scalar('total_loss', total_loss)

    # define training operation - use AdamOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
    train_op = optimizer.minimize(total_loss,
                                  name='train_op')

    return logits, train_op, total_loss

print('\nTesting optimize')
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
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
    #set up TensorBoard logging
    writer = tf.summary.FileWriter('./log', sess.graph)
    merged_summary = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    
    print("Training FCN...")
    print()
    for i in range(epochs):
        print("EPOCH #", i+1)
        for (image, label) in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], 
                               feed_dict={input_image: image, correct_label: label, 
                                          keep_prob: 0.5, learning_rate: 0.001})
            #step=tf.train.global_step(sess, tf.train.get_global_step())
            #writer.add_summary(summary, step)
            print("Loss: = {:.3f}".format(loss))
        print()
    pass

print('\nTesting train_nn')
tests.test_train_nn(train_nn)

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', '--mode', help='mode [1]: 0=Train, 1=Test', type=int, default=1)
    args=parser.parse_args()
    
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

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        if args.mode == 0:
            # TODO: Build NN using load_vgg, layers, and optimize function
            epochs = 50
            batch_size = 5

            # TF placeholders
            correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
            learning_rate = tf.placeholder(tf.float32, name='learning_rate')

            input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

            nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

            logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

            # TODO: Train NN using the train_nn function

            train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)
       
            # TODO: Save inference data using helper.save_inference_samples
            #helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

            # Save model result
            print("\nSaving model.")
            saver = tf.train.Saver()
            save_path = saver.save(sess, "fcn_model/fcn")
            print("\nModel saved.")
            
        elif args.mode == 1:
            # Load saved model
            saver = tf.train.import_meta_graph('fcn_model/fcn'+'.meta')
            saver.restore(sess, tf.train.latest_checkpoint('fcn_model/'))
            graph = tf.get_default_graph()
            img_input = graph.get_tensor_by_name('image_input:0')
            keep_prob = graph.get_tensor_by_name('keep_prob:0')
            final_layer = graph.get_tensor_by_name('final_layer:0')
            logits = tf.reshape(final_layer, (-1, num_classes))

            # TODO: Save inference data using helper.save_inference_samples
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, img_input)
     
        else:
            # OPTIONAL: Apply the trained model to a video
            # Load saved model
            saver = tf.train.import_meta_graph('fcn_model/fcn'+'.meta')
            saver.restore(sess, tf.train.latest_checkpoint('fcn_model/'))
            graph = tf.get_default_graph()
            img_input = graph.get_tensor_by_name('image_input:0')
            keep_prob = graph.get_tensor_by_name('keep_prob:0')
            final_layer = graph.get_tensor_by_name('final_layer:0')
            logits = tf.reshape(final_layer, (-1, num_classes))


            def process_frame(img):
                # Input image is a Numpy array, resize it to match NN input dimensions
                img_orig_size = (img.shape[0], img.shape[1])
                img_resized = scipy.misc.imresize(img, image_shape)

                # Process image with NN
                img_softmax = sess.run([tf.nn.softmax(logits)],
                                        {keep_prob: 1.0, img_input: [img_resized]})

                # Reshape to 2D image dimensions
                img_softmax = img_softmax[0][:, 1].reshape(image_shape[0],
                                                           image_shape[1])

                # Threshold softmax probability to a binary road judgement (>50%)
                segmentation = (img_softmax > 0.5).reshape(image_shape[0],
                                                           image_shape[1], 1)

                # Apply road judgement to original image as a mask with alpha = 50%
                mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
                mask = scipy.misc.toimage(mask, mode="RGBA")
                street_im = scipy.misc.toimage(img_resized)
                street_im.paste(mask, box=None, mask=mask)

                # Resize image back to original dimensions
                street_img_resized = scipy.misc.imresize(street_im, img_orig_size)

                # Output image as a Numpy array
                img_out = np.array(street_img_resized)
                return img_out
        
            # Process video frames
            print ("\nStart video creation")
            video_outfile = './video/project_video_out.mp4'
            video = VideoFileClip('./video/project_video.mp4')#.subclip(37,38)
            video_out = video.fl_image(process_frame)
            video_out.write_videofile(video_outfile, audio=False)
            print ("\nVideo creation completed")
            

if __name__ == '__main__':
    run()
