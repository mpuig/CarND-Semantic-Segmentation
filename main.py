import os.path
import tensorflow as tf
import helper
import warnings
import sys
from distutils.version import LooseVersion
import project_tests as tests

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
    print("Load Pretrained VGG Model into TensorFlow.")
    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    tensor_names = ['image_input:0', 'keep_prob:0', 'layer3_out:0', 'layer4_out:0', 'layer7_out:0']
    return [sess.graph.get_tensor_by_name(tn) for tn in tensor_names]

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    print("Create the layers for a fully convolutional network")

    # 1x1 convolution layer conversion
    layer_7 = tf.layers.conv2d(vgg_layer7_out,
        filters=num_classes, kernel_size=1, strides=(1,1), padding='SAME',
        kernel_initializer=tf.truncated_normal_initializer(stddev = 1e-3))

    layer_4 = tf.layers.conv2d(vgg_layer4_out,
        filters=num_classes, kernel_size=1, strides=(1,1), padding='SAME',
        kernel_initializer=tf.truncated_normal_initializer(stddev=1e-3))

    layer_3 = tf.layers.conv2d(vgg_layer3_out,
        filters=num_classes, kernel_size=1, strides=(1,1), padding='SAME',
        kernel_initializer=tf.truncated_normal_initializer(stddev=1e-3))

    # transposed convolution to upsample layer 7
    up_layer_7 = tf.layers.conv2d_transpose(layer_7,
        filters=num_classes, kernel_size=4, strides=(2,2), padding='SAME',
        kernel_initializer=tf.truncated_normal_initializer(stddev=1e-3))

    # first skip-connection layer with 4 and upsampled 7
    skip_1 = tf.add(layer_4, up_layer_7)

    # transposed convolution to upsample the previous combination (4_7)
    up_layer_4_7 = tf.layers.conv2d_transpose(skip_1,
        filters=num_classes, kernel_size=4, strides=(2,2), padding='SAME',
        kernel_initializer=tf.truncated_normal_initializer(stddev=1e-3))

    # second skip-connection layer with 3 and upsampled 4_7
    skip_2 = tf.add(layer_3, up_layer_4_7)

    # transposed convolution to upsample to original image
    up_layer_3_4_7 = tf.layers.conv2d_transpose(skip_2,
        filters=num_classes, kernel_size=16, strides=(8,8), padding='SAME',
        kernel_initializer=tf.truncated_normal_initializer(stddev=1e-3))

    return up_layer_3_4_7

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
    print("Build the TensorFLow loss and optimizer operations.")

    # TensorFlow pixelwise softmax cross-entropy loss
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    softmax = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)
    cross_entropy_loss = tf.reduce_mean(softmax)
    loss_operation = tf.reduce_mean(cross_entropy_loss)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_operation)
    # tf.summary.histogram("softmax", softmax)
    # tf.summary.scalar("loss", loss_operation)
    return logits, train_op, cross_entropy_loss

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
    print("Train neural network and print out the loss during training.")

    for epoch in range(epochs):
        print("EPOCH {}/{}".format(epoch+1, epochs), end="")
        for batch_x, batch_y in get_batches_fn(batch_size):
            _, loss = sess.run(
                [train_op, cross_entropy_loss],
                feed_dict={
                    input_image: batch_x,
                    correct_label: batch_y,
                    keep_prob: 0.8,
                    learning_rate: 0.0001}
                )
            print(".", end="")
            sys.stdout.flush()
        print ("-> Loss: ", loss)

tests.test_train_nn(train_nn)


def run():
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

    EPOCHS = 10
    BATCH_SIZE = 8

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN: load VGG & Create Layers
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        # Optimize NN
        learning_rate = tf.placeholder(tf.float32)
        correct_label = tf.placeholder(tf.float32, shape=(None, None, None, num_classes))
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # Train NN
        sess.run(tf.global_variables_initializer())
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        print("Save inference data")
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()