# train a keras model of pix2pix using tf-2
# code taken from https://www.tensorflow.org/alpha/tutorials/generative/pix2pix

import tensorflow as tf
import argparse
import pix2pix
import os

def main():
    # create dataset
    print("Creating Dataset...")
    train_dataset = tf.data.Dataset.list_files(args.input_dir + '/train/*.png')
    train_dataset = train_dataset.shuffle(pix2pix.settings.BUFFER_SIZE)
    train_dataset = train_dataset.map(pix2pix.utils.load_image_train,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(1)

    test_dataset = tf.data.Dataset.list_files(args.input_dir + '/val/*.png')
    # shuffling so that for every epoch a different image is generated
    # to predict and display the progress of our model.
    train_dataset = train_dataset.shuffle(pix2pix.settings.BUFFER_SIZE)
    test_dataset = test_dataset.map(pix2pix.utils.load_image_test)
    test_dataset = test_dataset.batch(1)

    print("Creating Pix2Pix Model...")
    model = pix2pix.train.Pix2PixModel(args.checkpoint_dir, args.output_dir)
    print("Training Model...")
    model.train(train_dataset=train_dataset, test_dataset=test_dataset, epochs=args.epochs)
    print("Saving Generator Weights...")
    model.generator.save(args.checkpoint_dir+'/generator_weights.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=0, help='training epochs')
    parser.add_argument('--input_dir', dest='input_dir', type=str,
                        default=0, help='input directory, contains train and val directories')
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', type=str,
                        default=0, help='directory where checkpoints are saved')
    parser.add_argument('--output_dir', dest='output_dir', type=str,
                        default=0, help='directory for saving final output')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    main()
