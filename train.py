import tensorflow as tf
import numpy as np
import os
import datetime
import data_loader
from model import CharCNN

flags = tf.flags

# data
tf.flags.DEFINE_float("dev_percentage", 0.001, "Percentage of the training data to use for validation")
flags.DEFINE_string('pos_data',    'data',   'pos data directory. ')
flags.DEFINE_string('neg_data',    'data',   'neg data directory. ')
flags.DEFINE_string('save_path',   'model/',     'training directory (models and summaries are saved there periodically)')

# model params
flags.DEFINE_integer('num_classes',        2,                            'class numbers')
flags.DEFINE_integer('rnn_size',        2048,                            'size of LSTM internal state')
flags.DEFINE_integer('attention_size',        2048,                            'size of Attention layer internal state')
flags.DEFINE_integer('num_highway_layers',  1,                              'number of highway layers')
flags.DEFINE_integer('char_vocab_size', 31,                             'character vocab size')
flags.DEFINE_integer('char_embed_size', 150,                             'dimensionality of character embeddings')
flags.DEFINE_string ('filters',         '[1,2,3,4,5,6,7]',              'CNN filter widths')
flags.DEFINE_string ('filter_sizes', '[50,100,150,200,200,200,200]', 'number of features in the CNN filters')
flags.DEFINE_float  ('dropout_keep',         0.5,                            'dropout_keep 1 = no dropout')

# optimization
flags.DEFINE_integer('max_seq_length',    35,   'number of timesteps to unroll for')
flags.DEFINE_integer('batch_size',          64,   'number of sequences to train on in parallel')
flags.DEFINE_integer('num_epochs',          25,   'number of full passes through the training data')
flags.DEFINE_float  ('max_grad_norm',       5.0,  'normalize gradients at')
flags.DEFINE_integer('max_word_length',     15,   'maximum word length')

# bookkeeping
flags.DEFINE_integer("evaluate_every", 10000, "Evaluate model on dev set after this many steps (default: 100)")
flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")

# Misc Parameters
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = flags.FLAGS

def train_step(x_batch, y_batch, sequence_length, sess, cnn):
    feed_dict = {
        cnn.input: x_batch,
        cnn.label: y_batch,
        cnn.sequence_length: sequence_length,
        cnn.dropout_keep_prob: FLAGS.dropout_keep
    }
    _, step, loss, accuracy, _, lr = sess.run(
        [cnn._train_op, cnn.global_step, cnn.loss, cnn.accuracy, cnn.clear_char_embedding_padding, cnn.lr],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}, learning rate {:g}".format(time_str, step, loss, accuracy, lr))

def dev_step(x_batch, y_batch, sequence_length, sess, cnn, max_dev_acc):
    feed_dict = {
        cnn.input: x_batch,
        cnn.label: y_batch,
        cnn.sequence_length: sequence_length,
        cnn.dropout_keep_prob: 1.0
    }
    step, loss, accuracy = sess.run(
        [cnn.global_step, cnn.loss, cnn.accuracy],
    feed_dict)
    if accuracy > max_dev_acc:
        max_dev_acc = accuracy
    time_str = datetime.datetime.now().isoformat()
    print("{}: Dev_phase!!!: \n step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    return max_dev_acc

def main(_):
    print("Loading data...")
    x, y, sequence_length = data_loader.read_data(FLAGS.pos_data, FLAGS.neg_data, FLAGS.max_word_length, FLAGS.max_seq_length)
    print("Data Size:", len(y))
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    seq_shuffled = sequence_length[shuffle_indices]
    dev_sample_index = -1 * int(FLAGS.dev_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    seq_train, seq_dev = seq_shuffled[:dev_sample_index], seq_shuffled[dev_sample_index:]
    del x, y, sequence_length, x_shuffled, y_shuffled, seq_shuffled
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        #session_conf.gpu_options.per_process_gpu_memory_fraction = 0.45
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = CharCNN(char_vocab_size=FLAGS.char_vocab_size,
                          char_embed_size=FLAGS.char_embed_size,
                          batch_size=FLAGS.batch_size,
                          max_word_length=FLAGS.max_word_length,
                          max_seq_length=FLAGS.max_seq_length,
                          filters=eval(FLAGS.filters),
                          filter_sizes=eval(FLAGS.filter_sizes),
                          num_classes=FLAGS.num_classes,
                          rnn_size=FLAGS.rnn_size,
                          attention_size=FLAGS.attention_size)

            save_path = os.path.join(FLAGS.save_path)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            saver = tf.train.Saver(tf.trainable_variables())
            for v in tf.trainable_variables():
                print("Save:", v.name)

            sess.run(tf.global_variables_initializer())

            check_point_dir = os.path.join(FLAGS.save_path)
            ckpt = tf.train.get_checkpoint_state(check_point_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("Created model with fresh parameters.")

            batches = data_loader.batch_iter(
                list(zip(x_train, y_train, seq_train)), FLAGS.batch_size, FLAGS.num_epochs
            )

            gloabl_max_acc = 0
            for batch in batches:
                x_batch, y_batch, seq_batch = zip(*batch)
                train_step(x_batch, y_batch, seq_batch, sess, cnn)
                current_step = tf.train.global_step(sess, cnn.global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    max_dev_acc = 0
                    print("\nEvaluation:")
                    batches_dev = data_loader.batch_iter(
                        list(zip(x_dev, y_dev, seq_dev)), FLAGS.batch_size, 1
                    )
                    for batch_dev in batches_dev:
                        x_batch_dev, y_batch_dev, seq_batch_dev = zip(*batch_dev)
                        max_dev_acc = dev_step(x_batch_dev, y_batch_dev, seq_batch_dev, sess, cnn, max_dev_acc)
                    print("During this evaluation phase, the max accuracy is:", max_dev_acc)
                    if max_dev_acc > gloabl_max_acc:
                        gloabl_max_acc = max_dev_acc
                    print("\n Until now, the max accuracy is:", gloabl_max_acc)
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, os.path.join(save_path, "model"), global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


if __name__ == '__main__':
    tf.app.run()



