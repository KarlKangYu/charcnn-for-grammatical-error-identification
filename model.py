import tensorflow as tf
import numpy as np


class CharCNN(object):
    """
    A simplified implementation of Sentence-Level Grammatical Error Identification as Sequence-to-Sequence Correction.
    Just binary classification.
    Using charcnn & highway instead of word embedding. And BiLSTM.
    """
    def __init__(self, char_vocab_size, char_embed_size, batch_size, max_word_length, max_seq_length,
                 filters, filter_sizes, num_classes, rnn_size, attention_size, num_highway_layers=1, max_grad_norm=5):

        assert len(filters) == len(filter_sizes)

        self.input = tf.placeholder(tf.int32, shape=[batch_size, max_seq_length, max_word_length], name="input")
        self.label = tf.placeholder(tf.float32, shape=[batch_size, num_classes], name="label")
        self.sequence_length = tf.placeholder(tf.int32, shape=[batch_size], name="sequence_length")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.variable_scope("Char_Embedding"):
            char_embedding = tf.get_variable("char_embedding", [char_vocab_size, char_embed_size], dtype=tf.float32)

            # this op clears embedding vector of first symbol (symbol at position 0, which is by convention the position
            # of the padding symbol). It can be used to mimic Torch7 embedding operator that keeps padding mapped to
            # zero embedding vector and ignores gradient updates. For that do the following in TF:
            # 1. after parameter initialization, apply this op to zero out padding embedding vector
            # 2. after each gradient update, apply this op to keep padding at zero

            self.clear_char_embedding_padding = tf.scatter_update(char_embedding, [0], tf.constant(0.0, shape=[1, char_embed_size]), name="clear_padding_embedding")

            input_embedded = tf.nn.embedding_lookup(char_embedding, self.input)
            input_embedded = tf.reshape(input_embedded, [-1, max_word_length, char_embed_size])  #[batch_size * max_seq_length, max_word_length, char_embed_size]

        with tf.variable_scope("Char_CNN"):
            input_cnn = self.charCNN(input_embedded, filters, filter_sizes)  # [batch_size x max_seq_length, cnn_size]  # where cnn_size=sum(filter_sizes)

        if num_highway_layers > 0:
            input_cnn = self.highway(input_cnn, num_highway_layers)

        with tf.variable_scope("LSTM"):
            input_cnn = tf.reshape(input_cnn, [batch_size, max_seq_length, -1])   # [batch_size, max_seq_length, cnn_size]
            length = self.sequence_length
            fw_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=1.0)
            bw_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=1.0)
            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)
            init_fw = fw_cell.zero_state(batch_size, dtype=tf.float32)
            init_bw = bw_cell.zero_state(batch_size, dtype=tf.float32)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs=input_cnn,
                                                                 initial_state_fw=init_fw, initial_state_bw=init_bw, sequence_length=length)
            rnn_outputs = tf.concat(outputs, 2)  #[batch_size, max_seq_length, rnn_size * 2]
            self.rnn_outputs = tf.identity(rnn_outputs, "BiLSTM_outputs")

        with tf.variable_scope("Attention"):
            attention_input_size = rnn_outputs.shape[-1]
            att_w = tf.get_variable("Attention_W", [attention_input_size, attention_size], dtype=tf.float32)
            att_b = tf.get_variable("Attention_b", [attention_size], dtype=tf.float32)
            u = tf.tanh(tf.tensordot(rnn_outputs, att_w, axes=[[2], [0]]) + att_b)
            context = tf.get_variable("Attention_context", [attention_size], dtype=tf.float32)
            att_logits = tf.tensordot(u, context, axes=[[2], [0]])
            alpha = tf.nn.softmax(logits=att_logits)
            attention_out = tf.reduce_sum(rnn_outputs * tf.expand_dims(alpha, -1), 1) #[batch_size, rnn_size * 2]

            attention_out = tf.nn.dropout(attention_out, self.dropout_keep_prob)

        with tf.variable_scope("Softmax"):
            softmax_w = tf.get_variable("SoftMax_W", [attention_out.shape[-1], num_classes], dtype=tf.float32)
            softmax_b = tf.get_variable("SoftMax_b", [num_classes], dtype=tf.float32)
            self.logits = tf.matmul(attention_out, softmax_w) + softmax_b
            self.predictions = tf.argmax(self.logits, 1, name="predictions")
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("Accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="Accuracy")

        with tf.name_scope("Train"):
            self.global_step = tf.contrib.framework.get_or_create_global_step()
            tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            for i in tvars:
                print("Variables:", i)
            grads,_ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), max_grad_norm)
            optimizer = tf.train.AdamOptimizer(0.001)
            self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

    def charCNN(self, input, filters, filter_sizes):
        max_word_length = input.shape[1]
        char_embed_size = input.shape[-1]

        input = tf.expand_dims(input, -1)   #[batch_size * max_seq_length, max_word_length, char_embed_size, 1]
        layers = list()

        for filter_width, filter_size in zip(filters, filter_sizes):
            reduced_length = max_word_length - filter_width + 1

            with tf.variable_scope("conv_filter_%d" % filter_width):
                w = tf.get_variable("conv_w", [filter_width, char_embed_size, 1, filter_size], dtype=tf.float32)
                b = tf.get_variable("conv_b", [filter_size], dtype=tf.float32)

                conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="VALID") + b

            pool = tf.nn.max_pool(tf.tanh(conv), [1, reduced_length, 1, 1], [1, 1, 1, 1], "VALID")  # [batch_size x 1 x 1 x filter_size]
            layers.append(tf.squeeze(pool))

        if len(filters) > 1:
            output = tf.concat(layers, 1)
        else:
            output = layers[0]

        return output

    def highway(self, input, num_highway_layers=1, f=tf.nn.relu):
        size = input.shape[-1]
        for i in range(num_highway_layers):
            with tf.variable_scope("HighWay_%d" % i):
                wh = tf.get_variable("Wh", [size, size], dtype=tf.float32)
                bh = tf.get_variable("bh", [size], dtype=tf.float32)
                wt = tf.get_variable("Wt", [size, size], dtype=tf.float32)
                bt = tf.get_variable("bt", [size], dtype=tf.float32)

                t = tf.sigmoid(tf.matmul(input, wt) + bt)
                g = f(tf.matmul(input, wh) + bh)

                output = t * g + (1 - t) * input
                input = output
        return output


    # def _length(self, input):
    #     input = tf.reduce_sum(input, axis=-1)
    #     relevant = tf.sign(tf.abs(input))
    #     length = tf.reduce_sum(relevant, axis=1)
    #     length = tf.cast(length, tf.int32)
    #     return length
















