import codecs
import numpy as np

def read_data(data_path_pos, data_path_neg, max_word_length, max_sequence_length):
    data = list()
    sequence_length = list()
    y = list()

    with codecs.open(data_path_pos, 'r') as f1:
        for line in f1.readlines():
            line = line.strip()
            line = line.strip('#')
            line = line.strip()
            words = line.split('#')
            sequence = []
            for word in words:
                word = word.split()[:max_word_length]
                word = word + [0] * max(0, max_word_length - len(word))
                sequence.append(word)
            sequence_length.append(min(max_sequence_length, len(sequence)))
            sequence = sequence[:max_sequence_length]
            sequence = sequence + [[0] * max_word_length] * max(0, max_sequence_length - len(sequence))
            data.append(sequence)
            y.append([1, 0])

    with codecs.open(data_path_neg, 'r') as f2:
        for line in f2.readlines():
            line = line.strip()
            line = line.strip('#')
            line = line.strip()
            words = line.split('#')
            sequence = []
            for word in words:
                word = word.split()[:max_word_length]
                word = word + [0] * max(0, max_word_length - len(word))
                sequence.append(word)
            sequence_length.append(min(max_sequence_length, len(sequence)))
            sequence = sequence[:max_sequence_length]
            sequence = sequence + [[0] * max_word_length] * max(0, max_sequence_length - len(sequence))
            data.append(sequence)
            y.append([0, 1])

    return np.array(data), np.array(y), np.array(sequence_length)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)) / batch_size)
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]