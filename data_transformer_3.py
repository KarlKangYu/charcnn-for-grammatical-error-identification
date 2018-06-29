import codecs
import sys

#要得到[batch_size, sequence_length, word_length]
def transformer(data_in, data_out, letter_vocab):
    id2letters = {}
    letters2id = {}
    with codecs.open(letter_vocab, "r") as f1:  # vocab_in_letters
        for line in f1.readlines():
            token, id = line.strip().split("##")
            id = int(id)
            id2letters[id] = token
            letters2id[token] = id
    #start_id = letters2id["<start>"]
    letters_count = len(letters2id)

    with codecs.open(data_in, 'r') as f2:
        with codecs.open(data_out, 'w') as f3:
            for line in f2.readlines():
                line = line.strip()
                line = line.strip('#')
                words = line.split('#')
                for word in words:
                    word = word.split()
                    word = letters2ids(word, letters2id)
                    word = [str(i) for i in word]
                    word = ' '.join(word)
                    f3.write(word + '#')
                f3.write('\n')

def letters2ids(letters, letters2id):
    #start_id = letters2id["<start>"]
    unk_str = "<unk>"
    return [letters2id.get(letter, letters2id[unk_str])
                              for letter in letters if len(letter) > 0]

if __name__ == '__main__':
    args = sys.argv
    data_in = args[1]
    data_out = args[2]
    letter_vocab = args[3]
    transformer(data_in, data_out, letter_vocab)