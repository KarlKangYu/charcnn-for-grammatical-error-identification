import codecs
import sys

#把句子除了'的标点换成<pun>
#把句子拆成以'#'分隔的单词，单词内部用空格分开每个字符

def generation(data_in, data_out):
    with codecs.open(data_in, 'r') as f1:
        with codecs.open(data_out, 'w') as f2:
            for line in f1.readlines():
                line = line.strip()
                words_line = line
                words_line = words_line.replace('.', ' <pun> ')
                words_line = words_line.replace(',', ' <pun> ')
                words_line = words_line.replace('?', ' <pun> ')
                words_line = words_line.replace('!', ' <pun> ')
                words_line = words_line.replace('(', ' <pun> ')
                words_line = words_line.replace(')', ' <pun> ')
                words_line = words_line.replace("'", " ' ")
                words_line = words_line.replace('"', ' <pun> ')
                words_line = words_line.replace("  ", " ")
                words = words_line.split()
                for word in words:
                    if word == '<pun>':
                        f2.write(word + '#')
                    else:
                        letters = list(word)
                        letters = ' '.join(letters)
                        f2.write(letters + '#')
                f2.write('\n')



if __name__ == '__main__':
    args = sys.argv
    data_in = args[1]
    data_out = args[2]
    generation(data_in, data_out)



