import numpy
import random

word_counts = dict()
tag_counts = dict()

lexical_prob_dict = dict()
bigram_prob_dict = dict()
bigram_dict = dict()
word_tag_count = dict()
prev_tag = "o"
k = 0.001


f = open('input.txt', 'w')
for line in open('gene-trainF18.txt', 'r'):
    token_tag = line.strip('\n')
    token_tag = line.split('\t')
    if len(token_tag) >= 3:
        f.write(token_tag[1] + '\t' + token_tag[2])

f.close()

for line in open('input.txt'):
    tag = line.strip('\n').split('\t')[1].lower()
    word = line.strip('\n').split('\t')[0].lower()
    if tag not in tag_counts.keys():
        tag_counts[tag] = 0
    tag_counts[tag] += 1
    if word not in word_counts.keys():
        word_counts[word] = 0
    word_counts[word] += 1

tag_array = []

for tag in tag_counts.keys():
    tag_array.append(tag)

N = len(tag_array)
bigrams_all = numpy.zeros(shape=(N,N))

# [prev_tag][tag]


f = open('input_text.txt', 'w')
for line in open('F18-assgn4-test.txt', 'r'):
    token = line.strip('\n')
    token = line.strip().split('\t')
    if len(token) >= 2:
        f.write(token[1] + ' ')
    else:
        f.write('\n')
f.close()

word_array = []
for word in word_counts:
    word_array.append(word)

def find_maximum(token_index, tag_index):
    temp_array = []
    for tags2 in range(0, N):
        string = tag_array[tag_index] + "|" + tag_array[tags2]
        if string not in bigram_dict:
            num = 0
        else:
            num = (bigram_dict[string])
        value = num / (tag_counts[tag_array[tags2]] + bigrams_all[tags2].sum())
        temp_array.append(seq_scr_array[tags2, token_index - 1] * value)
    return max(temp_array)

f = open('input.txt')

for line in f:
    token_tag = line.strip('\n')
    token_tag = token_tag.lower()
    array = token_tag.split()
    if len(array) != 2:
        continue
    token = array[0]
    tag = array[1]
    bigram_tag = tag + "|" + prev_tag
    if token+"|"+tag not in word_tag_count.keys():
        word_tag_count[token+"|"+tag] = 0
    word_tag_count[token+"|"+tag] += 1
    if bigram_tag not in bigram_dict.keys():
        bigram_dict[bigram_tag] = 0
    bigram_dict[bigram_tag] += 1
    prev_tag = tag
f.close()

for bigram_tag in bigram_dict.keys():
    prev_tag = tag_array.index(bigram_tag.split("|")[1])
    tag = tag_array.index(bigram_tag.split("|")[0])
    #print (bigram_dict[bigram_tag])
    bigrams_all[prev_tag][tag] = bigram_dict[bigram_tag]

for x in range(N):
    bigrams_all[x] /= bigrams_all[x].sum()
#print ("Calculated bigram probabilities")

#print (N)

word_index = 0
seq_scr_array = numpy.array
final_tag_array = []
f = open('input_text.txt')
line_no = 1
for line in f:
    input_line = line.strip('\n').lower()
    array = input_line.split()
    word_index = 0
    T = len(array)
    seq_scr_array = numpy.zeros((N, T))
    lexical_counts = numpy.zeros((N, T))
    prev_tag = "o"
    token = array[0]
    #print(array)

    for word in array:
        for tag in tag_array:
            if word not in word_counts.keys():
                #print (word)
                lexical_counts[tag_array.index("o")][word_index] = 1
            else:
                if word+"|"+tag not in word_tag_count:
                    #print (word + "|" + tag)
                    lexical_counts[tag_array.index(tag)][word_index] = 0
                else:
                    #print(word+"|"+tag, word_tag_count[word + "|" + tag])
                    lexical_counts[tag_array.index(tag)][word_index] = word_tag_count[word + "|" + tag]
        word_index += 1




 #   for tag in tag_array:
 #       print (lexical_counts[tag_array.index(tag)].sum())

    # init step
    for i in range(0, N):
        #print (lexical_counts)
        tag = tag_array[i]
        lexical_string = token + "|" + tag
        bigram_string = tag + "|" + prev_tag

        #print(word, tag, word_index, lexical_string, bigram_string)
        #print(lexical_counts[i][word_index])
        #print(lexical_counts[i].sum())
        if lexical_counts[i].sum() == 0:
            lexical_value = k
        else:
            lexical_value = (lexical_counts[i][0] ) / (lexical_counts[i].sum() + tag_counts[tag_array[i]] + lexical_counts[i].sum() * k)



        bigram_value = bigrams_all[tag_array.index(prev_tag)][tag_array.index(tag)]
        #print("Lexical counts",lexical_counts)
        seq_scr_array[i, 0] = lexical_value * bigram_value
        #word_index += 1
        #print (seq_scr_array)
    max_tag_index = numpy.nanargmax(seq_scr_array[:, 0])
    final_tag = tag_array[max_tag_index]
    final_token_tag = [token, final_tag]
    final_tag_array.append(final_token_tag)
    # iterative step
    for t in range(1, T):
 #       print (array[t])
        #word_index = 0
        for tags in range(0, N):
            token = array[t]
            tag = tag_array[tags]
            max_seq_scr = find_maximum(t, tags)
            lex_string = token + "|" + tag
            if lexical_counts[tags].sum() == 0:
                lex_value = k
            else:
                #print (t, len(array))
                lex_value = (lexical_counts[tags][t] + k) / (lexical_counts[tags].sum() + tag_counts[tag_array[tags]] + lexical_counts[tags].sum() * k)
            seq_scr_array[tags, t] = max_seq_scr * lex_value
            #print(seq_scr_array[i, t])
        max_tag_index = numpy.nanargmax(seq_scr_array[:, t])
        final_tag = tag_array[max_tag_index]
        final_token_tag = [token, final_tag]
        final_tag_array.append(final_token_tag)
#    print (final_tag_array)
#print (line_no)
    line_no += 1


#print (final_tag_array)

f = open("akash-iyengar-assgn4-output.txt", "w")
i = 0
for line in open('F18-assgn4-test.txt', 'r'):
    token = line.strip('\n')
    token = line.strip().split('\t')
    if len(token) >= 2:
        f.write(token[0] + '\t' + token[1] + '\t' + final_tag_array[i][1].upper() + '\n')
        i += 1
    else:
        f.write('\n')


'''for item in final_tag_array:
    f.write(item[1].upper() + '\n')
    if item[0] == ".":
        f.write("\n");'''
