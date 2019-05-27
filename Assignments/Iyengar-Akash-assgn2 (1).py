

import re
import json
import os
import sys

data_set_file_name = "berp-POS-training.txt"
training_file_name = "train.txt"
unaltered_test_file_name = "gold-test.txt"
test_file_name = "assgn2-test-set.txt"
output_file_name1 = "out.txt"
output_file_name = "Iyengar-Akash-assgn2-output.txt"
unknown_symbol = "UNK"
prefixPattern = "(^[0-9])"
bigram_word_separator = "|"
my_hmm_name = "hmm"
start_state = "."


def readFile(filename,prefix_pattern):
    with open(filename) as f:
        content = f.readlines()
        #print('content size is:' + str(len(content)))
    if prefix_pattern:
        #print("File Lines is:" + str(len(content)))
        lines = []
        altered_line = ""
        for i in range(0, len(content)):
            #print("Content[i]: " + content[i])
            m = re.search(prefix_pattern,content[i])
            if m:
                #print("Group is:" + m.group())
                altered_line = altered_line + content[i].strip() +","
            else:
                lines.append(altered_line.strip(","))
                lines.append("\n")
                altered_line = ""
        return lines
    else:
        return content

def partitionData(content):
    
    import os.path
    if( (os.path.exists(training_file_name) and os.path.exists(test_file_name) and os.path.exists(unaltered_test_file_name)) ):
        test = readFile(test_file_name,"")
        train = readFile(training_file_name,"")
        print('training size... '+str(len(train)) +"..... test size ...."+str(len(test)))
        return train,test
    else:
            from sklearn.model_selection import train_test_split
            train, test = train_test_split(content, test_size=0.3)
            print('training size... ' + str(len(train)) + "..... test size ...." + str(len(test)))
            return_test = []
            return_train = []
            #print('writing training .... ' + train[0])
            f = open(training_file_name, 'w')
            for i in range(0,len(train)):
                if train[i].strip():
                    sentences = train[i].strip().split(',')
                    for j in range(0, len(sentences)):
                        f.write(sentences[j].strip()+'\n')
                    f.write('\n')
            #print('writing unaltered .... ' + test[0])
            f = open(unaltered_test_file_name, 'w')
            for i in range(0,len(test)):
                sentences = test[i].strip().split(',')
                if test[i].strip():
                    sentences = test[i].strip().split(',')
                    for j in range(0, len(sentences)):
                        f.write(sentences[j].strip() + '\n')
                    f.write('\n')
            #print('wrote unaltered test file')
            f = open(test_file_name, 'w')
            #print('writing test .... ' + test[0])
            for i in range(0, len(test)):
                sentences = test[i].strip().split(',')
                if test[i].strip():
                    sentences = test[i].strip().split(',')
                    for j in range(0,len(sentences)):
                        line = sentences[j]
                        parts = line.split('\t')
                        if(len(parts) == 3):
                            f.write(parts[0] + "\t" + parts[1] +'\n')
                    f.write('\n')
            return train,test
        
        
def computeBaseLine(train):
    
    #step 1: GET THE COUNTS FOR EVERY WORD AND TAG PAIR. STORE THAT IN WORD_DICT
    word_dict = {} # this is going to be used to build the training data dict. This will have all the counts for a given word and tag
    for i in range(0,len(train)):
        line = train[i]
        parts_of_line = line.split('\t') # 1. position in sentence, the word in consideration, the tag associated with it.

        if(len(parts_of_line) == 3):
            sentence_pos = parts_of_line[0].strip()
            word = parts_of_line[1].strip()
            tag = parts_of_line[2].strip()

            if(word not in word_dict.keys()):
                #print('inserting new word')
                new_dict_tag_count = {}
                new_dict_tag_count[str(tag)] =  str(1); # CREATE THE INNER DICT
                word_dict[str(word)] = new_dict_tag_count # ADD TO THE OUTER DICT

            else:
                # if the word already exists
                inner_dict_of_tags = word_dict[word] # get the inner dict of tags
                #print("---------- tag vs tag.keys() -->" + str(tag.strip()) +" vs " + str(inner_dict_of_tags.keys()))
                if(tag not in inner_dict_of_tags.keys()):
                    #print('creating new tag')# if tag doesnt exist
                    inner_dict_of_tags[tag] = str(1)
                else:
                    #print('altering exising tag ... exisitng count is:' +str(inner_dict_of_tags[tag]))
                    count = int(inner_dict_of_tags[tag]) # get the existing tag count
                    count = count +1
                    inner_dict_of_tags[tag] = str(count)
                    word_dict[str(word)] = inner_dict_of_tags

    #STEP2: FOR ALL THE WORDS IN THE WORD DICT, COMPUTE THE MAX AND ASSIGN THAT TO THE TRAINING_DICT
    training_dict = {} # from word dict get the max count of the tags and store here.
    for key in word_dict:
        if(len(word_dict[key]) > 1):
            inner_dict = word_dict[key]
            maxKey = ""
            maxValue = 0
            for iKey in inner_dict:
                if(int(inner_dict[iKey]) > maxValue):
                    maxKey = iKey
                    maxValue = int(inner_dict[iKey])
                max_tag = {}
                max_tag[maxKey] = str(maxValue)
                training_dict[key] = max_tag;

        else:
            training_dict[key] = word_dict[key]
    #print("Training dict!")
    #for key in training_dict.keys():
        #print("key: " + key + "  value:" + str(training_dict[key]))

    #STEP 3: Read the test data from the file.
    testData = readFile(test_file_name,"")
    #print('testDat size: ' + str(len(testData)))
    outputs = []
    print(list(training_dict.keys()))
    for test_line in testData:
        if test_line:
            test_parts = test_line.split("\t")
            if(len(test_parts) == 2):
                word = test_parts[1].strip()
                print('word is: '+word +'   ' + str(word in training_dict.keys()))
                if(word in training_dict.keys()):
                    inner_dict = training_dict[word]
                    tag = list(inner_dict.keys())[0].strip()
                    #print('ttag:' + str(tag))
                else:
                    tag = unknown_symbol.strip()
                output = str(test_parts[0]) + "\t" + str(word) + "\t" + str(tag) + "\n"
                print('tag is:' + str(tag))
                print('output is:' + output)
                outputs.append(output)
            else:
                outputs.append(test_line)
        else:
            outputs.append(test_line)

    f = open(output_file_name1, 'w')
    #print('outputsize:' + str(len(outputs)))
    for i in range(0,len(outputs)):
        f.write(outputs[i])
    f.close()



    

def computeUnigrams(training_data):
    unigrams = {}
    total_words = 0
    for i in range(0, len(training_data)):
        line = training_data[i]
        line_parts = line.split('\t')
        if len(line_parts) == 3: # ignores new lines
            if(line_parts[2].strip() in unigrams.keys()):
                unigrams[line_parts[2].strip()] = int(unigrams[line_parts[2].strip()]) +1
            else:
                unigrams[line_parts[2].strip()] = int(1)
            total_words = total_words +1
        else:
            #print('empty line!')
            pass
    return unigrams,total_words

def computeBigrams(unigrams_counts, training_data):
    bigrams = {}
    unigrams_keys = list(unigrams_counts.keys())

    ## Creating the bigram table. Filling it full of zeros.
    for i in range(0,len(unigrams_keys)):
        for j in range(0,len(unigrams_keys)):
            bigram_key = unigrams_keys[i] + bigram_word_separator + unigrams_keys[j]
            bigrams[bigram_key] = 0;
    # print('Size of bigram table: ' + str(len(bigrams.keys()))) #1296
    ## Filling in the values of bigrams now.

    '''
        Formula : P(wi | wi-1 ) = P(wi-1,wi)/P(wi-1)
    '''
    # Step 1: Lets get all the bigram counts
    sentence = []
    for i in range(0,len(training_data)):
        m = re.match(prefixPattern, training_data[i]) # if it starts with a number then its part of the sentence.
        if m:
            sentence.append(training_data[i])
        else:
            # we encountered new line
            # now an entire sentence is read. Compute the Bigram count for it.
            pos_tags = []
            pos_tags.append(".") # Start of sentence marker also is "."
            for j in range(0, len(sentence)):
                parts = sentence[j].split("\t")
                pos_tags.append(parts[2].strip())
            for j in range(0,len(pos_tags)-1):
                #key = pos_tags[j] + bigram_word_separator + pos_tags[j+1]
                key = pos_tags[j+1] + bigram_word_separator + pos_tags[j]
                bigrams[key] = bigrams[key] + 1
            sentence.clear()
            #key = pos_tags[len(pos_tags) -1] + bigram_word_separator + "."
            #bigrams[key] = bigrams[key] + 1 # This isnt needed.
    bigram_counts = dict(bigrams)
    #print(bigrams)

    # Step 2: Divide by the unigram count of the first half of the key
    for key in bigrams.keys():
        #print(' for bigram: ' + key +" ..... " + key.split(bigram_word_separator)[1])
        #if bigrams[key] > 0: print("key is: " + key + " Count is: " + str(bigrams[key]))
        wiminus1 = key.split(bigram_word_separator)[1]
        unigram_count = unigrams_counts[wiminus1.strip()] # nothing but the unigram count of the wi-1th word
        #if bigrams[key] > 0: print("unigram key is:" + wiminus1+" Count is: " + str(unigram_prob))
        if unigram_count:
            #if bigrams[key] > 0: print("Old bigram:"+ str(bigrams[key]))
            bigrams[key] = float(bigrams[key])/float(unigram_count)
            #if bigrams[key] > 0: print("New bigram:" + str(bigrams[key]))
        else:
            print('error!!')
    return bigram_counts,bigrams

def bigram_verification(unigrams, bigrams, flag):
    if flag:
        bigram_verification = {}
        for key in bigrams.keys():
            minus1 = key.split(bigram_word_separator)[1]
            minus1 = minus1.strip()
            if (minus1 in unigrams.keys()):
                # print("key .. "+key +" ... prob:.." + str(bigrams[key]))
                if (minus1 in bigram_verification.keys()):
                    bigram_verification[minus1] = bigram_verification[minus1] + bigrams[key]
                else:
                    bigram_verification[minus1] = bigrams[key]
        for key in bigram_verification.keys():
            print("w(i-1)  is: " + key + ".. bigram count is: " + str(bigram_verification[key]))

def computeEmissionProbability(training_data, unigram_counts):
    emission_probability_counts = {}
    emission_probability = {}
    # step 1: get all the counts
    for line in training_data:
        parts = line.split('\t')
        if(len(parts) == 3): # ignore new lines.
            word = (parts[1].strip())
            tag = (parts[2].strip())
            key = word + bigram_word_separator +tag
            if key in emission_probability_counts.keys():
                emission_probability_counts[key] = emission_probability_counts[key] + 1
            else:
                emission_probability_counts[key] = 1
        else:
            pass
    for key in emission_probability_counts.keys():
        word = key.split(bigram_word_separator)[0]
        tag = key.split(bigram_word_separator)[1]
        if tag in unigram_counts.keys():
            emission_probability[key] = emission_probability_counts[key]/unigram_counts[tag]
        else:
            print('something drastic just happened.')
    return emission_probability_counts,emission_probability

def unigram_verification(unigrams, total_words, flag):
    unigram_prob = 0
    unigram_prob_dict = {}
    for key in unigrams:
        unigram_prob = unigram_prob + (unigrams[key]/total_words)
        unigram_prob_dict[key] = unigram_prob

    if flag:
        print("Unigram Prob sum: " + str(unigram_prob))
        for key in unigrams.keys():
            print("Unigram key is:" + key + " ...... count is...... " + str(unigrams[key]))
    return unigram_prob_dict

def emission_probability_verification(emission_probability,flag):
    if flag:
        emission_verification = {}
        for key in emission_probability.keys():
            #print("Emission key is:" + key +" .... "+str(emission_probability[key]))
            tag = key.split(bigram_word_separator)[1]
            word = key.split(bigram_word_separator)[0]
            if tag in emission_verification.keys():
                emission_verification[tag] = emission_verification[tag] + emission_probability[key]
            else:
                emission_verification[tag] = emission_probability[key]
        for key in emission_verification.keys():
            print("Emission key is:" + key + " .... " + str(emission_verification[key]))

def computeWordCount(training_data):
    word_count_dict = {}
    for line in training_data:
        parts = line.split('\t')
        if len(parts) == 3:
            tag = (parts[2]).strip()
            word = (parts[1]).strip()
            if word in word_count_dict.keys():
                word_count_dict[word] = word_count_dict[word] + 1
            else:
                word_count_dict[word] = 1
    return word_count_dict

def computeUnkFromTest(word_count_dict,test):
    unknown_word_list = []
    unknown_index_dict = {}
    for i in range(0,len(test)):
        line = test[i]
        parts = line.split('\t')
        if(len(parts) == 2):
            #print('yabadaadooo')
            word = (parts[1]).strip()
            if word not in word_count_dict.keys():
                unknown_word_list.append(word)
                #print('replacing .. ' + test[i].strip())
                test[i] = parts[0] +'\t' + unknown_symbol +'\n'
                unknown_index_dict[i] = parts[1].strip()
                #print('by .. ' + test[i].strip())
    unknown_word_list = list(set(unknown_word_list))
    return unknown_word_list,test,unknown_index_dict

def computeUnkFromTraining(word_count_dict,training_data):
    unknown_word_list = []
    word_count_dict[unknown_symbol] = 0
    for i in range(0,len(training_data)):
        line = training_data[i]
        parts = line.split('\t')
        if len(parts) == 3:
            word = (parts[1]).strip()
            tag = (parts[2]).strip()
            if(word in word_count_dict.keys()):
                if(word_count_dict[word] < 1 or word_count_dict[word] == 1):
                    word_count_dict[unknown_symbol] = word_count_dict[unknown_symbol] + 1
                    popped_count = word_count_dict.pop(word)
                    #print('poped word: '+word +" ... count: "+ str(popped_count))
                    unknown_word_list.append(word)
                    training_data[i] = parts[0] +'\t' + unknown_symbol + '\t' + tag
            else:
                print('something is very wrong here ... ' + word)
    #for i in range(0,len(unknown_word_list)):
        #print('unknown word .. ' + unknown_word_list[i])
    return unknown_word_list,training_data

def unknown_word_verification(unknown_word_list,training,flag):
    if flag:
        for i in range(0,len(unknown_word_list)):
            print("unk: " + str(unknown_word_list[i]))
        print(str(len(unknown_word_list)))
        training_unk_count = 0
        for i in range(0,len(training)):
            m = re.search(unknown_symbol,training[i])
            if m:
                print(training[i])
                training_unk_count = training_unk_count +1
        print(training_unk_count)



def addKBigramSmoothing(k, bigram_counts, unigrams_counts):
    smoothed_bigram_counts = {}
    smoothed_bigram_probability = {}
    vocab_size = k*len(bigram_counts.keys())
    for bigram_key in bigram_counts.keys():
        smoothed_bigram_counts[bigram_key] = bigram_counts[bigram_key] + k

    for bigram_key in smoothed_bigram_counts.keys():
        wiminus1 = bigram_key.split(bigram_word_separator)[1]
        smoothed_bigram_probability[bigram_key] = smoothed_bigram_counts[bigram_key]/ (unigrams_counts[wiminus1] + vocab_size)
    return smoothed_bigram_counts,smoothed_bigram_probability

def createModelFile(A, B, unigram_prob, word_count_dict, unknown_word_list):
    outer_dict = {}
    hmm_dict = {}
    # Step 1 : Build A matrix tagi|tagi-1
    A_dict = {}
    for key in A.keys():
        tagiminus1 = key.split(bigram_word_separator)[1]
        tagi = key.split(bigram_word_separator)[0]
        if(tagiminus1 in A_dict.keys()):
            inner_dict = A_dict[tagiminus1]
            inner_dict[tagi] = A[key]
            A_dict[tagiminus1] = inner_dict
        else:
            inner_dict = {}
            inner_dict[tagi] = A[key]
            A_dict[tagiminus1]  = inner_dict

    hmm_dict['A'] = A_dict # Verified using vim that sum for each state adds to 1, for bigram_prob. Dont know for smoothed one

    B_dict = {}
    for key in B.keys():
        tag = key.split(bigram_word_separator)[1]
        word = key.split(bigram_word_separator)[0]
        if(tag in B_dict.keys()):
            inner_dict = B_dict[tag]
            inner_dict[word] = B[key]
            B_dict[tag] = inner_dict
        else:
            inner_dict = {}
            inner_dict[word] = B[key]
            B_dict[tag]  = inner_dict

    exhaustive_list_of_words = set(word_count_dict.keys())
    exhaustive_list_of_words.add(unknown_symbol)

    for key in B_dict.keys():
        inner_dict = dict(B_dict[key]) # key is a tag, inner_dict is word1:prob1,word2:prob2 ...
        #print(inner_dict)
        exisiting_words = set(inner_dict.keys())
        #print(exisiting_words)
        omitted_words = list(exhaustive_list_of_words - exisiting_words)
        for i in range(0,len(omitted_words)):
            inner_dict[omitted_words[i]] = float(0)
        B_dict[key] = inner_dict


    hmm_dict['B'] = B_dict
    #print(B_dict)
  
    # Step 3 : In the A matrix, where ever you have a <TAG>|<START>, i.e. <TAG>|. in our case.. so pi[TAG] = P(<TAG>|.)
    pi_dict = A_dict[start_state]
    hmm_dict['pi'] = pi_dict
    outer_dict[my_hmm_name] = hmm_dict
    with open('model.json', 'w') as f:
        output = json.dumps(outer_dict, sort_keys=True,indent=4,separators=(',', ': '))
        f.write(output)
    f.close()

class ViterbiHMM(object): # base class for different HMM models
    def __init__(self, model_name):
        # model is (A, B, pi) where A = Transition probs, B = Emission Probs, pi = initial start state probability
        if model_name == None:
            print ("Please provide the model name. Exiting program")
            sys.exit()
        self.model = json.loads(open(model_name).read())["hmm"]
        self.A = self.model["A"] # transition matrix between the hidden states.
        self.states = list(self.A.keys()) # get the list of states
        self.N = len(self.states) # number of states of the model
        self.B = self.model["B"] # emission matrix
        sym = []
        for key in self.B:
            sym.append(key)
        self.symbols = sym
        self.M = len(self.symbols) # number of states of the model
        self.pi = self.model["pi"]
        return

    def viterbi_algorithm(self, observation_seq):
        vit = [{}]
        path = {}
        # Before running the recursive algirithm, we need to define our base cases.
        for state in self.states:
            pi_y = self.pi[state]
            pi_y = self.B[state][observation_seq[0]]
            vit[0][state] = self.pi[state] * self.B[state][observation_seq[0]]
            path[state] = [state]

        # Recursively fill in the table
        for t in range(1, len(observation_seq)):
            vit.append({})
            newpath = {}
            for y in self.states:
                (prob, state) = max((vit[t - 1][y0] * self.A[y0][y] * self.B[y][observation_seq[t]], y0) for y0 in self.states)
                vit[t][y] = prob
                newpath[y] = path[state] + [y]
                # Don't need to remember the old paths
            path = newpath
        n = 0  # if only one element is observed max is sought in the initialization values
        if len(observation_seq) != 1:
            n = t
        (prob, state) = max((vit[n][y], y) for y in self.states)
        return (prob, path[state])







if __name__ == '__main__':
    # step1 : read in the data file
    content = readFile(data_set_file_name,prefixPattern)

    # step2: partition the data file into test and training.
    # 70% training data, 30% test data.
    train,test = partitionData(content);
    computeBaseLine(train)
    #print(test)
    # step3: unigram
    unigrams_counts, total_words = computeUnigrams(train);
    unigram_prob = unigram_verification(unigrams_counts, total_words, False)

    # step 4: bigrams
    bigram_counts,bigram_prob = computeBigrams(unigrams_counts, train);
    # step 5: Bigram Verification
    bigram_verification(unigrams_counts, bigram_prob, False)

    # step 6: Handle Unknown words
    word_count_dict = computeWordCount(train)
    #print(word_count_dict)

    # step a : For all the words in the word_count_dict, where the count is = 1 rename them as unknown.
    # step b : For all words in the test data, if it does not appear in the training data, rename the word to UNK in the test
    unknown_word_list, training_with_unk = computeUnkFromTraining(word_count_dict,train)
    unknown_word_list, test, unknown_index_dict = computeUnkFromTest(word_count_dict, test)
    unknown_word_verification(unknown_word_list,training_with_unk,False)


    # step 7: Emission Probability
    emission_probability_counts,emission_probability = computeEmissionProbability(training_with_unk, unigrams_counts)
    emission_probability_verification(emission_probability,False)

    # step 8: Smoothing Bigrams
    #tag_tag_vocab_size = len(list(bigram_prob.keys()))
    smoothed_bigram_counts,smoothed_bigram_probability = addKBigramSmoothing(0.7, bigram_counts, unigrams_counts)
    #for key in bigram_prob.keys():
    #    print(key + " ..." + str(bigram_prob[key]) +"..." + str(smoothed_bigram_probability[key]) )

    # step 9: Create the model file for the HMM
    #createModelFile(bigram_prob,emission_probability,unigram_prob)
    createModelFile(smoothed_bigram_probability, emission_probability, unigram_prob, word_count_dict, unknown_word_list)

    # step 10: Call Viterbi
    models_dir = "."
    model_file = "model.json"
    hmm = ViterbiHMM(os.path.join(models_dir, model_file))
    print("Using the model from file: ", model_file)
    total1 = total2 = 0  # to keep track of total probability of distribution which should sum to 1
    #observations = ['well','what','are','omelets','anyways','uh','i','guess','it','could','be','french','or','california','or','just','plain','old','american','.']

    observations = []
    output_lines = []
    sub_outputs = []
    for i in range(0,len(test)):
        line = test[i]
        parts = line.split('\t')
        if len(parts) == 2:
            observations.append(parts[1].strip())
            sub_outputs.append(line)
            #print(observations)
        else:
            #print(observations)
            prob, hidden_states =  hmm.viterbi_algorithm(observations)
            #print ("Max Probability = ", prob, " Hidden State Sequence = ", hidden_states)
            #print("Hidden states size" + str(len(hidden_states)) +" Suboutputs size: " + str(len(sub_outputs)))
            for j in range(0,len(hidden_states)):
                sub_outputs[j] = sub_outputs[j].strip() + '\t' + hidden_states[j] + '\n'
            output_lines.extend(sub_outputs)
            output_lines.append('\n')
            sub_outputs = []
            observations = []



    f = open(output_file_name,'w')
    for i in range(0,len(output_lines)):
        #print( str(type(output_lines[i])))
        #print(output_lines[i])
        output = output_lines[i]
        if(i not in unknown_index_dict.keys()):
            f.write(output)
        else:
            parts = output.split('\t')
            print(parts)
            if(len(parts) == 3):
                unk_replaced_output = parts[0] + str("\t") + unknown_index_dict[i] + str('\t') + parts[2].strip() + '\n'
                print(unk_replaced_output)
                f.write(unk_replaced_output)
            else:
                f.write(output)
    f.close()











