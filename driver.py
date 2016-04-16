import csv
import itertools
import nltk
import numpy as np
import time
import RNNmodel
import os
import sys
from datetime import datetime

_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '500'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '500'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
wordEngStop = nltk.corpus.stopwords.words('english')
punctions = [',', '.', '!', ':', '@', ';', '#']
stopwords = punctions+wordEngStop

def train_sgd(model, X_train, y_train, X_test, y_test,learning_rate=0.005, nepoch=1000, evaluate_loss_after=1):
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            train_acc = float(model.calculate_acc(X_train,y_train))/float(len(y_train))*100
            test_acc = float(model.calculate_acc(X_test,y_test))/float(len(y_test))*100
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            # print ("epoch:%d Loss: %4.2f Train_Acc: %4.2f Test_Acc: %4.2f"%(epoch,loss,train_acc,test_acc))
            print("%s: Loss after num_examples_seen=%d epoch=%d: %4.2f Train_Acc: %4.2f Test_Acc: %4.2f" % (time, num_examples_seen, epoch, loss, train_acc,test_acc))
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            # model.print_pre(X_train[i],y_train[i])
            if len(X_train[i]) > 0:
                model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1


vocabulary_size = _VOCABULARY_SIZE
unknown_token = "UNKNOWN_TOKEN"

print("Reading CSV file...")
with open('data/train_data.csv', 'r') as f:
# with open('data/training.1600000.processed.noemoticon.csv',d 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # Split full comments into sentences
    sentences = itertools.chain(unicode(x[5], errors='ignore').lower() for x in reader)
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s" % (x) for x in sentences]
with open('data/train_data.csv', 'r') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # Split full comments into sentences
    values = itertools.chain((y[0]) for y in reader)
    # Append SENTENCE_START and SENTENCE_END
    values = ["%s" % (y) for y in values]
print("Parsed %d sentences." % (len(sentences)))
# tokenized_sentences = [[w for w in nltk.word_tokenize(sent) if w not in stopwords] for sent in sentences]
tokenized_sentences = [[w for w in nltk.word_tokenize(sent)] for sent in sentences]
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Found %d unique words tokens." % len(word_freq.items()))
# print("values",values)

vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

print("Using vocabulary size %d." % vocabulary_size)
print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

X_train = np.asarray([[word_to_index[w] for w in sent] for sent in tokenized_sentences])
Y_train = np.asarray([[int(s)] for s in values], dtype=np.int32)

with open('data/test_data.csv', 'r') as f:
# with open('data/training.1600000.processed.noemoticon.csv',d 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # Split full comments into sentences
    sentences_test = itertools.chain(unicode(x[5], errors='ignore').lower() for x in reader)
    # Append SENTENCE_START and SENTENCE_END
    sentences_test = ["%s" % (x) for x in sentences_test]
with open('data/test_data.csv', 'r') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # Split full comments into sentences
    values_test = itertools.chain((y[0]) for y in reader)
    # Append SENTENCE_START and SENTENCE_END
    values_test = ["%s" % (y) for y in values_test]
print("Parsed %d sentences." % (len(sentences_test)))
# tokenized_sentences = [[w for w in nltk.word_tokenize(sent) if w not in stopwords] for sent in sentences]
tokenized_sentences_test = [[w for w in nltk.word_tokenize(sent)] for sent in sentences_test]
word_to_index_test = dict([(w, i) for i, w in enumerate(index_to_word)])

for i, sent in enumerate(tokenized_sentences_test):
    tokenized_sentences_test[i] = [w if w in word_to_index_test else unknown_token for w in sent]

X_test = np.asarray([[word_to_index_test[w] for w in sent] for sent in tokenized_sentences_test])
Y_test = np.asarray([[int(s)] for s in values_test], dtype=np.int32)

np.random.seed(0)
model = RNNmodel.RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)
# t1 = time.time()
# model.sgd_step(X_train[10], Y_train[10], _LEARNING_RATE)
# t2 = time.time()
# print("SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.))
# RNNmodel.gradient_check_theano(model,[0,1,2,3], [1])
train_sgd(model, X_train, Y_train, X_test, Y_test, nepoch=1000, learning_rate=0.005)