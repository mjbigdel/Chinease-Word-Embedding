# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:51:14 2019

@author: Manoochehr.Joodi
"""
import re
import os
import errno
import pickle
import numpy as np
import codecs
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import crf   
#from hanziconv import HanziConv

START_SENT = "گ"
END_SENT = "چ"
UNK = "ژ"
PAD = "پ"

TAGB,TAGI,TAGE,TAGS=0,1,2,3

rNUM = '(-|\+)?\d+((\.|·)\d+)?%?'
rENG = '[A-Za-z_.]+'

LEARNING_RATE = 0.001
NUM_CLASSES = 4
L2_REGU_LAMBDA=0.0001
NUM_LAYERS = 1
CLIP=5



# getting test data function with unigram, bigram and trigrams
def get_test_data(filename,ngrams2id, usebigram, usetrigram,usefourgram):   
    x=[]
    with codecs.open(filename,'r','utf-8') as f:
        # for every line we split and then tag every word with proper id.
        for line in f:   
            line_x = []
            # for every line we remove spaces and then assume every 5 (2 left char and 2 right char)
            # char as a new context, for every context we also calculate bigram and trigram and add them too
            line=re.sub(u'\s+','',line.strip())
            contexs=window(line)
            for contex in contexs:
                charx=[]
                #contex window
                charx.extend([ngrams2id.get(c,ngrams2id[UNK]) for c in contex])
                #bigram feature
                if usebigram:
                    charx.extend([ngrams2id.get(bigram, ngrams2id[UNK]) for bigram in ngram(contex,2)])
                if usetrigram:
                    charx.extend([ngrams2id.get(trigram, ngrams2id[UNK]) for trigram in ngram(contex,3)])
                if usefourgram:
                    charx.extend([ngrams2id.get(fourgram, ngrams2id[UNK]) for fourgram in ngram(contex,4)])
                line_x.append(charx)
            x.append(line_x)
    return x


# getting ngrams function
def ngram(ustr,n=2):
    ngram_list=[]
    for i in range(len(ustr)-n+1):
        ngram_list.append(ustr[i:i+n])
    return ngram_list


# window return sentence as list of 5 length context, for every word we are 
# gettnig 4 neighbour, 2 left and 2 right for having more training data.
def window(ustr,left=2,right=2):
    sent=''
    for i in range(left):
        sent+=START_SENT
    sent+=ustr
    for i in range(right):
        sent+=END_SENT
    windows=[]
    for i in range(len(ustr)):
        windows.append(sent[i:i+left+right+1])
    return windows

# use pretrained embeddings function, for chars we are using pretrain file and for bigram, trigram 
# and fourgram we are using mean of embeddings of all unigrams of them
def reading_pretrained_embeddings(filename, ngrams2id):
    # Reading Pretrained Embeddings from file
    pretrain_embeddigs = {}
    with codecs.open(filename, "r", "utf-8") as f:
        for line in f:
            pre_train = line.split()
            if len(pre_train) > 2:
                word = pre_train[0]
                if word in ngrams2id:
                    vec = pre_train[1:]
                    pretrain_embeddigs[word] = vec                
    
    #print("pretraine embedding files reading finished ...")
    # making embeddings for all ngrams2id.
    embedding_dim = len(next(iter(pretrain_embeddigs.values())))
    out_of_vocab = 0
    out = np.ones((len(ngrams2id), embedding_dim))
    for ngram in ngrams2id.keys():
        if len(ngram) == 1:
            if ngram in pretrain_embeddigs.keys():        
                out[ngrams2id[ngram]]=np.array(pretrain_embeddigs[ngram])
            else:
                out_of_vocab+=1
                np.random.uniform(-0.8, 0.8, embedding_dim)
    for ngram in ngrams2id.keys():        
        #embedding for bigrams
        if len(ngram) == 2:            
            out[ngrams2id[ngram]]= (out[ngrams2id[ngram[0]]]+out[ngrams2id[ngram[1]]])/2
        #embedding for trigrams
        if len(ngram) == 3:
            out[ngrams2id[ngram]]= (out[ngrams2id[ngram[0]]]+out[ngrams2id[ngram[1]]]+out[ngrams2id[ngram[2]]])/3
        #embedding for fourgrams
        if len(ngram) == 4:
            out[ngrams2id[ngram]]= (out[ngrams2id[ngram[0]]]+out[ngrams2id[ngram[1]]]+out[ngrams2id[ngram[2]]]+out[ngrams2id[ngram[3]]])/4               
    #print('out_of_vocab characters: %d' % (out_of_vocab) )         
    return out,out_of_vocab


# Retriev word_embeddings and PAD word id from Saved pkl File
def ngrams_pretrain_embed(resources_path):
    
    ngrams2id = pickle.load(open(resources_path+'/ngram2id_750K.pkl', "rb"))["ngrams2id"]    
    
    # get ngrams2ids's pretrained embeddings
    word_embeddings, out_of_vocab = reading_pretrained_embeddings(resources_path+'/character.vec', ngrams2id)
    PAD_ID = ngrams2id[PAD]
    return ngrams2id,word_embeddings,PAD_ID


# ===-----------------------------------------------------------------------===
# Trainig Section
# ===-----------------------------------------------------------------------===

def padding(X,padding_word):
	max_len = 0
	for x in X:
		if len(x) > max_len:
			max_len = len(x)
	padded_X = np.ones((len(X), max_len), dtype=np.int32) * padding_word
	for i in range(len(X)):
		for j in range(len(X[i])):
			padded_X[i, j] = X[i][j]
	return padded_X

def padding3(X,padding_word):
	max_len = 0
	for x in X:
		if len(x) > max_len:
			max_len = len(x)
	padded_X = np.ones((len(X), max_len,14), dtype=np.int32) * padding_word
	for i in range(len(X)):
		for j in range(len(X[i])):
			padded_X[i, j] = X[i][j]
	return padded_X

# function for calculating umber of hits and number of all predictions.
def number_of_batch_hits(y_batch_pred,y_batch_true):
    num_hits = 0
    num_all_chars = 0    
    for i in range(len(y_batch_true)):
     #   print("---------------------")
        for j in range(len(y_batch_true[i])):
            num_all_chars = num_all_chars+1
            if y_batch_pred[i][j] == y_batch_true[i][j]:
               num_hits = num_hits+1    
    return num_hits, num_all_chars

# ----------------- Add Summary Function ------------------------------------------
def add_summary(writer, name, value, global_step):
    summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
    writer.add_summary(summary, global_step=global_step)


# --------------------- Tensorflow part ---------------------------------------------------

def create_tensorflow_model(vocab_size, embedding_size, hidden_layer_dim, word_embeddings, PAD_ID):
    print("Creating TENSORFLOW model")
     
    # Inputs have (batch_size, timesteps) shape.
    X = tf.placeholder(dtype=tf.int32,shape=[None,None,14],name='input_x')   
    # Labels have (batch_size,) shape.
    labels = tf.placeholder(dtype=tf.int32,shape=[None,None],name='input_y')
    # dropout_keep_prob is a scalar.
    dropout_keep_prob=tf.placeholder(dtype=tf.float32,name='dropout_keep_prob')
    # Calculate sequence lengths to mask out the paddings later on.
    seq_length = tf.reduce_sum(tf.cast( tf.not_equal(X[:,:,2], tf.ones_like(X[:,:,2]) * PAD_ID), tf.int32), 1)
        
    with tf.variable_scope('embeddings', reuse=tf.AUTO_REUSE):
        embedding_matrix = tf.Variable(word_embeddings, dtype=tf.float32, name='embedding')
        #embedding_matrix = tf.get_variable("embeddings", shape=[vocab_size, embedding_size])
        embeddings = tf.nn.embedding_lookup(embedding_matrix, X)
        embeddings = tf.reshape(embeddings,[tf.size(X[:,1,1]),-1,14*embedding_size])
    
    # embeddings shape (batch size, sentence length with padding, 1400)
    embeddings=tf.nn.dropout(tf.cast(embeddings,tf.float32),dropout_keep_prob)
    
    with tf.variable_scope('rnn_cell', reuse=tf.AUTO_REUSE):
            print ('rnn_cell is lstm')
            def lstm_cell():
                return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(hidden_layer_dim), output_keep_prob=dropout_keep_prob)
            
            def gru_cell():
                return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(hidden_layer_dim), output_keep_prob=dropout_keep_prob)

            
            stacked_fw_lstm = tf.nn.rnn_cell.MultiRNNCell(
                [gru_cell() for _ in range(NUM_LAYERS)])
            
            stacked_bw_lstm = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell() for _ in range(NUM_LAYERS)])
        
    with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):                        
            (forward_output,backword_output),_=tf.nn.bidirectional_dynamic_rnn(
                cell_fw=stacked_fw_lstm,
                cell_bw=stacked_bw_lstm,
                inputs=embeddings,
                sequence_length=seq_length,
                dtype=tf.float32
            )
            outputBidirection=tf.concat([forward_output,backword_output],axis=2)
            print ('outputBidirection is ok')    
            
    print ('Loss Starts ....')
    with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
        output=layers.fully_connected(
            inputs=outputBidirection,
            num_outputs=NUM_CLASSES,
            activation_fn=None            
            )
        print ('output is ok ....')
        #crf
        log_likelihood, transition_params = crf.crf_log_likelihood(
            output, labels, seq_length)
        print ('crf_log_likelihood is ok ....')
        
        loss = tf.reduce_mean(-log_likelihood)
        print ('loss crf_log_likelihood is ok ....')
    
    print ('train-op Starts ....')
    with tf.variable_scope('train_op', reuse=tf.AUTO_REUSE):
        optimizer=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        print ('optimizer is ok ....')
    
        tvars=tf.trainable_variables()
        print ('tvars is ok ....')
    
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
        print ('l2_loss is ok ....')
        
        loss=loss+L2_REGU_LAMBDA*l2_loss
        print ('loss L2 is ok ....')
        
        grads,_=tf.clip_by_global_norm(tf.gradients(loss,tvars),CLIP)
        print ('grads is ok ....')
        
        train_op=optimizer.apply_gradients(zip(grads,tvars))
        print ('train_op apply_gradients is ok ....')
               
    return X, labels, output, train_op, dropout_keep_prob, loss, transition_params, seq_length


#function for creating BIES file from id's and saving it as utf8 file
def ids_to_tag_file(test_pred,output_file):
    with codecs.open(output_file, 'w','utf-8') as f:
        for i in range(len(test_pred)):
            tag_list = ''
            for j in range(len(test_pred[i])):
                if test_pred[i][j]==0:
                    tag_list +='B'
                elif test_pred[i][j]==1:
                    tag_list+='I'
                elif test_pred[i][j]==2:
                    tag_list+='E'
                elif test_pred[i][j]==3:
                    tag_list+='S'
            f.write(''.join(tag_list).strip())            
            f.write('\n') 
#    print("result is ready ...")
   
