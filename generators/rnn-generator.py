
'''

SMILES generator based on Chembl dataset.

Usage:

    for training
    python3 rnn-generator.py 

    for SMILES generstion 
    python3 rnn-genertor.py model.npy 

While training your SMILES are supposed to be in a file chemcl.smi, 
also please check how current vocabulary fits your needs.

The trained model for generation about 87% of valid chembl-like SMILES 
is available in the models folder.

'''

import sys
import numpy as np
import theano
import theano.tensor as T
import lasagne
import datetime

import numpy as np
import os
import os.path
import math
import json
import time

#if you need always the same string point, please uncomment the following line.
#lasagne.random.set_rng(np.random.RandomState(1))

#Chembl's SMILES vocabulary.
chars = sorted("$#%()+-./0123456789=@ABCFGHIKLMNOPRSTVXZ[\]abcdegilnoprstu^");
vocab_size = len(chars);

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

N_HIDDEN = 512
LEARNING_RATE = .01
GRAD_CLIP = 100
NUM_EPOCHS = 50
BATCH_SIZE = 32
MAX_LENGTH = 100

def gen(probs):

   yt = np.zeros((BATCH_SIZE, MAX_LENGTH, vocab_size), np.int8);
   mt = np.zeros((BATCH_SIZE, MAX_LENGTH), np.int8);
 
   yt[:, 0, char_to_ix["^"] ] =1;
   mt[:, 0] = 1;

   for y in range(1, MAX_LENGTH):
      n = probs(yt, mt);
      p = n[: , y, :];     

      mt[:, y] = 1;
      for batch in range(BATCH_SIZE):
         yt[batch, y, np.random.choice(np.arange(vocab_size), p= p[batch]) ] = 1.0;      
      
   for batch in range(BATCH_SIZE):  
      st = "";    
      for q in range(MAX_LENGTH):
         w = np.argmax(yt[batch, q]);
         if(w == char_to_ix["$"]):
            print(st);
            break;
         if(q > 0):
            st += ix_to_char[ w ];
    

def gen_data(data):

    batch_size = len(data);
    seq_length = len(data[0]);
    for line in data:
        if len(line) > seq_length:
            seq_length = len(line);

    x = np.zeros((batch_size, seq_length, vocab_size), np.int8);
    y = np.zeros((batch_size, seq_length, vocab_size), np.int8);
    m = np.zeros((batch_size, seq_length), np.int8);

    cnt = 0;
    for line in data:
        line = line.replace("\n","");
        line = "^" + line + "$";

        n = len(line) - 1;
        m[ cnt, : n ] = 1;

        for i in range(n):
           x[cnt, i, char_to_ix[ line[i]] ] = 1;
           y[cnt, i, char_to_ix[ line[i+1]] ] = 1;
        cnt += 1;
    return x, y, m;

def main():   

    print("Building network ...");

    X = T.tensor3('X', dtype='int8');
    M = T.matrix('M', dtype='int8');
    Y = T.tensor3('Y', dtype='int8');

    l_in = lasagne.layers.InputLayer(shape=(None, None, vocab_size), input_var=X);
    l_mask = lasagne.layers.InputLayer(shape=(None, None), input_var=M);
    batchsize, seqlength,_ = l_in.input_var.shape;

    l_forward_1 = lasagne.layers.LSTMLayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        mask_input = l_mask,
        nonlinearity=lasagne.nonlinearities.tanh,
        only_return_final = False)

    l_forward_2 = lasagne.layers.LSTMLayer(
        l_forward_1, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh,
        mask_input = l_mask,
        only_return_final = False)

    l_shp = lasagne.layers.ReshapeLayer(l_forward_2, (-1, N_HIDDEN));
    l_soft = lasagne.layers.DenseLayer(l_shp, num_units=vocab_size, W = lasagne.init.Normal(),
                                       nonlinearity=lasagne.nonlinearities.softmax)
    l_out = lasagne.layers.ReshapeLayer(l_soft, (batchsize, seqlength, vocab_size));

    network_output = lasagne.layers.get_output(l_out, deterministic = False)
    network = lasagne.layers.get_output(l_out, deterministic = True)

    cost = T.nnet.categorical_crossentropy(network_output, Y);
    cost = cost * M;
    cost = cost.mean();

    all_params = lasagne.layers.get_all_params(l_out,trainable=True)

    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)

    print("Compiling functions ...")
    train = theano.function([X, M, Y], cost, updates=updates, allow_input_downcast=True)
    compute_cost = theano.function([X, M, Y], cost, allow_input_downcast=True)

    probs = theano.function([X, M],network,allow_input_downcast=True)

    if len(sys.argv) == 2:
        params = np.load(sys.argv[1]);
        lasagne.layers.set_all_param_values(l_out, params);

        print("Generation ...");
        #nothing special here about BATCH_SIZE
     
        for step in range(BATCH_SIZE):
           gen(probs);
        sys.exit(0);

    #basic is training
    print("Training ...")

    smi = open("chembl.smi", "r");
    epoch = 0;

    avg_cost = 0.0;
    start = datetime.datetime.now();

    #No validation here.
    try:
        while True:
            lines = [];
            for i in range(BATCH_SIZE):
               line = smi.readline();

               #Some molecules in Chembl are big enough to be called small...
               if len(line) > 0:
                   if(len(line) > 400):
                       continue;

                   lines.append(line);
               else:
                   smi.seek(0, 0);
                   end = datetime.datetime.now();
                   elapsed = end - start;

                   print("Epoch ", epoch, " finished: ", avg_cost, elapsed.seconds);
                   sys.stdout.flush();

                   epoch += 1;
                   avg_cost = 0.0;
                   start = datetime.datetime.now();
                   break;

            if epoch > NUM_EPOCHS:
               break;

            x, y, m = gen_data(lines);
            avg_cost += train(x, m, y);

    except KeyboardInterrupt:
       pass;

    np.save("generator.npy", lasagne.layers.get_all_param_values(l_out))
    print("Finished successfully!");

if __name__ == '__main__':
    main()

