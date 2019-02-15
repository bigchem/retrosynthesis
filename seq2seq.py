
import sys
import os
import pickle
import datetime
import h5py
import os
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.optimizers import Adam

from rdkit import Chem

N_HIDDEN = 256;
EMBEDDING_SIZE = 64;
BATCH_SIZE = 32;
NUM_EPOCHS = 20;
TOPK = 1;
max_predict = 160;

seed = 0;
tf.random.set_random_seed(seed);
np.random.seed(seed);

np.set_printoptions(threshold=np.nan);

chars = "^#()+-./123456789=@BCFHIKLMNOPSZ[\\]cdegilnorstu$";
vocab_size = len(chars)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

class suppress_stderr(object):
   def __init__(self):
       self.null_fds =  [os.open(os.devnull,os.O_RDWR)]
       self.save_fds = [os.dup(2)]

   def __enter__(self):
       os.dup2(self.null_fds[0],2)

   def __exit__(self, *_):
       os.dup2(self.save_fds[0],2)

       for fd in self.null_fds + self.save_fds:
          os.close(fd)


class AttentionLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs);

    def build(self, input_shape):
        self.Wa = self.add_weight( shape =(N_HIDDEN, N_HIDDEN),
                                name="Wa", trainable = True,
                                initializer = 'glorot_uniform');
        super(AttentionLayer, self).build(input_shape);

    def call(self, inputs):

        encoder = inputs[0];
        Wa = tf.tensordot(inputs[0], self.Wa, axes = [[2], [0]]);
        decoder = tf.transpose(inputs[1], (0,2,1));

        A = K.batch_dot(Wa, decoder);
        A = tf.exp(A);

        mask = tf.reshape(inputs[2], (tf.shape(inputs[2])[0], tf.shape(inputs[2])[1], 1));
        A = A * mask;

        A = tf.transpose(A, (0,2,1));
        A = A / tf.reshape( tf.reduce_sum(A, axis = 2, keepdims = True), (-1, tf.shape(inputs[1])[1] ,1));

        C = K.batch_dot(A, encoder);
        C = tf.concat([C, inputs[1]], axis= 2);

        return C;

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[1][1], 2*input_shape[1][2]);

def generate2(product, res, mdl):

   lines = [];
   for r in res:
      lines.append(product + " >> " + r);

   v = gen_data(lines);
   i = len(res[0]);

   n = mdl.predict(v[0]);
   p = n[:, i, :];

   p = np.exp(p) / np.sum(np.exp(p), axis = 1).reshape((len(lines), -1));
   return p;


def gen_data(lines):

   batch_size = len(lines);
   products = [];
   reactants = [];

   for line in lines:
      data = line.strip().split(">>");
      products.append(data[0].strip());
      reactants.append("^" + data[1].strip() + "$");

   seq_length_product = len(products[0]);
   seq_length_reactants = len(reactants[0]) - 1;

   for i in range(1, len(lines)):
      nl_product = len(products[i]);
      nl_reactants = len(reactants[i]) - 1;

      if nl_product > seq_length_product:
         seq_length_product = nl_product;
      if nl_reactants > seq_length_reactants:
         seq_length_reactants = nl_reactants;

   #straightforward and reverse
   xs = np.zeros((batch_size, seq_length_product), np.int8);
   rs = np.zeros((batch_size, seq_length_product), np.int8);

   ys = np.zeros((batch_size, seq_length_reactants), np.int8);
   yt = np.zeros((batch_size, seq_length_reactants, vocab_size), np.float32);

   ms = np.zeros((batch_size, seq_length_product), np.int8);
   mt = np.zeros((batch_size, seq_length_reactants), np.int8);

   for batch in range(batch_size):
      n = len(products[batch]);
      for i in range(n):
         xs[batch, i] = char_to_ix[ products[batch][i]];
         rs[batch, i] = char_to_ix[ products[batch][n-i-1]];
      ms [batch, :n] = 1;

      n = len(reactants[batch]) - 1;
      for i in range(n):
         ys[batch, i] =char_to_ix[ reactants[batch][i]];
         yt[batch, i, char_to_ix[ reactants[batch][i+1]]] = 1;

      mt [batch, :n] = 1;

   return [xs, rs, ms, ys, mt], yt;

def data_generator(fname):

   f = open(fname, "r");
   lines = [];

   while True:
      for i in range(BATCH_SIZE):
         line = f.readline();
         if len(line) == 0:
            f.seek(0,0);
            if len(lines) > 0:
               yield gen_data(lines);
               lines = [];
               break;
         lines.append(line);
      if len(lines) > 0:
          yield gen_data(lines);
          lines = [];


def gen(mdl, product):

   res = "";
   for i in range(1, 70):
      lines = [];
      lines.append( product + " >> " + res);

      v = gen_data(lines);

      n = mdl.predict(v[0]);
      p = n[0 , i-1, :];

      w = np.argmax(p);
      if w == char_to_ix["$"]:
         break;
      res += ix_to_char[w];

   return res;

def gen_beam(mdl, product):

   if "greedy" != "greedy":
      answer = [];

      reags = gen(mdl, product).split(".");
      sms = set();

      with suppress_stderr():
         for r in reags:
            r = r.replace("$", "");
            m = Chem.MolFromSmiles(r);
            if m is not None:
               sms.add(Chem.MolToSmiles(m));
         if len(sms):
            answer.append(sorted(list(sms)));
      return answer;

   beam_size = 5;

   lines = [];
   scores = [];
   final_beams = [];

   for i in range(beam_size):
       lines.append("");
       scores.append(0.0);

   for step in range(max_predict):

      if step == 0:
         p = generate2(product, [""], mdl);
         nr = np.zeros((vocab_size, 2));
         for i in range(vocab_size):
            nr [i ,0 ] = -math.log10(p[0][i]);
            nr [i ,1 ] = i;
      else:

         cb = len(lines);
         nr = np.zeros(( cb * vocab_size, 2));

         p = generate2(product, lines, mdl);
         for i in range(cb):
            for j in range(vocab_size):
               nr[ i* vocab_size + j, 0] = -math.log10(p[i,j]) + scores[i];
               nr[ i* vocab_size + j, 1] = i * 100 + j;

      y = nr [ nr[:, 0].argsort()] ;
      #print(y);
      #print("~~~~~~");

      new_beams = [];
      new_scores = [];

      for i in range(beam_size):

         c = ix_to_char[ y[i, 1] % 100 ];
         beamno = int( y[i, 1] ) // 100;

         if c == '$':
             added = lines[beamno] + c;
             if added != "$":
                final_beams.append( [ lines[beamno] + c, y[i,0] ]);
             beam_size -= 1;
         else:
             new_beams.append( lines[beamno] + c );
             new_scores.append( y[i, 0]);

      lines = new_beams;
      scores = new_scores;

      if len(lines) == 0: break;

   for i in range(len(final_beams)):
      final_beams[i][1] = final_beams[i][1] / len(final_beams[i][0]);

   #print("Raw beams");
   #print(final_beams);

   final_beams = list(sorted(final_beams, key=lambda x:x[1]))[:TOPK];
   answer = [];

   #print("final beans:");
   #print(final_beams);

   for k in range(TOPK):
      reags = set(final_beams[k][0].split("."));
      sms = set();

      with suppress_stderr():
         for r in reags:
            r = r.replace("$", "");
            m = Chem.MolFromSmiles(r);
            if m is not None:
               sms.add(Chem.MolToSmiles(m));
            #print(sms);
         if len(sms):
            answer.append(sorted(list(sms)));

   #print(answer);
   return answer;


def validate(mdl):

   ftest = "test.smi";

   NTEST = 10; #sum(1 for line in open(ftest,"r"));
   fv = open(ftest, "r");

   cnt = 0;
   ex = 0;
   longest = 0;
   partly = 0.0;
   partly_cor = 0.0;

   for step in tqdm(range( NTEST )):
      line = fv.readline();
      if len(line) == 0: break;

      reaction = line.split(">");
      product = reaction[0].strip();
      reagents = reaction[2];

      answer = [];
      reags = set(reagents.split("."));

      sms = set();

      with suppress_stderr():
         for r in reags:
            m = Chem.MolFromSmiles(r);
            if m is not None:
               sms.add(Chem.MolToSmiles(m));
         if len(sms):
            answer = sorted(list(sms));

      if len(answer) == 0: continue;

      cnt += 1;
      beams = [];
      #try:
      beams = gen_beam(mdl, product);
      #except:
      #    pass;

      if len (beams) == 0:
         continue;

      answer_s = set(answer);

      for beam in beams:
         beam = set(beam);
         right = answer_s.intersection(beam);
         if len(right) == 0: continue;
         if len(right) == len(answer):
             ex += 1;
             longest += 1;
             partly += 1;
             partly_cor += 1;

             print("CNT: ", cnt, ex /cnt *100.0, answer);
             break;

         substrate1 = max( list(right) , key=len);
         substrate2 = max( answer, key=len);

         partly_l = len(right) * 1.0 / len(answer);
         partly += partly_l;

         if substrate1 == substrate2:
            longest += 1;
            partly_cor += partly_l;


   fv.close();

   print("Total reactions: ", cnt);
   print("Partly: ", partly/cnt * 100.0);
   print("Longest: ", longest / cnt * 100.0);
   print("Partly longest:", partly_cor / cnt * 100.0);
   print("Exact: ", ex / cnt * 100.0);

   return ex / cnt * 100.0;

def main():

    print("Building network ...");

    l_in = layers.Input( shape= (None,));
    batch_size = tf.shape(l_in)[0];

    l_mask = layers.Input( shape= (None, ));
    l_rev = layers.Input( shape= (None,));

    l_embed = layers.Embedding(input_dim = vocab_size, output_dim = EMBEDDING_SIZE, input_length = None)(l_in);
    l_embed_r = layers.Embedding(input_dim = vocab_size, output_dim = EMBEDDING_SIZE, input_length = None)(l_rev);

    l_forward_1 = layers.GRU( N_HIDDEN, activation='tanh', return_sequences=True)(l_embed);
    l_forward_2 = layers.GRU( N_HIDDEN, activation='tanh', return_sequences=True)(l_embed_r);

    l_concat = layers.Add()([l_forward_1, l_forward_2]);
    l_concat_d = layers.Dropout(rate=0.1)(l_concat);

    #find the last hidden_state with the mask
    l_ind = layers.Lambda(lambda x: K.sum(x, axis=1, keepdims = True) -1)(l_mask);
    l_arange = tf.reshape(K.arange(0, batch_size, dtype='int32'), (batch_size, 1));
    l_ind = K.concatenate([l_arange, K.cast(l_ind, 'int32')], axis = 1);
    l_last = layers.Lambda(lambda x: tf.gather_nd(x, l_ind), output_shape=(batch_size, N_HIDDEN)) (l_concat_d);

    l_state = layers.Dropout(rate=0.1)(l_last);

    #decoder
    l_in_d = layers.Input( shape= (None, ));
    l_embed_d = layers.Embedding(input_dim = vocab_size, output_dim = EMBEDDING_SIZE, input_length = None)(l_in_d);
    l_mask_d = layers.Input( shape= (None, ));

    l_forward_d1 = layers.GRU( N_HIDDEN, activation='tanh', return_sequences=True) (l_embed_d, initial_state=[l_state]);
    l_forward_d2 = layers.GRU( N_HIDDEN, activation='tanh', return_sequences=True)(l_forward_d1);
    l_forward_d2_d = layers.Dropout(rate=0.1)(l_forward_d2);

    l_att = AttentionLayer()([l_concat_d, l_forward_d2_d, l_mask]);
    l_att_d = layers.Dropout(rate=0.1)(l_att);

    l_out = layers.TimeDistributed(layers.Dense(vocab_size)) (l_att);

    def masked_loss(y_true, y_pred):
       loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred);
       mask = tf.cast(tf.not_equal(tf.reduce_sum(y_true, -1), 0), 'float32');
       loss = tf.reduce_sum(loss * l_mask_d, -1) / tf.reduce_sum(l_mask_d, -1);
       loss = K.mean(loss);
       return loss;

    mdl = tf.keras.Model([l_in, l_rev, l_mask, l_in_d, l_mask_d], l_out);
    mdl.summary();

    adam = Adam(1e-4, clipvalue = 5);

    mdl.compile(optimizer = adam, loss = masked_loss);
    tf.keras.utils.plot_model(mdl, to_file= "lstm.png");
    print("Training ...");

    NTRAIN = sum(1 for line in open('train.smi'));
    NVALID = sum(1 for line in open('valid.smi'));
    fpath = "./";

    print("Number of points: ", NTRAIN, NVALID);

    #mdl.load_weights(fpath + "rnn-3.h5");

    #validate(mdl);
    #sys.exit(0);

    class GenCallback(tf.keras.callbacks.Callback):

       def __init__(self, **kwargs):
          pass;

       def on_batch_end(self, batch, logs):
          if np.isnan(logs["loss"]):
              print("{Problem batch: ", batch);
              sys.exit(0);

       def on_epoch_end(self, epoch, logs={}):

          for t in ["CN1CCc2n(CCc3c[n]c[n]c3)c3ccc(C)cc3c2C1",
                    "Clc1ccc2n(CC(c3ccc(F)cc3)(O)C(F)(F)F)c3CCN(C)Cc3c2c1",
                   "Ic1ccc2n(CC(=O)N3CCCCC3)c3CCN(C)Cc3c2c1"]:

             res = gen(mdl, t);
             print(t, " >> ", res);

             mdl.save_weights(fpath + "rnn-" + str(epoch) + ".h5", save_format="h5");
          return;

    callback = [ GenCallback() ];

    callback[0].on_epoch_end(0);
    history = mdl.fit_generator( generator = data_generator("train.smi"),
                                 steps_per_epoch = int(math.ceil(NTRAIN / BATCH_SIZE)),
                                 epochs = NUM_EPOCHS,
                                 validation_data = data_generator("valid.smi"),
                                 validation_steps = int(math.ceil(NVALID / BATCH_SIZE)),
                                 use_multiprocessing=False,
                                 shuffle = True,
                                 callbacks = callback);

    mdl.save_weights(fpath + "nadine-tf.h5", save_format="h5");

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(fpath + "nadine-loss.pdf");

main();

