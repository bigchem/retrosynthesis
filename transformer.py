
import sys
import datetime
import yaml
import math
import os
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt

from rdkit import Chem

#suppress INFO, WARNING, and ERROR messages of Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

import argparse
from tqdm import tqdm

#custom layers
from layers import PositionLayer, MaskLayerLeft, \
                   MaskLayerRight, MaskLayerTriangular, \
                   SelfLayer, LayerNormalization

#seed = 0;
#tf.random.set_random_seed(seed);
#np.random.seed(seed);

config = tf.ConfigProto()
config.gpu_options.allow_growth = True;
K.set_session(tf.Session(config=config))

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


chars = " ^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy$"; #UPTSO
#chars = " ^#()+-./123456789=@BCFHIKLMNOPSZ[\\]cdegilnorstu$"; #nadine
vocab_size = len(chars);

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

max_predict = 160;  #max for nadine database
TOPK = 1;
NUM_EPOCHS = 1000;

BATCH_SIZE = 64
N_HIDDEN = 512
EMBEDDING_SIZE = 64;
KEY_SIZE = EMBEDDING_SIZE;

def gen_data(data, progn = False):

    batch_size = len(data);

    #search for max lengths
    left = [];
    right = [];
    for line in data:
        line = line.split(">");
        #left.append( list(filter(None, re.split(token_regex, line[0].strip() ))) );
        left.append( line[0].strip());
        if len(line) > 1:
           #right.append( list(filter(None, re.split(token_regex, line[2].strip() ))) );
           right.append ( line[2].strip() );
        else:  right.append("")

    nl = len(left[0]);
    nr = len(right[0]);
    for i in range(1, batch_size, 1):
        nl_a = len(left[i]);
        nr_a = len(right[i]);
        if nl_a > nl:
            nl = nl_a;
        if nr_a > nr:
            nr = nr_a;

    #add start and end symbols
    nl += 2;
    if progn: nr = max_predict;
    else :  nr += 1;

    #products
    x = np.zeros((batch_size, nl), np.int8);
    mx = np.zeros((batch_size, nl), np.int8);

    #reactants
    y = np.zeros((batch_size, nr), np.int8);
    my = np.zeros((batch_size, nr), np.int8);

    #for output
    z = np.zeros((batch_size, nr, vocab_size), np.int8);

    for cnt in range(batch_size):
        product = "^" + left[cnt] + "$";
        reactants = "^" + right[cnt];

        if progn == False: reactants += "$";
        for i, p in enumerate(product):
           x[cnt, i] = char_to_ix[ p] ;

        mx[cnt, :i+1] = 1;
        for i in range( (len(reactants) -1) if progn == False else len(reactants ) ):
           y[cnt, i] = char_to_ix[ reactants[i]];
           if progn == False:
              z[cnt, i, char_to_ix[ reactants[i + 1] ]] = 1;

        my[cnt, :i+1] =1;

    return [x, mx, y, my], z;

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

      v = gen_data(lines, True);

      n = mdl.predict(v[0]);
      p = n[0 , i-1, :];

      w = np.argmax(p);
      if w == char_to_ix["$"]:
         break;
      res += ix_to_char[w];

   return res;


def generate2(product, T, res, mdl):

   lines = [];
   lines.append(product + " >> " + res);

   v = gen_data(lines, True);
   i = len(res);

   n = mdl.predict(v[0]);

   #increase temperature during decoding steps
   p = n[0, i, :] / T;
   p = np.exp(p) / np.sum(np.exp(p));

   return p;

def gen_beam(mdl, T, product, beam_size=7):

   if beam_size <= 1: #perform greedy search
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

   lines = [];
   scores = [];
   final_beams = [];

   for i in range(beam_size):
       lines.append("");
       scores.append(0.0);

   for step in range(max_predict):

      if step == 0:
         p = generate2(product, T, "", mdl);
         nr = np.zeros((vocab_size, 2));
         for i in range(vocab_size):
            nr [i ,0 ] = -math.log10(p[i]);
            nr [i ,1 ] = i;
      else:

         cb = len(lines);
         nr = np.zeros(( cb * vocab_size, 2));

         for i in range(cb):
            p = generate2(product, T, lines[i], mdl);
            for j in range(vocab_size):
               nr[ i* vocab_size + j, 0] = -math.log10(p[j]) + scores[i];
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
                final_beams.append( [ lines[beamno] + c, y[i,0]]);
             beam_size -= 1;
         else:
             new_beams.append( lines[beamno] + c );
             new_scores.append( y[i, 0]);

      lines = new_beams;
      scores = new_scores;

      if len(lines) == 0: break;

   for i in range(len(final_beams)):
      final_beams[i][1] = final_beams[i][1] / len(final_beams[i][0]);

   final_beams = list(sorted(final_beams, key=lambda x:x[1]))[:TOPK];
   answer = [];

   for k in range(TOPK):
      reags = set(final_beams[k][0].split("."));
      sms = set();

      with suppress_stderr():
         for r in reags:
            r = r.replace("$", "");
            m = Chem.MolFromSmiles(r);
            if m is not None:
               sms.add(Chem.MolToSmiles(m));
         if len(sms):
            answer.append([sorted(list(sms)), final_beams[k][1] ]);
   return answer;


def validate(ftest, mdls, Ts):

   NTEST = sum(1 for line in open(ftest,"r"));
   fv = open(ftest, "r");

   cnt = 0;
   ex = 0;
   longest = 0;

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

      for i in range(len(mdls)):
         try:
            beams += gen_beam(mdls[i], Ts[i], product);
         except:
            pass;

      if len (beams) == 0:
         continue;

      answer_s = set(answer);
      ans = [];

      for k in range(len(beams)):
         ans.append(beams[k][0]);

      for beam in ans:
         beam = set(beam);
         right = answer_s.intersection(beam);
         if len(right) == 0: continue;
         if len(right) == len(answer):
             ex += 1;
             longest += 1;

             print("CNT: ", cnt, ex /cnt *100.0, answer);
             break;

         substrate1 = max( list(right) , key=len);
         substrate2 = max( answer, key=len);

         if substrate1 == substrate2:
            longest += 1;

   fv.close();

   print("Total reactions: ", cnt);
   print("Longest: ", longest / cnt * 100.0);
   print("Exact: ", ex / cnt * 100.0);

   return ex / cnt * 100.0;

def buildNetwork(n_block, n_self):

    print("Building network ...");

    #product
    l_in = layers.Input( shape= (None,));
    l_mask = layers.Input( shape= (None,));

    #reagents
    l_dec = layers.Input(shape =(None,)) ;
    l_dmask = layers.Input(shape =(None,));

    #positional encodings for product and reagents, respectively
    l_pos = PositionLayer(EMBEDDING_SIZE)(l_mask);
    l_dpos = PositionLayer(EMBEDDING_SIZE)(l_dmask);

    l_emask = MaskLayerRight()([l_dmask, l_mask]);
    l_right_mask = MaskLayerTriangular()(l_dmask);
    l_left_mask = MaskLayerLeft()(l_mask);

    #encoder
    l_voc = layers.Embedding(input_dim = vocab_size, output_dim = EMBEDDING_SIZE, input_length = None);

    l_embed = layers.Add()([ l_voc(l_in), l_pos]);
    l_embed = layers.Dropout(rate = 0.1)(l_embed);

    for layer in range(n_block):

       #self attention
       l_o = [ SelfLayer(EMBEDDING_SIZE, KEY_SIZE) ([l_embed, l_embed, l_embed, l_left_mask]) for i in range(n_self)];

       l_con = layers.Concatenate()(l_o);
       l_dense = layers.TimeDistributed(layers.Dense(EMBEDDING_SIZE)) (l_con);
       l_drop = layers.Dropout(rate=0.1)(l_dense);
       l_add = layers.Add()( [l_drop, l_embed]);
       l_att = LayerNormalization()(l_add);

       #position-wise
       l_c1 = layers.Conv1D(N_HIDDEN, 1, activation='relu')(l_att);
       l_c2 = layers.Conv1D(EMBEDDING_SIZE, 1)(l_c1);
       l_drop = layers.Dropout(rate = 0.1)(l_c2);
       l_ff = layers.Add()([l_att, l_drop]);
       l_embed = LayerNormalization()(l_ff);

    #bottleneck
    l_encoder = l_embed;

    #l_embed = layers.Embedding(input_dim = vocab_size, output_dim = EMBEDDING_SIZE, input_length = None)(l_dec);
    l_embed = layers.Add()([l_voc(l_dec), l_dpos]);
    l_embed = layers.Dropout(rate = 0.1)(l_embed);

    for layer in range(n_block):

       #self attention
       l_o = [ SelfLayer(EMBEDDING_SIZE, KEY_SIZE)([l_embed, l_embed, l_embed, l_right_mask]) for i in range(n_self)];

       l_con = layers.Concatenate()(l_o);
       l_dense = layers.TimeDistributed(layers.Dense(EMBEDDING_SIZE)) (l_con);
       l_drop = layers.Dropout(rate=0.1)(l_dense);
       l_add = layers.Add()( [l_drop, l_embed]);
       l_att = LayerNormalization()(l_add);

       #attention to the encoder
       l_o = [ SelfLayer(EMBEDDING_SIZE, KEY_SIZE)([l_att, l_encoder, l_encoder, l_emask]) for i in range(n_self)];
       l_con = layers.Concatenate()(l_o);
       l_dense = layers.TimeDistributed(layers.Dense(EMBEDDING_SIZE)) (l_con);
       l_drop = layers.Dropout(rate=0.1)(l_dense);
       l_add = layers.Add()( [l_drop, l_att]);
       l_att = LayerNormalization()(l_add);

       #position-wise
       l_c1 = layers.Conv1D(N_HIDDEN, 1, activation='relu')(l_att);
       l_c2 = layers.Conv1D(EMBEDDING_SIZE, 1)(l_c1);
       l_drop = layers.Dropout(rate = 0.1)(l_c2);
       l_ff = layers.Add()([l_att, l_drop]);
       l_embed = LayerNormalization()(l_ff);

    l_out = layers.TimeDistributed(layers.Dense(vocab_size,
                                          use_bias=False)) (l_embed);

    mdl = tf.keras.Model([l_in, l_mask, l_dec, l_dmask], l_out);

    def masked_loss(y_true, y_pred):
       loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred);
       mask = tf.cast(tf.not_equal(tf.reduce_sum(y_true, -1), 0), 'float32');
       loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1);
       loss = K.mean(loss);
       return loss;

    def masked_acc(y_true, y_pred):
       mask = tf.cast(tf.not_equal(tf.reduce_sum(y_true, -1), 0), 'float32');
       eq = K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis = -1)), 'float32');
       eq = tf.reduce_sum(eq * mask, -1) / tf.reduce_sum(mask, -1);
       eq = K.mean(eq);
       return eq;

    mdl.compile(optimizer = 'adam', loss = masked_loss, metrics=['accuracy', masked_acc]);
    mdl.summary();

    return mdl;


def main():

    parser = argparse.ArgumentParser(description='Transformer retrosynthesis model.')
    parser.add_argument('--layers', type=int, default =3,
                    help='Number of layers in encoder\'s module. Default 3.');
    parser.add_argument('--heads', type=int, default =10,
                    help='Number of attention heads. Default 10.');

    parser.add_argument('--validate', action='store', type=str, help='Validation regime.', required=False);
    parser.add_argument('--model', type=str, default ='../models/retrosynthesis-long.h5', help='A model to be used during validation. Default file ../models/retrosynthesis-long.h5', required=False);
    parser.add_argument('--temperature', type=float, default =1.2, help='Temperature for decoding. Default 1.2', required=False);

    args = parser.parse_args();

    mdl = buildNetwork(args.layers, args.heads);

    if args.validate is not None:
        mdl.load_weights(args.model);
        acc= validate(args.validate, [mdl], [args.temperature]);
        sys.exit(0);


    #evaluate before training
    def printProgress():
       print("");
       for t in ["CN1CCc2n(CCc3cncnc3)c3ccc(C)cc3c2C1",
    	         "Clc1ccc2n(CC(c3ccc(F)cc3)(O)C(F)(F)F)c3CCN(C)Cc3c2c1",
    	         "Ic1ccc2n(CC(=O)N3CCCCC3)c3CCN(C)Cc3c2c1"]:
          res = gen(mdl, t);
          print(t, " >> ", res);

    printProgress();

    try:
        os.mkdir("storage");
    except:
        pass;

    print("Training ...")
    lrs = [];

    class GenCallback(tf.keras.callbacks.Callback):

       def __init__(self, eps=1e-6, **kwargs):
          self.steps = 0
          self.warm = 16000.0;

       def on_batch_begin(self, batch, logs={}):
          self.steps += 1;
          lr = 20.0 * min(1.0, self.steps / self.warm) / max(self.steps, self.warm);
          lrs.append(lr);
          K.set_value(self.model.optimizer.lr, lr)

       def on_epoch_end(self, epoch, logs={}):

          printProgress();
          if epoch > 90:
             mdl.save_weights("storage/tr-" + str(epoch+1) + ".h5", save_format="h5");

          if epoch % 100 == 0 and epoch > 0:
              self.steps = self.warm - 1;
          return;

    try:

        train_file = "../data/retrosynthesis-all.smi";

        NTRAIN = sum(1 for line in open(train_file));
        print("Number of points: ", NTRAIN);

        callback = [ GenCallback() ];
        history = mdl.fit_generator( generator = data_generator(train_file),
                                     steps_per_epoch = int(math.ceil(NTRAIN / BATCH_SIZE)),
                                     epochs = NUM_EPOCHS,
                                     use_multiprocessing=False,
                                     shuffle = True,
                                     callbacks = callback);

        mdl.save_weights("final.h5", save_format="h5");

        #save raw data
        np.savetxt("lr.txt", lrs);
        np.savetxt("loss.txt", history.history['loss']);
        np.savetxt('accuracy.txt', history.history['masked_acc']);

        # summarize history for accuracy
        plt.plot(history.history['masked_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("accuracy.pdf");

        plt.clf();
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("loss.pdf");

        plt.clf();
        # summarize history for learning rate
        plt.plot(lrs)
        plt.title('model lr')
        plt.ylabel('lr')
        plt.xlabel('epoch')
        plt.legend(['lr'], loc='upper left')
        plt.savefig("lr.pdf");

    except KeyboardInterrupt:
       pass;

if __name__ == '__main__':
    main();
