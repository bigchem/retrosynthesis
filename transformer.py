
import sys
import datetime
import yaml
import math
import os
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN

config = tf.ConfigProto()
config.gpu_options.allow_growth = True;
K.set_session(tf.Session(config=config))

from rdkit import Chem

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

seed = 0;

tf.random.set_random_seed(seed);
np.random.seed(seed);

np.set_printoptions(threshold=np.nan)

#may be different
chars = "^#%()+-./0123456789=>@ABCFGHIKLMNOPRSTVXZ[\]abcdegilnoprstu$";
vocab_size = len(chars);

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

max_predict = 160;
TOPK = 1;
NUM_EPOCHS = 50;

BATCH_SIZE = 32
N_HIDDEN = 512
EMBEDDING_SIZE = 64;
KEY_SIZE = EMBEDDING_SIZE;

N_SELF = 8;
N_BLOCK = 2;

def GetPosEncodingMatrix(max_len, d_emb):
	pos_enc = np.array([
		[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
		if pos != 0 else np.zeros(d_emb)
			for pos in range(max_len)
			])
	pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])
	pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])
	return pos_enc

#calc positional encodings only one time
#then just copy slices needed to the batches
GEO = GetPosEncodingMatrix(2048, EMBEDDING_SIZE);


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
    mx = np.zeros((batch_size, nl, nl), np.int8);
    px = np.zeros((batch_size, nl, EMBEDDING_SIZE), np.float32);

    #reactants
    y = np.zeros((batch_size, nr), np.int8);
    py = np.zeros((batch_size, nr, EMBEDDING_SIZE), np.float32);

    #mask for self-attention( ~triangular)
    my = np.zeros((batch_size, nr, nr), np.int8);
    #mask for the decoder attention
    mz = np.zeros((batch_size, nr, nl), np.int8);

    #for output
    z = np.zeros((batch_size, nr, vocab_size), np.int8);

    for cnt in range(batch_size):

        product = "^" + left[cnt] + "$";
        reactants = "^" + right[cnt];

        if progn == False: reactants += "$";

        for i, p in enumerate(product):
           x[cnt, i] = char_to_ix[ p] ;
           px[cnt, i] = GEO[i+1, :EMBEDDING_SIZE ];

        mx[cnt, :, :i+1] = 1;
        mz[cnt, :, :i+1] = 1;

        for i in range( (len(reactants) -1) if progn == False else len(reactants ) ):
           y[cnt, i] = char_to_ix[ reactants[i]];
           py[cnt, i] = GEO[i+1, :EMBEDDING_SIZE ];
           if progn == False:
              z[cnt, i, char_to_ix[ reactants[i + 1] ]] = 1;
           my [cnt, i, :i+1] = 1;
        my[cnt, i:, :i+1] =1;

    return [x, mx, px, y, my, mz, py], z;

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


def generate2(product, res, mdl):

   #print(product);

   lines = [];
   lines.append(product + " >> " + res);

   v = gen_data(lines, True);
   i = len(res);

   n = mdl.predict(v[0]);
   p = n[0, i, :];

   p = np.exp(p) / np.sum(np.exp(p));
   return p;

def gen_beam(mdl, product, greedy = False, beam_szie = 7):

   if greedy:
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

   #beam-search

   lines = [];
   scores = [];
   final_beams = [];

   for i in range(beam_size):
       lines.append("");
       scores.append(0.0);

   for step in range(max_predict):

      if step == 0:
         p = generate2(product, "", mdl);
         nr = np.zeros((vocab_size, 2));
         for i in range(vocab_size):
            nr [i ,0 ] = -math.log10(p[i]);
            nr [i ,1 ] = i;
      else:

         cb = len(lines);
         nr = np.zeros(( cb * vocab_size, 2));

         for i in range(cb):
            p = generate2(product, lines[i], mdl);
            for j in range(vocab_size):
               nr[ i* vocab_size + j, 0] = -math.log10(p[j]) + scores[i];
               nr[ i* vocab_size + j, 1] = i * 100 + j;

      y = nr [ nr[:, 0].argsort()] ;

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

   final_beams = list(sorted(final_beams, key=lambda x:x[1]))[:TOPK];
   answer = [];
   scores = [];

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
            answer.append(sorted(list(sms)));
            scores.append(final_beams[k][1]);

   return answer, scores;


class LayerNormalization(tf.keras.layers.Layer):
	def __init__(self, eps=1e-6, **kwargs):
		self.eps = eps
		super(LayerNormalization, self).__init__(**kwargs)
	def build(self, input_shape):
		self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                             initializer= tf.keras.initializers.Ones(), trainable=True)
		self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                             initializer= tf.keras.initializers.Zeros(), trainable=True)
		super(LayerNormalization, self).build(input_shape)
	def call(self, x):
		mean = K.mean(x, axis=-1, keepdims=True)
		std = K.std(x, axis=-1, keepdims=True)
		return self.gamma * (x - mean) / (std + self.eps) + self.beta
	def compute_output_shape(self, input_shape):
		return input_shape;

class SelfLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        self.denom = math.sqrt(EMBEDDING_SIZE);
        super(SelfLayer, self).__init__(**kwargs);

    def build(self, input_shape):

        self.K = self.add_weight( shape =(EMBEDDING_SIZE, KEY_SIZE),
                                name="K", trainable = True,
                                initializer = 'glorot_uniform');
        self.V = self.add_weight( shape =(EMBEDDING_SIZE, KEY_SIZE),
                                name="V", trainable = True,
                                initializer = 'glorot_uniform');
        self.Q = self.add_weight( shape =(EMBEDDING_SIZE, KEY_SIZE),
                                name="Q", trainable = True,
                                initializer = 'glorot_uniform');
        super(SelfLayer, self).build(input_shape);

    def call(self, inputs):

        Q = tf.tensordot(inputs[0], self.Q, axes = [[2], [0]]);
        K = tf.tensordot(inputs[1], self.K, axes = [[2], [0]]);
        V = tf.tensordot(inputs[2], self.V, axes = [[2], [0]]);

        A = tf.keras.backend.batch_dot(Q, tf.transpose(K, (0,2,1)));
        A = A / self.denom;

        A = tf.exp(A) * inputs[3];
        A = A / tf.reshape( tf.reduce_sum(A, axis = 2), (-1, tf.shape(inputs[0])[1] ,1));

        A = layers.Dropout(rate = 0.1) (A);
        return tf.keras.backend.batch_dot(A, V);

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1], KEY_SIZE);

def validate(mdl, ftest="../data/retrosynthesis-test.smi"):

   NTEST = sum(1 for line in open(ftest,"r"));
   fv = open(ftest, "r");

   cnt = 0;
   ex = 0;
   longest = 0;

   for step in tqdm(range( NTEST )):
      line = fv.readline();
      if len(line) == 0: break;

      reaction = line.split(">");
      product = reaction[2].strip();
      reagents = reaction[0].strip();

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
      scores = [];
      try:
         beams, scores = gen_beam(mdl, product);
      except:
         pass;

      #print(product, answer, beams);
      if len (beams) == 0:
         continue;

      answer_s = set(answer);
      found = False;
      part = False;

      score = 0.0;
      for no in range(len(beams)):
         beam = set(beams[no]);
         score = scores[no];

         right = answer_s.intersection(beam);
         if len(right) == 0: continue;
         if len(right) == len(answer):
             ex += 1;
             found = True;
             print("CNT: ", cnt, ex /cnt *100.0, answer);

         substrate1 = max( list(right) , key=len);
         substrate2 = max( answer, key=len);

         if substrate1 == substrate2:
            longest += 1;
            part = True;

   fv.close();

   print("Total reactions: ", cnt);
   print("Longest: ", longest / cnt * 100.0);
   print("Exact: ", ex / cnt * 100.0);

   return ex / cnt * 100.0;

def main():

    print("Building network ...");

    l_in = layers.Input( shape= (None,));
    l_mask = layers.Input( shape= (None, None,));
    l_pos = layers.Input(shape=(None, EMBEDDING_SIZE,));

    #encoder
    l_embed = layers.Embedding(input_dim = vocab_size, output_dim = EMBEDDING_SIZE, input_length = None, name="embed")(l_in);
    l_embed = layers.Add()([l_embed, l_pos]);
    l_embed = layers.Dropout(rate = 0.1)(l_embed);

    l_encoders = [];

    for layer in range(N_BLOCK):

       #self attention
       l_o = [ SelfLayer() ([l_embed, l_embed, l_embed, l_mask]) for i in range(N_SELF)];

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

       l_encoders.append(l_embed);

    #bottleneck
    #l_encoder = l_embed;

    l_dec = layers.Input(shape =(None,)) ;
    l_dmask = layers.Input(shape =(None, None,));
    l_emask = layers.Input(shape = (None, None,));
    l_dpos = layers.Input(shape =(None, EMBEDDING_SIZE,));

    l_embed = layers.Embedding(input_dim = vocab_size, output_dim = EMBEDDING_SIZE, input_length = None)(l_dec);
    l_embed = layers.Add()([l_embed, l_dpos]);

    for layer in range(N_BLOCK):

       #self attention
       l_o = [ SelfLayer()([l_embed, l_embed, l_embed, l_dmask]) for i in range(N_SELF)];

       l_con = layers.Concatenate()(l_o);
       l_dense = layers.TimeDistributed(layers.Dense(EMBEDDING_SIZE)) (l_con);
       l_drop = layers.Dropout(rate=0.1)(l_dense);
       l_add = layers.Add()( [l_drop, l_embed]);
       l_att = LayerNormalization()(l_add);

       #attention to the encoder
       l_o = [ SelfLayer()([l_att, l_encoders[layer], l_encoders[layer], l_emask]) for i in range(N_SELF)];
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

    mdl = tf.keras.Model([l_in, l_mask, l_pos, l_dec, l_dmask, l_emask, l_dpos], l_out);

    def masked_loss(y_true, y_pred):
       loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred);
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

    mdl.compile(optimizer = 'adam', loss = masked_loss, metrics=[masked_acc]);
    mdl.summary();

    fpath = str(BATCH_SIZE) + "-" + str(EMBEDDING_SIZE) + "-" + str(N_HIDDEN) + "-" + str(N_SELF) + "-" + str(N_BLOCK) + "/";
    try:
       os.mkdir(fpath);
    except:
       pass;

    try:
       mdl.load_weights(fpath + "retrosynthesis-12.h5");
    except: pass;

    if len(sys.argv) == 2 and sys.argv[1] == "validate":
       validate(mdl);
       sys.exit(0);


    #evaluate before training
    def try_synthesis():
        testmols = ["CN1CCc2n(CCc3cncnc3)c3ccc(C)cc3c2C1",
        	    "Clc1ccc2n(CC(c3ccc(F)cc3)(O)C(F)(F)F)c3CCN(C)Cc3c2c1",
   	            "Ic1ccc2n(CC(=O)N3CCCCC3)c3CCN(C)Cc3c2c1"];
        for t in testmols:
           res = gen(mdl, t);
           print(t, " >> ", res);

    try_synthesis();

    print("Training ...")
    class GenCallback(tf.keras.callbacks.Callback):

       def __init__(self, eps=1e-6, **kwargs):
          self.basic = 256 **-0.5
          self.warm = 8000**-1.5
          self.steps = 0

       def on_batch_begin(self, batch, logs={}):
          self.steps += 1;
          lr = self.basic * min(self.steps**-0.5, self.steps*self.warm);
          K.set_value(self.model.optimizer.lr, lr)

       def on_epoch_end(self, epoch, logs={}):

          try_synthesis();
          mdl.save_weights(fpath + "retrosynthesis-" + str(epoch) + ".h5", save_format="h5");
          return

    try:

        train_file = "../data/retrosynthesis-train.smi";
        valid_file = "../data/retrosynthesis-valid.smi";

        NTRAIN = sum(1 for line in open(train_file));
        NVALID = sum(1 for line in open(valid_file));

        print("Number of points: ", NTRAIN, NVALID);

        callback = [ GenCallback()];
        history = mdl.fit_generator( generator = data_generator(train_file),
                                     steps_per_epoch = int(math.ceil(NTRAIN / BATCH_SIZE)),
                                     epochs = NUM_EPOCHS,
                                     validation_data = data_generator(valid_file),
                                     validation_steps = int(math.ceil(NVALID / BATCH_SIZE)),
                                     use_multiprocessing=False,
                                     shuffle = True,
                                     callbacks = callback);

        mdl.save_weights(fpath + "final.h5", save_format="h5");

        # summarize history for accuracy
        plt.plot(history.history['masked_acc'])
        plt.plot(history.history['val_masked_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(fpath + "accuracy.pdf");

        plt.clf();

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(fpath + "loss.pdf");

    except KeyboardInterrupt:
       pass;

    print("Finished successfully!");

if __name__ == '__main__':
    main();
