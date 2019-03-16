
#S. Zheng, X. Yan, Y. Yang, J. Xu.
# Identifying Structure-Property Relationships through SMILES
# Syntax Analysis with Self-Attention Mechanism
# DOI:10.1021/acs.jcim.8b00803

import sys
import datetime
import yaml
import math
import os
import re
import glob
import collections
import csv
import time
import traceback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3";

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, ReduceLROnPlateau
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from rdkit import Chem
from tqdm import tqdm
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allow_growth = True;
tf.logging.set_verbosity(tf.logging.ERROR);
K.set_session(tf.Session(config=config))

NUM_EPOCHS = 200;
BATCH_SIZE = 64;
CV = 5;
N_HIDDEN = 128;
EMBEDDING_SIZE = 64;
LSTM_SIZE = 256;
DA_SIZE = 64;   #parameter Da in the article
R_SIZE = 5;     #parameter R in the article

seed = 0;
tf.random.set_random_seed(seed);
np.random.seed(seed);

np.set_printoptions(threshold=np.inf)

#Helper functions for statistics: regression (q2, rmsep)
#and classification(AUC, OT, TP, TN, FP, FN).

def calcStatistics(p,y):

   y_mean = np.mean(y);
   press = 0.0;
   ss = 0.0;
   for i in range(len(y)):
      press += ( p[i] - y[i] ) * ( p[i] - y[i] );
      ss += ( y[i] - y_mean ) * ( y[i] - y_mean );

   q2 = 1.0 - press / ss;
   rmsep = math.sqrt(press / len(y));

   return q2, rmsep;

def spc_step(data, threshold):

   tn = 0.0;
   tp = 0.0;
   fp = 0.0;
   fn = 0.0;

   for point in data:
      if(point[1] > 1.0e-3):
         if(point[0] >= threshold):
            tp += 1;
         else:
            fn += 1;
      else:
         if(point[0] < threshold):
            tn += 1;
         else:
            fp += 1;

   if(fp + tn == 0):
      fpr = 0.0;
   else:
      fpr = fp / (fp + tn);

   tpr = tp / (tp + fn);
   return fpr, tpr;

def spc(data):

   maxh = -sys.float_info.max;
   minh = +sys.float_info.max;

   x = [];
   for point in data:
      x.append(point[0]);
      if(point[0] > maxh):
         maxh = point[0];
      if(point[0] < minh):
         minh = point[0];


   step =  (maxh - minh) / 1000.0;

   h = minh - 0.1 * step;

   spc_data = [];

   prev_fpr = -1.0;
   prev_tpr = -1.0;

   while (h <= maxh + 0.1* step):
      fpr, tpr = spc_step(data, h);
      if(fpr != prev_fpr):
          spc_data.append((fpr, tpr, h));

      prev_fpr = fpr;
      prev_tpr = tpr;

      h += step;

   spc_data = sorted(spc_data, key= lambda tup: tup[0]);
   return spc_data;

def auc(data):
   S = 0.0;

   x1 = data[0][0];
   y1 = data[0][1];

   for i in range(1, len(data)):
      x2 = data[i][0];
      y2 = data[i][1];

      S += (x2 - x1) * (y1 + y2);
      x1 = x2;
      y1 = y2;

   S *= 0.5;
   return round(S,4);


def optimal_threshold(data):

   ot1 = 0.0;
   ot2 = 0.0;

   m1 = 10;
   m2 = -10;

   for point in data:
      r = point[1] - (1.0 - point[0]);
      if (r>=0):
         if(r <= m1):
            m1 = r;
            ot1 = point[2];
      else:
         if(r > m2):
            m2 = r;
            ot2 = point[2];


   ot = (ot1 + ot2) / 2.0;
   return round(ot,5);

def tnt(data, threshold):

   tn = 0;
   tp = 0;
   fp = 0;
   fn = 0;

   for point in data:
      if(point[1] > 1.0e-3):
         if(point[0] >= threshold):
            tp += 1;
         else:
            fn += 1;
      else:
         if(point[0] < threshold):
            tn += 1;
         else:
            fp += 1;

   return [tp, fp, tn, fn, (tp + tn) / (tp +tn + fp +fn)];

def auc_acc(data):
   spc_data = spc(data);
   v_auc = auc(spc_data);
   v_ot = optimal_threshold(spc_data);
   v_tnt = tnt(data, v_ot);

   return v_auc, v_tnt[4], v_ot;

#suppress debug output from RDKit
class suppress_stderr(object):
   def __init__(self):
       self.null_fds = [os.open(os.devnull,os.O_RDWR)]
       self.save_fds = [os.dup(2)]
   def __enter__(self):
       os.dup2(self.null_fds[0],2)
   def __exit__(self, *_):
       os.dup2(self.save_fds[0],2)
       for fd in self.null_fds + self.save_fds:
          os.close(fd)

def LoadDataSets():

    print("Loading all datasets: ");
    vocab = set();
    dataset = {};
    props = [];

    props = glob.glob("data/*.csv");
    prop_size = len(props);

    for i, prop in enumerate(props):
       print("Loading file: ", prop);

       first_row = True;
       ind_mol = -1;
       ind_val = -1;

       line = 1;
       for row in csv.reader(open(prop, "r")):
          if first_row:
             first_row = False;
             ind_mol = row.index("mol");
             ind_val = row.index("value");
             continue;

          line += 1;

          mol = row[ind_mol];
          val = row[ind_val];

          if len(mol) == 0: continue;
          with suppress_stderr():
             m = Chem.MolFromSmiles(mol);
             if m is None:
                continue;

          canon = Chem.MolToSmiles(m);
          vocab = vocab | set(canon);
          if canon not in dataset:
              x = np.zeros(prop_size);
              x.fill(np.nan);
              x[i] = val;
              dataset[canon] = x;
          else:
             dataset[canon][i] = val;

    DS = [];

    for i, prop in enumerate(props):
        total = 0;
        for key in dataset:
            if not np.isnan(dataset[key][i]): total += 1;
        DS.append([prop, total]);

    vocab = sorted(vocab);
    return DS, dataset, vocab;

#size, dataset and implicit vocabulary
DS, dataset, vocab = LoadDataSets();

chars = vocab;
vocab_size = len(chars);

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

def gen_data(data, y_min, y_max):

    batch_size = len(data);
    #search for max lengths
    nl = len(data[0][0]);
    for i in range(1, batch_size, 1):
        nl_a = len(data[i][0]);
        if nl_a > nl:
            nl = nl_a;

    x = np.zeros((batch_size, nl), np.int8);
    rev = np.zeros((batch_size, nl), np.int8);

    mx = np.zeros((batch_size, nl), np.int8);
    z = np.zeros((batch_size) if (y_min != y_max or (y_min == -1 and y_min == y_max)) else (batch_size, 2), np.float32);

    for cnt in range(batch_size):
        n = len(data[cnt][0]);
        for i in range(n):
           x[cnt, i] = char_to_ix[ data[cnt][0][i]] ;
           rev[cnt, i] = char_to_ix [ data[cnt][0][n-i-1]];
        mx[cnt, :i+1] = 1;

        if y_min != y_max:
           z [cnt ] = 0.9 + 0.8 * (data[cnt][1] - y_max) / (y_max - y_min);
        elif y_min == y_max and y_min == -1:
           z [cnt ] = data[cnt][1];
        else:
           z [cnt , int(data[cnt][1]) ] = 1;
    return [x, rev, mx], z;

def data_generator(params):

   inds_key = 5;

   key = params[1];
   mols = params[2];
   y_min = params[3];
   y_max = params[4];

   lines = [];
   keys = iter(params[inds_key]);
   try:
       while True:
          try:
             canon = mols[next(keys)];
             if not np.isnan(dataset[canon][key]):
                lines.append( [canon, dataset[canon][key] ]);
             if len(lines) == BATCH_SIZE:
                yield gen_data(lines, y_min, y_max);
                lines = [];
          except StopIteration:
             if len(lines):
                yield gen_data(lines, y_min, y_max);
             lines = [];
             keys = iter(params[inds_key]);
   finally:
        pass;
#generator ends

def findBoundaries(prop):

    print(prop, DS[prop][0], DS[prop][1], end=" ");

    x = [];
    mols = [];

    for key in dataset:
        if not np.isnan(dataset[key][prop]):
            x.append(dataset[key][prop]);
            mols.append(key);

    hist= np.histogram(x)[0];
    if np.count_nonzero(hist) > 2:
        y_min = np.min(x);
        y_max = np.max(x);

        add = 0.01 * (y_max - y_min);
        y_max = y_max + add;
        y_min = y_min - add;

        print("regression: ", y_min, " to ", y_max);
        return ["regression", prop, mols, y_min, y_max];

    else:
        print("classification");
        return ["classification", prop, mols, 0, 0];

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

        self.W1 = self.add_weight( shape =(DA_SIZE, 2*LSTM_SIZE),
                                name="W1", trainable = True,
                                initializer = 'glorot_uniform');
        self.W2 = self.add_weight( shape =(R_SIZE, DA_SIZE),
                                name="W2", trainable = True,
                                initializer = 'glorot_uniform');
        super(SelfLayer, self).build(input_shape);

    def call(self, inputs):

        x = inputs[0];
        mask = inputs[1];

        x = tf.transpose(x, (2,0,1)) * mask;
        x = tf.transpose(x, (1,2,0));

        #in the article: softmax(W2 * tanh( W1 * H^T)) * H
        Q = tf.tensordot(self.W1, tf.transpose(x, (0,2,1)), axes = [[1], [1]]);
        Q = tf.math.tanh(Q);
        Q = tf.tensordot(self.W2, Q, axes = [[1], [0]]);

        A = tf.exp(Q) * mask;
        A = tf.transpose(A, (1,0,2));

        A = A / tf.reshape( tf.reduce_sum(A, axis = 2), (-1, R_SIZE ,1));
        A = layers.Dropout(rate = 0.1) (A);

        return tf.keras.backend.batch_dot( A, x);

    def compute_output_shape(self, input_shape):
        return (input_shape[0], R_SIZE, 2*LSTM_SIZE);

def main():

    print("Building network ...");
    def createNetwork(regression = True):

       l_in = layers.Input( shape= (None,), name = "Forward");
       l_rev = layers.Input( shape = (None, ), name="Reverse");
       l_mask = layers.Input( shape= (None,), name ="Mask");

       #one embedding for forward and reverse
       l_embed = layers.Embedding(input_dim = vocab_size, output_dim = EMBEDDING_SIZE, input_length = None, name="embed");
       l_embed_f = layers.Dropout(rate = 0.2)(l_embed(l_in));
       l_embed_r = layers.Dropout(rate = 0.2)(l_embed(l_rev));

       #recurrency
       l_rnn_f = layers.LSTM(units = LSTM_SIZE, return_sequences = True, name="LSTM-forward")(l_embed_f);
       l_rnn_r = layers.LSTM(units = LSTM_SIZE, return_sequences = True, name="LSTM-reverse")(l_embed_r);
       l_concat = layers.Concatenate(axis = 2) ([l_rnn_f, l_rnn_r]);
       l_concat_d = layers.Dropout(rate = 0.2)(l_concat);

       #sel attention
       l_internal = SelfLayer(name = "Self-Attention") ([l_concat_d, l_mask]);
       l_flat = layers.Flatten()(l_internal);

       def buildRegression():
          l_dense =layers.Dense(N_HIDDEN, activation='tanh') (l_flat);
          l_drop = layers.Dropout(rate=0.2)(l_dense);
          l_out = layers.Dense(1, activation='linear', name="Regression") (l_drop);
          mdl = tf.keras.Model([l_in, l_rev, l_mask], l_out);
          mdl.compile (optimizer = 'adam', loss = 'mse', metrics=['mse'] );
          return mdl;

       def buildClassification():
          l_dense =layers.Dense(N_HIDDEN, activation='tanh') (l_flat);
          l_drop = layers.Dropout(rate=0.2)(l_dense);
          l_out = layers.Dense(2, activation='softmax', name="Classification") (l_drop);
          mdl = tf.keras.Model([l_in, l_rev, l_mask], l_out);
          mdl.compile (optimizer = 'adam', loss = 'binary_crossentropy', metrics=['acc'] );
          return mdl;

       mdl = buildRegression() if regression else buildClassification();
       #mdl.summary();
       #plot_model(mdl, to_file='att.png', show_shapes= True);
       return mdl;

    print("Training ...")

    for propId in range(len(DS)):

       params = findBoundaries(propId);
       mols = params[1];

       try:

          print("Training propery: ", DS[propId]);
          NALL = DS[propId][1];

          print("Number of all points: ", NALL);

          #collect mols for train/valid/predict
          inds = np.arange(NALL, dtype=np.int32);
          np.random.shuffle(inds);

          #cross-validation
          total = NALL;
          cv = np.arange(total, dtype=np.int32);
          cv_n = int(math.ceil(total / CV));

          cv_inds = [];
          for i in range(0, total, cv_n):
             cv_inds.append(range(i, (i + cv_n) if (i + cv_n) < total else total, 1));

          y_min = params[3];
          y_max = params[4];

          y_real = [];
          y_pred = [];

          for cv_i in range(len(cv_inds)):

             fpath = os.getcwd() + "/" + str(propId) + "-" + str(cv_i) + "/";
             try:
                os.mkdir(fpath);
             except:
                pass;

             ind_t = [];
             int_p = [];

             for cv_j in range(len(cv_inds)):
                if ( cv_j == cv_i ):
                   ind_p = list(cv_inds[cv_j]);
                else:
                   ind_t += list(cv_inds[cv_j]);

             inds_train = inds[ ind_t, ... ];
             inds_test = inds[ ind_p , ... ];

             NVAL = int(len(inds_train) * 0.1);
             NTEST = len(inds_test);
             NTRAIN = len(inds_train);

             inds_valid = inds_train[-NVAL:];
             inds_train = inds_train[:-NVAL];

             print("==========================================================");
             print("CV: ", cv_i, len(inds_train), len(inds_valid), len(inds_test));

             train_params = params[:];
             train_params.append(inds_train);

             valid_params = params[:];
             valid_params.append(inds_valid);

             mdl = createNetwork(params[0] == "regression");
             history = mdl.fit_generator( generator = data_generator(train_params),
                                         steps_per_epoch = int(math.ceil(NTRAIN / BATCH_SIZE)),
                                         epochs = NUM_EPOCHS,
                                         validation_data = data_generator(valid_params),
                                         validation_steps = int(math.ceil(NVAL / BATCH_SIZE)),
                                         use_multiprocessing=False,
                                         shuffle = True,
                                         callbacks = [ ModelCheckpoint(fpath, monitor='val_loss',
                                                                      save_best_only= True, save_weights_only= True,
                                                                      mode='auto', period=1),
                                                       ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                                                      patience = 10, min_lr = 1e-6, verbose= 1)]) ;
             mdl.load_weights(fpath);

             plt.clf();
             plt.plot(history.history['loss'][1:])
             plt.plot(history.history['val_loss'][1:])
             plt.title('Model loss')
             plt.ylabel('loss')
             plt.xlabel('epoch')
             plt.legend(['train', 'test'], loc='upper left')
             plt.savefig(fpath + "loss.pdf");
             plt.clf();

             test_params = params[:];
             test_params.append(inds_test);

             #Ask the generator do not scale target values.
             test_params[3] = -1 if params[0] == "regression" else 0;
             test_params[4] = -1 if params[0] == "regression" else 0;

             test_gen = data_generator(test_params);
             for i in range( int(math.ceil( NTEST / BATCH_SIZE))):
                d, z = next(test_gen);
                if params[0] == "regression":
                   y = (mdl.predict(d) - 0.9) / 0.8 * (y_max - y_min) + y_max;
                else:
                   y = mdl.predict(d);

                for i in range(y.shape[0]):
                   y_pred.append(y[i, 0 if params[0] == "regression" else 1]);
                   y_real.append(z[i] if params[0] == "regression" else z[i, 1]);

          if params[0] == "regression":
             q2, rmsep = calcStatistics(y_pred, y_real);
             print("Done cross validation (regression) q2, rmsep: ", q2, rmsep);

             plt.clf();
             fig, ax = plt.subplots();
             ax.scatter(y_real, y_pred);
             line = mlines.Line2D([0,1],[0,1], color = 'black');
             transform = ax.transAxes;
             line.set_transform(transform);
             ax.add_line(line);

             plt.xlabel('real values');
             plt.ylabel('predicted values');
             plt.savefig(fpath + "plot.pdf");

          else:

             auc, acc, ot = auc_acc( list(zip(y_pred, y_real)) );
             print("Done cross validation (classification) auc, acc: ", auc, acc);

             sp = spc(list(zip( y_pred, y_real)));
             fpr, tpr, _ = zip(*sp);

             plt.clf();
             fig, ax = plt.subplots();
             ax.scatter(fpr, tpr);
             line = mlines.Line2D([0,1],[0,1], color = 'black');
             transform = ax.transAxes;
             line.set_transform(transform);
             ax.add_line(line);

             plt.xlabel('False positive rate');
             plt.ylabel('True positive rate');
             plt.xlim(0,1);
             plt.ylim(0,1);
             plt.savefig(fpath + "roc.pdf");

       except KeyboardInterrupt:
          pass;

    print("Finished successfully!");

if __name__ == '__main__':
    main();
