
import sys
import datetime
import yaml
import math

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN


seed = 0;
tf.random.set_random_seed(seed);
np.random.seed(seed);

np.set_printoptions(threshold=np.nan)

vocab = "$#%()+-./0123456789=@ABCFGHIKLMNOPRSTVXZ[\]abcdegilnoprstu^";
chars = sorted(set(vocab))
vocab_size = len(chars);

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

SEQ_LENGTH = 200
EMBEDDING_SIZE = vocab_size
N_HIDDEN = 512
LEARNING_RATE = .01

NUM_EPOCHS = 200
BATCH_SIZE = 32

def gen_data(data):

    batch_size = len(data);
    length = len(data[0]);

    for i in range(1, batch_size):
       nl = len(data[i]);
       if nl > length:
          length = nl;

    x = np.zeros((batch_size, length, vocab_size), np.int8);
    y = np.zeros((batch_size, length, vocab_size), np.int8);

    cnt = 0;
    for line in data:

        line = line.replace("\n","");
        line = "^" + line + "$";

        n = len(line) - 1;
        y[cnt, n:, char_to_ix["$"] ] = 1;

        for i in range(n):
           x[cnt, i, char_to_ix[ line[i]]] = 1;
           y[cnt, i, char_to_ix[ line[i+1]] ] = 1;
        cnt += 1;

    return [x], y;



def gen2(mdl, fp):

   length = 100;
   bn = 5;

   yt = np.zeros((bn, length, vocab_size), np.int8);
   yt[:, 0, char_to_ix["^"] ] = 1;

   for y in range(1, length-1):

      n = mdl.predict([yt]);
      p = n[: , y, :];

      for batch in range(bn):
         z = np.exp(p[batch]) / np.sum(np.exp(p[batch]));
         yt[batch, y, np.random.choice(np.arange(vocab_size), p= z)] = 1;

   for batch in range(bn):
      st = "";
      for q in range(length):
         w = np.argmax(yt[batch, q]);
         if(w == char_to_ix["$"]):
            break;
         if(q > 0):
            st += ix_to_char[ w ];
      print(st, file=fp);
      fp.flush();

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


def main():

    print("Building network ...");

    l_in = layers.Input( shape= (None, EMBEDDING_SIZE, ));

    l_forward_1 = layers.LSTM( N_HIDDEN, activation='tanh', return_sequences=True)(l_in);
    l_forward_2 = layers.LSTM( N_HIDDEN, activation='tanh', return_sequences=True)(l_forward_1);

    l_out = layers.TimeDistributed(layers.Dense(vocab_size)) (l_forward_2);
    mdl = tf.keras.Model([l_in], l_out);

    mdl.summary();

    def masked_loss(y_true, y_pred):
       loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred);
       mask = tf.cast(tf.not_equal(tf.reduce_sum(y_true, -1), 0), 'float32');
       loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1);
       loss = K.mean(loss);
       return loss;

    adagrad = tf.keras.optimizers.Adagrad(lr=LEARNING_RATE, epsilon=1e-06, clipvalue=100.0);
    mdl.compile(optimizer = adagrad, loss = masked_loss);

    print("Training ...")

    class GenCallback(tf.keras.callbacks.Callback):
       def on_epoch_end(self, epoch, logs={}):
          gen2(mdl, sys.stdout);
          mdl.save_weights("rnn-" + str(epoch+1) + ".h5", save_format="h5");

    try:
        #mdl.load_weights("rnn.h5");
        gen2(mdl, sys.stdout);
        #sys.exit(0);

        train_file = "/home/carpovpv/study-tensorflow/gen/train.smi";
        NTRAIN = sum(1 for line in open(train_file));

        history = mdl.fit_generator( generator = data_generator(train_file),
                                     steps_per_epoch = int(math.ceil(NTRAIN / BATCH_SIZE)),
                                     epochs = 50,
                                     use_multiprocessing=False,
                                     callbacks = [ GenCallback() ]);

        mdl.save_weights("rnn.h5", save_format="h5");

        plt.plot(history.history['loss']);
        plt.title('model loss');
        plt.ylabel('loss');
        plt.xlabel('epoch');
        plt.legend(['train'], loc='upper right');
        plt.savefig('rnn-train.pdf');

    except KeyboardInterrupt:
       pass;

    mdl.save_weights("rnn-final.h5", save_format="h5");
    print("Finished successfully!");

if __name__ == '__main__':
    main()
