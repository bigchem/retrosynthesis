
import sys
import math
import numpy as np
import theano
import theano.tensor as T
import lasagne
import csv

DESCR_SIZE = 24;
N_HIDDEN = 64;
LEARNING_RATE = 0.001;

#For Hinge Loss calculation
F0 = 0.5;
lasagne.random.set_rng(np.random.RandomState(1));

def get_batch_idx(N, batch_size):
    num_batches = (N + batch_size - 1) / batch_size;
    for i in range(int(num_batches)):
        start, end = i * batch_size, (i + 1) * batch_size;
        idx = slice(start, end);
        yield idx;

data = [];

bio = csv.reader(open("data","r"), delimiter='\t');
for row in bio:
   x = [];
   for item in row:
      if item == '' or item == 'N/A':
         x.append(-1);
      else:
         x.append(float(item));
   data.append(x);


min_x = np.zeros( DESCR_SIZE );
max_x = np.zeros( DESCR_SIZE );

min_x.fill(sys.float_info.max);
max_x.fill(-sys.float_info.max);

for i in range(len(data)):
   for j in range(DESCR_SIZE):
      v = data[i][j+2];
      if v == -1.0: continue;
      if v < min_x[j] : min_x[j] = v;
      if v > max_x[j] : max_x[j] = v;

for i in range(DESCR_SIZE):
   if max_x[i] > 1.0:
      print("Normalize descriptor values : ", i);
      for row in range(len(data)):
         data[row][i+2] = 0.9 + 0.8 * (data[row][i+2] - max_x[i]) / (max_x[i] - min_x[i]);


mix = [];

for row in range(len(data)):
   interval = data[row][1];
   for nextrow in range(row+1, len(data)):
      nextinterval = data[nextrow][1];
      if nextinterval > interval:
         x = data[row][:];
         x.extend(data[nextrow]);
         mix.append(x);

NALL = len(mix);
NTRAIN = int(0.8 * NALL);
NVALID = NALL - NTRAIN;

print("Loaded ", NALL, " pairs of samples.");
print("Shuffling...");

inds = np.arange(NALL, dtype=np.int32);
np.random.shuffle(inds);

inds_t = inds[:NTRAIN];
inds_v = inds[NTRAIN:];

train_x = np.zeros( (NTRAIN*2, DESCR_SIZE), dtype = np.float32);
train_mask = np.ones (( NTRAIN*2, DESCR_SIZE), dtype = np.int8);

valid_x = np.zeros( (NVALID*2, DESCR_SIZE), dtype = np.float32);
valid_mask = np.zeros( (NVALID*2, DESCR_SIZE), dtype = np.float32);
valid_y = np.zeros( (NVALID*2, 1), dtype = np.float32);

for i in range(len(inds_t)):

   x = mix [inds_t[i]];
   x1 = x[:2+DESCR_SIZE];
   x2 = x[2+DESCR_SIZE:];
   for j in range(DESCR_SIZE):

      train_x[i*2, j] = x1[j+2];
      if x1[j+2] == -1.0:
         train_mask [i*2, j] = 0;

      train_x[i*2+1, j] = x2[j+2];
      if x2[j+2] == -1.0:
         train_mask [i*2+1, j] = 0;

for i in range(len(inds_v)):

   x = mix [inds_v[i]];
   x1 = x[:2+DESCR_SIZE];
   x2 = x[2+DESCR_SIZE:];
   for j in range(DESCR_SIZE):

      valid_x[i*2, j] = x1[j+2];
      if x1[j+2] == -1.0:
         valid_mask [i*2, j] = 0;

      valid_x[i*2+1, j] = x2[j+2];
      if x2[j+2] == -1.0:
         valid_mask [i*2+1, j] = 0;

      valid_y [i*2] = x1[0];
      valid_y [i*2 + 1] = x2[0];


X = T.matrix('X', dtype='float32');

W_embed = theano.shared(np.random.uniform(-1.0 / math.sqrt(DESCR_SIZE),
                        1.0 / math.sqrt(DESCR_SIZE) ,
                        (DESCR_SIZE, DESCR_SIZE)));
learning = theano.shared(np.array( LEARNING_RATE, dtype='float32' ));

#Our network consists of two networks. The first one encodes the input and substitutes
#input descriptors with their calculated embeddings. So, for missing descriptors there
#will be no back propagation. The second network  will  do the training. The first and
#second network share embedding matrix.

l_in = lasagne.layers.InputLayer(shape=(None, DESCR_SIZE), input_var=X);
l_z = lasagne.layers.DenseLayer(l_in, num_units= DESCR_SIZE, W = W_embed,
                                      nonlinearity=lasagne.nonlinearities.sigmoid);
t_descr = lasagne.layers.get_output(l_z, deterministic = True);

#function to calculate our descriptors
f_descr = theano.function([X], t_descr, allow_input_downcast = True);

#Second network
X2 = T.matrix('X', dtype='float32');
l_in2 = lasagne.layers.InputLayer(shape=(None, DESCR_SIZE), input_var=X2);
t_batch,_ = l_in2.input_var.shape;

l_embed = lasagne.layers.DenseLayer(l_in2, num_units= DESCR_SIZE, W = W_embed,
                                           nonlinearity=lasagne.nonlinearities.sigmoid);

l_dense = lasagne.layers.DenseLayer(l_embed, num_units= N_HIDDEN,
                                             nonlinearity=lasagne.nonlinearities.rectify);
l_norm = lasagne.layers.StandardizationLayer(l_dense);
l_drop = lasagne.layers.DropoutLayer(l_norm, p = 0.5);
l_out = lasagne.layers.DenseLayer(l_drop, num_units = 1,
                                          nonlinearity=lasagne.nonlinearities.linear);

#function for evaluation
t_score = lasagne.layers.get_output(l_out, deterministic = True);
f_score = theano.function([X2], t_score, allow_input_downcast = True);

#reshape for training
l_shp = lasagne.layers.ReshapeLayer(l_out, shape=(T.cast(t_batch/2, 'int8'), 2));
t_shp = lasagne.layers.get_output(l_shp, deterministic = False);

t_ones = T.ones(shape=(T.cast(t_batch/2, 'int8'), 1), dtype='float32');
t_minus = t_ones - 2.0;

t_weights = T.concatenate([t_minus, t_ones], axis= 1);
t_mul = T.mul(t_shp, t_weights);
loss = T.mean(T.nnet.relu(F0 - T.sum(t_mul, axis=1)));

all_params = lasagne.layers.get_all_params(l_out, trainable=True);
updates = lasagne.updates.adagrad(loss, all_params, learning);

f_train = theano.function([X2], loss, updates=updates, allow_input_downcast=True);
f_loss = theano.function([X2], loss, allow_input_downcast=True);

#cycle training
batch_size = 64;

bestparams = lasagne.layers.get_all_param_values(l_out);
best_train_error = -1;
best_val_error = -1;
last_val_cycle = 0;

tr_log = open("log.txt", "w");

for epoch in range(1000):

   train_error = 0.0;
   it = 0;

   valid_error = 0.0;
   iv = 0;

   for idx in get_batch_idx(NTRAIN, batch_size):
      x = train_x[idx];
      z = f_descr(x);
      m = train_mask[idx];
      xz = x * m + z * (1 - m);
      train_error += f_train(xz);
      it = it + 1;

   train_error = train_error / it / batch_size * 2.0;

   for idx in get_batch_idx(NVALID, batch_size):
      x = valid_x[idx];
      z = f_descr(x);
      m = valid_mask[idx];
      xz = x * m + z * (1 - m);
      valid_error += f_loss(xz);
      iv = iv + 1;

   valid_error = valid_error / iv / batch_size * 2.0;

   last_val_cycle += 1;

   if(best_val_error == -1):
      best_val_error = valid_error;
      best_train_error = train_error;
   else:
      if(best_train_error > train_error and best_val_error > valid_error):
         best_train_error = train_error;
         best_val_error = valid_error;

         bestparams = lasagne.layers.get_all_param_values(l_out);
         print("Save params");
         last_val_cycle = 0;

   print("Epoch:", epoch, " train loss ", train_error, " valid loss ", valid_error);
   print(epoch, train_error, valid_error, file= tr_log);
   learning *= 0.999;

lasagne.layers.set_all_param_values(l_out, bestparams);

f = open('f', "w");

#prognosis
for idx in get_batch_idx(NVALID, batch_size):
   x = valid_x[idx];
   y = valid_y[idx];
   z = f_descr(x);
   m = valid_mask[idx];
   xz = x * m + z * (1 - m);
   s =  f_score(xz);

   for i in range(x.shape[0]):
      if valid_y[i][0] == -1: continue;
      print(s[i][0], valid_y[i][0], file=f);

f.close();

np.save("bio", bestparams);
