import numpy as np
import theano
import theano.tensor as T
import lasagne
import sys

class TemperatureLayer(lasagne.layers.Layer):

   def __init__(self, incoming, temperature, **kwargs):
      super(TemperatureLayer, self).__init__(incoming, **kwargs)
      self.temperature = temperature;

   def get_output_for(self, input, **kwargs):
      x = T.log(input) / self.temperature;
      x = T.exp(x);               
      return x / T.sum(x);

class RnnModel:

   SEQ_LENGTH = 200 + 3;
   N_HIDDEN = 512;
   GRAD_CLIP = 100;
   LEARNING_RATE = 0.01;

   def __init__(self, vocabulary, file_weights):

      self.vocab = vocabulary;
      self.vocab_size = len(self.vocab);

      self.char_to_ix = { ch:i for i,ch in enumerate(self.vocab) }
      self.ix_to_char = { i:ch for i,ch in enumerate(self.vocab) }

      self.start_seq = self.char_to_ix["^"];
      self.end_seq = self.char_to_ix["$"];

      self.Temperature = theano.shared(np.array(1.0, dtype=theano.config.floatX));

      #pred - for generation
      #train - training procedure
      #valid - validation procedure
      #params - all weights...

      self.pred, self.train, self.valid, self.params = \
           self.buildRecurrentNetwork(file_weights); 

   def get_batch_idx(self, N, batch_size):
      num_batches = (N + batch_size - 1) / batch_size

      for i in range(int(num_batches)):
         start, end = i * batch_size, (i + 1) * batch_size
         idx = slice(start, end)

         yield idx

   def toOneHot(self, data):
      x = np.zeros((1, self.SEQ_LENGTH, self.vocab_size), np.float32);
      for i in range(len(data)):
         x[0,i,self.char_to_ix[data[i]]] = 1.0;
      return x;   

   #generating N symbols from RNN
   def generate(self, bait):

      sample_ix = [];
      raw = bait;

      cnt = len(bait);
      x = self.toOneHot(bait);
      m = np.zeros( (1, self.SEQ_LENGTH), np.int16);

      print("Bait: ", bait, " shape ", x.shape);

      for i in range(cnt, self.SEQ_LENGTH):
  
         m[0, :i] = 1;

         p = self.pred(x, m);
         ix = np.random.choice(np.arange(self.vocab_size), p = p.ravel())
         sample_ix.append(ix);

         if(self.ix_to_char[ix] == "$"):
            break;
       
         x[:,i, sample_ix[-1]] = 1;
         raw += self.ix_to_char[ix];


      print("RAW: ", raw);
      return raw[1:];

   #for training
   def gen_data(self, data, return_target = True):     

      data = data.replace("\n","");
      data = "^" + data + "$";

      sm_length = len(data);
      batch_size = sm_length;

      #print(data, len(data), batch_size * SEQ_LENGTH * vocab_size);
      x = np.zeros((batch_size, self.SEQ_LENGTH, self.vocab_size), np.int16);
      y = np.zeros( batch_size, np.int16);
      m = np.zeros((batch_size, self.SEQ_LENGTH), np.int16);

      for n in range(batch_size):
         m[n, :n] = 1;
         for i in range(n):
            x[n,i,self.char_to_ix[data[i]]] = 1;
            #print(data[i], end="");

         #print(" => ", end="");
         if(return_target):
            y[n] = self.char_to_ix[data[n]];
           #print(data[n]);
      return x, y, m;

 
   def setTemperature(self, Temperature):
      self.Temperature.set_value(Temperature);

   def buildRecurrentNetwork(self, weights):

      l_in = lasagne.layers.InputLayer(shape=(None, None, self.vocab_size));
      l_mask = lasagne.layers.InputLayer(shape=(None, None));

      l_forward_1 = lasagne.layers.LSTMLayer(
                     l_in, self.N_HIDDEN, grad_clipping=self.GRAD_CLIP,
                     nonlinearity=lasagne.nonlinearities.tanh,
                     mask_input = l_mask);

      l_drop = lasagne.layers.DropoutLayer(l_forward_1);

      l_forward_2 = lasagne.layers.LSTMLayer(
                      l_drop, self.N_HIDDEN, grad_clipping=self.GRAD_CLIP,
                      nonlinearity=lasagne.nonlinearities.tanh,
                      only_return_final=True, mask_input = l_mask);

      l_vals = lasagne.layers.DenseLayer(l_forward_2, num_units=self.vocab_size, 
                                      nonlinearity=lasagne.nonlinearities.softmax);

      #additional temperature layer for generation
      l_out = TemperatureLayer(l_vals, self.Temperature);

      target_values = T.vector('target_output', 'int32');

      #deterministic output fo the RNN for generation
      network = lasagne.layers.get_output(l_out, deterministic = True);
      probs = theano.function([l_in.input_var, l_mask.input_var], network);

      #network outpur for training
      network_output = lasagne.layers.get_output(l_vals, deterministic = False);

      #network output for validation
      val_output = lasagne.layers.get_output(l_vals, deterministic = True);

      train_cost = T.nnet.categorical_crossentropy(network_output, target_values).mean();
      val_cost =  T.nnet.categorical_crossentropy(val_output, target_values).mean();

      all_params = lasagne.layers.get_all_params(l_vals,trainable=True)

      updates = lasagne.updates.adagrad(train_cost, all_params, self.LEARNING_RATE);
      train = theano.function([l_in.input_var, l_mask.input_var, target_values], train_cost, updates=updates);

      compute_cost = theano.function([l_in.input_var, l_mask.input_var, target_values], train_cost);
      compute_val = theano.function([l_in.input_var, l_mask.input_var, target_values], val_cost);

      params = np.load(weights);
      lasagne.layers.set_all_param_values(l_vals, params);

      return probs, train, compute_val, l_vals;
 
   def fineTune(self, data):
 
      val_lost = [];

      #11% of the whole data for validation
      TV = 0.11;  
      num_epochs = 5;

      total = len(data);
      tv = int(TV * total);

      training = data[:total - tv];
      validation = data [-tv:];

      init_val = lasagne.layers.get_all_param_values(self.params);

      bestval = None;
      params = None;

      improved = False;

      for it in range(1, num_epochs + 1):
         avg_cost = 0.0;
         avg_val = 0.0;

         for o in range(len(training)):
            x,y,m = self.gen_data(training[o], True);
            for u in self.get_batch_idx(x.shape[0], 80):
               avg_cost += self.train(x[u], m[u], y[u]);

         for o in range(len(validation)): 
            x,y,m = self.gen_data(validation[o], True);
            for u in self.get_batch_idx(x.shape[0], 80):
               avg_val += self.valid(x[u], m[u], y[u]);
    
         val_lost.append(avg_val);

         if(it == 1): 
            bestval = avg_val;
            params = lasagne.layers.get_all_param_values(self.params);         
         else:
            if(avg_val < bestval):
               bestval = avg_val;    
               params = lasagne.layers.get_all_param_values(self.params);
               improved = True;          
  
         print("   Epoch {} average training loss = {}, validation loss = {}".format(it, avg_cost, avg_val));

      if(improved == True):
         lasagne.layers.set_all_param_values(self.params, params);
         print("Replacing the parameters of the network.");   
      else:
         lasagne.layers.set_all_param_values(self.params, init_val);

      return val_lost;       
   
   def save(self, name):

      params = lasagne.layers.get_all_param_values(self.params);
      np.save(name, params);
        
