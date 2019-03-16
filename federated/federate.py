
import sys
import os
import math
import pickle

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdReducedGraphs
from rdkit.Chem import Descriptors

import numpy as np
import theano
import theano.tensor as T
from theano import shared
import lasagne

#For single task and federated training use the same meta-parameters.

FED_EPOCHS = 10000;
VAL_PER = 0.2;
PER_TEST = 0.15;
LEARNING_RATE = 0.0075;
BATCH_SIZE = 64;
MAXEPOCHS = 10000;
MAXEPOCHS_WITHOUT_IMPROVEMENT = 1000;

#For batch slicing.
def get_batch_idx(N, batch_size):
   num_batches = (N + batch_size - 1) / batch_size

   for i in range(int(num_batches)):
      start, end = i * batch_size, (i + 1) * batch_size
      idx = slice(start, end)

      yield idx

#All data in one file for this toy example. Load everything. 
#Molecules a stored in a global list, whereas each property has its own list of indexes to that pool.
def loadDataSet(fname, descr_names):

   limits = {};
   mols = [];
   sms = [];

   descrs = np.zeros( (0, len(descr_names)), dtype=np.float32);

   suppl = Chem.SDMolSupplier(fname);
   for mol in suppl:
      if(mol is None):
         continue;
 
      for prop in mol.GetPropNames():         

         if "converted" in prop:
            if prop not in limits:
                
               limits[prop]= { "min": sys.float_info.max, \
                               "max" : -sys.float_info.max, \
                               "count": 0, \
                               "avg" : 0.0, \
                               "mols" : [], \
                               "vals" : []};
            
            val = float(mol.GetProp(prop));

            if(limits[prop]["min"] > val):
               limits[prop]["min"] = val;
            if(limits[prop]["max"] < val):
               limits[prop]["max"] = val;            
           
            limits[prop]["count"] += 1;
            limits[prop]["avg"] += val;
            limits[prop]["vals"].append(val);
            
            smiles = Chem.MolToSmiles(mol);

            idx = -1;
            for i in range(len(sms)):
               if(sms[i] == smiles):
                  idsx = i;
                  break;
               
            if(idx == -1):
               mols.append(mol);
               sms.append(smiles);

               tmp_descr = np.zeros( (1, len(descr_names)), dtype=np.float32);
       
               for op in mol.GetPropNames():
                  if op in descr_names:
                     index = descr_names.index(op);
                     tmp_descr[0][index] = float(mol.GetProp(op));

               descrs = np.append(descrs, tmp_descr, axis=0);
               idx = len(mols) - 1;

            limits[prop]["mols"].append(idx);            


   props = [];
         
   for it in limits:
      props.append(it);
      limits[it]["avg"] = limits[it]["avg"] / limits[it]["count"] * 1.0;

   props = sorted(props);  
   return props, limits, mols, descrs;

#Federated learning training procedure.
#W all weights, W3 - weights of common parts.
#l_out - lasagne output layer
#l_dense_3 - lasagne common part output
#train and val theano functions, x and y the data

def federate( a, x, y):

   W = a[0];
   W3 = a[1];
   l_out = a[2];
   l_dense_3 = a[3];
   train = a[5];
   val = a[6];

   lasagne.layers.set_all_param_values(l_out, W);

   #set common weights
   lasagne.layers.set_all_param_values(l_dense_3, W3);

   #A flag to stop training.
   stop = False;

   #Shuffle the data.
   N = x.shape[0];
   inds = np.arange(N, dtype=np.int32);
   np.random.shuffle(inds);

   x_s = np.zeros_like(x);
   y_s = np.zeros_like(y);

   for i in range(N):
      x_s [i] = x[ inds[i] ];
      y_s [i] = y[ inds[i] ];

   #Split to training and validation datasets.
   nval = int(N * VAL_PER);
   ntrain = N - nval;
   
   x_val = x_s[:nval];
   y_val = y_s[:nval];

   x_train = x_s[nval:];
   y_train = y_s[nval:];

   #print("Training shape (federated): ", x_train.shape);
   #print("Validation shape (federated): ", x_val.shape);

   bestparams = lasagne.layers.get_all_param_values(l_out);
   best_train_error = -1;
   best_val_error = -1;
   last_val_cycle = 0;

   for epoch in range(MAXEPOCHS):

      if(stop == True):
         break;

      val_error = 0.0;
      train_error = 0.0;
      ic = 0;
      for idx in get_batch_idx(ntrain, BATCH_SIZE):
         train_error += train(x_train[idx], y_train[idx]);   
         ic = ic + 1;   

      train_error = train_error / ic;
     
      i= 0;
      for idx in get_batch_idx(nval, BATCH_SIZE):
         val_error += val(x_val[idx], y_val[idx]);
         ic = ic + 1;

      val_error = val_error / ic;

      #print("\tEpoch: {}, training loss = {}, validation loss = {}".format(epoch, train_error, val_error));

      last_val_cycle += 1;

      if(best_val_error == -1):
         best_val_error = val_error;
         best_train_error = train_error;
      else:
         if(best_train_error > train_error and best_val_error > val_error):
            best_train_error = train_error;
            best_val_error = val_error;

            bestparams = lasagne.layers.get_all_param_values(l_out);
            #print("===Save interim params.===");
            #sys.stdout.flush();

            last_val_cycle = 0;
            stop = True;

            break;

      if(last_val_cycle > MAXEPOCHS_WITHOUT_IMPROVEMENT):
         #print("No validation improvement(:");
         break;

   lasagne.layers.set_all_param_values(l_out, bestparams);

   #return new matrixes to the outer functions...

   W3a = lasagne.layers.get_all_param_values(l_dense_3);
   Wa = lasagne.layers.get_all_param_values(l_out);

   return Wa, W3a; 

def buildOneModel(x, y):
   
   NumFeatures = x.shape[1];
   N = x.shape[0];

   X = T.matrix('inputs', dtype='float32');
   Y = T.vector('outputs', dtype='float32');

   l_in = lasagne.layers.InputLayer(shape=(None, NumFeatures));
   
   #common part
   l_dense_1 = lasagne.layers.DenseLayer(l_in, 4000);

   l_dense_2 = lasagne.layers.DenseLayer(l_dense_1, 2000);
   l_drop_2 = lasagne.layers.DropoutLayer(l_dense_2, p=0.25);

   l_dense_3 = lasagne.layers.DenseLayer(l_drop_2, 1000);
   l_drop_3 = lasagne.layers.DropoutLayer(l_dense_3, p = 0.25);

   l_dense_3 = lasagne.layers.DenseLayer(l_drop_2, 1000);
   l_drop_3 = lasagne.layers.DropoutLayer(l_dense_3, p = 0.25);

   #common part

   #individual
   l_dense_4 = lasagne.layers.DenseLayer(l_drop_3, 1000);
   l_drop_4 = lasagne.layers.DropoutLayer(l_dense_4, p=0.25);

   l_out = lasagne.layers.DenseLayer(l_drop_4, 1, nonlinearity = lasagne.nonlinearities.sigmoid);

   prediction = lasagne.layers.get_output(l_out, X, deterministic = True);
   pred = theano.function([X], prediction);   
   val_loss = lasagne.objectives.squared_error(prediction, Y);
   val_loss = val_loss.mean();
   val = theano.function( [X, Y], outputs = val_loss);

   training = lasagne.layers.get_output(l_out, X, deterministic = False);
   loss = lasagne.objectives.squared_error(training, Y);
   loss = loss.mean();

   params = lasagne.layers.get_all_params(l_out, trainable = True);
   updates = lasagne.updates.sgd(loss, params, learning_rate = LEARNING_RATE);
   train = theano.function( [X, Y], outputs = loss, updates=updates);

   #shuffle the data
   inds = np.arange(N, dtype=np.int32);
   np.random.shuffle(inds);

   x_s = np.zeros_like(x);
   y_s = np.zeros_like(y);

   for i in range(N):
      x_s [i] = x[ inds[i] ];
      y_s [i] = y[ inds[i] ];

   #split to training and validation datasets
   nval = int(N * VAL_PER);
   ntrain = N - nval;
   
   x_val = x_s[:nval];
   y_val = y_s[:nval];

   x_train = x_s[nval:];
   y_train = y_s[nval:];

   #print("Training shape: ", x_train.shape);
   #print("Validation shape: ", x_val.shape);

   bestparams = lasagne.layers.get_all_param_values(l_out);
   best_train_error = -1;
   best_val_error = -1;
   last_val_cycle = 0;

   for epoch in range(MAXEPOCHS):
      val_error = 0.0;
      train_error = 0.0;
      ic = 0;
      for idx in get_batch_idx(ntrain, BATCH_SIZE):
         train_error += train(x_train[idx], y_train[idx]);   
         ic = ic + 1;   

      train_error = train_error / ic;
     
      i= 0;
      for idx in get_batch_idx(nval, BATCH_SIZE):
         val_error += val(x_val[idx], y_val[idx]);
         ic = ic + 1;

      val_error = val_error / ic;

      #print("\tEpoch: {}, training loss = {}, validation loss = {}".format(epoch, train_error, val_error));

      last_val_cycle += 1;

      if(best_val_error == -1):
         best_val_error = val_error;
         best_train_error = train_error;
      else:
         if(best_train_error > train_error and best_val_error > val_error):
            best_train_error = train_error;
            best_val_error = val_error;

            bestparams = lasagne.layers.get_all_param_values(l_out);
            #print("===>> Epoch: {}, train: {}, validation: {}".format(epoch, train_error, val_error));
            #sys.stdout.flush();

            last_val_cycle = 0;

      if(last_val_cycle > MAXEPOCHS_WITHOUT_IMPROVEMENT):
         #print("No validation improvement(:");
         break;

   lasagne.layers.set_all_param_values(l_out, bestparams);
   W3 = lasagne.layers.get_all_param_values(l_dense_3);
 
   return [bestparams, W3, l_out, l_dense_3, pred, train, val];


def calcStatistics(p,y):

   y_mean = np.mean(y);
   press = 0.0;
   ss = 0.0;
   for i in range(y.shape[0]):
      press += ( p[i] - y[i] ) * ( p[i] - y[i] );
      ss += ( y[i] - y_mean ) * ( y[i] - y_mean );

   q2 = 1.0 - press / ss;
   rmsep = math.sqrt(press / y.shape[0]);

   return q2, rmsep;

def calcPrognosis(p, pred, x):
   for idx in get_batch_idx(x.shape[0], BATCH_SIZE):
      nd = pred(x[idx]);
      for i in range(nd.shape[0]):
         p.append(nd[i][0]);
   return p; 

def main():

   descr_fn = "../data/admet-ochem-descr.txt";
   with open(descr_fn) as f: 
      descrs = f.readline();
 
   descr_names = descrs.strip("\n").split(" ");
   ind_fn = "../data/admet-nums.txt";

   print("Loading indexes...");
   with open(ind_fn) as f:
      lines = f.readlines();  

   indexes = {};

   for i in range(0, len(lines), 2):
      prop = lines[i].strip("\n");
      inds = lines[i + 1].strip("\n").split(",");    
      indexes[prop] = inds;
   
 
   fn = "../data/admet-ochem.sdf";
   print("Loading data from the file {}".format(fn));

   '''
   props, limits, mols, descrs = loadDataSet(fn, descr_names);

   with open("dump.pkl", 'wb') as f:
      pickle.dump([props, limits, mols, descrs], f);
   '''

   with open("../data/admet.pkl", "rb") as f:
      props, limits, mols, descrs = pickle.load(f);

   #print("shape of: ", len(mols), descrs.shape);


   if(len(props) == 0):
      print("something went wrong! The number of properties loaded is equal to zero! Check input file.");
      sys.exit(1);

   print("Loaded: {} properties.\nThey are:".format(len(props)));
   for prop in props:
      print("\t", prop);

   print("The number of molecules: {}".format(len(mols)));

   if(len(descrs) != len(mols)):
      print("Something went wrong! The number of descriptors calculated must be the same as the number of molecules.");
      sys.exit(1);

   '''
   for prop in props:
      print(prop);
      z = np.arange( (len(limits[prop]["mols"])), dtype=np.int32);
      np.random.shuffle(z);
      for i in range(z.shape[0]):
         print(z[i], end=",");
      print("");       
   sys.exit(0);
   '''

   nd = len(descrs[0]);

   #scaling
   descr_props = {};
   for i in range(nd):
      m = np.mean(descrs[... ,i]);
      sigma = np.std(descrs[... ,i]);
      descr_props[ descr_names[i]] = { "mean" : m, "std" : sigma };

      descrs[... , i] = (descrs[... , i] - m) / sigma;
    
   Models = {};
   
   #To determine which model will be the coordinator.
   main_prop = "";
   main_q2 = -sys.float_info.max;

   for prop in props:
      print("Building the model for ", prop, end="");
      
      points = limits[prop]["mols"];
      vals = limits[prop]["vals"];
      inds = indexes[prop];

      cnt = len(points);

      x = np.zeros((cnt, nd), dtype=np.float32);      
      y = np.zeros((cnt, ), dtype=np.float32);
     
      for j in range(cnt):
         x[j] = descrs[ points[ int(inds[j]) ] ];
         y[j] = vals[ int(inds[j]) ];        

      #scaling to 0.1 - 0.9, add extra space to both ends

      add = 0.01 * (limits[prop]["max"] - limits[prop]["min"]);
      max_y = limits[prop]["max"] + add;
      min_y = limits[prop]["min"] - add;

      #print("Max: {}; Min: {}".format(max_y, min_y));

      for i in range(cnt):
         y[i] = 0.9 + 0.8 * (y[i] - max_y) / (max_y - min_y);
    
      #print("Dimensions: ", x.shape, y.shape);
      
      per = int(cnt * PER_TEST);     

      #x_p and y_p for external prediction    
      x_p = x[ : per ];
      x_t = x[ per : ];

      y_p = y[ : per ];
      y_t = y[ per : ];
        
      #For each property we will store all cross-validation models.
      Models[prop] = [];

      total = x_t.shape[0];
      cv = np.arange(total, dtype=np.int32);
      cv_n = int(total / 5);

      inds = [];
      for i in range(0, total, cv_n):
         inds.append(range(i, (i + cv_n) if (i + cv_n) < total else total, 1));
         #print("Inds: " , i, inds[ len(inds) -1]);

      p = [];
      for cv_i in range(len(inds)):

         ind_t = [];
         int_p = [];
         
         for cv_j in range(len(inds)):
            if ( cv_j == cv_i ):
               ind_p = list(inds[cv_j]);
            else:
               ind_t += list(inds[cv_j]);

         #print("CV ", cv_i + 1, len(ind_t), len(ind_p), total);
         mdl = buildOneModel(x_t[ ind_t, ... ], y_t[ind_t , ... ]);

         calcPrognosis(p, mdl[4], x_t[ind_p, ...]);
         Models[prop].append(mdl);

      for i in range(total):
         y_t[i] = (y_t[i] - 0.9) / 0.8 * (max_y - min_y) + max_y;
         p[i] = (p[i] - 0.9) / 0.8 * (max_y - min_y) + max_y;

      q2, rmse = calcStatistics(p, y_t);
   
      y1 =[]; y2 =[]; y3 =[]; y4 =[]; y5 =[];
      
      calcPrognosis(y1, Models[prop][0][4], x_p);
      calcPrognosis(y2, Models[prop][1][4], x_p);
      calcPrognosis(y3, Models[prop][2][4], x_p);
      calcPrognosis(y4, Models[prop][3][4], x_p);
      calcPrognosis(y5, Models[prop][4][4], x_p);

      for i in range(len(y1)):
         y1[i] = (y1[i] + y2[i] + y3[i] + y4[i] + y5[i]) / 5.0;
      
      #rescale back external validation
      for i in range(len(y1)):
         y1[i] = (y1[i] -0.9) / 0.8 * (max_y - min_y) + max_y;
         y_p[i] = (y_p[i] -0.9) / 0.8 * (max_y - min_y) + max_y;

      _, rmsep = calcStatistics(y1, y_p);

      print(" (q2, rmse, rmsep): ", q2, rmse, rmsep);
      sys.stdout.flush();

      for i in range(len(Models[prop])):
         fname = prop + str(i);
         fname = fname.replace("/","-");
         fname = fname.replace(" ","-");
   
         np.save("single/" + fname, Models[prop][i][0]);

      #maybe also some averaging here...
      Models[prop] = Models[prop][ np.random.randint(5) ];

      if(q2 >= main_q2):
         main_q2 = q2;
         main_prop = prop;
      
   #finish building single models
   if(main_prop == ""):
      printf("Something went wrong! No property selected to be the coordinator!");
      sys.exit(0);
 
   print("Begin federated learning cycle with coordinator {}".format(main_prop));
   sys.stdout.flush();
  
   for fed_epoch in range(FED_EPOCHS):
      W_prev = Models[main_prop][1]; 
      W_new = [];   

      for prop in props:
         if(prop == main_prop and fed_epoch == 0): 
            continue;

         Models[prop][1] = W_prev;
         #print("Training federatedly {}".format(prop));

         points = limits[prop]["mols"];
         vals = limits[prop]["vals"];
         inds = indexes[prop];

         cnt = len(points);

         x = np.zeros((cnt, nd), dtype=np.float32);      
         y = np.zeros((cnt, ), dtype=np.float32);
     
         for j in range(cnt):
            x[j] = descrs[ points[ int(inds[j]) ] ];
            y[j] = vals[ int(inds[j]) ];        

         #scaling to 0.1 - 0.9, add extra space to both ends

         add = 0.01 * (limits[prop]["max"] - limits[prop]["min"]);
         max_y = limits[prop]["max"] + add;
         min_y = limits[prop]["min"] - add;

         #print("Max: {}; Min: {}".format(max_y, min_y));

         for i in range(cnt):
            y[i] = 0.9 + 0.8 * (y[i] - max_y) / (max_y - min_y);
    
         #print("Dimensions: ", x.shape, y.shape);
         per = int(cnt * PER_TEST);     
 
         x_p = x[ : per ];
         x_t = x[ per : ];

         y_p = y[ : per ];
         y_t = y[ per : ];

         Wa, W3a = federate( Models[prop], x_t, y_t);         
         p = []; 
         calcPrognosis(p, Models[prop][4], x_p);

         W_new.append(W3a);

         #rescale back
         for i in range(len(p)):
            p[i] = (p[i] -0.9) / 0.8 * (max_y - min_y) + max_y;
            y_p[i] = (y_p[i] -0.9) / 0.8 * (max_y - min_y) + max_y;

         q2, rmsep = calcStatistics(p, y_p);
         print("Federated statistics {}: q2= {}, RMSEP= {}".format(prop, q2, rmsep));
         sys.stdout.flush();

      #one cycle
      upd = {};
      for j in range(len(W_new[0])):
         upd[j] = [];
         for i in range(len(W_new)):
            delta = W_new[i][j] - W_prev[j];
            upd[j].append(delta);

      for j in range(len(W_new[0])):
         u = np.average(upd[j], axis=(0));
         W_prev[j] = W_prev[j] + u;       

   #end of federated cycles
   for prop in props:
      fname = prop;
      fname = fname.replace("/","-");
      fname = fname.replace(" ","-");
   
      np.save("federated/" + fname, lasagne.layers.get_all_param_values(Models[prop][2]));


if __name__ == "__main__":
   main();


