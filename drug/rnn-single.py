
import sys
import numpy as np
import theano
import theano.tensor as T
import lasagne

from rdkit import Chem
import exposedlstm as ep
import utils as ut


in_text = open('mic-t.smi', 'r').readlines();
in_valid = open('mic-v.smi', 'r').readlines();

vocab = open('chars.txt', 'r').read()[0:-1];
chars = sorted((set(vocab)))

data_size, val_size, vocab_size = len(in_text), len(in_valid), len(chars)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

start_seq = char_to_ix["^"];
end_seq = char_to_ix["$"];

lasagne.random.set_rng(np.random.RandomState(1))

#maximum 98% of the entire chembl Database 
SEQ_LENGTH = 200 + 3

N_HIDDEN = 512
LEARNING_RATE = .01
GRAD_CLIP = 100
PRINT_FREQ = 1000
BATCH_SIZE = 64
NUM_EPOCHS = 150

def gen_data(data=in_text, return_target=True):

    data = data.replace("\n","");
    data = "^" + data + "$";

    sm_length = len(data);
    batch_size = sm_length;    

    #print(data, len(data), batch_size * SEQ_LENGTH * vocab_size);

    x = np.zeros((batch_size, SEQ_LENGTH, vocab_size), np.int16);
    y = np.zeros( batch_size, np.int16);
    m = np.zeros((batch_size, SEQ_LENGTH), np.int16);

    for n in range(batch_size):
        m[n, :n] = 1;

        for i in range(n):
            x[n,i,char_to_ix[data[i]]] = 1;
            #print(data[i], end="");

        #print(" => ", end="");
        if(return_target):
           y[n] = char_to_ix[data[n]];
           #print(data[n]);

    return x, y, m;


def try_it_out(N, probs):

   sample_ix = []
   x = np.zeros((1, SEQ_LENGTH, vocab_size), np.int16);
  
   x[0, 0, char_to_ix["^"]  ] = 1;
  
   for t in range(1, 200):
      m = np.zeros((1, SEQ_LENGTH), np.int16);
      m[0, :t] = 1;
      
      p = probs(x, m);
     
      p0 = p[..., :59];
      p1 = p[..., 59:];
 
      ix = np.random.choice(np.arange(vocab_size), p=p0.ravel())          
      sample_ix.append(ix)

      if(ix_to_char[ix] == "$"):        
        break;      

      x[0,t, sample_ix[-1]] = 1;
     
   random_snippet = "".join(ix_to_char[ix] for ix in sample_ix)    
   return [random_snippet, p1];

   '''
   sys.stdout.flush();

   ta = 0;
   cn = 0;

   for line in random_snippet.splitlines():
      m = Chem.MolFromSmiles(line);
      ta = ta + 1;
      if(m):
         cn = cn + 1;
       
   rate = round(cn * 100.0 / ta);
   print("Valid smiles: {}".format(rate));

   return rate;
   '''


def main(num_epochs=NUM_EPOCHS):
   
    print("Loading pretrained weights...");
    params = np.load("chembl-one.npy");

    print("Building network ...")
   
    l_in = lasagne.layers.InputLayer(shape=(None, None, vocab_size));
    l_mask = lasagne.layers.InputLayer(shape=(None, None));

    l_forward_1 = lasagne.layers.LSTMLayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh,
        mask_input = l_mask)

    l_forward_2 = ep.ExposedLSTMLayer(
        l_forward_1, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh,
        only_return_final=True,
        mask_input = l_mask)

    cell_out = lasagne.layers.SliceLayer(l_forward_2, indices=slice(0, N_HIDDEN), axis=-1)
    hid_out = lasagne.layers.SliceLayer(l_forward_2, indices=slice(N_HIDDEN, None), axis=-1)

    l_out = lasagne.layers.DenseLayer(hid_out, num_units=vocab_size, 
                                      W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)

    l_pred = lasagne.layers.ConcatLayer([l_out, cell_out]);

    target_values = T.vector('target_output', dtype='int32');

    network_output = lasagne.layers.get_output(l_out, deterministic = False)
    network = lasagne.layers.get_output(l_out, deterministic = True)
    network2 = lasagne.layers.get_output(l_pred, deterministic = True)

    cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()
    all_params = lasagne.layers.get_all_params(l_out,trainable=True)

    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)

    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values, l_mask.input_var], cost, updates=updates)
    compute_cost = theano.function([l_in.input_var, target_values, l_mask.input_var], cost)
 
    probs = theano.function([l_in.input_var, l_mask.input_var], network2);
       
    lasagne.layers.set_all_param_values(l_out, params);
   
    print("Training ...")

    try:
        for it in range(100, num_epochs):
 
            avg_cost = 0;
            avg_val = 0;          
           
            for p in range(data_size):
                 
                x,y,mask = gen_data( in_text[p] )
                
                for idx in ut.get_batch_idx(x.shape[0], BATCH_SIZE):            
                   avg_cost += train(x[idx], y[idx], mask[idx]);
                
                if (p % 1000 == 0):
                   smile, internal= try_it_out(100, probs);

                   screened = p * 100.0 / data_size;
                   print("Database screened: {} => {}".format(screened, smile));
                   sys.stdout.flush();

                   #np.save("/opt/pavel/chembl-{}-{}-{}".format(it, screened, rate), lasagne.layers.get_all_param_values(l_out))
                   #break;
                                           
            for v in range(val_size):
               x,y,mask = gen_data(in_valid[v]);
    
               for idx in ut.get_batch_idx(x.shape[0], BATCH_SIZE):
                  avg_val += compute_cost(x[idx], y[idx], mask[idx]);
               #break;  

                                     
            print("Epoch {} average training loss = {}, validation loss = {}".format(it, avg_cost, avg_val));
            np.save("chembl-{}".format(it), lasagne.layers.get_all_param_values(l_out))
            #break;
    
        np.save("chembl-rnn", lasagne.layers.get_all_param_values(l_out))

    except KeyboardInterrupt:
       np.save("chembl-rnn", lasagne.layers.get_all_param_values(l_out))
       pass

if __name__ == '__main__':
    main()

