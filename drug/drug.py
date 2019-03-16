
''' 
   Drug Search program: recurrent neural networks, 
   monte-carlo search trees and molecular docking.


   Helmholtz-Zentrum Munchen, STB
   P. Karpov, I. Tetko, 2018

   carpovpv@gmail.com
'''

import numpy as np
import sys as sys
import math as math

import chemfilter as cf
import rnnmodel as rn
import montecarlo as mc
import moldb 

from rdkit import Chem


def main():
 
   print("Starting the program.");

   ##!!!
   Pretrained = "../data/chembl-one.npy";
   ActiveMols = "../data/mic.smi";

   print("Loading active molecules: ", ActiveMols);

   ##!!!
   StartTemperature = 0.8;
   TemperatureStep = 0.05;
  
   print("Loading the vocabulary.");
   chars = open('../data/chars.txt', 'r').read()[0:-1];   
   chars = sorted(set(chars))

   print("Loading and building RNN functions.");
   rnn = rn.RnnModel(chars, Pretrained);

   print("Loading filters.");
   filters = [];

   filters.append( cf.NAtomsFilter() );
   #filters.append( cf.SaLightFilter() );
   filters.append( cf.MwFilter() );
   filters.append( cf.LogPFilter() );
   filters.append( cf.SaFilter() );

   bm = cf.OChemFilter();

   #filters.append( cf.MicFilter() );
   #filters.append( cf.OChemFilter() );
   #filters.append( cf.VinaFilter() );

   #moving threshold

   tv = 4.0;
   bm.setThreshold(tv);

   print("Loading and preparing the database.");
   db = moldb.MolDb();  

   cnt = 0;

   #load our actives
   #dont check uniqness. This allows to increase the training dataset.

   with open(ActiveMols) as f:
      lines = f.readlines();
      for line in lines:
         m, canon = cf.checkSmile(line);
         if(m is not None):
            db.addMol(canon, line);    
            cnt += 1;

   print("Loaded {} molecules.".format(cnt));

   temp_mols = [];

   def rnn_before():
      temp_mols.clear();

   def rnn_after(cycle):
      
      rewards = [];

      print(temp_mols);

      cnt = 0;
      writer = Chem.SDWriter('temp/mol.sdf')
      for ik in range(len(temp_mols)):
         if(temp_mols[ik][0] == True):
            try:
               writer.write(Chem.MolFromSmiles(temp_mols[ik][1]));
               cnt += 1;
            except:
               print("Write exception: ", temp_mols[ik][1]);
               temp_mols[ik][0] = False;

      writer.close();

      if(cnt == 0):
         for k in temp_mols:
            rewards.append((0,0));   
         return rewards;   
      
      res = bm.calcScore(None);

      print(res);
      print(len(res));

      i = 0;
      for k in temp_mols:
         if(k[0] == True):
            score = k[2] + res[i][0];

            rmse = -1.0;
            if(len(res[i]) ==3):
               rmse = res[i][2];

            print("I = ", i, score, res[i][1]);
            rewards.append((score, res[i][1]) );
                      
            if(score > 0.7):
               idx = db.addMol(k[1], k[3], True, cycle, score, rmse);
               db.updateTargetValue(idx, res[i][1]);
            else:
               #prevent UNIQUE constraint failure
               try:
                  idx = db.addMol(k[1], k[3], False);
               except:
                  print("Unique constraint");

            i += 1;
         else:
            rewards.append( (k[2], 0.0 ));

      return rewards;


   #apply state in the MC workflow
   #the function returns the score 
   def rnn_generate(x, N = 10):
      bait = "^";
      bait += x;
      x = bait;
   
      score = 0.0;
      raw = [];

      for i in range(N):
         raw.append(rnn.generate(x));
 
      print("Generated: ", raw);
 
      added = False;     

      #applying filters
      for line in raw:
         m, canon = cf.checkSmile(line);
         if(m is not None): 

            added = True;

            if(canon in temp_mols): 
               temp_mols.append([False, canon, 0.0, line]);
               return;

            idx, oldscore, oldhits = db.search(canon);
   
            #The structure has been generated before, but it is not active.
            if(idx == 2): 
               temp_mols.append([False, canon, 0.0, line]);
               return; 
  
            #The structure was generated before and it is active. But we are not
            #interested in the same molecules so decrese the score by some decay factor.
            if(idx == 1):
               temp_mols.append([False, canon, oldscore - 0.01 * oldhits, line]);
               return;

            allow = True;
            score = 0.0;
       
            try: 
               for fl in filters:
                  if(fl.calcScore(m) == False):
                     allow = False;
                     break;
                  else:    
                     score += fl.score;
            except:
               print("Exception. Continue.", sys.exc_info()[0]);

            temp_mols.append([allow, canon, score, line]);

      #No valid SMILES found.

      if(added == False):
         temp_mols.append([False, "", 0.0, line]);

   #end rnn_generate_function



   #loop: training monte-carlo -> new data
   max_cycle = 100;
   mtree_max = 100;

   ##!!!
   for cycle in range(1, max_cycle + 1):

      print("=======CYCLE {} =======".format(cycle));
     
      
      pretrain_without_improvement = 0;
      max_trials = 10;
      '''           
      for pretrain in range(1,max_trials):
   
         data = db.selectMols();
         vals = rnn.fineTune(data);

         id_min = np.argmin(vals);
         if(id_min == 0):
            pretrain_without_improvement += 1;
            print("Nothing improved after training. Step: {}".format(pretrain_without_improvement));            
         else:
            break;
     
      sys.stdout.flush();

      if(pretrain_without_improvement >= max_trials):
         print("Stop training. No improvement.");
         break;

      #update only last values 
      db.updateLearning(cycle, vals);
      
      #save weights each cycle
      rnn.save("weights/weights-{}".format(cycle));
      
      
      #increase the temperature each cycle      
      rnn.setTemperature(StartTemperature + TemperatureStep * (cycle - 1));
      '''
      print("Start Monte Carlo Tree search");

      mtree = mc.MonteCarloNode('^', 5);   
      mtree.setSimulationFunction(rnn_generate, rnn_before, rnn_after, chars, cycle);   
      for i in range(mtree_max):
         mtree.selectAction();
         print("MTREE ", i);
         sys.stdout.flush();
         #mtree.printTree();
      
      #define new threshold
      nt = db.averageTarget(cycle);
      print("Average value for cycle {} is {}".format(cycle, nt));
         
      if(tv < 8):
         tv += 0.2;
      else:
         #increase target value, but not much...
         tv += 0.2 / cycle;
      
      sys.stdout.flush();

   rnn.save("weights/weights-end");
   print("Finish procedure. Relax!");


if __name__ == "__main__" :
   main();


