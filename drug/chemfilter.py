
from subprocess import Popen, PIPE

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem, rdReducedGraphs
from rdkit import DataStructs

import numpy as np
import os
import os.path
import math

import theano
import theano.tensor as T
from theano import shared
import lasagne

import sascore

def checkSmile(smile):
   canon = "";
   m = Chem.MolFromSmiles(smile);        
   if(m is not None):
      canon = Chem.MolToSmiles(m);
   return m, canon;   

def run_external(cmd):
 
   proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
   out, err = proc.communicate()
   exitcode = proc.returncode
      
   return exitcode, out, err

class BasicFilter:   
   def __init__(self, name):
      self.name = name;

      self.score = 0.0;
      self.value = 0.0;
 
   def calcScore(self, smiles):
      return True;

#all compounds with logp < 9 are good and we will
#process them further, otherwise stop
class LogPFilter(BasicFilter): 
   
   def __init__(self):
      BasicFilter.__init__(self, "LogP");

   def calcScore(self, m):

      self.score = 0.0;      
      self.value = Descriptors.MolLogP(m);
      if(self.value <= 4):
         self.score = 0.1;
         return True;  
      
      return False;
   
class NAtomsFilter(BasicFilter): 
   
   def __init__(self):
      BasicFilter.__init__(self, "NAtoms");

   def calcScore(self, m):

      self.score = 0.0;      
      self.value = m.GetNumHeavyAtoms();
      if(self.value < 14):
         self.score = 0.1;
         return True;  
      
      return False;
      
class MwFilter(BasicFilter): 
   
   def __init__(self):
      BasicFilter.__init__(self, "MW");

   def calcScore(self, m):

      self.score = 0.0;      
      self.value = Descriptors.MolWt(m);
      if(self.value <= 300):
         self.score = 0.1;
         return True;

      return False;


class SaFilter(BasicFilter): 
   
   def __init__(self):
      BasicFilter.__init__(self, "SA");

   def calcScore(self, m):

      self.score = 0.0;
      sa = sascore.calculateScore(m);      
      if(sa >= 2.0 and sa <= 6.0):
         self.score = (1.0 + (sa - 6.0) / 4.0);
         self.value = sa;
         return True;

      return False;
 
class SaLightFilter(BasicFilter): 
   
   def __init__(self):
      BasicFilter.__init__(self, "SA");

   def calcScore(self, m):

      self.score = 0.0;
      sa = sascore.calculateScore(m);      
      if(sa <= 5.0 and sa > 1.5):
         self.score = 1.0 - (5.0 - sa) / 3.5;
         self.value = sa;
         return True;

      return False;

class OChemFilter(BasicFilter):

   def __init__(self):
      BasicFilter.__init__(self, "OChem");
      self.OChemThreshold = 5;

   def setThreshold(self, newThreshold):
      self.OChemThreshold = newThreshold;

   def calcScore(self, m):
 
      if(os.path.exists("temp/res.csv")): 
         os.remove("temp/res.csv");                        

 
      print("Start ochem");
      os.chdir("../ochem-predictor");
      exitcode, out, err = run_external(["./predict", "--file=../work/temp/mol.sdf", "--out=../work/temp/res.csv", "--model=mic"]);
 
      res = [];

      #print(out, err, exitcode);
      if(exitcode == 0):                 
         
         fp = open("../work/temp/res.csv", "r");
         d = fp.read().split("\n", -1);
         fp.close();          

         for i in range(1, len(d)):
            dd = d[i].split(",",-1);

            print(dd);
           
            try:
               vs = -1 * float(dd[2]) ;
               rmse = float(dd[5]);
 
               self.score = vs - 5.0;                  
               self.value = vs;

               res.append((self.score, self.value, rmse));

            except:
               self.score = -1.0;
               self.value = 0.0;
               res.append((self.score, self.value, -1.0));

      else:    
         print(out, err);

      print("Finish ochem");
      os.chdir("../work/");
      return res;

 
class VinaFilter(BasicFilter):

   def __init__(self):
      BasicFilter.__init__(self, "Vina");
      self.VinaThreshold = -9.5;

   def setThreshold(self, newThreshold):
      self.VinaThreshold = newThreshold;

   def calcScore(self, m):
      if(AllChem.EmbedMolecule(m,useRandomCoords=True) == 0):
         AllChem.MMFFOptimizeMolecule(m)
         if(m):
             
             writer = Chem.SDWriter('temp/mol.sdf')
             writer.write(m);
             writer.close();
  
             exitcode, out, err = run_external(["data/pkr1-rnn", "mol.sdf"]);
             if(exitcode == 0):

                 try:
                    out = str(out);
                    out = out.replace("b'", "");
                    out = out.replace("\\n'", "");
                   
                    print("vina output: {}".format(out));

                    vs = float(out);
                    if(vs <= self.VinaThreshold):
                       self.score = (1.0 + (vs + 15.0) / (-15.0 + self.VinaThreshold + 0.00001));          
                       self.value = vs;
                       return True;
                 except:
                    print("VINA failed.");

      return False;


class MicFilter(BasicFilter): 
   
   def __init__(self):
      BasicFilter.__init__(self, "MIC");
     
      self.threshold = 3.0; 
      self.NFeatures = 2363;
      self.prognosis = [];

      print("Starting load Lasagne prognosis model...");

      #load all models from cross-valiation
      self.prognosis.append(self.buildModel( "work/062/mic-0.mdl.npy"));
      self.prognosis.append(self.buildModel( "work/062/mic-1.mdl.npy"));
      self.prognosis.append(self.buildModel( "work/062/mic-2.mdl.npy"));
      self.prognosis.append(self.buildModel( "work/062/mic-3.mdl.npy"));
      self.prognosis.append(self.buildModel( "work/062/mic-4.mdl.npy"));

   def setThreshold(self, newThreshold):
      self.threshold = newThreshold;

   def buildModel(self, fname):

      print("Loading Lasagne model {}".format(fname));

      X = T.matrix('inputs', dtype='float32');
      Target_Values = T.matrix('targets', dtype='float32');

      l_in = lasagne.layers.InputLayer(shape=(None, self.NFeatures));
      l_1 = lasagne.layers.DenseLayer(l_in, 512, nonlinearity = lasagne.nonlinearities.sigmoid);

      l_out = lasagne.layers.DenseLayer(l_1, 1, nonlinearity = lasagne.nonlinearities.sigmoid);

      prediction = lasagne.layers.get_output(l_out, X, deterministic=True);
      pred = theano.function([X], prediction);

      weights = np.load(fname);
      lasagne.layers.set_all_param_values(l_out, weights);

      return pred;   

   def CalculateDescriptors(self, m):

      arr = np.zeros((1,0)).astype(np.float32);
      fp0 = AllChem.GetMorganFingerprintAsBitVect(m, 2);
      DataStructs.ConvertToNumpyArray(fp0, arr);

      fp1 = rdReducedGraphs.GetErGFingerprint(m);
      arr = np.append(arr, fp1);

      return arr[np.newaxis, ...].astype(np.float32);
   
   def scale(self, y):
      return 10.0 + 9.0 / 0.8 * (y - 0.9);

   def calcScore(self, m):

      descr = self.CalculateDescriptors(m);
      r = np.zeros(5, dtype=np.float32);

      i = 0;
      for pred in self.prognosis:
         r[i] = pred(descr);
         i += 1;

      v = self.scale(np.mean(r));

      self.score = (v - self.threshold) / 10.0;
      self.value = v;

      return True;

