import sys
import math
import numpy as np

#MonteCarlo Tress

chars = None;
rnn_generate = None;
rnn_before = None;
rnn_after = None;

class MonteCarloNode:
  
   def __init__(self, state, limit):
      
      self.state = state;

      self.nVisits = 0;
      self.Visits = 0;
      self.epsilon = 1e-5;
      self.limit = limit;
      self.cycle = 0;   

      self.children = [];

      if(state == '^'):
         self.start_chars = "$cnosBCFHINOPS";
         for i in range(len(self.start_chars)):
            self.children.append( MonteCarloNode( self.start_chars[i], self.limit * 2));


   def setSimulationFunction(self, fun_generate, fun_before, fun_after, vocab, frame):

      global chars;
      global rnn_generate;
      global rnn_before;
      global rnn_after;     

      if(chars is None):
         rnn_generate = fun_generate;
         rnn_before = fun_before;
         rnn_after = fun_after; 
         chars = vocab;
     
      self.cycle = frame;
 
   def expand(self):
      global chars;
      print("expanding");
      for i in range(len(chars)):
         self.children.append( MonteCarloNode( chars[i], self.limit * 2));
   
   def printTree(self):
      print(self.state, self.Visits, self.nVisits);
      for i in range(len(self.children)):
         self.children[i].printTree();

   def selectAction(self):

      vs = [];
      rnn_before();

      for it in range(10):

         visited = [];
         cur = self;
         x = "";

         if(cur.state != "^"):
            x = cur.state;
            visited.append(cur);


         while(cur.isLeaf() == False):
            cur = cur.select();
            visited.append(cur);
            x = x + cur.state;
      
         if(cur.state == "^" or cur.nVisits >= cur.limit):
            cur.expand();
            node = cur.select();
            x = x + str(node.state);

         vs.append(visited);
         self.simulate(x);

      rewards = rnn_after(self.cycle);

      for i in range(len(vs)):
         for child in vs[i]:
            child.update(rewards[i][0]);


   def select(self):      
      
      bv = -sys.float_info.max;
      selected = None;

      for child in self.children:
         nv = child.nVisits + child.epsilon;
         uct = child.Visits / nv + math.sqrt(math.log(child.nVisits + 1) / nv) +  \
               np.random.random(1) * self.epsilon;
         if(uct > bv):
            bv = uct;
            selected = child;

      if(selected == None):
         selected = self.children[int(np.random.random(1) * len(self.children))];
 
      return selected;

   def isLeaf(self):
      return len(self.children) == 0;

   def simulate(self, x):    
      rnn_generate(x);
   
   def update(self, reward):
      self.nVisits += 1;
      self.Visits += reward;


