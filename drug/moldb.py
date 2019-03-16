
import random
import sqlite3
import numpy as np

class MolDb: 

   def __init__(self):
  
      print("Preparing the database...");

      self.conn = sqlite3.connect('screening.db');
      self.cur = self.conn.cursor();

      
      self.cur.execute("create table if not exists mols(id integer primary key, " 
                       "canon text, smile text, cycle integer, hits integer, score real, tv real, rmse real)");
    
      self.cur.execute("create index idx_canon on mols(canon)");

      #table for structures that were generated but didn't pass filters
      self.cur.execute("create table if not exists alls(smile text unique)");
      self.cur.execute("create table if not exists epochs(id integer primary key, cycle integer, val real)"); 

      #clean   all data from previous runs
      #self.cur.execute("delete from mols");
      self.cur.execute("delete from alls");
      self.cur.execute("delete from epochs");
      

      print("The database is ready.");

   #return 1 if the molecule in our active dataset 
   #return 2 if the molecule in our decoys
   #return 0 if there is no such molecule in the database 

   def search(self, canon, updateHits = True):
      #actives
      self.cur.execute("select count(*) from mols where canon = ?", (canon, ));
      row = self.cur.fetchone();
      if(row[0] != 0):
         
         if(updateHits == True):
            self.cur.execute("select id, score, hits from mols where canon = ?", (canon, ));
            row = self.cur.fetchone();

            idx = row[0];
            score = row[1];
            hits = row[2];

            #increment hit counter 
            self.cur.execute("update mols set hits = hits + 1 where id = ?", (idx, ));
            return 1, score, hits;
      
         return 1, None, None;
 
      #decoys
      self.cur.execute("select count(*) from alls where smile= ?", (canon, ));
      row = self.cur.fetchone();
      if(row[0] != 0):
         return 2, None, None;

      return 0, None, None;
 
   def addMol(self, canon, smile, active = True, cycle = 0, score = 0, rmse= -1.0):
      if(active == True):
         self.cur.execute("insert into mols(canon, smile, cycle, hits, score, rmse) values(?,?,?,0,?,?)", (canon, smile, cycle, score, rmse, ));
      else:
         self.cur.execute("insert into alls(smile) values(?)", (canon, ));        
      return self.cur.lastrowid;

   def updateLearning(self, cycle, vals):
      for i in vals:
         self.cur.execute("insert into epochs(cycle, val) values(?,?)", (cycle, i, ));

   def averageTarget(self, cycle):
      self.cur.execute("select avg(tv) from mols where cycle = ?", (cycle, ));
      row = self.cur.fetchone();
      return row[0]; 

   def updateTargetValue(self, idx, val):
      self.cur.execute("update mols set tv = ? where id = ?", (val, idx, ));

   def selectMols(self):
      #first original dataset
      self.cur.execute("select smile from mols where cycle = 0"); 
      rows = self.cur.fetchall();
      
      cnt_original = len(rows);
      data = [];
      for i in rows:
         data.append(i[0]);

      #select the same number of smiles from later epochs randomly
      #how many additional data we have?
      self.cur.execute("select count(*) from mols where cycle > 0");
      rows = self.cur.fetchone();
      
      cnt_new = rows[0];

      print("The number of generated smiles: {}".format(cnt_new));
      
      if(cnt_new <= cnt_original):
         #select all additional data
         self.cur.execute("select smile from mols where cycle > 0");
         rows = self.cur.fetchall();
         for i in rows:
            data.append(i[0]);
      else:
         #randomly choose cnt_real records
         indexes = np.arange(0, cnt_new, 1, np.int32); 
         np.random.shuffle(indexes);
         indexes = indexes[:cnt_original].tolist();

         self.cur.execute("select smile from mols where cycle > 0");
         row = self.cur.fetchone();

         currow = 0;
         while( row is not None):
            if( currow in indexes):
               data.append(row[0]);
            row = self.cur.fetchone();
            currow += 1;
      
      random.shuffle(data);
      return data;


   def __del__(self):

      print("Committing the work...");
      self.conn.commit();
      self.conn.close();
   
 
