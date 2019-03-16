import sys
import h5py
import numpy as np

f = [];

for i in range(39,49,1):
   f.append(h5py.File("16-64-512-8-2/retrosynthesis-" + str(i) + ".h5", "r+"));

keys = list(f[0].keys());
for key in keys:
   groups = list(f[0][key]);
   if len(groups):
      for group in groups:
         items = list(f[0][key][group].keys());
         for item in items:
            data = [];
            for i in range(len(f)):
               data.append(f[i][key][group][item]);           
            avg = np.mean(data, axis = 0);
            del f[0][key][group][item];
            f[0][key][group].create_dataset(item, data=avg);
            

for fp in f:
   fp.close();


