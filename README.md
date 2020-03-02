# Transformer for Retrosynthesis
The repository contains the source code and trained models for the paper "A Transformer model for retrosynthesis" https://link.springer.com/chapter/10.1007/978-3-030-30493-5_78. The retrosynthesis task is treated as a machine translation problem where the source language is a molecule one wants to synthesize, and the target language is the set of reactants suitable for the synthesis.  

# Dependencies

The code has been tested within Linux Ubuntu and OpenSuse environments with the following versions of the major components:

1. python v.3.4.6 or higher
2. tensorflow v.1.12 
3. rdkit v.2018.09.2
4. python additional libraries: yaml, h5py, matplotlib, argparse, tqdm. 

# How to train models from scratch

To train a new model use the command:

python3 transformer.py --train=file_with_reactions 

The format of the file is as follows. Each line contains a single reaction with one product. The product is written first, then  all reactants and reagents. Sample files are located in the data subdirectory of the project. The model will be trained for one tousands of epochs with cyclic learning rate schedule (see the article). After training the weights at 600, 700, 800, 900, 999 epochs will be averaged and the final model will be stored in final.h5 file in the current directory. If you reaction dataset contains some not common elements then you have to increase model's vocabulary on line 53 (transformer.py). 

# Using the trained models 

To infer model prediction use the command:

python3 transformer.py --model=final.h5 --predict=file_with_products.smi --beam=5 --temperature=1.0

It is possible to apply greedy search setting the beam size to 1. From our experience there is no valuable differences to use beam size more than 5. 

# Retrain the models

To retrain a model for a particular reaction dataset use the command:

python3 transformer.py --retrain=original.h5 --train=file_with_reactions

The new model will be trained for 100 epochs with decreasing learning rate and last 10 epochs will be average for the fianl weights. Again the final model will be saved to final.h5 in the current directory. 


