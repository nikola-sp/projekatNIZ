
# coding: utf-8

# In[3]:

import numpy as np
import sys
import argparse
from seq2seqLSTM import translate
from seq2seqLSTM import learning
from seq2seqLSTM import save_model


# In[1]:

parser = argparse.ArgumentParser(description='Enlish to French translator.')
subparsers = parser.add_subparsers()
stranslate = subparsers.add_parser('translate', aliases=['t'])
stranslate.add_argument('text', help='English text file')
stranslate.set_defaults(which='translate')

learn = subparsers.add_parser('learn', aliases=['l'])
learn.add_argument('X', help='learning input (English)')
learn.add_argument('Y', help='learning output (French)')
learn.add_argument('-vs', '--vocab_size', help='vocabulary size (default=0 if vocab.txt should stay unchanged)', type=int, default=0)
learn.add_argument('-lcn', '--layer_cell_number', help='number of cells in a layer', type=int, default=1000)
learn.set_defaults(which='learn')

print("Reading input")
args=vars(parser.parse_args())

if args['which'] == 'translate':
    translate(args['text'])
else:
    print("Learning - train & test")
    model=learning(args['X'], args['Y'], args['vocab_size'], args['layer_cell_number'])
    print("Saving model")
    save_model(model)

