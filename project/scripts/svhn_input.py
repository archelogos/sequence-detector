# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import gzip
import os
import re
import sys
import tarfile
import h5py
import glob

import numpy as np
import tensorflow as tf
from scipy import ndimage
from scipy.misc import imresize
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from PIL import Image

SVHN_data = {}

# def maybe_download(filename, expected_bytes, force=False):
#   """Download a file if not present, and make sure it's the right size."""
#   downloaded = False
#   if force or not os.path.exists(DATA_FOLDER + filename):
#     print('Attempting to download:', filename)
#     filename, _ = urlretrieve(URL + filename, DATA_FOLDER + filename)
#     downloaded = True
#     print('\nDownload Complete!')
#   if downloaded:
#     statinfo = os.stat(filename)
#   else:
#     statinfo = os.stat(DATA_FOLDER + filename)
#   if statinfo.st_size == expected_bytes:
#     print('Found and verified', filename)
#   else:
#     raise Exception(
#       'Failed to verify ' + filename + '. Can you get to it with a browser?')
#   if downloaded:
#     return filename
#   else:
#     return DATA_FOLDER + filename
#
# def maybe_extract(filename, force=False):
#   root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
#   if os.path.isdir(root) and not force:
#     # You may override by setting force=True.
#     print('%s already present - Skipping extraction of %s.' % (root, filename))
#   else:
#     print('Extracting data for %s. This may take a while. Please wait.' % root)
#     # Extract tar in the folder where it is placed
#     sr = root.split('/')
#     sr1 = root.split('/')[0:len(sr)-1]
#     sr2 = "/".join(sr1)
#     tar = tarfile.open(filename)
#     sys.stdout.flush()
#     tar.extractall(sr2)
#     tar.close()
#     print('Completed!')
#   data_folders = root
#   print(data_folders)
#   return data_folders

def load_data_pickle(pickle_file):
  """
    Info
  """
  if os.path.isfile(pickle_file):
    try:
      with open(pickle_file, 'rb') as f:
        print('Loaded Pickle')
        save = pickle.load(f)
        SVHN_data = save
        del save  # hint to help gc free up memory
    except:
      print("Unexpected error:", sys.exc_info()[0])
      raise


### PREPROCESSING!!



# def label_zero(labels):
#   for label in labels:
#     #print(label)
#     L = label[0]
#     for i in range(1,L+1):
#       if label[i] == 10:
#         label[i] = 0
#
#   return labels
#
# def get_metadata(filename):
#   f = h5py.File(filename)
#
#   metadata= {}
#   metadata['height'] = []
#   metadata['label'] = []
#   metadata['left'] = []
#   metadata['top'] = []
#   metadata['width'] = []
#
#   def print_attrs(name, obj):
#     vals = []
#     if obj.shape[0] == 1:
#         vals.append(obj[0][0])
#     else:
#         for k in range(obj.shape[0]):
#             vals.append(f[obj[k][0]][0][0])
#     metadata[name].append(vals)
#
#   for item in f['/digitStruct/bbox']:
#       f[item[0]].visititems(print_attrs)
#   return metadata
#
# def image_processing(image, metadata):
#   original = Image.open(image)
#   L = len(metadata['label'])
#   aux = []
#   for i in range(L):
#     left = metadata['left'][i]
#     top = metadata['top'][i]
#     right = metadata['left'][i] + metadata['width'][i]
#     bottom = metadata['top'][i] + metadata['height'][i]
#     cropped = original.crop((left, top, right, bottom)) # crop with bbox data
#     pix = np.array(cropped)
#     pix_resized = imresize(pix, (IMAGE_SIZE,IMAGE_SIZE)) # resize each digit
#     pix_gs = np.dot(pix_resized[...,:3], [0.299, 0.587, 0.114]) # grayscale
#     aux.append(pix_gs)
#
#   sequence = np.hstack(aux) # horizontal stack
#   sequence_resized = imresize(sequence, (IMAGE_SIZE,IMAGE_SIZE)) # resize
#   sequence_resized = sequence_resized.reshape((IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)).astype(np.float32) # 1 channel
#   return sequence_resized
#
# def label_processing(metadata):
#   L = len(metadata['label'])
#   seq_labels = np.ones([N+1], dtype=int) * 10
#   seq_labels[0] = L # labels[i][0] = L to help the loss function. 6xLinearModels: s0...s5 and L
#   for i in range(1,L+1):
#     seq_labels[i] = metadata['label'][i-1]
#   return seq_labels
