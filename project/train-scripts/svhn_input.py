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

import numpy as np
from six.moves import cPickle as pickle

SVHN_data = {}

def load_data_pickle(pickle_file):
  """Load data from a pickle file"""
  if os.path.isfile(pickle_file):
    try:
      with open(pickle_file, 'rb') as f:
        print('Loaded Pickle')
        save = pickle.load(f)
        SVHN_data['train_dataset'] = save['train_dataset']
        SVHN_data['train_labels'] = save['train_labels']
        SVHN_data['valid_dataset'] = save['valid_dataset']
        SVHN_data['valid_labels'] = save['valid_labels']
        SVHN_data['test_dataset'] = save['test_dataset']
        SVHN_data['test_labels'] = save['test_labels']
        del save  # hint to help gc free up memory
    except:
      print("Unexpected error:", sys.exc_info()[0])
      raise
