from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import gzip
import os
import re
import sys

import numpy as np
from scipy import ndimage
from scipy.misc import imresize
from six.moves import cPickle as pickle
from PIL import Image


N = 5
IMAGE_SIZE = 32
NUM_CHANNELS = 1
RTP_FOLDER = 'svhn/data'


def load_pickle(pickle_file):
	with open(pickle_file, 'rb') as f:
		save = pickle.load(f)
		metadata = save['metadata']
		del save  # hint to help gc free up memory
		return metadata

def rtp_data_processing(img_index):
	print('Processing %s.png' % img_index)

	image_proc = np.ndarray([IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS], dtype='float32')
	labels = np.ndarray([N+1], dtype=int)

	print('Getting Metadata')
	all_metadata = load_pickle(pickle_file=os.path.join(RTP_FOLDER, 'metadata.pickle'))

	print('Processing Images and Labels')
	image_file = os.path.join(RTP_FOLDER, img_index)+'.png'
	i = int(img_index)-1
	metadata = {}
	metadata['label'] = np.array(all_metadata['label'][i]).astype(int)
	metadata['height'] = np.array(all_metadata['height'][i]).astype(int)
	metadata['width'] = np.array(all_metadata['width'][i]).astype(int)
	metadata['top'] = np.array(all_metadata['top'][i]).astype(int)
	metadata['left'] = np.array(all_metadata['left'][i]).astype(int)
	L = len(metadata['label'])
	if L <= N:
		sequence = image_processing(image_file, metadata)
		seq_labels = label_processing(metadata)
		image_proc = sequence
		labels = seq_labels

	labels = label_zero(seq_labels)
	return image_file, image_proc, labels

def image_processing(image, metadata):
	original = Image.open(image)
	L = len(metadata['label'])
	aux = []
	for i in range(L):
		left = metadata['left'][i]
		top = metadata['top'][i]
		right = metadata['left'][i] + metadata['width'][i]
		bottom = metadata['top'][i] + metadata['height'][i]
		cropped = original.crop((left, top, right, bottom)) # crop with bbox data
		pix = np.array(cropped)
		pix_resized = imresize(pix, (IMAGE_SIZE,IMAGE_SIZE)) # resize each digit
		pix_gs = np.dot(pix_resized[...,:3], [0.299, 0.587, 0.114]) # grayscale
		aux.append(pix_gs)

	sequence = np.hstack(aux) # horizontal stack
	sequence_resized = imresize(sequence, (IMAGE_SIZE,IMAGE_SIZE)) # resize
	sequence_resized = sequence_resized.reshape((IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)).astype(np.float32) # 1 channel
	return sequence_resized


def label_processing(metadata):
	L = len(metadata['label'])
	seq_labels = np.ones([N+1], dtype=int) * 10
	seq_labels[0] = L # labels[i][0] = L to help the loss function. 6xLinearModels: s0...s5 and L
	for i in range(1,L+1):
		seq_labels[i] = metadata['label'][i-1]
	return seq_labels


def label_zero(labels):
	L = labels[0]
	for i in range(1,L+1):
		if labels[i] == 10:
			labels[i] = 0
	return labels
