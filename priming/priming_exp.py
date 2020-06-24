#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:15:55 2020

@author: danny
"""

import torch
import torch.nn as nn
import sys
import tables
import numpy as np

sys.path.append('/Users/sebastiaanscholten/Documents/speech2image-master/PyTorch/functions')
sys.path.append("/Users/sebastiaanscholten/Documents/speech2image-master/vgsexperiments/experiments")

import random
from encoders import img_encoder
from costum_layers import multi_attention
from data_split import split_data_flickr
from helper import print_file_list, split_data_objects
import pandas as pd
import json

data_loc = '/Users/sebastiaanscholten/Documents/speech2image-master/vgsexperiments/experiments/priming/features/priming_features_oldconfig.h5'
dtype = torch.FloatTensor
#
def check_word_occurence(testedwords,testset,results,jsonpath):
    path = jsonpath

    with open(path, "r") as json_file:
        data = json.load(json_file)
    returnedcorrectly = []
    for idx,result in enumerate(results):
        testword = testedwords[idx]._v_name.split("_",-1)[0]
        resultslist = []
        for res in result:
            filename = testset[res]._v_name.replace('flickr_', '')+".jpg"
            for word in data[filename]:
                found = False
                if word == testword:
                    resultslist.append(1)
                    found = True
                    break
            if found == False:
                resultslist.append(0)
        returnedcorrectly.append(resultslist)
    return returnedcorrectly, results

def cosine(emb_1, emb_2):
    return torch.matmul(emb_1, emb_2.t())

def results_at_n(embeddings_1, embeddings_2, n):
    results = []
    for index, emb in enumerate(embeddings_1):
        sim = cosine(emb, embeddings_2)
        sorted, indices = sim.sort(descending=True)
        results.append(indices[0:n])
    return results

def iterate_data(h5_file):
    for x in h5_file.root:
        yield x

def embed_images(imageiterator):
    # set to evaluation mode
    img_net.eval()
    for images in imageiterator:
        img = images
        img = torch.FloatTensor(img)
        img = img_net(img)
        try:
            image = torch.cat((image, img.data))
        except:
            image = img.data
    image_embeddings = image
    return image_embeddings

def only_iterate_images(f_nodes, batchsize, visual, shuffle=True):
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):
        # take a batch of nodes of the given size
        excerpt = f_nodes[start_idx:start_idx + batchsize]
        images = []
        for ex in excerpt:
            # extract and append the visual features
            images.append(eval('ex.' + visual + '._f_list_nodes()[0].read()'))
            # retrieve the audio features
            # padd to the given output size
        # reshape the features and recast as float64
        images = np.float64(images)
        yield images
def randomizer():

    return 2

def mini_batcher(nodes, batchsize, dtype = torch.FloatTensor, audio = 'mfcc',
                 max_len = 1024,randomprimes=False):

    if randomprimes:
        randomorder = random.sample(range(len(nodes)), len(nodes))

    for start_idx in range(0, len(nodes) - batchsize + 1, batchsize):
        # take a batch of nodes of the given size               
        excerpt = nodes[start_idx:start_idx + batchsize]
        filenames = []
        primes = []
        targets = []
        prime_l = []
        target_l = []
        for idx,ex in enumerate(excerpt):
            # retrieve the audio features

            if randomprimes:
                p = eval(f'nodes[{randomorder[idx]}].{audio}.prime.read().transpose()')
            else:
                p = eval(f'ex.{audio}.prime.read().transpose()')

            t = eval(f'ex.{audio}.target.read().transpose()')

            p_frames = p.shape[1]
            t_frames = t.shape[1]
            # pad to max_len if sentence is shorter
            if p_frames < max_len:
                p = np.pad(p, [(0, 0), (0, max_len - p_frames )], 'constant')
            if t_frames < max_len:
                t = np.pad(t, [(0, 0), (0, max_len - t_frames )], 'constant')

            # truncate to max_len if sentence is longer
            if t_frames > max_len:
                t = t[:,:max_len]
                t_frames = max_len
            if p_frames > max_len:
                p = p[:,:max_len]
                p_frames = max_len

            prime_l.append(p_frames)
            target_l.append(t_frames)
            primes.append(p)
            targets.append(t)
            filenames.append(ex._v_name)
        max_length = max(prime_l)
        # reshape the features and recast as float64
        primes = np.float64(primes)
        # truncate all padding to the length of the longest utterance
        primes = dtype(primes[:,:, :max_length])

        max_length = max(target_l)
        # reshape the features and recast as float64
        targets = np.float64(targets)
        # truncate all padding to the length of the longest utterance
        targets = dtype(targets[:,:, :max_length])


        yield primes, targets, prime_l, target_l,filenames

# priming encoder, which expects prime-target pairs
class priming_encoder(nn.Module):
    def __init__(self, config):
        super(priming_encoder, self).__init__()
        conv = config['conv']
        rnn = config['rnn']
        att = config ['att']
        self.max_len = rnn['max_len']
        self.Conv = nn.Conv1d(in_channels = conv['in_channels'],
                                  out_channels = conv['out_channels'],
                                  kernel_size = conv['kernel_size'],
                                  stride = conv['stride'],
                                  padding = conv['padding']
                                  )

        self.RNN = nn.GRU(input_size = rnn['input_size'],
                               hidden_size = rnn['hidden_size'],
                               num_layers = rnn['n_layers'],
                               batch_first = rnn['batch_first'],
                               bidirectional = rnn['bidirectional'],
                               dropout = rnn['dropout']
                               )

        self.att = multi_attention(in_size = att['in_size'],
                                   hidden_size = att['hidden_size'],
                                   n_heads = att['heads']
                                   )

    def forward(self, input, l):

        x = self.Conv(input).permute(0,2,1).contiguous()
        # correct the sequence length for subsampling
        cor = lambda l, ks, stride : int((l - (ks - stride)) / stride)
        l = [cor(y, self.Conv.kernel_size[0], self.Conv.stride[0]) for y in l]
        x, hx = self.apply_rnn(x, l)
        x = nn.functional.normalize(self.att(x), p=2, dim=1)
        return x
    # primed_encode expects the same input formats, but paired prime/target pairs
    # each with a seperate list indicate batch lengths in frames.
    def primed_encode(self, prime, target, prime_l, target_l):

        # correct the sequence lengths for subsampling
        cor = lambda l, ks, stride : int((l - (ks - stride)) / stride)
        prime_l = [cor(y, self.Conv.kernel_size[0], self.Conv.stride[0]) for y in prime_l]
        target_l = [cor(y, self.Conv.kernel_size[0], self.Conv.stride[0]) for y in target_l]

        # first encode the prime
        x = self.Conv(prime).permute(0,2,1).contiguous()
        x, hx = self.apply_rnn(x, prime_l)
        # encode the target
        x = self.Conv(target).permute(0,2,1).contiguous()
        # add appropriate prime hidden state as initial hidden state
        # for the target
        x, hx_ = self.apply_rnn(x, target_l, hx)

        # Pool the sequence using attention
        x = nn.functional.normalize(self.att(x), p=2, dim=1)
        return x
    # for the priming experiment, optionally give the encoder an initial
    # hidden state from the prime stimulus
    def apply_rnn(self, input, l, hx = None):
        input = nn.utils.rnn.pack_padded_sequence(input, l, batch_first = True,
                                                  enforce_sorted = False
                                                  )
        if hx is None:
            x, hx = self.RNN(input)
        else:
            x, hx = self.RNN(input, hx)

        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
        return x, hx


# create config dictionaries with all the parameters for your encoders
audio_config = {'conv':{'in_channels': 39, 'out_channels': 64,
                        'kernel_size': 6, 'stride': 2,'padding': 0,
                        'bias': False
                        },
                'rnn':{'input_size': 64, 'hidden_size': 1024,
                       'n_layers': 4, 'batch_first': True,
                       'bidirectional': True, 'dropout': 0,
                       'max_len': 1024
                       },

                'att':{'in_size': 2048, 'hidden_size': 128, 'heads': 1
                       }
                }
# calculate the required output size of the image encoder
out_size = audio_config['rnn']['hidden_size'] * 2 ** \
           audio_config['rnn']['bidirectional'] * audio_config['att']['heads']
image_config = {'linear':{'in_size': 2048, 'out_size': out_size},
                'norm': True
                }

img_net = img_encoder(image_config)
cap_net = priming_encoder(audio_config)

cap_state = torch.load('/Users/sebastiaanscholten/Documents/speech2image-master/PyTorch/flickr_audio/results/caption_model.32',map_location='cpu')
cap_net.load_state_dict(cap_state)

img_state = torch.load('/Users/sebastiaanscholten/Documents/speech2image-master/PyTorch/flickr_audio/results/image_model.32',map_location='cpu')
img_net.load_state_dict(img_state)

cap_net.eval()

for param in cap_net.parameters():
    param.requires_grad = False

data = tables.open_file(data_loc, mode='r+')
data_nodes = data.root._f_list_nodes()

batcher = mini_batcher(data_nodes, 5, dtype)

split_loc = "/Users/sebastiaanscholten/Documents/speech2image-master/preprocessing/testfolder2/test/dataset.json"
flickr_loc = "/Users/sebastiaanscholten/Documents/speech2image-master/preprocessing/prep_data/flickr_features_27jan_working_8000.h5"
flickr_file = tables.open_file(flickr_loc, mode="r+")
f_nodes_flickr = [node for node in iterate_data(flickr_file)]

train, val, test = split_data_flickr(f_nodes_flickr, split_loc)

#contains word caption information with images, needed for creating images to look for
textcp = pd.read_csv("/Users/sebastiaanscholten/Documents/speech2image-master/vgsexperiments/experiments/Results_isolated_word_recognition/documents/textcp.csv")
#words that are being tested
testlist = ["dog","man","boy","girl","woman","people"
            ,"dogs","shirt","child","ball","person"
            ,"children","men","girls","bike","rock","camera"
            ,"boys","hat","player","jacket","basketball","swing"
            ,"car", "wall", "hair","football","sunglasses","head"
            ,"shorts","dress","table","water","grass","bench","snow"
            ,"air","field","street","mouth","dirt","mountain","pool"
            ,"ocean","sand","building","soccer","park","face"]

#creates list of images to look for, which is the list of images that primes and targets are taken from
wavfiles,wavfilenames,word_appended_finallist,finallist,finallistdict = print_file_list(textcp,testlist)
#takes the chosen images out of the testset
images_test = split_data_objects(test,finallist)
#creates iterator for the images chosen
imageiterator = only_iterate_images(images_test,5,"resnet",shuffle=False)
#returns those embeddings
images = embed_images(imageiterator)

for prime, target, pl, tl, filename in batcher:
    primed = cap_net.primed_encode(prime, target, pl, tl)
    normal = cap_net(target, tl)

    try:
        primeresults = torch.cat((primeresults, primed))
    except:
        primeresults = primed

    try:
        targetresults = torch.cat((targetresults, normal))
    except:
        targetresults = normal



primes_precision_at_10 = results_at_n(primeresults,images,10)


targets_precision_at_10 = results_at_n(targetresults,images,10)

imagecaptiondict = "/Users/sebastiaanscholten/Documents/speech2image-master/vgsexperiments/experiments/data/imagecaptiontextdictionary.json"

resultsat10_primes, images_primes = check_word_occurence(data_nodes,images_test,primes_precision_at_10,imagecaptiondict)
resultsat10_targets, images_targets = check_word_occurence(data_nodes,images_test,targets_precision_at_10,imagecaptiondict)

wordlist = []

for idx, result in enumerate(resultsat10_primes):
    wordlist.append(data_nodes[idx]._v_name)
    print("For word: ", data_nodes[idx]._v_name.replace('flickr_', ''), "we retrieved: ", result, "\n")
    imagefilenames = []
    for res in images_primes[idx]:
        imagefilenames.append(images_test[res]._v_name.replace('flickr_', '') + ".jpg")
    images_primes[idx] = imagefilenames
resultsdf = pd.DataFrame(list(zip(wordlist, resultsat10_primes, images_primes)), columns=["Tested", "Results", "Files"])
resultsdf.to_csv("results_primes.csv", index=False)

wordlist = []
for idx, result in enumerate(resultsat10_targets):
    wordlist.append(data_nodes[idx]._v_name)
    print("For word: ", data_nodes[idx]._v_name.replace('flickr_', ''), "we retrieved: ", result, "\n")
    imagefilenames = []
    for res in images_targets[idx]:
        imagefilenames.append(images_test[res]._v_name.replace('flickr_', '') + ".jpg")
    images_targets[idx] = imagefilenames
resultsdf = pd.DataFrame(list(zip(wordlist, resultsat10_targets, images_targets)), columns=["Tested", "Results", "Files"])
resultsdf.to_csv("results_targets.csv", index=False)


