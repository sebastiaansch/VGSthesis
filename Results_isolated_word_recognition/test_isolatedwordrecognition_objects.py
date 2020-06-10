
import os
import tables
import argparse
import torch
import sys
import json

sys.path.append('/Users/sebastiaanscholten/Documents/speech2image-master/PyTorch/functions')
sys.path.append('../')
sys.path.append('../Mytrainer')

from mytrainer import personaltrainer
from helper import create_noun_set, print_file_list, split_data_objects
import pandas as pd
from encoders import img_encoder, audio_rnn_encoder
from data_split import split_data_flickr





parser = argparse.ArgumentParser(description='Create and run an articulatory feature classification DNN')

# args concerning file location
parser.add_argument('-data_loc', type = str, default = '/Users/sebastiaanscholten/Documents/speech2image-master/vgsexperiments/experiments/Generating_Flickrwords_mfcc/mfcc/objects_49words_mfcc_features.h5')

parser.add_argument('-flickr_loc', type = str, default = '/Users/sebastiaanscholten/Documents/speech2image-master/preprocessing/prep_data/flickr_features_27jan_working_8000.h5',
                    help = 'location of the Flickr feature file, default: /prep_data/flickr_features.h5')

parser.add_argument('-split_loc', type=str,
                    default='/Users/sebastiaanscholten/Documents/speech2image-master/preprocessing/testfolder2/test/dataset.json',
                    help='location of the json file containing the data split information')
parser.add_argument('-results_loc', type=str,
                    default='/Users/sebastiaanscholten/Documents/speech2image-master/PyTorch/flickr_audio/results/',
                    help='location of the json file containing the data split information')
# args concerning training settings
parser.add_argument('-batch_size', type=int, default=10, help='batch size, default: 100')
parser.add_argument('-cuda', type=bool, default=False, help='use cuda, default: True')
# args concerning the database and which features to load
parser.add_argument('-visual', type=str, default='resnet',
                    help='name of the node containing the visual features, default: resnet')
parser.add_argument('-cap', type=str, default='mfcc',
                    help='name of the node containing the audio features, default: mfcc')
parser.add_argument('-gradient_clipping', type=bool, default=True, help='use gradient clipping, default: True')

args = parser.parse_args()

# create config dictionaries with all the parameters for your encoders

audio_config = {'conv': {'in_channels': 39, 'out_channels': 64, 'kernel_size': 6, 'stride': 2,
                         'padding': 0, 'bias': False}, 'rnn': {'input_size': 64, 'hidden_size': 1024,
                                                               'num_layers': 4, 'batch_first': True,
                                                               'bidirectional': True, 'dropout': 0},
                'att': {'in_size': 2048, 'hidden_size': 128, 'heads': 1}}
# automatically adapt the image encoder output size to the size of the caption encoder
out_size = audio_config['rnn']['hidden_size'] * 2 ** audio_config['rnn']['bidirectional'] * audio_config['att']['heads']
image_config = {'linear': {'in_size': 2048, 'out_size': out_size}, 'norm': True}

# open the data file
data_file = tables.open_file(args.data_loc, mode='r+')
flickr_file = tables.open_file(args.flickr_loc, mode="r+")
# check if cuda is available and user wants to run on gpu
cuda = args.cuda and torch.cuda.is_available()
if cuda:
    print('using gpu')
else:
    print('using cpu')


# flickr doesnt need to be split at the root node
def iterate_data(h5_file):
    for x in h5_file.root:
        yield x


f_nodes_mfcc = [node for node in iterate_data(data_file)]

f_nodes_flickr = [node for node in iterate_data(flickr_file)]

# split the database into train test and validation sets. default settings uses the json file
# with the karpathy split
train, val, test = split_data_flickr(f_nodes_flickr, args.split_loc)

textcp = pd.read_csv("/Users/sebastiaanscholten/Documents/speech2image-master/vgsexperiments/experiments/Results_isolated_word_recognition/documents/textcp.csv")

testlist = ["dog","man","boy","girl","woman","people"
            ,"dogs","shirt","child","ball","person"
            ,"children","men","girls","bike","rock","camera"
            ,"boys","hat","player","jacket","basketball","swing"
            ,"car", "wall", "hair","football","sunglasses","head"
            ,"shorts","dress","table","water","grass","bench","snow"
            ,"air","field","street","mouth","dirt","mountain","pool"
            ,"ocean","sand","building","soccer","park","face"]

wavfiles,wavfilenames,word_appended_finallist,finallist,finallistdict = print_file_list(textcp,testlist)

mfcc_test = f_nodes_mfcc
images_test = split_data_objects(test,finallist)


#####################################################

# network modules
img_net = img_encoder(image_config)
cap_net = audio_rnn_encoder(audio_config)

# list all the trained model parameters
models = os.listdir(args.results_loc)
caption_models = [x for x in models if 'caption' in x]
img_models = [x for x in models if 'image' in x]

# run the image and caption retrieval
img_models.sort()
caption_models.sort()

# create a trainer with just the evaluator for the purpose of testing a pretrained model
trainer = personaltrainer(img_net, cap_net, args.visual, args.cap)
trainer.set_only_audio_batcher()
trainer.set_only_image_batcher()
# optionally use cuda

if cuda:
    trainer.set_cuda()
trainer.set_evaluator([1, 5, 10])

jsonpath = "/Users/sebastiaanscholten/Documents/speech2image-master/vgsexperiments/experiments/data/imagecaptiontextdictionary.json"
for img, cap in zip(img_models, caption_models):
    epoch = img.split('.')[1]
    # load the pretrained embedders
    trainer.load_cap_embedder(args.results_loc + cap)
    trainer.load_img_embedder(args.results_loc + img)

    # calculate the recall@n
    trainer.set_epoch(epoch)

    resultsat10, whichimages = trainer.word_precision_at_n(mfcc_test, images_test, jsonpath, 10, args.batch_size)

    wordlist = []
    for idx,result in enumerate(resultsat10):
        wordlist.append(mfcc_test[idx]._v_name)
        print("For word: ", mfcc_test[idx]._v_name.replace('flickr_', ''), "we retrieved: ", result, "\n")
        imagefilenames = []
        for res in whichimages[idx]:
            imagefilenames.append(images_test[res]._v_name.replace('flickr_', '')+".jpg")
        whichimages[idx] = imagefilenames
    resultsdf = pd.DataFrame(list(zip(wordlist, resultsat10, whichimages)),columns = ["Tested","Results","Files"])
    resultsdf.to_csv("results_forcedalignment_49words.csv",index=False)



    print("done!")
    print("okay")
