import sys

import pandas as pd

sys.path.append('/Users/sebastiaanscholten/Documents/speech2image-master/PyTorch/functions')
from trainer import flickr_trainer
from evaluate import evaluate
from minibatchers import iterate_audio
import numpy as np
import json
import tables


import torch
class evaluating(evaluate):
    def results_at_1(self):
        embeddings_1 = self.caption_embeddings
        embeddings_2 = self.image_embeddings
        results = []
        for index, emb in enumerate(embeddings_1):
            sim = self.dist(emb, embeddings_2)
            idx = torch.argmax(sim)
            results.append(idx)
        return results
    def results_at_n(self,n):
        embeddings_1 = self.caption_embeddings
        embeddings_2 = self.image_embeddings
        results = []
        for index, emb in enumerate(embeddings_1):
            sim = self.dist(emb, embeddings_2)
            sorted, indices = sim.sort(descending = True)
            results.append(indices[0:n])
        return results
    def return_word_embeddings(self,speechdata):
        embeddings_1 = self.caption_embeddings
        wordembeddings = pd.DataFrame()
        # data_file = tables.open_file("embeddingstest.h5", mode='a')
        # append_name = 'flickr_'

        # for x, emb in zip(speechdata,embeddings_1):
        #
        #     # one group for each image file which will contain its vgg16 features and audio captions
        #     node = data_file.create_group("/", x._v_name)
        #     embeddingnode = data_file.create_group(node, "embedding")
        #     emb = np.array(emb)
        #     shape = emb.shape[-1]
        #     embedding = data_file.create_earray(embeddingnode, 'emb', tables.Float32Atom(),obj=np.array())
        #     embedding.append(emb)
        # data_file.close()
        # node_list = data_file.root._f_list_nodes()
        #

        for ex, emb in zip(speechdata,embeddings_1):
            embfile = ex._v_name
            emb = np.array(emb).astype('float64')
            wordembeddings = wordembeddings.append({"File": embfile,"Embedding":emb},ignore_index=True)


        return wordembeddings



    def sep_embed_data(self, speechiterator,imageiterator):
        # set to evaluation mode
        self.embed_function_1.eval()
        self.embed_function_2.eval()

        for speech in speechiterator:
            cap, lengths = speech
            sort = np.argsort(- np.array(lengths))
            cap = cap[sort]
            lens = np.array(lengths)[sort]
            cap = self.dtype(cap)
            cap = self.embed_function_2(cap, lens)
            cap = cap[torch.LongTensor(np.argsort(sort))] #used to be: cap = cap[torch.cuda.LongTensor(np.argsort(sort))]
            try:
                caption = torch.cat((caption, cap.data))
            except:
                caption = cap.data
        for images in imageiterator:
            img = images
            img = img[sort]
            img = self.dtype(img)
            img = self.embed_function_1(img)
            img = img[torch.LongTensor(np.argsort(sort))]
            try:
                image = torch.cat((image, img.data))
            except:
                image = img.data
        self.image_embeddings = image
        self.caption_embeddings = caption

class personaltrainer(flickr_trainer):
    def only_audio_batcher(self, data, batch_size, shuffle):
        return only_iterate_audio(data,batch_size,self.cap,shuffle)
    def only_image_batcher(self, data, batch_size, shuffle):
        return only_iterate_images(data,batch_size,self.vis,shuffle)

    def set_evaluator(self, n):
        self.evaluator = evaluating(self.dtype, self.img_embedder, self.cap_embedder)
        self.evaluator.set_n(n)
    def set_only_audio_batcher(self):
        self.audiobatcher = self.only_audio_batcher
    def set_only_image_batcher(self):
        self.imagebatcher = self.only_image_batcher

    def retrieve_best_image(self,speechdata,imgdata,batch_size):
        speechiterator = self.audiobatcher(speechdata, 5,shuffle=False)
        imageiterator = self.imagebatcher(imgdata, 5, shuffle=False)

        self.evaluator.sep_embed_data(speechiterator,imageiterator)
        return self.evaluator.results_at_1()
    def word_precision_at_n(self, speechdata, imgdata, path, n, batch_size):
        speechiterator = self.audiobatcher(speechdata, 5,shuffle=False)
        imageiterator = self.imagebatcher(imgdata, 5, shuffle=False)

        self.evaluator.sep_embed_data(speechiterator,imageiterator)
        results = self.evaluator.results_at_n(n)
        self.evaluator.return_word_embeddings(speechdata)
        returnedcorrectly, results = check_word_occurence(speechdata,imgdata,results,path)
        wordembeddings = self.evaluator.return_word_embeddings(speechdata)

        return returnedcorrectly, results, wordembeddings
    def word_precision_at_n_phonemes(self, speechdata, imgdata, path, n, batch_size):
        speechiterator = self.audiobatcher(speechdata, 5,shuffle=False)
        imageiterator = self.imagebatcher(imgdata, 5, shuffle=False)

        self.evaluator.sep_embed_data(speechiterator,imageiterator)
        results = self.evaluator.results_at_n(n)
        return check_word_occurence_phonemes(speechdata,imgdata,results,path)

def only_iterate_audio(f_nodes, batchsize, audio, shuffle=True):
    frames = 2048
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):
        # take a batch of nodes of the given size
        excerpt = f_nodes[start_idx:start_idx + batchsize]
        speech = []
        lengths = []
        for ex in excerpt:
            # extract and append the visual features
            # retrieve the audio features
            sp = eval('ex.' + audio + '._f_list_nodes()[0].read().transpose()')
            # padd to the given output size
            n_frames = sp.shape[1]
            if n_frames < frames:
                sp = np.pad(sp, [(0, 0), (0, frames - n_frames)], 'constant')
            # truncate to the given input size
            if n_frames > frames:
                sp = sp[:, :frames]
                n_frames = frames
            lengths.append(n_frames)
            speech.append(sp)

        max_length = max(lengths)
        # reshape the features and recast as float64
        speech = np.float64(speech)
        # truncate all padding to the length of the longest utterance
        if max_length < 7:
            speech = speech[:, :, :6]
        else:
            speech = speech[:, :, :max_length]

        # reshape the features into appropriate shape and recast as float32
        speech = np.float64(speech)
        yield speech, lengths

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

def check_word_occurence(testedwords,testset,results,jsonpath):
    path = jsonpath

    with open(path, "r") as json_file:
        data = json.load(json_file)
    returnedcorrectly = []
    for idx,result in enumerate(results):
        testword = testedwords[idx]._v_name.replace('flickr_', '').split("_",-1)[0]
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



def check_word_occurence_phonemes(testedwords,testset,results,jsonpath):
    path = jsonpath

    with open(path, "r") as json_file:
        data = json.load(json_file)
    returnedcorrectly = []
    for idx,result in enumerate(results):
        testword = testedwords[idx]._v_name.replace('flickr_', '').split("_",-1)[2]
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