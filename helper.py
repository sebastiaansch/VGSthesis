import pandas as pd
import numpy as np
import os
from nltk.tokenize import RegexpTokenizer
from os import path
from pydub import AudioSegment
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer

def tokenize(path):
    text = pd.read_csv(path, sep="\t")
    text = text.iloc[:, 1]
    tokenizer = RegexpTokenizer(r'\w+')
    wordlist = []
    for sentence in text:
        sentence = sentence.lower()
        words = tokenizer.tokenize(sentence)
        wordlist.extend(words)
    wordlist = [x.lower() for x in wordlist]
    sr = pd.Series(wordlist)
    worddictionary = dict(sr.value_counts())
    return wordlist, worddictionary

def return_postag_keys(x,postag):
    nouns = [words for idx,words in enumerate(x) if x[idx][1] == postag]
    return nouns

def create_noun_set(wordlist,data_file):
    mfcc = [node for node in iterate_data(data_file)]
    mfccset = []
    for word in mfcc:
        raw_word = word._v_name.replace('flickr_', '')
        for wordlistword in wordlist:
            if raw_word == wordlistword:
                mfccset.append(word)
    return mfccset

def iterate_data(h5_file):
    for x in h5_file.root:
        yield x

def dicttodf(dictionary):
    dfwordlist = pd.DataFrame.from_dict(dictionary,orient="index")
    dfwordlist.reset_index(level=0, inplace=True)
    keys = dfwordlist.iloc[:,0]
    values = dfwordlist.iloc[:,1]
    dfwordlist.columns = ["words","counts"]
    return dfwordlist

def MP3toWAV(audio,fromdirectory,todirectory):
    sound = AudioSegment.from_mp3(os.path.join(fromdirectory,audio))
    sound.export(os.path.join(todirectory,audio[:-4] + ".wav"), format="wav")

def print_file_list(textcp,inputwords):
    tokenizer = RegexpTokenizer(r'\w+')
    #loop through the words
    word_appended_finallist = []
    finallist = []
    finallistdict = defaultdict(list)
    for inputword in inputwords:
        wavfiles = []
        #find captions that contain these words
        for idx, captions in enumerate(textcp.captions):
            if inputword in tokenizer.tokenize(captions):
                wavfiles.append(textcp.original.iloc[idx])
        wavfilenames = []
        #preprocess them for file copying
        for word in wavfiles:
            word = word.replace("#","_")
            word = word.replace(".jpg","")
            wordwithinputword = inputword + "_" + word
            wavfilenames.append(word)
        count = 0
        #move the files
        for word in wavfilenames:
            count = count + 1
            word_appended_finallist.append(wordwithinputword)
            finallist.append(word)
            finallistdict[word].append(inputword)
            if count == 50:
                count = 0
                break
    return wavfiles,wavfilenames,word_appended_finallist,finallist,finallistdict


def split_data_objects(f_nodes, splitlist):
    testobjects = []
    for x in f_nodes:
        name = x._v_name.replace('flickr_', '')
        for captionfile in splitlist:
            # inefficient way to remove caption number from files for comparison
            file = captionfile.split('_', -1)[0:2]
            file = "_".join(file)
            if name == file:
                testobjects.append(x)
                break
    return testobjects