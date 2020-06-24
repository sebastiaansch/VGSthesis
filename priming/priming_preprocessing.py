#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:17:15 2020

@author: danny
"""

import glob 
import os
import tables

import numpy as np
import python_speech_features.base as base
import python_speech_features.sigproc as sigproc

from scipy.io.wavfile import read

params = {}
params['alpha'] = 0.97
params['nfilters'] = 40
params['ncep'] = 13
params['t_window'] = .025
params['t_shift'] = 0.01
params['feat'] = 'mfcc'
params['use_deltas'] = True
params['use_energy'] = True
params['windowing'] = np.hamming
params['delta_n'] = 2

stimuli_loc = '/home/danny/Downloads/Archive'
data_loc = '/home/danny/Downloads/priming_features.h5' 
primes = glob.glob(f'{stimuli_loc}/49words_primes/*.wav') 
targets = glob.glob(f'{stimuli_loc}/49words_targets/*.wav') 

f_atom = tables.Float32Atom() 

stimuli_locs = [[x, y] for x in primes for y in targets if '_'.join(x.split('/')[-1].split('_')[:-1]) == y.split('/')[-1].split('.')[:-1][0]]

assert len(primes) == len(stimuli_locs)

output_file = tables.open_file(data_loc, mode='a') 

for prime, target in stimuli_locs:

    prime_data = read(os.path.join(prime))
    target_data = read(os.path.join(target))
    
    node_name = target.split('/')[-1].split('.')[0]
    node = output_file.create_group("/", node_name)
    
    feature_node = output_file.create_group(node, params['feat'])
    
    # sampling frequency
    fs = prime_data[0]    
    fs_target = target_data[0]   
    assert(fs_target == fs)
    # set the fft size to the power of two equal to or greater than 
    # the window size.
    window_size = int(fs * params['t_window'])
    exp = 1
    while True:
        if np.power(2, exp) - window_size >= 0:
            fft_size = np.power(2, exp)
            break
        else:
            exp += 1
    
    prime_features = base.mfcc(prime_data[1], samplerate = fs,
                               winlen = params['t_window'], 
                               winstep = params['t_shift'], 
                               numcep = params['ncep'], 
                               nfilt = params['nfilters'], 
                               nfft = fft_size, lowfreq = 0, 
                               highfreq = None, 
                               preemph = params['alpha'], ceplifter = 0, 
                               appendEnergy = params['use_energy'], 
                               winfunc = params['windowing']
                               )

    target_features = base.mfcc(target_data[1], samplerate = fs,
                                winlen = params['t_window'], 
                                winstep = params['t_shift'], 
                                numcep = params['ncep'], 
                                nfilt = params['nfilters'], 
                                nfft = fft_size, lowfreq = 0, 
                                highfreq = None, 
                                preemph = params['alpha'], ceplifter = 0, 
                                appendEnergy = params['use_energy'], 
                                winfunc = params['windowing']
                                )
    
    if params['use_deltas']:
                
        single_delta = base.delta(prime_features, params['delta_n'])
        double_delta = base.delta(single_delta, params['delta_n'])
        prime_features= np.concatenate([prime_features, single_delta,
                                        double_delta], 1)
        
        single_delta = base.delta(target_features, params['delta_n'])
        double_delta = base.delta(single_delta, params['delta_n'])
        target_features= np.concatenate([target_features, single_delta,
                                         double_delta], 1)
    
    feature_shape= np.shape(prime_features)[1]    
    prime_node = output_file.create_earray(feature_node, 'prime', f_atom, 
                                           (0, feature_shape), 
                                           expectedrows = 5000
                                           )
    prime_node.append(prime_features)

    feature_shape= np.shape(target_features)[1]        
    target_node = output_file.create_earray(feature_node, 'target', f_atom, 
                                            (0, feature_shape),
                                            expectedrows = 5000
                                            )
    target_node.append(target_features)

output_file.close()
    
    
    
    
    
    