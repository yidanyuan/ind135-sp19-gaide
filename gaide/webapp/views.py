from django.shortcuts import render
from django.core import serializers
from django.utils import timezone
from django.http import HttpResponseRedirect
from webapp.models import *

import tensorflow as tf
from tensorflow.keras.models import load_model
#import models to be used
import numpy as np
import pandas as pd
import librosa
import multiprocessing
import sys
import os

model_path = os.getcwd()+'/webapp/trained_models/'
stella_model = load_model(model_path+'new_stella.h5')
please_model = load_model(model_path+'new_please.h5')
call_model = load_model(model_path+'new_call.h5')
word_list = ["please","call","stella"]

def homepage(request):
    return render(request, 'webapp/homepage.html')

def index(request):
    return render(request, 'webapp/homepage_guest.html')

def login(request):
    return render(request, 'webapp/login.html')

def speaker(request):
    return render(request, 'webapp/speaker.html')
# Create your views here.

def extract_feature(file_name):
    try:
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None, None
    return (mfccs)

def audio_split_on_slience(file_name, dest_folder, word_list):
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
    import os
    
    #check if destination folder is already there, if so nevermind, if not create new folder
    if not os.path.exists(dest_folder):
        print('creating audio destination folder'.format(dest_folder))
        os.makedirs(dest_folder)
        
    sound = AudioSegment.from_wav(file_name)
    chunks = split_on_silence(sound, 
    # must be silent for at least 40ms
    min_silence_len=100,

    # consider it silent if quieter than -26 dBFS
    silence_thresh=-30
    )
    
    #save chopped chunks separately into dest_folder
    for i, chunk in enumerate(chunks):
        if i < len(word_list):
            chunk.export(dest_folder +word_list[i]+".wav", format="wav")
            #chunk.export(os.getcwd() + "/splitwav/" + dest_folder +"/"\
                         #+"chunk{0}_".format(i) + word_list[i] +\
                         #"_" + num + ".wav", format="wav")
    
        else:
            chunk.export(dest_folder +"/" +"chunk{0}.wav".format(i), format="wav")
            #chunk.export(os.getcwd() + "/splitwav/" + dest_folder +"/"\
                         #+"chunk{0}.wav".format(i), format="wav")
    
    #return: array showing chopped chunks
    return chunks


def detect(request):
	test_seg = AudioSegment.from_wav(os.getcwd()+'/webapp/test/arabian/arabian.wav')
	test_seg_chunk = audio_split_on_slience(os.getcwd()+'/wenapp/test/arabian/arabian.wav',\
                                        os.getcwd()+'/webapp/test/arabian/', word_list)
	file_name_please = os.getcwd()+'/webapp/test/arabian/please.wav'
	file_name_call = os.getcwd()+'/webapp/test/arabian/call.wav'
	file_name_stella = os.getcwd()+'/webapp/test/arabian/stella.wav'
	feature1 = extract_feature(file_name_please)
	feature2 = extract_feature(file_name_call)
	feature3 = extract_feature(file_name_stella)

	X_please = feature1
	X_call = feature2
	X_stella = feature3
	Y_please = please_model.predict_classes(X_please.reshape((1,40)))
	Y_call= call_model.predict_classes(X_call.reshape((1,40)))
	Y_stella = stella_model.predict_classes(X_stella.reshape((1,40)))
	print("Result for 'Please': ",Y_please)
	print("Result for 'call': ",Y_call)
	print("Result for 'Stella': ",Y_stella)
	return

