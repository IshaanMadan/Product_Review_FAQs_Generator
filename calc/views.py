from django.shortcuts import render
from transformers import pipeline
from django.http import HttpResponse
import tensorflow as tf
from django.http import JsonResponse
assert tf.__version__.startswith('2')
tf.random.set_seed(1234)
import sys
import warnings
warnings.filterwarnings("ignore")
from . transformers_services import *
sys.path.append('main_model/BERT-SQuAD')
from bert import QA
import os
import re
import pickle
import textwrap
from django.views.decorators.csrf import csrf_exempt
# !pip install tensorflow-datasets==1.2.0
import tensorflow_datasets as tfds
from . services import *
print("Loading Bert model....!")

bert_model = QA('bert-large-uncased-whole-word-masking-finetuned-squad')
# bert_model = QA('bert-large-uncased-whole-word-masking-finetuned-squad')
# print("Loading Bert model....!")
# bert_model=pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad', tokenizer='bert-base-uncased')
print("Bert Model loaded...!")

# Hyper-parameters
NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1

model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)
print("Loading Transformer model...!")
model.load_weights('main_model/model_weights.h5')
print("Transformers model loaded")
print("Loading Distilbert model loaded...!")
distillbert_model = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad', tokenizer='distilbert-base-uncased')
print("Distilbert model loaded..!")

@csrf_exempt
def review(request):
    return render(request,'home.html')

@csrf_exempt
def get_bert(request):
    res="This is result"
    results={}
    if request.method == 'POST':
        
        sentence=request.POST['doc']
        output = predict(model,preprocess_sentence(sentence))
        
        ques=[]
        for i in output.split('?'):
            if len(i)!=0:
                a=str(i)+"?"
                a=a.strip()
                ques.append(a)
        wrapper = textwrap.TextWrapper(width=80)
        print("before opts",preprocess_sentence(sentence)) 
        print("before optq",ques)
        ques=optimize(sentence,ques)
        ques=[i.strip().capitalize() for i in ques]

        print("Your review :-\n",wrapper.fill(sentence),'\n\nGenerating FAQs...\n')
        # for i,j in enumerate(ques):
            # results[i]=j
        # for i in ques:
            # print(i)
            # output = bert_model(question = i, context = sentence)
            # results[i]=output['answer']

        for i in ques:
            # results[i]=ques
            # print(results[i])
            # print("Ques: ",i)
            ans=bert_model.predict(sentence,i)
            results[i]=str(ans['answer']).capitalize()
            print("Answer: ",ans['answer'])
            print("\n")
    return JsonResponse(results)
    # return response(ques)
@csrf_exempt
def distillbert(request):
    return render(request,'distillbert.html')
@csrf_exempt
def get_ditillbertreview(request):
    results={}
    if request.method == 'POST':
        
        sentence=request.POST['doc']
        output = predict(model,preprocess_sentence(sentence))
        
        ques=[]
        for i in output.split('?'):
            if len(i)!=0:
                a=str(i)+"?"
                a=a.strip()
                ques.append(a)
        wrapper = textwrap.TextWrapper(width=80)
        print("before opts",preprocess_sentence(sentence)) 
        print("before optq",ques)
        ques=optimize(sentence,ques)
        ques=[i.strip().capitalize() for i in ques]
        results={}
        for i in ques:
            # print(i)
            output = distillbert_model(question = i, context = sentence)
            results[i]=output['answer']
    return JsonResponse(results)



