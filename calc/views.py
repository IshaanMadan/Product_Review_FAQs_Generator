from django.shortcuts import render
from django.http import HttpResponse
import tensorflow as tf
from django.http import JsonResponse
assert tf.__version__.startswith('2')
tf.random.set_seed(1234)
import sys
import warnings
warnings.filterwarnings("ignore")
from . transformers import *
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
bert_model = QA('bert-large-uncased-whole-word-masking-finetuned-squad')
# bert_model = QA('bert-large-uncased-whole-word-masking-finetuned-squad')

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

model.load_weights('main_model/model_weights.h5')


def home(request):
    return render(request,'home.html')
    # return HttpResponse('home.html')

@csrf_exempt
def review(request):
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

        for i in ques:
            results[i]=ques
            print(results[i])
            print("Ques: ",i)
            ans=bert_model.predict(sentence,i)
            results[i]=str(ans['answer']).capitalize()
            print("Answer: ",ans['answer'])
            print("\n")
    return JsonResponse(results)
    # return response(ques)



