import tensorflow as tf 
import os
import re
import pickle
import textwrap
import sys
import warnings
warnings.filterwarnings("ignore")
from . constants import *
from . transformers_services import *

tf.keras.backend.clear_session()
with open("main_model/tokenizer_data.pkl", 'rb') as f:
    data = pickle.load(f)
    tokenizer = data['tokenizer']
    MAX_LENGTH = data['maxlen']
print("Tokenizer loaded")
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
VOCAB_SIZE = tokenizer.vocab_size + 2


def preprocess_sentence(sentence):
  sentence = sentence.lower().strip()
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)
  sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
  sentence = sentence.strip()
  return sentence

def loss_function(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  
  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')(y_true, y_pred)

  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
  loss = tf.multiply(loss, mask)

  return tf.reduce_mean(loss)

def evaluate(model,sentence):
  sentence = preprocess_sentence(sentence)

  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  output = tf.expand_dims(START_TOKEN, 0)

  for i in range(MAX_LENGTH):
    predictions = model(inputs=[sentence, output], training=False)

    # select the last word from the seq_len dimension
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # return the result if the predicted_id is equal to the end token
    if tf.equal(predicted_id, END_TOKEN[0]):
      break

    # concatenated the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)

def predict(model,sentence):
  prediction = evaluate(model,sentence)

  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

  print('Input: {}'.format(sentence))
  print('Output: {}'.format(predicted_sentence))

  return predicted_sentence

def optimize(sent,ques):
  cleanques=[]
  for i in ques:
    i=i.split("?")[0].strip()
    if len(i)!=0:
      a=str(i)+"?"
      a=a.strip()
      cleanques.append(a)
  sent=preprocess_sentence(sent)
  finalques=cleanques.copy()
  for index,que in enumerate(questions):
    keywords=[]
    if que.lower() in cleanques:
      if que.lower()=='how is the product?':
        pass
      else:
        keywords=list(freq_used[index])
        found=0
        for key in keywords:
          if key in  sent:
            found=1
        if found==0:
          if que.lower() in finalques:
            finalques.remove(que.lower())
  return finalques
