import tensorflow as tf
assert tf.__version__.startswith('2')
tf.random.set_seed(1234)
import sys
import warnings
warnings.filterwarnings("ignore")
from transformers import *
sys.path.append('BERT-SQuAD')
from bert import QA
import os
import re
import pickle
import textwrap
# !pip install tensorflow-datasets==1.2.0
import tensorflow_datasets as tfds
bert_model = QA('bert-large-uncased-whole-word-masking-finetuned-squad')

tf.keras.backend.clear_session()
with open("tokenizer_data.pkl", 'rb') as f:
    data = pickle.load(f)
    tokenizer = data['tokenizer']
    # num_words = data['num_words']
    MAX_LENGTH = data['maxlen']
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
VOCAB_SIZE = tokenizer.vocab_size + 2
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

model.load_weights('model_weights.h5')


# Maximum number of samples to preprocess
MAX_SAMPLES = 50000

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



def evaluate(sentence):
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


def predict(sentence):
  prediction = evaluate(sentence)

  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

  print('Input: {}'.format(sentence))
  print('Output: {}'.format(predicted_sentence))

  return predicted_sentence
# sentence="Sound quality is awesome. \
# Bass i/s good. \
# Build quality is good. \
# Price is affordable. \
# Overall product is worth buying."


# sentence="Excellent phone,Amazing camera, nice fast charging, good battery backup."
# sentence="Bass quality superb, calling quality is really very nice, all over can say it's very nice earphone, if any loves earphone and can handle the wire very carefully then go for it."
# sentence ="Nice earphones.Good noise cancellation."
# data.iloc[35]['ques']
freq_used=[['gaming','gamming'],['camera','cam'],['battery'],['sound'],['charging','charger','vooc'],['colour','color'],['fingerprint','finger','fingerprints'],['display'],['design'],['ram'],['delivery'],['build quality'],['processor'],['product','mobile','phone'],['build quality','built quality'],['mic'],['sound','audio quality'],['bass'],['wire quality','cable'],['buttons'],['product','earphones'],['look','colour'],['noise cancellation','noise cancelling'],['call quality'],['design'],['fit'],['tangle'],['treble'],['durablity','durable','durability']]
questions=['How is the gaming experience?','How is the Camera?','How is the battery?','How is the sound?','Does it support fast charging?','How is the colour of the phone?','How is the fingerprint reader?','How is display?', 'How is the design of the phone?','How is the RAM Management of the phone?','How is the delivery services?','How is the build quality?','How is the processor of the phone?','How is the product?' ,'How is build quality?','How is the mic?','How is the sound?','How is bass?','How is the wire quality?','How are the buttons?','How is the product?','How is the look?','How is the noise cancellation?','How is the call quality?','How is the design?','How is the fit?','Does it tangle?','How is the treble?','How is the durablity?']
def optimize(sent,ques):
  cleanques=[]
  for i in ques:
    i=i.split("?")[0].strip()
    if len(i)!=0:
      a=str(i)+"?"
      a=a.strip()
      cleanques.append(a)
  sent=preprocess_sentence(sentence)
  finalques=cleanques.copy()
  for index,que in enumerate(questions):
    keywords=[]

    # print(que)
    if que.lower() in cleanques:
      if que=='How is the product?':
        pass
      else:
        # print(que)
        keywords=list(freq_used[index])
        found=0
        for key in keywords:
          if key in  sent:
            found=1
        if found==0:
          # pass
          finalques.remove(que.lower())
  return finalques


def GenerateFAQs(sentence):
  results={}
  sentence=input("Please Enter the review:- \n")
  output = predict(sentence)
  ques=[]
  for i in output.split('?'):
    if len(i)!=0:
      a=str(i)+"?"
      a=a.strip()
      ques.append(a)

# ko="Excellent phone,Amazing camera, nice fast charging, good battery backup."

  wrapper = textwrap.TextWrapper(width=80) 
  ques=optimize(sentence,ques)

  print("Your review :-\n",wrapper.fill(sentence),'\n\nGenerating FAQs...\n')
  for i in ques:
    print("Ques: ",i)
    ans=bert_model.predict(sentence,i)
    results[i]=ans['answer']
    # print("Answer: ",ans['answer'])
    # print("\n")
  return results