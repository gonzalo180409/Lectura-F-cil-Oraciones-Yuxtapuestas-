import spacy
from joblib import load
from pprint import pprint
import numpy as np
import requests
import re
from openai import OpenAI
import time
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd

#API fillmask task
#API_URL1 = "https://api-inference.huggingface.co/models/bertin-project/bertin-roberta-base-spanish"
#API similarity sentences
API_URL2 = "https://api-inference.huggingface.co/models/hiiamsid/sentence_similarity_spanish_es"
#API LLAMA2
#API_URL3 = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-hf"

#Huggingface token
headers = {"Authorization": "Bearer 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"}
#List for storing sentences detected as juxtaposed
global oraciones_yuxtapuestas
oraciones_yuxtapuestas = []
global oraciones_NO_yuxtapuestas
oraciones_NO_yuxtapuestas = []
#List for storing sentences after fillmask processing
global oraciones_post_fillmask 
oraciones_post_fillmask = []
#List for storing the results of similarity between original and transformed sentences
global similar
similar = []
#List for storing the results of similarity between original and transformed sentences
global oraciones_pre_fillmask
oraciones_pre_fillmask = []

def load_senteces (txt):
  """
  Loads the sentences from oraciones.txt and processes them with spaCy library.
  Loads the sentences considered as juxtaposed into the global variable: oraciones_yuxtapuestas[]
  
  Parameter:
  txt (.txt): Text file containing the sentences to be processed
  """
  modelo_svm = load('modelo_svm.joblib')
  nlp = spacy.load("es_core_news_md")
  #nlp = spacy.load("es_core_news_md")
  #Loads the sentences from oraciones.txt
  with open(txt, "r", encoding="utf-8") as archivo:
    oraciones = archivo.read().splitlines()
  for oracion in oraciones:
    doc = nlp(oracion)
    for sent in doc.sents:
      for token in sent:
        print(token.text+'| '+token.pos_ +' '+ token.dep_)
      bool = es_yuxta(sent)
      #The algorithm does not recognize the sentence structure, so it is processed by the binary classification model.
      if(bool == 2):
        vector_oracion = sent.vector
        vector_oracion_reshaped = np.array(vector_oracion).reshape(1, -1)
        prediccion = modelo_svm.predict(vector_oracion_reshaped)
        if(prediccion[0] == 1): #is juxtaposed
          print(f"La predicción del modelo para la oración '{sent}' es: True")
          oraciones_yuxtapuestas.append(str(sent))
        else:
          print(f"La predicción del modelo para la oración '{sent}' es: False")
          oraciones_NO_yuxtapuestas.append(str(sent))
      #The algorithm recognizes that the sentence is juxtaposed
      elif(bool == 1):
        print(f"La predicción para la oración '{sent}' es: True")
        oraciones_yuxtapuestas.append(str(sent))
      #The algorithm recognizes the sentence is NOT juxtaposed
      else:
        print(f"La predicción para la oración '{sent}' es: False")
        oraciones_NO_yuxtapuestas.append(str(sent))

def transformar_oraciones (lista):
  """
  Transforms juxtaposed sentences so that they can be processed by the fillmask task.
  Replace punctuation marks with ". <mask>,"

  Parameters:
  lista (List): List of sentences to be transformed except the period at the end of the sentence.

  Returns:
  List: List of sentences transformed
  """
  for sent in lista:
    sent_text = str(sent)
    modificada = re.sub(r'[^\w\s](?=[^\s]*\s)', '. <mask>,', sent_text)
    oraciones_pre_fillmask.append(modificada)

def pos_verb(sent):
  """
  Given a sentence, identifies its verbs and return their position.

  Parameters:
  sent (String): Sentence to analyze.

  Returns:
  List: List of integers indicating verb positions.
  """
  res = []
  for token in sent:
    if token.pos_ == 'VERB' or (token.pos_ =='AUX' and token.dep_ == 'cop'):
       res.append(token.idx)
  return res

def pos_punc(sent):
  """
  Given a sentence, identifies punctuation marks, except for the period at the end of the sentence.

  Parameters:
  sent (String): Sentence to analyze.

  Returns:
  List: List of integers indicating punctuation marks positions.
  """
  res = []
  for token in sent:
    if token.is_punct:
      if token.is_sent_end == False:
        res.append(token.idx)
  return res

def pos_nexo(sent):
    """
    Given a sentence, identifies nexus or discourse connectors.

    Parameters:
    sent (String): Sentence to analyze.

    Returns:
    List: List of integers indicating nexus or discourse connectors positions.
    """
    res = []
    for i, token in enumerate(sent):
        if token.dep_ == 'cc' or token.dep_ == 'mark':
            if i < len(sent) - 1:
                next_token = sent[i + 1]
                if (next_token.pos_ == 'VERB' and next_token.dep_ == 'conj') or next_token.dep_ != 'conj':
                    res.append(token.idx)
            # else:
            #     res.append(token.idx)
        if token.pos_ == 'PART' and token.dep_ == 'advmod' : #no obstante
          res.append(token.idx)
    return res

def es_yuxta(sent):
  """
  Given a sentence, identify whether it is juxtaposed or not.

  Parameters:
  sent (String): Sentence to analyze.

  Returns: 
  int : 0 -> The algorithm detects that it is NOT juxtaposed.
  int : 1 -> The algorithm detects that it is juxtaposed.
  int : 2 -> The algorithm is unable to detect the juxtaposition, so it will be further processed by the binary classification model.
  """
  i = 0
  l_v = pos_verb(sent)
  l_n = pos_nexo(sent)
  l_p = pos_punc(sent)
  print('Verbos: '+str(l_v))
  print('Nexos: '+str(l_n))
  print('Punct: '+str(l_p))
  if not l_p or l_n:
    return 0
  if len(l_v) > len(l_p)+2:
    return 2
  for i in range(len(l_v) - 1) :
    verbo_actual = l_v[i]
    verbo_siguiente = l_v[i + 1]
    puntuacion_entre_verbos2 = [p for p in l_p if verbo_actual < p < verbo_siguiente]
    if not puntuacion_entre_verbos2:
      continue
    nexos_entre_verbos = [n for n in l_n if verbo_actual < n < verbo_siguiente]  
    if not nexos_entre_verbos:
      return 1
    else :
      return 0


def queryS(payload):
  """
  Query with which the similarity task is called.

  Parameters:
  json: payload.

  Return:
  json: Similarity senteces percentages.
  """
  response = requests.post(API_URL2, headers=headers, json=payload)
  return response.json()

def similarity (originals, transformed):
  while True:
    for original, trans in zip(originals, transformed):
      payload = {
        "inputs": {
        "source_sentence": original,
        "sentences": trans
        }
      }
      output = queryS(payload)
      if 'error' in output:
        print("Esperando 25 segundos a que el modelo de similitudes se cargue...")
        time.sleep(25)
        break
      else:
        similar.append(output)
    else: break

 
def fillmaskGPT (sent):
  client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="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",
  )
  user_message = "\n".join(sent)
  chat_completion = client.chat.completions.create(
    messages=[
      {
        "role": "system",
        "content": "Sustituye la etiqueta <mask> por un nexo o marcador discursivo que mejor se adecúe al significado de las oraciones: "
      },
      {
        "role": "user",
        "content": user_message  
      },
    ],
    model="gpt-3.5-turbo",
  )
  response = chat_completion.choices[0].message.content
  oraciones = response.split('\n')
  for o in oraciones:
    oraciones_post_fillmask.append([str(o)])
  

if __name__ == "__main__":
  txt = "oraciones.txt"
  load_senteces(txt)
  if not oraciones_yuxtapuestas and not oraciones_NO_yuxtapuestas:
    print('No se han detectado oraciones.')
  else : 
    transformar_oraciones(oraciones_yuxtapuestas)
    fillmaskGPT(oraciones_pre_fillmask)
    similarity(oraciones_yuxtapuestas, oraciones_post_fillmask)
    # Aplanar la lista de oraciones transformadas y similitudes
    oraciones_transformadas_flat = [item for sublist in oraciones_post_fillmask for item in sublist]
    similitudes_flat = [item for sublist in similar for item in sublist]

    # Crear un DataFrame
    df = pd.DataFrame({
        'Oración Yuxtapuesta': oraciones_yuxtapuestas,
        'Oración Transformada': oraciones_transformadas_flat,
        'Similitud': similitudes_flat
    })
    df_no_yuxtapuestas = pd.DataFrame({
    'Oración No Yuxtapuesta': oraciones_NO_yuxtapuestas
    })
    print(df_no_yuxtapuestas)

    # Crear una figura vacía de Matplotlib
    fig = plt.figure(figsize=(12, 8))  # Aumenta el tamaño de la figura para acomodar ambas tablas

    # Agregar el subplot para la primera tabla (yuxtapuestas)
    ax1 = fig.add_subplot(211)  # 2 filas, 1 columna, posición 1
    ax1.axis('tight')
    ax1.axis('off')
    table1 = ax1.table(cellText=df.values, colLabels=df.columns, cellLoc = 'center', loc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.auto_set_column_width(col=list(range(len(df.columns))))

    # Agregar espacio entre las tablas
    fig.subplots_adjust(hspace=0.05)

    # Agregar el subplot para la segunda tabla (no yuxtapuestas)
    ax2 = fig.add_subplot(212)  # 2 filas, 1 columna, posición 2
    ax2.axis('tight')
    ax2.axis('off')
    table2 = ax2.table(cellText=df_no_yuxtapuestas.values, colLabels=df_no_yuxtapuestas.columns, cellLoc = 'center', loc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.auto_set_column_width(col=[0])  # Solo hay una columna

    plt.show()