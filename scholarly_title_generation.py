import json

from pyserini.search.faiss import AutoQueryEncoder

import pandas as pd

import numpy as np
from translate import Translator
translator=Translator(to_lang='en',from_lang='es')
from tqdm import tnrange
from sklearn.metrics import jaccard_score
import ijson
import json
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as palm
import torch
import numpy as np


import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
encoder = AutoQueryEncoder(encoder_dir='facebook/contriever', pooling='mean')


model2 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
tokenizerss = AutoTokenizer.from_pretrained("google/flan-t5-large")

docid_to_vector = {}



def encode(model, query, hypothesis_documents):
    all_emb_c = []
    for c in [query] + hypothesis_documents:
        
        if not c:
            c = "sample" 
        c_emb = model.encode(c)
        all_emb_c.append(np.array(c_emb))
    all_emb_c = np.array(all_emb_c)
    avg_emb_c = np.mean(all_emb_c, axis=0)
    hyde_vector = avg_emb_c.reshape((1, len(avg_emb_c)))
    return hyde_vector

def vector_search(vector, index, documents):
    
    D, I = index.search(np.array(vector).astype("float32"), k=documents)
    return D, I



def split_into_sentences(text):
    return text.split('.')


def get_embeddings(model, tokenizer, sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings



def find_top_k_sentences(query, documents, k):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    z=[]
  
    for i in range(0,len(documents)):
      t=-10**9
      fin=""
      for sentence in split_into_sentences(documents[i]):
  
        query_embeddings = get_embeddings(model, tokenizer, query[i])
        document_embeddings = get_embeddings(model, tokenizer, sentence)
        similarities = cosine_similarity(query_embeddings, document_embeddings)

        if(similarities>t):
          fin=sentence
          t=similarities
      # print("+++")
      
      z.append("Title:"+query[i] + ". Abstract:"+ fin )
    return(z)


output_json = {"task":"LAMP_T5","golds":[]}
i=0

def create_index(df):

    values = df.abstract.to_list()
    embeddings = []
    for value in values:
        embeddings.append(encoder.encode(value))
   
    embeddings = np.array([embedding for embedding in embeddings]).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index = faiss.IndexIDMap(index)
    index.add_with_ids(embeddings, df.id.values)
    return index

def generate(given_line):
    response = []
    palm.configure(api_key='AIzaSyBtSrjCdGx-POg7nEKmOFbpNYCMCgOGePQ')
    for i in range(0,3):
        response_t = palm.generate_text(prompt=given_line)
        
        response.append(response_t.result) 
    return response

def get_documents(I,df):
    documents = []
    for i in I:
        text_for_desired_id = df.loc[df['id'] == str(i), 'abstract'].values[0]
        documents.append(text_for_desired_id)
    return documents

def get_title(I,df):
    documents = []
    for i in I: 
        text_for_desired_id = df.loc[df['id'] == str(i), 'title'].values[0]
        documents.append(text_for_desired_id)
    return documents

iterator = 0
output_json = []
output_file_path = "predict-scholarly.json"

final_json = {"task":"LaMP_5","golds":[]}
with open("train_scholarly.json", 'r', encoding='utf-8') as f:
        
        json_parser = ijson.items(f, 'item')

        
        for item in json_parser:
            iterator = iterator + 1
            
            original_query = item['input']
            given_line = item['input'][56:]
            

             
            new_query = "Paraphrase the article: "+ given_line
            
            response = []
           
            
            df = pd.DataFrame(item['profile'])

            index = create_index(df)

            response = generate(new_query)

            dense_encoding_vector = encode(encoder,given_line, response)

            length_profile = len(item['profile'])

            length_profile = int(length_profile/2)

            length_profile = max(1,length_profile)
      
            length_profile = min(6,length_profile)
            D, I = vector_search(dense_encoding_vector, index,length_profile)


            I = I.flatten().tolist()
           

            documents = get_documents(I,df)
            
            title = get_title(I,df)

            #query = given_line[74:]

            top_k_sentences = find_top_k_sentences(title, documents, 1)

            k = ""
            for i in range(0,len(top_k_sentences)):
                k=k+top_k_sentences[i]+", "
            
            query = item['input'][:54]
            

            query = query + " while using writing style of the user's title of previous papers written by the user"

           
            output = palm.generate_text(prompt=original_query+". Previous Papers: "+ k)
            
            if not output.result:
                output = palm.generate_text(prompt=original_query)

            print(iterator)

            output_json.append([item['id'],output.result])

         
            dict = {}
            dict["id"]=item['id']
            dict["output"]=output.result
            final_json["golds"].append(dict)
            with open(output_file_path, 'w') as json_file:
                json.dump(final_json, json_file, indent=4)
            