from flask import Flask,render_template,request,flash
import numpy as np
from datetime import datetime
import random
import pandas as pd
import readtime

from nltk.cluster.util import cosine_distance
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import networkx as nx
import re
from rouge import Rouge
import nltk
nltk.download('punkt')
nltk.download('stopwords')


import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge import Rouge


from bs4 import BeautifulSoup
from urllib.request import urlopen



#nltk-extractive
def sentence_similarity(sent1, sent2, stopwords=None):    
     if stopwords is None:        
       stopwords = []     
     sent1 = [w.lower() for w in sent1]    
     sent2 = [w.lower() for w in sent2]     
     all_words = list(set(sent1 + sent2))    
     vector1 = [0] * len(all_words)    
     vector2 = [0] * len(all_words)        
     for w in sent1:
         if w in stopwords:
             continue
         vector1[all_words.index(w)] += 1
     for w in sent2:
         if w in stopwords:
             continue        
         vector2[all_words.index(w)] += 1     
     return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    return similarity_matrix

def nltk_summarizer(raw_text):
    raw_text=  re.sub(r'[^a-zA-z0-9.,!?/:;\"\'\s]', '', raw_text)
    stop_words = stopwords.words('english')
    summarize_text = []
    # Step 1 - Read text and tokenize
    sentences = sent_tokenize(raw_text)
    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    print("Indexes of top ranked_sentence order are ", ranked_sentence)
    print(len(raw_text))
    top_n=3
    for i in range(top_n):
        summarize_text.append("".join(ranked_sentence[i][1]))
    #Step 5 -output the summarize text
    a=(" ".join(summarize_text))
    # print(len(a))
    # b=""" 
    # The German Johannes Gutenberg introduced printing in Europe. His invention had a decisive contribution in spread of mass-learning and in building the basis of the modern society.

    # Gutenberg major invention was a practical system permitting the mass production of printed books. The printed books allowed open circulation of information, and prepared the evolution of society from to the contemporary knowledge-based economy.
    # """
    # x=Rouge()
    # y=x.get_scores(a,b)
    # print(y)
    return a


    # raw_text="""Johannes Gutenberg (1398 – 1468) was a German goldsmith and publisher who introduced printing to Europe. His introduction of mechanical movable type printing to Europe started the Printing Revolution and is widely regarded as the most important event of the modern period. It played a key role in the scientific revolution and laid the basis for the modern knowledge-based economy and the spread of learning to the masses.

    # Gutenberg many contributions to printing are: the invention of a process for mass-producing movable type, the use of oil-based ink for printing books, adjustable molds, and the use of a wooden printing press. His truly epochal invention was the combination of these elements into a practical system that allowed the mass production of printed books and was economically viable for printers and readers alike.

    # In Renaissance Europe, the arrival of mechanical movable type printing introduced the era of mass communication which permanently altered the structure of society. The relatively unrestricted circulation of information—including revolutionary ideas—transcended borders, and captured the masses in the Reformation. The sharp increase in literacy broke the monopoly of the literate elite on education and learning and bolstered the emerging middle class."""
    # m=""
    # m=nltk_summarizer(raw_text)
    # print(nltk_summarizer(raw_text))


 #T5-abstractive   
def Abstractivesummarization(text):
  
    model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    tokens_input = tokenizer.encode("summarize: "+text, return_tensors='pt',max_length=tokenizer.model_max_length,truncation=True)

    summary_ids = model.generate(tokens_input, min_length=100, max_length=150,length_penalty=4.0)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def get_text(m):
    raw_text=str(m)
    page=urlopen(raw_text)
    soup=BeautifulSoup(page,'lxml')
    fetched_text=' '.join(map(lambda p:p.text,soup.find_all('p')))
    print(fetched_text)
    return fetched_text


app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('index.html')



@app.route("/ans1",methods=["GET","POST"])
def ans():
    app.secret_key = 'huyhyhyhyhyhyhyhyhyh'
    
    l= request.form
    #text-input
    text=l["Text"]
    #url-input
    url=l["url"]
    #file-input
    from urllib.request import urlopen
    f=request.files['file']
    f1 = request.files['file'].filename
    filedata=""
    if((not (f1.lower().endswith(('.txt','.doc')))) and text=="" and url==""):
        return render_template("error.html")
    elif((f1.lower().endswith(('.txt','.doc')))):
        f.save(f.filename) 
        text_file = open(f1, "r")
        filedata = text_file.read()
        text_file.close()

    if(text!="" and url=="" and filedata==""):
        an=text
    elif(text=="" and url!="" and filedata==""):
        try:
            an=get_text(url)
        except:
            return render_template("error.html")
    
    elif(text=="" and url=="" and filedata!=""):
        an=filedata





#     #an=l["Name"]
#     print(an)
#     au=l["url"]
#     print(au)
    
#     f=request.files['file']
#     f1 = request.files['file'].filename
#     print(f)
#     f.save(f.filename) 
    
#     print("name",f1)
#     read_text=open(f1).read()
#     print("reas=",read_text)
#     text_file = open(f1, "r")
 
# #read whole file to a string
#     data = text_file.read()
 
# #close file
#     text_file.close()
 
#     print(data)
#     an=data
#     if(au==""):
#         print("url is null")
#     else:
#         an= get_text(au)
    

    result = readtime.of_text(an)
    rdbefore=str(result.seconds)+" Seconds"

    if "abs" in l:
        v=Abstractivesummarization(an)
        lenthofoutput=len(v.split())
        lenthofinput=len(an.split())
        t="Abstractive Summarization "
        result = readtime.of_text(v)
        rdafter=str(result.seconds)+" Seconds"
        x=Rouge()
        y=x.get_scores(an,v,avg=True)
        y1=y["rouge-1"]
        y2=y["rouge-2"]
        y3=y["rouge-l"]
        return render_template("index.html",ansss=v,lentO=lenthofoutput,lentI=lenthofinput,typeof=t,readtafter=rdafter,readtbefore=rdbefore,ro1=y1,ro2=y2,ro3=y3)

    print(l,an)
    s=nltk_summarizer(an)
    lenthofoutput=len(s.split())
    lenthofinput=len(an.split())
    t1="Extractive Summarization "
    result = readtime.of_text(s)
    rdafter=str(result.seconds)+" Seconds"
    print(result.seconds)
    print(s)
    #Rouge-score
    x=Rouge()
    y=x.get_scores(an,s,avg=True)
    y1=y["rouge-1"]
    y2=y["rouge-2"]
    y3=y["rouge-l"]
  

    return render_template("index.html",ansss=s,lentO=lenthofoutput,lentI=lenthofinput,typeof=t1,readtafter=rdafter,readtbefore=rdbefore,ro1=y1,ro2=y2,ro3=y3)


if __name__ == "__main__":
    app.run(debug=True)