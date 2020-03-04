#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 19:12:44 2020

@author: Hiloni
"""

import nltk,re
from nltk.tokenize import sent_tokenize
from nltk import load


def ngrammer(text):
    _POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle' #tag for every word based on sentence.
    tagger = load(_POS_TAGGER) #loading a pre-trained model

     #split sentences
    sentences=sent_tokenize(text) #converts to list of sentences. can do '.' split but not always true coz if i am saying 6.78 it will take as poimt or what if i have a ?
    print ('NUMBER OF SENTENCES: ',len(sentences))
    
    nounAfterAdj=[]# holds the adverb-adjective pairs found inthe text(list of 2-grams)

    # for each sentence
    for sentence in sentences:
        
        terms = nltk.word_tokenize(sentence)   #tokenize the sentence (list of all words in sentence) (can do split a t space but not always works so use this)
        
        tagged_terms=tagger.tag(terms)#do POS tagging on the tokenized sentence
        
       
        for i in range(len(tagged_terms)-1):# for every tagged term
            term1=tagged_terms[i] #current term
            term2=tagged_terms[i+1] # following term
        
        #re.match checks if it starts with same prefix. re.look looks for whole word
            if re.match('JJ',term1[1]) and re.match('NN',term2[1]): # current term is an adverb, next one is an adjective
                nounAfterAdj.append((term1[0].lower(),term2[0].lower()))# add the adverb-adj pair to the list
        

    return nounAfterAdj


def process(text1, text2):
    
    nounAfterAdj1 = ngrammer(text1)
    nounAfterAdj2 = ngrammer(text2)
    
    print(nounAfterAdj1, nounAfterAdj2)
    count = 0
    
    for i in nounAfterAdj1:
        for j in nounAfterAdj2:
            if(i[0] == j[0] and i[1] == j[1]):
                count += 1
                
    return count  
    
    
    
if __name__=='__main__':
    text1 = "I have great food and amazing drinks. The place has great music"
    text2 = "If you like great food, go to place. They have best steak."
    #text1 = "This is a really good product! It is Great color, quality, design, and amazingly stylish. We have this placed in our library as seating and it works great as a sofa or a nice place to lie down. Also, I like to put the full back down and pull it out to use it as seating all around when i'm having a party or gathering. This increases seating exponentially in a stylish way. It's pretty much like having a large and stylish bench. It is not very big so I wouldn't recommend using this as your main living room sofa. However, as a multi-function den piece this works great."
    #text2 = "This is a really good product! It is Great color, quality, design, and amazingly stylish. We have this placed in our library as seating and it works great as a sofa or a nice place to lie down. Also, I like to put the full back down and pull it out to use it as seating all around when i'm having a party or gathering. This increases seating exponentially in a stylish way. It's pretty much like having a large and stylish Bench. It is not very big so I wouldn't recommend using this as your main living room sofa. However, as a multi-function den piece this works great."
    print (process(text1, text2))
