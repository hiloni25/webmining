#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 18:45:42 2020

@author: Hiloni 
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import os
path = '//Users/Hilony/Desktop/Web Mining Proj/Final/'

'''f = open(path)
read = f.read().strip()
read.split('/n')
print(len(read))'''

reviews=[]
sentiments=[]
for i,j,k in os.walk(path):
    print(k)
    rev_train = []
    sentiments_train = []
    rev_test = []
    sentiments_test = []
    for file in k:
        if(file.endswith('.txt')):
            path1 = path+file
            print(path1)
            f=open(path1)
            for line in f:
                #print(line)
                if(len(line.strip().split('\t')) == 2):
                    review,sentiment=line.strip().split('\t')
                    #print(review,sentiment)
                    reviews.append(review.lower())    
                    sentiments.append(int(sentiment[0]))
            f.close()
#print(reviews,sentiments)
        length = len(reviews) 
        newl = int(length*0.8)
        print(length,newl)
        print(len(reviews[:newl]), len(sentiments[:newl]))
        rev_train += reviews[:newl]
        sentiments_train += sentiments[:newl]
        rev_test += reviews[newl:]
        sentiments_test += sentiments[newl:]

#print(rev_test,sentiments_test)
print(len(rev_train), len(sentiments_train), len(rev_test), len(sentiments_test))
        
#Build a counter based on the training dataset
print(type(rev_train))
counter = TfidfVectorizer()
counter.fit(rev_train)

#count the number of times each term appears in a document and transform each doc into a count vector
counts_train = counter.transform(rev_train)#transform the training data
counts_test = counter.transform(rev_test)#transform the testing data

#train classifier
clf = MLPClassifier()

#train all classifier on the same datasets
clf.fit(counts_train,sentiments_train)

#use hard voting to predict (majority voting)
pred=clf.predict(counts_test)

#print accuracy
print (accuracy_score(pred,sentiments_test))


#print(sentiments_test)

#rev_train,labels_train=loadData('reviews_train.txt')
#rev_test,labels_test=loadData('reviews_test.txt')

