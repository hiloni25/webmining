#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:21:47 2020

@author: Hiloni
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:19:11 2020

@author: Hiloni
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import os
import nltk,re
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


#read the reviews and their polarities from a given file
def getlabel(x):
    if x=='comp.windows.x' or x=='comp_os.ms-windows.misc' or x=='comp.sys.ibm.pc.hardware'  or x=='comp.graphics': #or x=='comp.sys.mac.hardware'
        return 'comp'
    elif x=='rec.sport.baseball' or x=='rec.sport.hockey':
        return 'sport'
    elif x=='talk.politics.misc' or x=='talk.politics.mideast' or x=='talk.politics.guns' :
        return 'politics'
    elif x=='rec.autos' or x=='rec.motorcycles':
        return 'rec'

def loadData(parent_dir):
    text_train=[]
    text_test=[]
    label_train=[]
    label_test=[]
    entries=os.listdir(parent_dir)
    entries.remove('.DS_Store')
    
    for kid_dir in entries:
        path=os.path.join(parent_dir, kid_dir)
        kid_dirs=os.listdir(path)

        
        for file in kid_dirs:
            files=os.path.join(path,file)

            news=[]
            first_emptyLine=False
            f=open(files,encoding="ISO-8859-1")
            read = f.read().strip()
            #print(read)
            
            read.split('/n/n')
            #news.append()
           
            if kid_dir== 'comp.windows.x' or kid_dir== 'rec.sport.baseball' or kid_dir== 'talk.politics.misc' or kid_dir=='rec.autos': 
                text_test.append(read[1:])
                #text_test.append(termL)
                label_test.append(getlabel(kid_dir))
            elif kid_dir=='comp.sys.ibm.pc.hardware'  or kid_dir=='comp.graphics' or kid_dir=='rec.sport.hockey' or kid_dir=='talk.politics.mideast' or kid_dir=='talk.politics.guns'  or kid_dir=='rec.autos' or kid_dir=='rec.motorcycles':
                text_train.append(read[1:])
                label_train.append(getlabel(kid_dir))
            else:
                continue
            f.close()
            
            
    return text_test, label_test, text_train, label_train

text_tests, label_tests, text_trains, label_trains=loadData('/Users/Hilony/Downloads/20_newsgroups')

counter = CountVectorizer()
counter.fit(text_trains)

#counter = TfidfVectorizer()
#counter.fit(text_trains) 


#count the number of times each term appears in a document and transform each doc into a count vector
counts_train = counter.transform(text_trains)#transform the training data
counts_test = counter.transform(text_tests)#transform the testing data

#train classifier
#clf = MLPClassifier() #62
clf = LogisticRegression(solver='liblinear')

#clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5) #60%
#clf = clf = MultinomialNB()
#clf = svm.LinearSVC() #60
#clf = svm.SVC(kernel='linear') #59%
#clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5) #59.7%
#clf =  ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0) -45%

#train all classifier on the same datasets
clf.fit(counts_train,label_trains)

#use hard voting to predict (majority voting)
pred=clf.predict(counts_test)

#print accuracy
print (accuracy_score(pred,label_tests))   #- 62%