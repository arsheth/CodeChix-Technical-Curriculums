#!/usr/bin/python
# -*- coding: utf-8 -*-

import psycopg2
import sys
import codecs
import numpy as np
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer

con = None

train_feature_List = []
labels = []
test_feature_List = []

test_item_name = []

try:
    maxValue = 0
    tagValue = ''
    con = psycopg2.connect(database='darkweb', user='postgres',password='123') 
    cur = con.cursor()
    cur.execute('SELECT * from train_people where id<100')   
    training_set = cur.fetchall()  


    for training_row in training_set:
    	labels.append(training_row[8])
    	train_feature_List.append(training_row[1:7]);
            # +training_row[30:]);



    cur.execute('SELECT * from train_people')   
    test_set = cur.fetchall()  

    for test_row in test_set:
        test_item_name.append(test_row[0])
    	test_feature_List.append(test_row[1:7]);
           
    

    final_training_data = np.array(train_feature_List)
    final_test_data = np.array(test_feature_List)
    
    
    forest = LogisticRegression(random_state=0)
    forest = forest.fit( final_training_data, labels )
    result = forest.predict(final_test_data)
   


    for x,y in zip(result, test_item_name):
        query='update train_people set is_come_less_50_pred_lr=%s where id=%s'
        data=(x, y)
        cur.execute(query,data)
    con.commit()    
    
        
except psycopg2.DatabaseError, e:
    print 'Error %s' % e    
    sys.exit(1)
    
    
finally:
    
    if con:
        con.close()