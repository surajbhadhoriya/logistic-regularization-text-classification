# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:30:30 2019

@author: SURAJ BHADHORIYA
"""
#import the libraries
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

#load data
df=pd.read_csv("C:/Users/SURAJ BHADHORIYA/Desktop/SINGLE-FILES/amazon_baby_subset.csv")

#json file
import json
important_word=json.loads(open("C:/Users/SURAJ BHADHORIYA/Desktop/SINGLE-FILES/important_words.json").read())
print(important_word)

#remove punctuation
df['review']=df['review'].str.replace('[^\w\s]','')
df['review'].head()
print(df['review'])

#fillna values
df=df.fillna({'review':''})
#create colounm to all importtant_word how many time they repeat
for word in important_word:
    adf[word]=df['review'].apply(lambda s: s.split().count(word))
    
# no pf +ive and -ive 1's  
pos_1=len(df[df['sentiment']==1]) 
neg_1=len(df[df['sentiment']==-1])  
print(pos_1)
print(neg_1) 

#how much perfect word contain in review
df['contain_perfect']=df['perfect']>=1
sum_contain_perfect=df['contain_perfect'].sum()
print(sum_contain_perfect)     
    
#convert df to multi dimension array
    
feature1=important_word
label=["sentiment"]

#form multidimension array for bag of words
def get_numpy_data(dataframe,features,label):
    dataframe['constant']=1
    features=['constant']+features
    features_frame=dataframe[features]
    feature_matrix=features_frame.as_matrix()
    label_sarray=dataframe[label]
    label_array=label_sarray.as_matrix()
    return(feature_matrix,label_array)

x,y=get_numpy_data(df,feature1,label) 

print(x)
print(y)
#split data
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#apply logistic regression with regularization to over come overfitting
model=LogisticRegression(penalty='l2', C=0.1,max_iter=100)
model.fit(X_train,y_train)

#accuracy
accuracy=model.score(X_test,y_test)
print(accuracy)
acc=model.score(X_train,y_train)

#prediction
pre=model.predict(x[27140:27150])
print(pre)
   

        
    
