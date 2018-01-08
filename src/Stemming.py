'''
Created on 13-Oct-2017

@author: mohit
'''
import os
from nltk import stem

stemmer = stem.PorterStemmer()

path = 'Z:\\Lectures\\Machine_Learning\\Assignments\\Homework_2\\train\\ham'
for filename in os.listdir(path):
    file = open(path+'\\'+filename, "r")
    with open('Z:\\Lectures\\Machine_Learning\\Assignments\\Homework_2\\train\\stemmed_ham\\'+filename,'a+') as write_file:
        for line in file:
            for word in line.split():
                #if word not in stopwords:
                    write_file.write(stemmer.stem(word)+'\n')


path = 'Z:\\Lectures\\Machine_Learning\\Assignments\\Homework_2\\train\\spam'
for filename in os.listdir(path):
    file = open(path+'\\'+filename, "r")
    with open('Z:\\Lectures\\Machine_Learning\\Assignments\\Homework_2\\train\\stemmed_spam\\'+filename,'a+') as write_file:
        for line in file:
            for word in line.split():
                #if word not in stopwords:
                    write_file.write(stemmer.stem(word)+'\n')
                

path = 'Z:\\Lectures\\Machine_Learning\\Assignments\\Homework_2\\test\\ham'
for filename in os.listdir(path):
    file = open(path+'\\'+filename, "r")
    with open('Z:\\Lectures\\Machine_Learning\\Assignments\\Homework_2\\test\\stemmed_ham\\'+filename,'a+') as write_file:
        for line in file:
            for word in line.split():
                #if word not in stopwords:
                    write_file.write(stemmer.stem(word)+'\n')
                

path = 'Z:\\Lectures\\Machine_Learning\\Assignments\\Homework_2\\test\\spam'
for filename in os.listdir(path):
    file = open(path+'\\'+filename, "r")
    with open('Z:\\Lectures\\Machine_Learning\\Assignments\\Homework_2\\test\\stemmed_spam\\'+filename,'a+') as write_file:
        for line in file:
            for word in line.split():
                #if word not in stopwords:
                    write_file.write(stemmer.stem(word)+'\n')