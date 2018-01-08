'''
Created on 11-Oct-2017

@author: mohit
'''
import os
import math
import sys
from collections import Counter

def generatewordlist(directory):
    path = os.getcwd()+directory
    trainwordsList = []
    allemail = [os.path.join(path,f) for f in os.listdir(path)]
    for emailfile in allemail: 
        with open(emailfile,'r') as m:
            for line in m:
                for word in line.split(): 
                    lowerword=word.lower()
                    if(lowerword.isalnum()):
                        trainwordsList.append(lowerword)      
    return trainwordsList


def trainMultinomiaNB(trainhamwordscount, trainspamwordscount, counttrainhamwordsdict, counttrainspamwordsdict, counttrainwordsdict):
    condprobhamDict = {}
    condprobspamDict = {}
    priortrainham = float(trainhamwordscount)/(trainhamwordscount + trainspamwordscount)
    priortrainspam = float(trainspamwordscount)/(trainhamwordscount + trainspamwordscount)
    
    for elem in counttrainwordsdict:
        condprobham = 0.0
        condprobspam = 0.0
        trainhamwordcount = 0
        trainspamwordcount = 0      
        if elem in counttrainhamwordsdict:
            trainhamwordcount = counttrainhamwordsdict[elem]
        if elem in counttrainspamwordsdict:
            trainspamwordcount = counttrainspamwordsdict[elem]
            
        condprobham = float(trainhamwordcount+1)/(trainhamwordscount + len(counttrainwordsdict))        
        condprobspam = float(trainspamwordcount+1)/(trainspamwordscount+ len(counttrainwordsdict))
        condprobhamDict[elem] = condprobham
        condprobspamDict[elem] = condprobspam
        
    return priortrainham, priortrainspam, condprobhamDict, condprobspamDict
 

def filefromdirectory(emailfile):
    testwordsList = []
    with open(emailfile,'r') as m:
        for line in m:
            for word in line.split(): 
                lowerword=word.lower()
                if(lowerword.isalnum()):
                    testwordsList.append(lowerword)
    return testwordsList

        
def applyMultinomialNB(counttrainwordsdict, priortrainham, priortrainspam, condprobhamDict, condprobspamDict, stemtestwordsList):
    ishamclassified = False
    hamprobsum = math.log(priortrainham)
    spamprobsum = math.log(priortrainspam)
    counttestwordsdict = Counter(stemtestwordsList)
    for elem in counttestwordsdict:
        count = counttestwordsdict[elem]
        for i in range(0, count):
            if elem in counttrainwordsdict:
                 hamprobsum += math.log(condprobhamDict[elem])
                 spamprobsum += math.log(condprobspamDict[elem])
    
    if(hamprobsum >= spamprobsum):
        ishamclassified = True        
    return ishamclassified



def main(stopwordsList, traindirectorypoststemming, testdirectorypoststemming, isstopwordused):
    trainhamDirectory = traindirectorypoststemming + "/stemmed_ham"
    trainspamDirectory = traindirectorypoststemming + "/stemmed_spam"
    testhamDirectory = testdirectorypoststemming + "/stemmed_ham"
    testspamDirectory = testdirectorypoststemming + "/stemmed_spam"
      
    #generating word list for train file now
    updatedtrainhamwordsList = []
    stemtrainhamwordsList = generatewordlist(trainhamDirectory)
    if isstopwordused == True:
        updatedtrainhamwordsList = [x for x in stemtrainhamwordsList if x not in stopwordsList]
    else:
        updatedtrainhamwordsList = stemtrainhamwordsList
    trainhamwordscount = len(updatedtrainhamwordsList)
    counttrainhamwordsdict = Counter(updatedtrainhamwordsList)     
    
    updatedtrainspamwordsList = []
    stemtrainspamwordsList = generatewordlist(trainspamDirectory)
    if isstopwordused == True:
        updatedtrainspamwordsList = [x for x in stemtrainspamwordsList if x not in stopwordsList]
    else:
        updatedtrainspamwordsList = stemtrainspamwordsList
    trainspamwordscount = len(updatedtrainspamwordsList)
    counttrainspamwordsdict = Counter(updatedtrainspamwordsList) 
    
    stemtrainwordList = [] + updatedtrainhamwordsList;
    stemtrainwordList.extend(updatedtrainspamwordsList)
    counttrainwordsdict = Counter(stemtrainwordList)
    
    priortrainham, priortrainspam, condprobhamDict, condprobspamDict = trainMultinomiaNB(trainhamwordscount, trainspamwordscount, counttrainhamwordsdict, counttrainspamwordsdict, counttrainwordsdict)
        
    positivecase = 0
    totalcase = 0
    testhampath = os.getcwd()+testhamDirectory    
    allhamemail = [os.path.join(testhampath,f) for f in os.listdir(testhampath)]
    for emailfile in allhamemail:
        totalcase += 1 
        updatedstemtestwordsList = []       
        stemtestwordsList = filefromdirectory(emailfile)
        if isstopwordused == True:
            updatedstemtestwordsList = [x for x in stemtestwordsList if x not in stopwordsList]
        else:
            updatedstemtestwordsList = stemtestwordsList     
        ishamclassified = applyMultinomialNB(counttrainwordsdict, priortrainham, priortrainspam, condprobhamDict, condprobspamDict, updatedstemtestwordsList)
        if ishamclassified == True:
            positivecase += 1    
    
    testspampath = os.getcwd()+testspamDirectory   
    allspamemail = [os.path.join(testspampath,f) for f in os.listdir(testspampath)]
    for emailfile in allspamemail:
        totalcase += 1     
        updatedstemtestwordsList = []       
        stemtestwordsList = filefromdirectory(emailfile)
        if isstopwordused == True:
            updatedstemtestwordsList = [x for x in stemtestwordsList if x not in stopwordsList]
        else:
            updatedstemtestwordsList = stemtestwordsList     
        ishamclassified = applyMultinomialNB(counttrainwordsdict, priortrainham, priortrainspam, condprobhamDict, condprobspamDict, updatedstemtestwordsList)
        if ishamclassified == False :
            positivecase += 1
    
    if isstopwordused == True:
         print("Accuracy achieved with stop words  :", (float(positivecase)/totalcase)*100)
    else:
         print("Accuracy achieved without stop words  :", (float(positivecase)/totalcase)*100)  


#traindirectorypoststemming = "Z:\\Lectures\\Machine_Learning\\Assignments\\Homework_2\\train"
#testdirectorypoststemming = "Z:\\Lectures\\Machine_Learning\\Assignments\\Homework_2\\test"
#stopwordsfile = "Z:\\Lectures\\Machine_Learning\\Assignments\\Homework_2\\stopwords.txt"
traindirectorypoststemming = sys.argv[2]
testdirectorypoststemming = sys.argv[3]
stopwordsfile = sys.argv[1]


stopwordsList = []
with open(stopwordsfile) as stopfile:
    for line in stopfile:
        stopwordsList.append(line.rstrip())
        
main(stopwordsList, traindirectorypoststemming, testdirectorypoststemming, False)
main(stopwordsList, traindirectorypoststemming, testdirectorypoststemming, True)