import pandas as pd
import os
import math
import sys
from collections import Counter

def sigmoid(z):
    try:    
        return math.exp(z)/float(1.0 + math.exp(z))
    except OverflowError:
        return 1.0


def classify(matrix, row, featureweightsList):
    sumweight = 0.00
    count = 0
    for weight in featureweightsList:
        sumweight += float(weight) * matrix[row][count]
        count += 1
    return sigmoid(sumweight) 


def trainLogisticRegression(dataframe, featureweightsList, deltaWeightList, regularizationFactor):
    #setting up prediction first
    learningrate = 0.0001
    matrix = dataframe.values
    rows = dataframe.shape[0]
    columns = dataframe.shape[1]    
    
    for i in range(0,10):
        predictionList = []
        for j in range(0, rows):
            predictionList.append(classify(matrix, j, featureweightsList))
        
        for elem in deltaWeightList:
            elem = 0.00
            
        for i in range(0,len(featureweightsList)):
            for j in range(0, rows):
                deltaWeightList[i] = deltaWeightList[i] + matrix[j][i] * (matrix[j][columns-1] - predictionList[j])
                
        for j in range(0, len(featureweightsList)):            
            featureweightsList[j] = featureweightsList[j] + learningrate*(deltaWeightList[j] - (regularizationFactor * featureweightsList[j])) 
     
    #print('feature weight list :', featureweightsList)       
    return featureweightsList


def testLogisticRegression(testdataframe, featureweightsList, isStopWordUsed):
    predictiontestList = []
    matrix = testdataframe.values
    totalrow = testdataframe.shape[0]
    totalcolumn = testdataframe.shape[1]
    correctCount = 0
    for j in range(0, totalrow):
        prediction = classify(matrix, j, featureweightsList)
        predictiontestList.append(prediction)
        if(prediction >= 0.5):
            if(matrix[j][totalcolumn - 1] == 1):
                correctCount += 1
        elif(prediction < 0.5):
            if(matrix[j][totalcolumn - 1] == 0):
                correctCount += 1 
          
    if(isStopWordUsed == True):
        print('accuracy achieved over the test set using stop words is :', (float(correctCount)/totalrow) * 100)
    else:
        print('accuracy achieved over the test set without using stop words is :', (float(correctCount)/totalrow) * 100)            

            
def generateFileTable(readfile, isStopWordUsed, stopwordsList): 
    with open(readfile,'r') as m:
        trainwordsList=[]
        for line in m:
            for word in line.split(): 
                lowerword=word.lower()
                if(lowerword.isalnum()):
                    if(isStopWordUsed == False):
                        trainwordsList.append(lowerword)
                    else:
                        if(lowerword not in stopwordsList):
                            trainwordsList.append(lowerword)      
    return trainwordsList


def main(stopwordsList, traindirectorypoststemming, testdirectorypoststemming, isStopWordUsed, regularizationFactor):
    trainhamDirectory = traindirectorypoststemming + "/stemmed_ham"
    trainspamDirectory = traindirectorypoststemming + "/stemmed_spam"
    testhamDirectory = testdirectorypoststemming + "/stemmed_ham"
    testspamDirectory = testdirectorypoststemming + "/stemmed_spam"
    featureweightsList = []
    deltaWeightList = []
      
    #generating word list for train file now
    path = os.getcwd()+trainhamDirectory
    hamfilewordsList = []   
    trainwordList = []
    allhamemail = [os.path.join(path,f) for f in os.listdir(path)]
    for emailfile in allhamemail:
        tempList = generateFileTable(emailfile, isStopWordUsed, stopwordsList)
        hamfilewordsList.append(tempList)
        trainwordList.extend(tempList)


    path = os.getcwd()+trainspamDirectory
    spamfilewordsList = []
    allspamemail = [os.path.join(path,f) for f in os.listdir(path)]
    for emailfile in allspamemail:
        tempList = generateFileTable(emailfile, isStopWordUsed, stopwordsList)
        spamfilewordsList.append(tempList)
        trainwordList.extend(tempList)
    
    #dictionary for the feature word map generation     
    featurewordDict = Counter(trainwordList)    
    featureKeyList =  featurewordDict.keys()
    finalfeatureKeyList = ['defweight']
    featureweightsList.append(0.00)
    deltaWeightList.append(0.00)
    for i in range(0, len(featureKeyList)):
        featureweightsList.append(0.00)
        deltaWeightList.append(0.00)
    finalfeatureKeyList.extend(featureKeyList)
    finalfeatureKeyList.append('classifierhamspam')
        
        
    dataframe = pd.DataFrame(columns=finalfeatureKeyList)
    rownumber = 0   
    for hamfile in hamfilewordsList:       
        hamDict = Counter(hamfile)
        tempFileList = [1]
        for featureName in featureKeyList:
            if(featureName in hamDict):
                tempFileList.append(hamDict[featureName])
            else:                
                tempFileList.append(0)
            
        tempFileList.append(0)
        dataframe.loc[rownumber] = tempFileList
        rownumber+=1    
        
    for spamfile in spamfilewordsList:
        spamDict = Counter(spamfile)
        tempFileList = [1]
        for featureName in featureKeyList:
            if(featureName in spamDict):
                tempFileList.append(spamDict[featureName])
            else:                
                tempFileList.append(0)
            
        tempFileList.append(1)
        dataframe.loc[rownumber] = tempFileList 
        rownumber+=1    
    
    featureweightsList = trainLogisticRegression(dataframe, featureweightsList, deltaWeightList, regularizationFactor)


    #Reading the test files and preparing for testing
    path = os.getcwd()+testhamDirectory
    testhamfilewordsList = []   
    testwordList = []
    alltesthamemail = [os.path.join(path,f) for f in os.listdir(path)]
    for emailfile in alltesthamemail:
        tempList = generateFileTable(emailfile, isStopWordUsed, stopwordsList)
        testhamfilewordsList.append(tempList)
        testwordList.extend(tempList)

    path = os.getcwd()+testspamDirectory
    testspamfilewordsList = []
    alltestspamemail = [os.path.join(path,f) for f in os.listdir(path)]
    for emailfile in alltestspamemail:
        tempList = generateFileTable(emailfile, isStopWordUsed, stopwordsList)
        testspamfilewordsList.append(tempList)
        testwordList.extend(tempList)
                
        
    testdataframe = pd.DataFrame(columns=finalfeatureKeyList)
    rownumber = 0   
    for testhamfile in testhamfilewordsList:       
        testhamDict = Counter(testhamfile)
        tempFileList = [1]
        for featureName in featureKeyList:
            if(featureName in testhamDict):
                tempFileList.append(testhamDict[featureName])
            else:                
                tempFileList.append(0)
            
        tempFileList.append(0)
        testdataframe.loc[rownumber] = tempFileList
        rownumber+=1    
        
    for testspamfile in testspamfilewordsList:
        testspamDict = Counter(testspamfile)
        tempFileList = [1]
        for featureName in featureKeyList:
            if(featureName in testspamDict):
                tempFileList.append(testspamDict[featureName])
            else:                
                tempFileList.append(0)
            
        tempFileList.append(1)
        testdataframe.loc[rownumber] = tempFileList 
        rownumber+=1
     
     
    testLogisticRegression(testdataframe, featureweightsList, isStopWordUsed)   

traindirectorypoststemming = sys.argv[2]
testdirectorypoststemming = sys.argv[3]
stopwordsfile = sys.argv[1]

stopwordsList = []
with open(stopwordsfile) as stopfile:
    for line in stopfile:
        stopwordsList.append(line.rstrip())

regularizationFactor = 0.50       
main(stopwordsList, traindirectorypoststemming, testdirectorypoststemming, False, regularizationFactor)
main(stopwordsList, traindirectorypoststemming, testdirectorypoststemming, True, regularizationFactor)