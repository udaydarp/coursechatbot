# -*- coding: utf-8 -*-
"""
Spyder Editor
"""
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
import math
from nltk.corpus import wordnet as wn
import nltk

nltk.download('corpora/wordnet')

import re
from geotext import GeoText

###################################################################################
# Sentence Bank and Intent identification
###################################################################################
from nltk.stem.lancaster import LancasterStemmer
# word stemmer
stemmer = LancasterStemmer()

training_data = []
training_data.append({"class":"greeting", "sentence":"how are you?"})
training_data.append({"class":"greeting", "sentence":"how is your day?"})
training_data.append({"class":"greeting", "sentence":"good day"})
training_data.append({"class":"greeting", "sentence":"how is it going today?"})
training_data.append({"class":"greeting", "sentence":"good morning!"})
training_data.append({"class":"greeting", "sentence":"good afternoon!"})
training_data.append({"class":"greeting", "sentence":"good evening!"})
training_data.append({"class":"greeting", "sentence":"namastey!"})
training_data.append({"class":"greeting", "sentence":"hey!"})
training_data.append({"class":"greeting", "sentence":"hi!"})

training_data.append({"class":"search", "sentence":"i want to find machine learning courses in mumbai"})
training_data.append({"class":"search", "sentence":"can you give me a list of management programs in india?"})
training_data.append({"class":"search", "sentence":"give me short-term diploma courses in computer science"})
training_data.append({"class":"search", "sentence":"show me bioengineering programs near delhi or chandigarh with 1 to 2 years duration"})
training_data.append({"class":"search", "sentence":"i am searching for machine learning courses"})
training_data.append({"class":"search", "sentence":"are there any commencing before 10 Aug 2018?"})
training_data.append({"class":"search", "sentence":"are there any commencing after 10 Aug 2018?"})
training_data.append({"class":"search", "sentence":"i want list of programs with fees 100000 rs"})
training_data.append({"class":"search", "sentence":"i need list of courses taught in carnegie mellon university"})
training_data.append({"class":"search", "sentence":"i need list of programs conducted by new york university"})

training_data.append({"class":"view", "sentence":"display the courses"})
training_data.append({"class":"view", "sentence":"show me the list of courses"})
training_data.append({"class":"view", "sentence":"list out the programs searched"})
training_data.append({"class":"view", "sentence":"i want to view the course list"})
training_data.append({"class":"view", "sentence":"yes"})

training_data.append({"class":"stop", "sentence":"thanks for sharing the list!"})
training_data.append({"class":"stop", "sentence":"no i am done"})
training_data.append({"class":"stop", "sentence":"that will be it for now"})
training_data.append({"class":"stop", "sentence":"that is all i was looking for"})
training_data.append({"class":"stop", "sentence":"good night"})
training_data.append({"class":"stop", "sentence":"that is what i needed"})
training_data.append({"class":"stop", "sentence":"ok am bored now"})
training_data.append({"class":"stop", "sentence":"ok stop it"})
training_data.append({"class":"stop", "sentence":"i dont need more"})
training_data.append({"class":"stop", "sentence":"good day!"})

training_data.append({"class":"showfees", "sentence":"what are the fees?"})
training_data.append({"class":"showfees", "sentence":"what are the damages?"})
training_data.append({"class":"showfees", "sentence":"how much does it cost?"})
training_data.append({"class":"showfees", "sentence":"what do i need to pay?"})
training_data.append({"class":"showfees", "sentence":"what amount do i pay?"})
training_data.append({"class":"showfees", "sentence":"does it cost a bomb?"})

training_data.append({"class":"structure", "sentence":"what is the course structure?"})
training_data.append({"class":"structure", "sentence":"what are the modules?"})
training_data.append({"class":"structure", "sentence":"what is the course schedule?"})

training_data.append({"class":"univs", "sentence":"which universities offer the courses?"})
training_data.append({"class":"univs", "sentence":"which universities offer the programs?"})
training_data.append({"class":"univs", "sentence":"which universities conduct the courses?"})
training_data.append({"class":"univs", "sentence":"which universities conduct the programs?"})
training_data.append({"class":"univs", "sentence":"which universities have the courses?"})
training_data.append({"class":"univs", "sentence":"which universities have the programs?"})
training_data.append({"class":"univs", "sentence":"list of universities that offer the courses"})
training_data.append({"class":"univs", "sentence":"list of universities that offer the programs"})
training_data.append({"class":"univs", "sentence":"list of universities that conduct the courses"})
training_data.append({"class":"univs", "sentence":"list of universities that conduct the programs"})
training_data.append({"class":"univs", "sentence":"list of universities that have the courses"})
training_data.append({"class":"univs", "sentence":"list of universities that have the programs"})
training_data.append({"class":"univs", "sentence":"which institutes offer the courses?"})
training_data.append({"class":"univs", "sentence":"which institutes offer the programs?"})
training_data.append({"class":"univs", "sentence":"which institutes conduct the courses?"})
training_data.append({"class":"univs", "sentence":"which institutes conduct the programs?"})
training_data.append({"class":"univs", "sentence":"which institutes have the courses?"})
training_data.append({"class":"univs", "sentence":"which institutes have the programs?"})
training_data.append({"class":"univs", "sentence":"list of institutes that offer the courses"})
training_data.append({"class":"univs", "sentence":"list of institutes that offer the programs"})
training_data.append({"class":"univs", "sentence":"list of institutes that conduct the courses"})
training_data.append({"class":"univs", "sentence":"list of institutes that conduct the programs"})
training_data.append({"class":"univs", "sentence":"list of institutes that have the courses"})
training_data.append({"class":"univs", "sentence":"list of institutes that have the programs"})
training_data.append({"class":"univs", "sentence":"which colleges offer the courses?"})
training_data.append({"class":"univs", "sentence":"which colleges offer the programs?"})
training_data.append({"class":"univs", "sentence":"which colleges conduct the courses?"})
training_data.append({"class":"univs", "sentence":"which colleges conduct the programs?"})
training_data.append({"class":"univs", "sentence":"which colleges have the courses?"})
training_data.append({"class":"univs", "sentence":"which colleges have the programs?"})
training_data.append({"class":"univs", "sentence":"list of colleges that offer the courses"})
training_data.append({"class":"univs", "sentence":"list of colleges that offer the programs"})
training_data.append({"class":"univs", "sentence":"list of colleges that conduct the courses"})
training_data.append({"class":"univs", "sentence":"list of colleges that conduct the programs"})
training_data.append({"class":"univs", "sentence":"list of colleges that have the courses"})
training_data.append({"class":"univs", "sentence":"list of colleges that have the programs"})
training_data.append({"class":"univs", "sentence":"which schools offer the courses?"})
training_data.append({"class":"univs", "sentence":"which schools offer the programs?"})
training_data.append({"class":"univs", "sentence":"which schools conduct the courses?"})
training_data.append({"class":"univs", "sentence":"which schools conduct the programs?"})
training_data.append({"class":"univs", "sentence":"which schools have the courses?"})
training_data.append({"class":"univs", "sentence":"which schools have the programs?"})
training_data.append({"class":"univs", "sentence":"list of schools that offer the courses"})
training_data.append({"class":"univs", "sentence":"list of schools that offer the programs"})
training_data.append({"class":"univs", "sentence":"list of schools that conduct the courses"})
training_data.append({"class":"univs", "sentence":"list of schools that conduct the programs"})
training_data.append({"class":"univs", "sentence":"list of schools that have the courses"})
training_data.append({"class":"univs", "sentence":"list of schools that have the programs"})

training_data.append({"class":"restart", "sentence":"no that is not what i was looking for"})
training_data.append({"class":"restart", "sentence":"no"})
training_data.append({"class":"restart", "sentence":"no i didnt want that"})
training_data.append({"class":"restart", "sentence":"no this was not what i wanted"})
training_data.append({"class":"restart", "sentence":"no this is incorrect"})
training_data.append({"class":"restart", "sentence":"no i expected something else"})
training_data.append({"class":"restart", "sentence":"i was not searching for this"})

training_data.append({"class":"aboutme", "sentence":"whats your name?"})
training_data.append({"class":"aboutme", "sentence":"why are you named cb?"})
training_data.append({"class":"aboutme", "sentence":"what can you help me with?"})
training_data.append({"class":"aboutme", "sentence":"how much course data you have?"})

# print ("%s sentences of training data" % len(training_data))

# capture unique stemmed words in the training corpus
corpus_words = {}
class_words = {}
# turn a list into a set (of unique items) and then a list again (this removes duplicates)
classes = list(set([a['class'] for a in training_data]))
for c in classes:
    # prepare a list of words within each class
    class_words[c] = []

# loop through each sentence in our training data
for data in training_data:
    # tokenize each sentence into words
    for word in nltk.word_tokenize(data['sentence']):
        # ignore a some things
        if word not in ["?", "'s"]:
            # stem and lowercase each word
            stemmed_word = stemmer.stem(word.lower())
            # have we not seen this word already?
            if stemmed_word not in corpus_words:
                corpus_words[stemmed_word] = 1
            else:
                corpus_words[stemmed_word] += 1

            # add the word to our words in class list
            class_words[data['class']].extend([stemmed_word])

# we now have each stemmed word and the number of occurances of the word in our training corpus (the word's commonality)
# print ("Corpus words and counts: %s \n" % corpus_words)
# also we have all words in each class
# print ("Class words: %s" % class_words)

###################################################################################
# calculate a score for a given class taking into account word commonality
###################################################################################
def calculate_class_score_commonality(sentence, class_name, show_details=True):
    score = 0
    # tokenize each word in our new sentence
    for word in nltk.word_tokenize(sentence):
        # check to see if the stem of the word is in any of our classes
        if stemmer.stem(word.lower()) in class_words[class_name]:
            # treat each word with relative weight
            score += (1 / corpus_words[stemmer.stem(word.lower())])

            if show_details:
                print ("   match: %s (%s)" % (stemmer.stem(word.lower()), 1 / corpus_words[stemmer.stem(word.lower())]))
    return score

class Intent(object):
    intentType=''
    response=''
    filterQuery=''

# Method to split text into individual parts for intent identification
def splitText(inputText):
    return sent_tokenize(inputText)

# Method to show some dynamic text in relation to intentType
def getIntentResponse(intentType):
    if (intentType == 'greeting'):
        return '*** Greets me back ***'
    elif (intentType == 'search'):
        return '*** Searching... ***'
    elif (intentType == 'stop'):
        return '*** Phew! Done for the day! ***'
    elif (intentType == 'restart'):
        return '*** Back to square one! ***'
    elif (intentType == 'univs'):
        return '*** Fetching universities ***'
    elif (intentType == 'showfees'):
        return '*** Fetching fees ***'
    elif (intentType == 'structure'):
        return '*** Fetching course structure ***'
    elif (intentType == 'view'):
        return '*** Showtime! ***'
    elif (intentType == 'aboutme'):
        return '*** Now I can brag! ;) ***'
    else:
        return '*** Thinking ***'
        
# Method to use machine learning techniques to check intent
def getIntent(inputText):
    high_class = None
    high_score = 0
    intent = Intent()
    # loop through our classes
    for c in class_words.keys():
        # calculate score of sentence for each class
        score = calculate_class_score_commonality(inputText, c, show_details=False)
        # keep track of highest score
        if score > high_score:
            high_class = c
            high_score = score
            
    intent.intentType = high_class
    intent.response = getIntentResponse(intent.intentType)
    #intent.response = '<<Your intent is: '+high_class+'>>'
    intent.filterQuery = 'df["country_name"]=="India"'
    return intent


# Method to identify intents from a user response
def identifyIntents (response):
    lstIntents = []
    #Split the response into distinct sentences
    lstParts = splitText(response)
    
    # For each sentence identify the intent
    for part in lstParts:
        intent = getIntent(part)
        lstIntents.append(intent)
    # 
    return lstIntents

##################################################
# Identify entities. Not in use currently
##################################################
class Socket(object):
    cities=''
    countries=''
    commencement_date_from=''
    commencement_date_to=''
    commencement_date_comp_operator=''
    fee_amount=''
    fee_currency=''
    fee_comparison_operator=''
    duration_from=''
    duration_to=''
    duration_unit=''
    program_name=''
    course_name=''
    
def to_number(s):
    try:
        s1 = float(s)
        return s1
    except ValueError:
        return 0
    except TypeError:
        return 0

month_dict = {"jan":"01",
              "feb":"02",
              "mar":"03",
              "apr":"04",
              "may":"05",
              "jun":"06",
              "jul":"07",
              "aug":"08",
              "sep":"09",
              "oct":"10",
              "nov":"11",
              "dec":"12"
              }

def getNumericMonth(month):
    if month.isnumeric():
        month = "0" + month
        return month[len(month)-2:len(month)]
    else:
        strMonth = month.lower()[0:3]
        return month_dict[strMonth]

#getNumericMonth('01')
#getNumericMonth('1')
#getNumericMonth('jan')
#getNumericMonth('nov')
#getNumericMonth('12')

def findDate(inpString):
    day = ''
    month = ''
    year = ''
    # check for dmy format
    datePattern = re.compile(r'(([1-9]|[0][1-9]|[1-2][0-9]|[3][0-1])[ \./-](Jan|Feb|Mar|Apr|May|Jun|July|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December|[0]*[1-9]|[1][0-2])[ \./-](\d{4}))')
    validDate = datePattern.findall(inpString)
    
    if (validDate == None or validDate == []):
        # check for ymd format
        datePattern = re.compile(r'((\d{4})[ \./-]([0]*[1-9]|[1][0-2])[ \./-]([0]*[1-9]|[1-2][0-9]|[3][0-1]))')
        validDate = datePattern.findall(inpString)
    
        if (validDate == None or validDate == []):
            # check for month year format
            datePattern = re.compile(r'((Jan|Feb|Mar|Apr|May|Jun|July|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)[ \./-](\d{4}))')
            validDate = datePattern.findall(inpString)
            if (validDate != None and validDate != []):
                # valid month-year date
                print("my:",validDate)
                day = '01'
                month = validDate[0][1]
                year = validDate[0][2]
        else:
            # valid ymd date
            print("ymd:",validDate)
            year = validDate[0][1]
            month = validDate[0][2]
            day = validDate[0][3]
    else:
        # valid dmy date
        print("dmy:",validDate)
        day = validDate[0][1]
        month = validDate[0][2]
        year = validDate[0][3]
        
    if (validDate == None or validDate == []):
        #print('No valid date in:',inpString)
        return None
    else:
        #print('Valid date found:', validDate)
        #return validDate[0][0]
        
        #return date in y-m-d format
        numMonth = getNumericMonth(month)
        return year + "-" + numMonth + "-" + day

#findDate("before 01 Jan 2018")
#findDate("before May 2018")
#findDate("before 05-Nov-2018")
#findDate("before 2018-03-31")
#findDate("after 5 August 2018")


def findAmount(inpString):
    amountPattern = re.compile(r'[^ ][\d]+[,\.\d]*(^month|day|year)[$ ]')
    amount=amountPattern.findall(inpString)
    #print(amount)
    return amount

def findCurrency(inpString):
    currencyPattern=re.compile(r'[$₹]|inr|usd|Rs\.*|rupees|dollars')
    currency=currencyPattern.findall(inpString.lower())
    #print(currency)
    return currency

# check for duration entered and convert the figure to days
# here 1 month = 30 days, 1 year = 360 days
def findDuration(inpString):
    durationPattern=re.compile(r'(\d+) *(month|year|day)')
    duration=durationPattern.findall(inpString.lower())
    
    if duration == None or duration == '' or duration == []:
        return None
    
    numValue = duration[0][0]
    unitValue = duration[0][1]
    
    durationInDays = int(numValue)
    
    if unitValue.lower() == 'day':
        durationInDays = durationInDays
    elif unitValue.lower() == 'month':
        durationInDays = durationInDays * 30
    else:
        durationInDays = durationInDays * 360
        
    return durationInDays

###################################################################################
# Global variables. Reference as "global" in the methods that you use them
###################################################################################

amount = ''
currency = ''
duration = ''
date = ''
cities = ''
countries = ''
filterQuery = ''
showuniv = False
showfees = False
showstructure = False
showcity = False
showcountry = False
showduration = False
showdate = False
outputColumns = '\'program_name\''
    
###################################################################################
# Method to initialize the global variables on every search error to reset the data
###################################################################################
def clearEntities():
    global amount
    global currency
    global duration
    global date
    global cities
    global countries
    global filterQuery
    global outputColumns
    global showuniv
    global showfees
    global showstructure
    global showcity
    global showcountry
    global showduration
    global showdate
    
    amount = ''
    currency = ''
    duration = ''
    date = ''
    cities = ''
    countries = ''
    filterQuery = ''
    showuniv = False
    showfees = False
    showstructure = False
    showcity = False
    showcountry = False
    showduration = False
    showdate = False
    outputColumns = '\'program_name\''

###################################################################################
# Method to identify all the entities/sockets within the search
###################################################################################
def findEntities(inpString):
    global amount
    global currency
    global duration
    global date
    global cities
    global countries
    global filterQuery
    global showfees
    global showuniv
    global showcity
    global showcountry
    global showduration
    global showdate
    
    andText = ''
    
    loc = GeoText(inpString)
    if  loc != None:
        if loc.cities != None and loc.cities != []:
            #filterQuery = filterQuery + andText + 'df["city"].isin (['
            comma = ''
            
            #if cities != None and cities != '':
            #    comma = ','
            cities = ''
                
            for city in loc.cities:
                #filterQuery = filterQuery + comma + '"' + city + '"'
                cities = cities + comma + '"' + city + '"'
                comma = ','
            #filterQuery = filterQuery + '])'
            #andText = ' & '
            
        if loc.countries != None and loc.countries != []:
            #filterQuery = filterQuery + andText + 'df["country_name"].isin (['
            comma = ''
            #if countries != None and countries != '':
            #    comma = ','
            countries = ''
            
            for country in loc.countries:
                #filterQuery = filterQuery + comma + '"' + country + '"'
                countries = countries + comma + '"' + country + '"'
                comma = ','
            #filterQuery = filterQuery + '])'
            #andText = ' & '

    localDate = findDate(inpString)
    if localDate != None:
        date = localDate
    
    localAmount = findAmount(inpString)
    if (localAmount != None and localAmount != []):
        amount = localAmount
        
    localCurrency = findCurrency(inpString)
    if (localCurrency != None and localCurrency != []):
        currency = localCurrency
        
    localDuration = findDuration(inpString)
    if (localDuration != None and localDuration != []):
        duration = localDuration

    filterQuery = ''
    
    if cities != None and cities != '':
        filterQuery = filterQuery + andText + '(df["cityName"].isin (['
        filterQuery = filterQuery + cities
        filterQuery = filterQuery + ']))'
        andText = ' & '
        showcity = True
    
    if countries != None and countries != '':
        filterQuery = filterQuery + andText + '(df["country_name"].isin (['
        filterQuery = filterQuery + countries
        filterQuery = filterQuery + ']))'
        andText = ' & '
        showcountry = True
        
    if (date != None and date != ''):
        filterQuery = filterQuery + andText + '(df["start_date"] == "' + date + '")'
        andText = ' & '
        showdate = True
    
    if (amount != None and amount != [] and amount != ''):
        filterQuery = filterQuery + andText + '(df["tution_1_money"] == ' + amount[0] + ')'
        andText = ' & '
        showfees = True

    if (currency != None and currency != [] and currency != ''):
        filterQuery = filterQuery + andText + '(df["tution_1_currency"] == "' + currency[0] + '")'
        andText = ' & '
        showfees = True

    if (duration != None and duration != [] and duration != ''):
        filterQuery = filterQuery + andText + '(df["durationInDays"] == ' + str(duration) + ')'
        andText = ' & '
        showduration = True
    
    #print("CB: <<The student is searching for a course in ",loc.cities," city in ",loc.countries," with cost ",currency," ",amount, " and commencement date ", date, " with duration ", duration,">>")
    #print("CB: <<FilterQuery:", filterQuery,">>")
    return filterQuery

###################################################################################
# Method to display results of the search
###################################################################################
def displayResults():
    global filterQuery
    global outputColumns
    global showuniv
    global showfees
    global showstructure
    global showcity
    global showcountry
    global showduration
    global showdate
    
    outputColumns = '\'program_name\''
    
    if showcity == True:
        outputColumns = outputColumns + ',' + '\'cityName\''
    
    if showcountry == True:
        outputColumns = outputColumns + ',' + '\'country_name\''
        
    if showduration == True:
        outputColumns = outputColumns + ',' + '\'duration\''
        
    if showdate == True:
        outputColumns = outputColumns + ',' + '\'start_date\''
        
    if showfees == True:
        outputColumns = outputColumns + ',' + '\'tution_1_currency\',\'tution_1_money\''
        
    if showuniv == True:
        outputColumns = outputColumns + ',' + '\'university_name\''
        
    if showstructure == True:
        outputColumns = outputColumns + ',' + '\'structure\''
        
    if filterQuery != '':
        filterQueryExec = 'data = df['+filterQuery+'][['+outputColumns+']]'
        print(filterQueryExec)
        exec(filterQueryExec)
        if (len(data) == 0):
            print("CB: Sorry I did not find any courses matching your search :(. Try again")
            clearEntities()
        else:
            print("CB:I found ", len(data), " courses:")
            print(data)

##################################################################################
# Main chat program: This is the one that will process user inputs and respond!
##################################################################################

# Initialization
result = None
stopSearch = False
df = pd.read_csv("C:\\Uday\\AEGIS\\NLP\\ChatBot\\201709301651_masters_portal.csv")

########################################
# Convert the duration to value in days
# to make this uniform across all rows
# Add column for duration in days
########################################
durationInDays = []
for idx,d in enumerate(df['duration']):
    if (isinstance(d, str)):
        durn=findDuration(d)
        durationInDays.append(to_number(durn))
    else:
        durationInDays.append(to_number(0))

df = df.assign(durationInDays=durationInDays)

########################################
# Extract name of city and store in a 
# separate column.
########################################
cityName = []
cityPattern=re.compile(r'(\[\'([A-Za-z]+)\'\])')
for idx,c in enumerate(df['city']):
    if (isinstance(c, str)):
        city = cityPattern.findall(c)
        if (city != None and city != []):
            cityName.append(city[0][1])
        else:
            cityName.append(c) # whatever is in the column
    else:
        cityName.append(c) # whatever is in the column
        
df = df.assign(cityName=cityName)

randomResultSize=20

# Greet user
lstGreetings = ["Hello!","Hi There!","Hi! How are you doing today?","Welcome to my world!","Namastey!"]
print("CB:",lstGreetings[np.random.randint(0,len(lstGreetings))])
print("CB: How may I help you?")

# Start the endless interaction!
while (True):
    response=input("You: ")
    
    lstIntents = identifyIntents(response)
    for intent in lstIntents:
        if intent.intentType == 'greeting':
            print ("CB: ", intent.response)
        elif intent.intentType == 'search':
            print ("CB: ", intent.response)
            filterQuery = findEntities(response)
            filterQueryExec = 'data = df['+filterQuery+'][['+outputColumns+']]'
            print(filterQueryExec)
            exec(filterQueryExec)
            resultSize = len(data)
            if resultSize == 0:
                filterQuery = '' # clear the query
                print("CB: Sorry I did not find any courses matching your search :(. Try again")
                clearEntities()
            elif resultSize > 50:
                print ("CB: I found ", resultSize, " courses matching your search. Tell me the program types or location or university you want to look for and we can narrow down the list further")
                print ("CB: You dont want me to dump so many on you ;)!")
            else:
                print("CB: I found ",resultSize," courses matching your search. Do you want to view them or filter them further?")
        elif intent.intentType == 'view':
            print ("CB: ", intent.response)
            displayResults()
        elif intent.intentType == 'stop':
            print ("CB: ", intent.response)
            stopSearch = True
        elif intent.intentType == 'restart':
            print ("CB: ", intent.response)
        elif intent.intentType == 'structure':
            showstructure = True
            displayResults()
        elif intent.intentType == 'showfees':
            showfees = True
            displayResults()
        elif intent.intentType == 'univs':
            showunivs = True
            displayResults()
        else:
            print ("CB: I cant understand your intent :(! Please have a human communicate with me!")
    
    if (stopSearch):
        break
# End while

# End of the chat session
print("CB: It was nice talking to you! Have a good day! If you like me, give me 5*s ;)")
                
###################################################################################
# This section below is to play around with the code. Keep this commented once done.
#df[(df["cityName"].isin (["Mumbai"])) & (df["durationInDays"] == 540)][['program_name','durationInDays','cityName','duration']]
#to_number(df[df["cityName"].isin (["Mumbai"])]['durationInDays'])==to_number(540)
#df[df["cityName"].isin (["Mumbai"]) & to_number(df["durationInDays"]) == to_number(540)][['program_name','durationInDays','cityName','duration']]
#df[df[df["cityName"].isin (["Mumbai"])]['durationInDays']==to_number(540)]['program_name']
#df['durationInDays']==12
# df[(df["country_name"].isin (["India"])) & (df["durationInDays"] == 540)][['program_name','country_name','duration','university_name','tution_1_currency','tution_1_money']]
###################################################################################