# -*- coding: utf-8 -*-
"""
Spyder Editor
"""
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
from nltk.corpus import stopwords
from datetime import datetime
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

#from nltk.corpus import wordnet as wn

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
training_data.append({"class":"greeting", "sentence":"hello!"})
training_data.append({"class":"greeting", "sentence":"hey there!"})

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
training_data.append({"class":"search", "sentence":"i am looking for courses with ranking less than 50"})
training_data.append({"class":"search", "sentence":"top ranked courses"})
training_data.append({"class":"search", "sentence":"i want list of top ranked courses"})
training_data.append({"class":"search", "sentence":"i want list of top 50 courses"})

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
training_data.append({"class":"stop", "sentence":"bye"})
training_data.append({"class":"stop", "sentence":"good bye"})
training_data.append({"class":"stop", "sentence":"stop"})

training_data.append({"class":"showfees", "sentence":"what are the fees?"})
training_data.append({"class":"showfees", "sentence":"what are the damages?"})
training_data.append({"class":"showfees", "sentence":"how much does it cost?"})
training_data.append({"class":"showfees", "sentence":"what do i need to pay?"})
training_data.append({"class":"showfees", "sentence":"what amount do i pay?"})
training_data.append({"class":"showfees", "sentence":"does it cost a bomb?"})
training_data.append({"class":"showfees", "sentence":"show me the fees"})
training_data.append({"class":"showfees", "sentence":"show me the program fees"})
training_data.append({"class":"showfees", "sentence":"show me the course fees"})

training_data.append({"class":"structure", "sentence":"what is the course structure?"})
training_data.append({"class":"structure", "sentence":"what are the modules?"})
training_data.append({"class":"structure", "sentence":"what is the course schedule?"})
training_data.append({"class":"structure", "sentence":"show me the structure"})
training_data.append({"class":"structure", "sentence":"show me the program structure"})
training_data.append({"class":"structure", "sentence":"show me the course structure"})

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

training_data.append({"class":"showrank", "sentence":"show me the rankings"})
training_data.append({"class":"showrank", "sentence":"how are they ranked"})
training_data.append({"class":"showrank", "sentence":"what are the rankings"})

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

training_data.append({"class":"showdebug", "sentence":"show me debug"})
training_data.append({"class":"showdebug", "sentence":"debug"})
training_data.append({"class":"showdebug", "sentence":"turn on debug"})
training_data.append({"class":"showdebug", "sentence":"start debugging"})

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
        # ignore some things
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
    elif (intentType == 'showrank'):
        return '*** Fetching rankings ***'
    elif (intentType == 'showdebug'):
        return '*** Turning on debug ***'
    elif (intentType == 'structure'):
        return '*** Fetching course structure ***'
    elif (intentType == 'view'):
        return '*** Showtime! ***'
    elif (intentType == 'aboutme'):
        return '*** Now I can brag! ;) ***'
    else:
        return '*** Thinking ***'

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
class Entity(object):
    amount = ''
    amountCompOp = '=='
    currency = ''
    duration = ''
    durationCompOp = '=='
    date = ''
    dateCompOp = '=='
    rank = 1000
    cities = ''
    countries = ''
    showuniv = False
    showfees = False
    showstructure = False
    showrank = False
    showcity = False
    showcountry = False
    showduration = False
    showdate = False
    outputColumns = '\'program_name\''
    
    def __init__(self):
        self.amount = ''
        self.amountCompOp = '=='
        self.currency = ''
        self.duration = ''
        self.durationCompOp = '=='
        self.date = ''
        self.dateCompOp = '=='
        self.rank = 1000
        self.cities = ''
        self.countries = ''
        self.showuniv = False
        self.showfees = False
        self.showstructure = False
        self.showrank = False
        self.showcity = False
        self.showcountry = False
        self.showduration = False
        self.showdate = False
        self.outputColumns = '\'program_name\''
        
    def dump(self):
        print("*************************************************")
        print("Entity object:")
        print("Amount:", self.amount)
        print("AmountCompOp:", self.amountCompOp)
        print("Currency:", self.currency)
        print("Duration:", self.duration)
        print("DurationCompOp:", self.durationCompOp)
        print("Date:", self.date)
        print("DateCompOp:", self.dateCompOp)
        print("Rank:", self.rank)
        print("Cities:", self.cities)
        print("Countries:", self.countries)
        print("ShowUniv:", self.showuniv)
        print("ShowFees:", self.showfees)
        print("ShowStructure:", self.showstructure)
        print("ShowRank:", self.showrank)
        print("ShowCity:", self.showcity)
        print("ShowCountry:", self.showcountry)
        print("ShowDuration:", self.showduration)
        print("ShowDate:", self.showdate)
        print("OutputColumns:", self.outputColumns)
        print("*************************************************")
    
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

# Tests:
#getNumericMonth('01')
#getNumericMonth('1')
#getNumericMonth('jan')
#getNumericMonth('nov')
#getNumericMonth('12')

###################################################################################
# Method to determine comparison operator based on entered text
###################################################################################
def getComparisonOperator(notString, prefixString):
    notString = notString.lower()
    prefixString = prefixString.lower()
    
    if prefixString == 'before' or prefixString == 'earlier than' or prefixString == 'less than' or prefixString == 'under' or prefixString == 'below':
        if notString == 'not':
            return '>'
        else:
            return '<'
    elif prefixString == 'after' or prefixString == 'later than' or prefixString == 'greater than' or prefixString == 'more than' or prefixString == 'above':
        if notString == 'not':
            return '<'
        else:
            return '>'
    else:
        return '=='

###################################################################################
# Find date entered in text and return y-m-d format along with comparison operator
###################################################################################
def findDate(inpString):
    day = ''
    month = ''
    year = ''
    notString = ''
    prefixString = ''
    
    # check for dmy format
    sepPat = '[ \./-]'
    dayPat = '[1-9]|[0][1-9]|[1-2][0-9]|[3][0-1]'
    monthPat = 'Jan|Feb|Mar|Apr|May|Jun|July|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December|[0]*[1-9]|[1][0-2]'
    yearPat = '\d{4}'
    notPat = 'not'
    prefixPat = 'before|after|by|on|earlier than|later than'
    
    datePattern = re.compile(r'(('+notPat+')*[ ]*('+prefixPat+')*[ ]*('+dayPat+')'+sepPat+'('+monthPat+')'+sepPat+'('+yearPat+'))')
    validDate = datePattern.findall(inpString)
    
    if (validDate == None or validDate == []):
        # check for ymd format
        datePattern = re.compile(r'(('+notPat+')*[ ]*('+prefixPat+')*[ ]*('+yearPat+')'+sepPat+'('+monthPat+')'+sepPat+'('+dayPat+'))')
        validDate = datePattern.findall(inpString)
    
        if (validDate == None or validDate == []):
            # check for month year format
            datePattern = re.compile(r'(('+notPat+')*[ ]*('+prefixPat+')*[ ]*('+monthPat+')'+sepPat+'('+yearPat+'))')
            validDate = datePattern.findall(inpString)
            if (validDate != None and validDate != []):
                # valid month-year date
                print("my:",validDate)
                day = '01'
                notString = validDate[0][1]
                prefixString = validDate[0][2]
                month = validDate[0][3]
                year = validDate[0][4]
        else:
            # valid ymd date
            print("ymd:",validDate)
            notString = validDate[0][1]
            prefixString = validDate[0][2]
            year = validDate[0][3]
            month = validDate[0][4]
            day = validDate[0][5]
    else:
        # valid dmy date
        print("dmy:",validDate)
        notString = validDate[0][1]
        prefixString = validDate[0][2]
        day = validDate[0][3]
        month = validDate[0][4]
        year = validDate[0][5]
        
        print("Not and Prefix:", notString, prefixString)
    if (validDate == None or validDate == []):
        #print('No valid date in:',inpString)
        return None, None
    else:
        #print('Valid date found:', validDate)
        #return validDate[0][0]
        
        #return date in y-m-d format
        numMonth = getNumericMonth(month)
        return getComparisonOperator(notString, prefixString), (year + "-" + numMonth + "-" + day)

# Tests:
#findDate("before 01 Jan 2018")
#findDate("not before May 2018")
#findDate("not earlier than 05-Nov-2018")
#findDate("not later than 2018-03-31")
#findDate("after 5 August 2018")

###################################################################################
# Find amount entered (comma separated, non-comma separated
###################################################################################
def findAmount(inpString):
    notPat = 'not'
    prefixPat = 'less than|more than|under|above|below'
    amountPat = '[\d]+[,\.\d]*(?!month|year|day|months|years|days)'
    
    amountPattern = re.compile(r'(('+notPat+')*[ ]*('+prefixPat+')*[ ]*('+amountPat+'))')
    amountValue = amountPattern.findall(inpString)
    
    if amountValue == None or amountValue == [] or amountValue == '':
        return None, None
    
    #print ("AmountValue:", amountValue)
    notString = amountValue[0][1]
    prefixString = amountValue[0][2]
    compOperator = getComparisonOperator(notString, prefixString)
    
    amount = amountValue[0][3]
    
    #print ("notString:", notString, "prefixString:", prefixString, "compOperator:", compOperator,"amount:", amount)
    return compOperator, amount

###################################################################################
# Find currency entered in text ($s and Rs only as of now)
###################################################################################
def findCurrency(inpString):
    currencyPattern=re.compile(r'[$₹]|inr|usd|Rs\.*|rupees|dollars')
    currency=currencyPattern.findall(inpString.lower())
    #print(currency)
    if currency == None or currency == []:
        return None
    
    return currency[0]

###################################################################################
# Check for duration entered and convert the figure to days
# Here 1 month = 30 days, 1 year = 360 days
###################################################################################
def findDuration(inpString):
    notPat = 'not'
    prefixPat = 'more than|greater than|less than'
    numDurationPat = '\d+'
    unitDurationPat = 'month|year|day'
    
    durationPattern=re.compile(r'(('+notPat+')*[ ]*('+prefixPat+')[ ]*('+numDurationPat+')[ ]*('+unitDurationPat+'))')
    duration=durationPattern.findall(inpString.lower())
    
    if duration == None or duration == '' or duration == []:
        return None, None
    
    notString = duration[0][1]
    prefixString = duration[0][2]
    numValue = duration[0][3]
    unitValue = duration[0][4]
    
    #print (duration, "|", notString, "|", prefixString, "|", numValue, "|", unitValue)
    
    compOperator = getComparisonOperator(notString, prefixString)
    
    durationInDays = int(numValue)
    
    if unitValue.lower() == 'day':
        durationInDays = durationInDays
    elif unitValue.lower() == 'month':
        durationInDays = durationInDays * 30
    else:
        durationInDays = durationInDays * 360
        
    return compOperator, durationInDays

###################################################################################
# Method to identify cities and countries in a text
###################################################################################
def findCitiesAndCountries(inpString):
    cityFound = False
    countryFound = False
    
    locCities = []
    locCountries = []

    loc = GeoText(inpString)
        
    if loc != None:
        if loc.cities != None and loc.cities != []:                
            for city in loc.cities:
                locCities.append(city)
            cityFound = True
            
        if loc.countries != None and loc.countries != []:
            for country in loc.countries:
                locCountries.append(country)
            countryFound = True
    
    if not cityFound or not countryFound:
        # split string into individual words, make camel case and check again
        stop_words = set(stopwords.words('english'))
        stop_words.remove('not')
        
        for w in inpString.split():
            w = w.capitalize()
            if not w.lower() in stop_words:
                loc = GeoText(w)
                
                if loc != None:
                    if cityFound == False and loc.cities != None and loc.cities != []:
    
                        for city in loc.cities:
                            locCities.append(city)
                        
                    if countryFound == False and loc.countries != None and loc.countries != []:
                        
                        for country in loc.countries:
                            locCountries.append(country)
    
    return locCities, locCountries

##################################################################
# Method to detect if ranking has been specified in the search
##################################################################
def findRanking(inpString):
    topString = ''
    strRank = ''
    numRank1 = ''
    numRank2 = ''
    rankPattern = re.compile(r'(((top)+[ ]*([\d]*)(ranked|ranking|rank)*)|((ranking|ranked|rank)+[A-Za-z ]*([\d]*)))')
    #rankPattern = re.compile(r'([A-Za-z(?!top)]*)')
    
    rank = rankPattern.findall(inpString)
    
    #print(rank)
    if rank == None or rank == []:
        return None
    else:
        if rank[0][1] != '':
            topString = rank[0][2]
            numRank1 = rank[0][3]
            strRank = rank[0][4]
            numRank2 = rank[0][5]
        else:
            strRank = rank[0][6]
            numRank2 = rank[0][7]
            
        if topString != None and topString != '':
                if strRank != None and strRank != '':
                    if numRank1 == None or numRank1 == '' or numRank1 == []:
                        numRank1 = 10 # top-10
                    
        if numRank1 != None and numRank1 != '' and numRank1 != []:
            return numRank1
        else:
            return numRank2

#findRanking("top ranked")
#findRanking("  top ranked  ")
#findRanking("i am looking for top ranked courses")
#findRanking("top 20 rank")
#findRanking(" ranking less than 50")
        
###################################################################################
# Global variables. Reference as "global" in the methods that you use them
###################################################################################

entity = Entity()
prev_entity = entity
bot_name = 'N.I.C.E.'
bot_full_name = '"Naturally" Intelligent Chatbot for Educational courses'
showdebug = False
course_vocab = [] # unique list of course-related words in the dataframe
df_course_list = [] # per row list of course-related words in the dataframe
df_course_matrix = []
nn_words = []
row_map = []

###################################################################################
# Method to initialize the global variables on every search error to reset the data
###################################################################################
def clearEntities():    
    global entity
    global prev_entity
    global course_vocab
    global df_course_list
    global df_course_matrix
    global nn_words
    global row_map

    if showdebug:
        print("DBG: Clearing:", entity.dump())
        print("DBG: Setting entity to:", prev_entity.dump())
    entity = prev_entity
    prev_entity.__init__()
    course_vocab = [] # unique list of course-related words in the dataframe
    df_course_list = [] # per row list of course-related words in the dataframe
    df_course_matrix = []
    nn_words = []
    row_map = []

###################################################################################
# Method to go backup a successfully executed entity
###################################################################################
def saveEntity():
    global entity
    global prev_entity

    if showdebug:
        print("DBG: Saving:", entity.dump())
    prev_entity = entity

##################################################################
# Method to build dataset query
##################################################################
def buildQuery():    
    global entity
    
    andText = ''
    filterQuery = ''
    
    if entity.cities != None and entity.cities != '':
        filterQuery = filterQuery + andText + '(df["cityName"].str.lower().isin (['
        filterQuery = filterQuery + entity.cities
        filterQuery = filterQuery + ']))'
        andText = ' & '
        entity.showcity = True
        if showdebug:
            print("DBG:showcity:", entity.showcity)
    
    if entity.countries != None and entity.countries != '':
        filterQuery = filterQuery + andText + '(df["country_name"].str.lower().isin (['
        filterQuery = filterQuery + entity.countries
        filterQuery = filterQuery + ']))'
        andText = ' & '
        entity.showcountry = True
        
    if (entity.date != None and entity.date != ''):
        filterQuery = filterQuery + andText + '(df["start_date_conv"] ' + entity.dateCompOp + '"' + entity.date + '")'
        andText = ' & '
        entity.showdate = True
    
    if (entity.amount != None and entity.amount != [] and entity.amount != ''):
        filterQuery = filterQuery + andText + '(df["tution_1_money"] ' + entity.amountCompOp + ' ' + entity.amount + ')'        
        andText = ' & '
        entity.showfees = True

    # Default value of rank = 1000 means no rank check required
    # Comparison will always be 'less than' entered value
    if (entity.rank != None and entity.rank != [] and entity.rank != '' and entity.rank != 1000):
        filterQuery = filterQuery + andText + '(df["university_rank"] < '+ str(entity.rank) + ')'
        andText = ' & '
        entity.showrank = True
        
    if (entity.currency != None and entity.currency != [] and entity.currency != ''):
        filterQuery = filterQuery + andText + '(df["tution_1_currency"] == "' + entity.currency + '")'
        andText = ' & '
        entity.showfees = True

    if (entity.duration != None and entity.duration != [] and entity.duration != ''):
        filterQuery = filterQuery + andText + '(df["durationInDays"] ' + entity.durationCompOp + ' ' + str(entity.duration) + ')'
        andText = ' & '
        entity.showduration = True
    
    setOutputColumns()
    
    if showdebug:
        print("DBG:buildQuery:",filterQuery)
        
    return filterQuery

###################################################################################
# Method to identify all the entities/sockets within the search
###################################################################################
def findEntities(inpString):
    global entity
        
    (locCities, locCountries) = findCitiesAndCountries(inpString)
        
    if locCities != None and locCities != []:
        comma = ''
        entity.cities = ''
            
        for city in locCities:
            entity.cities = entity.cities + comma + '"' + city.lower() + '"'
            comma = ','
            entity.showcity = True
            if showdebug == True:
                print("DBG:showcity:", entity.showcity)
            
    if locCountries != None and locCountries != []:
        comma = ''
        entity.countries = ''
        
        for country in locCountries:
            entity.countries = entity.countries + comma + '"' + country.lower() + '"'
            comma = ','
            entity.showcountry = True

    (locDateCompOp, localDate) = findDate(inpString)
    if localDate != None:
        entity.date = localDate
        entity.dateCompOp = locDateCompOp
        entity.showdate = True
    
    (localAmountCompOp, localAmount) = findAmount(inpString)
    if (localAmount != None and localAmount != []):
        entity.amount = localAmount
        entity.amountCompOp = localAmountCompOp
        entity.showfees = True
        
    locRank = findRanking(inpString)
    if locRank != None and locRank != [] and locRank != '':
        entity.rank = locRank
        entity.showrank = True
        
    localCurrency = findCurrency(inpString)
    if (localCurrency != None and localCurrency != []):
        entity.currency = localCurrency
        entity.showfees = True
        
    (localDurationCompOp, localDuration) = findDuration(inpString)
    if (localDuration != None and localDuration != []):
        entity.duration = localDuration
        entity.durationCompOp = localDurationCompOp
        entity.showduration = True

    filterQuery = buildQuery()
    
    if showdebug:
        print("DBG:findEntities:",filterQuery)
        
    #print("CB: <<The student is searching for a course in ",loc.cities," city in ",loc.countries," with cost ",currency," ",amount, " and commencement date ", date, " with duration ", duration,">>")
    #print("CB: <<FilterQuery:", filterQuery,">>")
    return filterQuery

def setOutputColumns():
    global entity
    
    entity.outputColumns = '\'program_name\''
    
    if entity.showcity == True:
        entity.outputColumns = entity.outputColumns + ',' + '\'cityName\''
    
    if entity.showcountry == True:
        entity.outputColumns = entity.outputColumns + ',' + '\'country_name\''
        
    if entity.showduration == True:
        entity.outputColumns = entity.outputColumns + ',' + '\'duration\''
        
    if entity.showdate == True:
        entity.outputColumns = entity.outputColumns + ',' + '\'start_date\''
        
    if entity.showfees == True:
        entity.outputColumns = entity.outputColumns + ',' + '\'tution_1_currency\',\'tution_1_money\''
        
    if entity.showuniv == True:
        entity.outputColumns = entity.outputColumns + ',' + '\'university_name\''
        
    if entity.showstructure == True:
        entity.outputColumns = entity.outputColumns + ',' + '\'structure\''
        
    if entity.showrank == True:
        entity.outputColumns = entity.outputColumns + ',' + '\'university_rank\''
        
###################################################################################
# Method to display results of the search
###################################################################################
def displayResults():
    global entity
    global bot_name
        
    results=''
    filterQueryToExecute = ''
    #dummyfilterQuery = 'data=df[(df["country_name"].isin(["India"]))]["program_name"]'

    #try:
    
    filterQuery = buildQuery()
    
    setOutputColumns()
    
    if filterQuery != '':
        filterQueryToExecute = ''
        filterQueryToExecute = 'df['+filterQuery+']'
        
        #if showdebug:
        #    print("DBG:Executing:",dummyfilterQuery)
        
        #exec(dummyfilterQuery)
        
        if showdebug:
            print("DBG:displayResults:",filterQueryToExecute)
        
        results = eval(filterQueryToExecute)
        
        if (len(results) == 0):
            print(bot_name, ": Sorry I did not find any courses matching your search :(. Try searching on another value")
            clearEntities()
        else:
            dataIndexSorted = findRelevantResults("dummy sentence", results)
            
            print (dataIndexSorted)
            
            if dataIndexSorted.empty:
                filterQueryToExecute = 'df['+filterQuery+'][['+entity.outputColumns+']]'
                if showdebug:
                    print("DBG:displayResults:",filterQueryToExecute)
            
                results = eval(filterQueryToExecute)
                
                print(bot_name,": I found ", len(results), " courses:")
                print("===================================================")
                print(results)
                print("===================================================")
            else:
                print(bot_name,": I found ", len(dataIndexSorted), " courses, sorted by relevance:")
                print("===================================================")
                #print(results.sort_values(by=dataIndexSorted["index"], axis=0, ascending = False))
                for (idx, r) in results.iterrows():
                    print(idx)
                #print(results.sort_values(by=np.array(dataIndexSorted["index"]), axis=0, ascending = False))
                print(results.loc[np.array(dataIndexSorted["index"])])
                print("===================================================")
            saveEntity()
    #except:
        #print(bot_name, ": Something went wrong. Try again.")

#####################################################################
# Function to find cosine similarity between two sentences
#####################################################################
def cos_sim(a, b):
    """Takes 2 vectors a, b and returns the cosine similarity according 
    to the definition of the dot product
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0
    else:
    	return dot_product / (norm_a * norm_b)

#####################################################################
# Build vocabulary for university_name, program_name, program_type
#####################################################################
def buildCourseVocabulary(data):
    global course_vocab
    global df_course_list
    global row_map
    
    # Create unique list of data words
    course_vocab = []
    df_course_list = []
    row_map = []
    
    counter = 0
    for (idx, d) in data.iterrows():
        row = {}
        u = d["university_name"]
        words = u.split()
        for w in words:
            w = stemmer.stem(w.lower())
            course_vocab.append(w)
            if row.get(w):
                continue;
            else:
                row[w] = 1
    
        pn = d["program_name"]
        words = pn.split()
        for w in words:
            w = stemmer.stem(w.lower())
            course_vocab.append(w)
            if row.get(w):
                continue;
            else:
                row[w] = 1
    
        pt = d["program_type"]
        words = pt.split()
        for w in words:
            w = stemmer.stem(w.lower())
            course_vocab.append(w)
            if row.get(w):
                continue;
            else:
                row[w] = 1
        
        df_course_list.append(row)
        row_map.append({'data_indx':idx, 'clist_indx':counter})
        counter = counter + 1
    
    course_vocab = list(set(course_vocab)) # unique list
    
############################################################################
# Function to return data index for the corresponding index from courselist
############################################################################
def get_data_index(clist_idx):
    global row_map
    
    #print("clist_idx:", clist_idx)
    #print(row_map)
    row = next((row for row in row_map if row['clist_indx'] == clist_idx), None)
    #print(row)
    return row['data_indx']

#######################################################################
# Create Score matrix for course words in a list with the presence (1)
# or absence of words in the vocabulary
#######################################################################
def buildScoreMatrix(data):
    global course_vocab
    global df_course_matrix
    global df_course_list
    
    # Create matrix of presence or absence of a word per row of data
    df_course_matrix = []
    for d in df_course_list:
        row = []
        for v in course_vocab:
            vfound = False
            for w in d.keys():
                if v == w:
                    row.append(1)
                    vfound = True
                    break
            if not vfound:
                row.append(0)
        df_course_matrix.append(row)

##########################################################################
# Create Score matrix for a sentence with the presence (1) or absence (0)
# of words in the vocabulary
##########################################################################
def getSentenceMatrix(sentence):
    global course_vocab
    global nn_words
    
    # tokenize the sentence and create the matrix
    text = nltk.word_tokenize(sentence)
    pos=nltk.pos_tag(text)
    
    words=nn_words # Take from the already existing list
    for p in pos:
        if p[1] == 'NNP':
            words.append(p[0])
            nn_words.append(p[0])
            
    row = []
    
    for v in course_vocab:
        vfound = False
        for w in words:
            w = stemmer.stem(w.lower())
            if v == w:
                row.append(1)
                vfound = True
                break
        if not vfound:
            row.append(0)
    return row

###################################################################################################
# Function to identify course names, program name, program type entered in the search and match
# rows in the dataset. If matches found, the row indices will be returned in the sorted order of
# relevance.
###################################################################################################
def findRelevantResults(sentence, data):
    global df_course_matrix
    
    # call method to build the vocabulary for course names
    buildCourseVocabulary(data)
    buildScoreMatrix(data)
    
    # create matrix for the sentence
    row = getSentenceMatrix(sentence)
    if row == None or row == []:
        return data # No course name in search text
    
    row_array = np.array(row)
    
    data_cos = pd.DataFrame(data=None, columns=('index','cosine'))
    matchFound = False
    
    for (idx, d) in enumerate(df_course_matrix):
        d_array = np.array(d)
        cosine = cos_sim(row_array, d_array)
        #print(cosine)
        #data["cosine"][idx] = cosine
        #print("Row:", str(idx), "Cosine["+str(idx)+"]:",str(cosine))
        if cosine >= 0.05:
            data_cos = data_cos.append({'index':int(get_data_index(idx)), 'cosine':cosine}, ignore_index = True)
            matchFound = True
            #print(data_cos)
    
    if matchFound:
        data_cos = data_cos.sort_values(by="cosine", ascending = False)
    
    print("CB: Data with cosine scores:")
    #print("==========================================================")
    #print(data_cos[["university_name","program_name","program_type","cosine"]])
    print("==========================================================")
    print(data_cos)
    print("==========================================================")
    return data_cos

##################################################################################
# Main chat program: This is the one that will process user inputs and respond!
##################################################################################
# Startup
print("***************************************************")
print("***                ", bot_name, "                   ***")
print("***                 STARTING UP                 ***")
print("***************************************************")
print("") # blank line
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

# Create new column "durationInDays" and add to dataset
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

# Create new column "cityName" and add to dataset        
df = df.assign(cityName=cityName)

################################################
# Convert date from string into datetime object.
################################################
start_date_conv = []
for idx,d in enumerate(df['start_date']):
    objDate = None
    if (isinstance(d, str)):
        objDate = datetime.strptime(d.strip(), '%Y-%m-%d %H:%M:%S')
        if (objDate != None):
            start_date_conv.append(objDate)
        else:
            start_date_conv.append(None)
    else:
        start_date_conv.append(None)

# Create new column "cityName" and add to dataset        
df = df.assign(start_date_conv = start_date_conv)

# Greet user
lstGreetings = ["Hello!","Hi There!","Hi! How are you doing today?","Welcome to my world!","Namastey!"]
print(bot_name, ":", lstGreetings[np.random.randint(0,len(lstGreetings))])
print(bot_name, ": How may I help you?")

# Start the endless interaction!
while (True):
    response=input("You: ")
    
    lstIntents = identifyIntents(response)
    for intent in lstIntents:
        if intent.intentType == 'greeting':
            print ("\n", bot_name, ": ", intent.response)
        elif intent.intentType == 'search':
            #try:
            print ("\n", bot_name, ": ", intent.response)
            saveEntity()
            filterQuery = findEntities(response)
            filterQueryExec = 'data = df['+filterQuery+']'
            if showdebug:
                print("DBG:",filterQueryExec)
            exec(filterQueryExec)
            resultSize = len(data)
            if resultSize == 0:
                filterQuery = '' # clear the query
                print("\n", bot_name, ": Sorry I did not find any courses matching your search :(. Try searching on another value")
                clearEntities()
            else:
                dataIndexSortedByCosine = findRelevantResults(response, data)
                
                resultSize = len(dataIndexSortedByCosine)
                
                if resultSize == 0:
                    resultSize = len(data)
                    
                if resultSize > 50:
                    print (bot_name, ": I found ", resultSize, " courses matching your search.")
                    print (bot_name, ": Tell me the program types or location or university you want to look for and we can narrow down the list further")
                    print (bot_name, ": *** You dont want me to dump so many on you ;)! ***")
                    saveEntity()
                else:
                    print(bot_name, ": I found ",resultSize," courses matching your search. Do you want to view them or filter them further?")
                    saveEntity()
            #except:
                #print("\n", bot_name, ": Something went wrong. Try again.")
        elif intent.intentType == 'view':
            print ("\n", bot_name, ": ", intent.response)
            displayResults()
        elif intent.intentType == 'stop':
            print ("\n", bot_name, ": ", intent.response)
            stopSearch = True
        elif intent.intentType == 'restart':
            print ("\n", bot_name, ": ", intent.response)
            clearEntities()
        elif intent.intentType == 'structure':
            print ("\n", bot_name, ": ", intent.response)
            entity.showstructure = True
            displayResults()
        elif intent.intentType == 'showfees':
            print ("\n", bot_name, ": ", intent.response)
            entity.showfees = True
            displayResults()
        elif intent.intentType == 'showrank':
            print ("\n", bot_name, ": ", intent.response)
            entity.showfees = True
            displayResults()
        elif intent.intentType == 'univs':
            print ("\n", bot_name, ": ", intent.response)
            entity.showuniv = True
            displayResults()
        elif intent.intentType == 'showdebug':
            print ("\n", bot_name, ": ", intent.response)
            showdebug = True
        else:
            print ("\n", bot_name, ": I cant understand your intent :(! Please have a human communicate with me!")
    
    if (stopSearch):
        break
# End while

# End of the chat session
print(bot_name, ": It was N.I.C.E. talking to you! Have a good day!")
print(bot_name, ": *** If you like me, give me 5 *s ;) ***")
print("") # blank line
print("***************************************************")
print("***                ", bot_name, "                   ***")
print("***                 SHUTTING DOWN               ***")
print("***************************************************")
                
###################################################################################
# This section below is to play around with the code. Keep this commented once done.
#df[(df["cityName"].isin (["Mumbai"])) & (df["durationInDays"] == 540)][['program_name','durationInDays','cityName','duration']]
#to_number(df[df["cityName"].isin (["Mumbai"])]['durationInDays'])==to_number(540)
#df[df["cityName"].isin (["Mumbai"]) & to_number(df["durationInDays"]) == to_number(540)][['program_name','durationInDays','cityName','duration']]
#df[df[df["cityName"].isin (["Mumbai"])]['durationInDays']==to_number(540)]['program_name']
#df['durationInDays']==12
# df[(df["country_name"].isin (["India"])) & (df["durationInDays"] == 540)][['program_name','country_name','duration','university_name','tution_1_currency','tution_1_money']]
#df[(df["cityName"].str.lower().isin (["mumbai"])) & (df["start_date_conv"] <"1919-01-01")]
#d=df[(df["country_name"].isin(["India"]))][["program_name"][0]]
#np.array(d)[0]
#d=df[((df["program_type"].isin(["Master"])) & ((df["university_name"].str.contains("Armenia")) | df["university_name"].str.contains("American University") | df["program_name"].str.contains("Political Science", "American University") | df["program_name"].str.contains("International Affairs") | df["structure"].str.contains("International Affairs") | df["structure"].str.contains("American University") | df["structure"].str.contains("Armenia")))][["university_name","program_name","program_type"]]
#np.array(d)[0:2]
###################################################################################