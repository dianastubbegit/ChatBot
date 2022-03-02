"""

Class:  CSC 525 - Module 8 - Option 2 - Portfolio Project - Diana Stubbe - Dr. Issa - 01/09/2022
Topic:  NLP Chatbot Final Version

Source:
TechVidvan. (n.d.). Create Chatbot with Python & Artificial Intelligence. Cloudflare.
https://techvidvan.com/tutorials/chatbot-project-python-ai/

Weather API: http://api.openweathermap.org/

Description:
This Chatbot can conversationally chat and provide the current weather in certain cities around the world.
It makes use of the OpenWeatherMap API and returns the forecast in the language for that country.
The Chatbot is AI-based contextual Chatbot that will maintain the context
or in which sense or proportion the user is asking a query. Further, using deep learning techniques in Python,
the logic will construct a Sequential model for the training sets of data.
The intents, patterns, and responses will be used to train the chatbot.
The user’s query will be mapped to the intents class using neural networks,
which will maintain context and then return a random response.

"""

import nltk, random, json, pickle
# nltk.download('punkt');nltk.download('wordnet')
import requests
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer

lemmatizer = WordNetLemmatizer()
context = {}


class Testing:
    def __init__(self):
        # load the intent file
        self.intents = json.loads(open('intents.json').read())
        # load the training_data file with training data
        data = pickle.load(open("training_data", "rb"))
        self.words = data['words']
        self.classes = data['classes']
        self.model = load_model('chatbot_model.h5')
        # set the error threshold value
        self.ERROR_THRESHOLD = 0.5
        self.ignore_words = list("!@#$%^&*?")

    def clean_up_sentence(self, sentence):
        # tokenize each sentence (user's query)
        sentence_words = word_tokenize(sentence.lower())
        # lemmatize the word to a root word and filter symbols words
        sentence_words = list(map(lemmatizer.lemmatize, sentence_words))
        sentence_words = list(filter(lambda x: x not in self.ignore_words, sentence_words))
        return set(sentence_words)

    def wordvector(self, sentence):
        # initialize Count Vectorizer
        # txt.split helps to tokenize a single character
        cv = CountVectorizer(tokenizer=lambda txt: txt.split())
        sentence_words = ' '.join(self.clean_up_sentence(sentence))
        words = ' '.join(self.words)

        # fit the words into cv and transform into one-hot encoded vector
        vectorize = cv.fit([words])
        word_vector = vectorize.transform([sentence_words]).toarray().tolist()[0]
        return np.array(word_vector)

    def classify(self, sentence):
        # predict to which class (tag) the user's query belongs to
        results = self.model.predict(np.array([self.wordvector(sentence)]))[0]
        # store the class name and probability of that class
        results = list(map(lambda x: [x[0], x[1]], enumerate(results)))
        # accept the class probability which are greater than threshold value,0.5
        results = list(filter(lambda x: x[1] > self.ERROR_THRESHOLD, results))

        # sort the class probability value in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []

        for i in results:
            return_list.append((self.classes[i[0]], str(i[1])))
        return return_list

    def results(self, sentence, userID):
        # if the context is maintained then filter class(tag) accordingly
        if sentence.isdecimal():
            if context[userID] == "historydetails":
                return self.classify('ordernumber')
        return self.classify(sentence)

    def response(self, sentence, userID='DianaStubbe'):
        # get class of users query
        results = self.results(sentence, userID)
        print(sentence, results)
        # store random response to the query
        ans = ""
        if results:
            while results:
                for i in self.intents['intents']:
                    # check if tag == query's class
                    if i['tag'] == results[0][0]:
                        # if the class contains key as "set"
                        # then store the key as userid along with its value in
                        # the context dictionary
                        if 'set' in i and not 'filter' in i:
                            context[userID] = i['set']
                        # if the tag doesn't have any filter return the response
                        if not 'filter' in i:
                            ans = random.choice(i['responses'])
                            print("Query:", sentence)
                            print("Bot:", ans)
                        # if a class has a key as filter then check if the context dictionary key's value
                        # is same as the filter value to return the random response
                        if userID in context and 'filter' in i and i['filter'] == context[userID]:
                            if 'set' in i:
                                context[userID] = i['set']
                            ans = random.choice(i['responses'])

                results.pop(0)
        # if ans contains some value then return response to user's query else return some message
        return ans if ans != "" else "Sorry! I am still learning.\nYou can train me by " \
                                     "providing more information. "
