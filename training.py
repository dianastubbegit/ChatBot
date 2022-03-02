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

import nltk
import random
import json
import pickle

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import flatten
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()


class Training:
    def __init__(self):
        # read and load the intent file
        data_file = open('intents.json').read()
        self.intents = json.loads(data_file)['intents']
        self.ignore_words = list("!@#$%^&*?")
        self.process_data()

    def process_data(self):
        # fetch patterns and tokenize into words
        self.pattern = list(map(lambda x: x["patterns"], self.intents))
        self.words = list(map(word_tokenize, flatten(self.pattern)))

        # fetch classes (tags) and store in documents with tokenized patterns
        self.classes = flatten([[x["tag"]] * len(y) for x, y in zip(self.intents, self.pattern)])
        self.documents = list(map(lambda x, y: (x, y), self.words, self.classes))

        # lower case and filter special the symbols from words
        self.words = list(map(str.lower, flatten(self.words)))
        self.words = list(filter(lambda x: x not in self.ignore_words, self.words))

        # lemmatize the words and sort the class and word lists
        self.words = list(map(lemmatizer.lemmatize, self.words))
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))

    def train_data(self):
        # initialize and set analyzer=word to vectorize words, not characters
        cv = CountVectorizer(tokenizer=lambda txt: txt.split(), analyzer="word", stop_words=None)
        # create the training sets for the model
        training = []
        for doc in self.documents:
            # lower case and lemmatize the pattern words
            pattern_words = list(map(str.lower, doc[0]))
            pattern_words = ' '.join(list(map(lemmatizer.lemmatize, pattern_words)))

            # train or fit the vectorizer with all words
            # and transform into one-hot encoded vector
            vectorize = cv.fit([' '.join(self.words)])
            word_vector = vectorize.transform([pattern_words]).toarray().tolist()[0]

            # create output for the respective input
            # output size will be equal to total numbers of classes
            output_row = [0] * len(self.classes)

            # if the pattern is from the current class put 1 in list else 0
            output_row[self.classes.index(doc[1])] = 1
            cvop = cv.fit([' '.join(self.classes)])
            out_p = cvop.transform([doc[1]]).toarray().tolist()[0]

            # store vectorized word list with its class
            training.append([word_vector, output_row])

        # shuffle training sets to avoid the model training on same data again
        random.shuffle(training)
        training = np.array(training, dtype=object)
        train_x = list(training[:, 0])  # patterns
        train_y = list(training[:, 1])  # classes
        print(train_y)
        return train_x, train_y

    def build(self):
        # load the data from the train_data function
        train_x, train_y = self.train_data()
        # create a Sequential model with 3 layers
        model = Sequential()
        # input layer with latent dimension of 128 neurons and ReLU activation function
        model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        model.add(Dropout(0.5))  # Dropout to avoid overfitting
        # second layer with the latent dimension of 64 neurons
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        # fully connected output layer with softmax activation function
        model.add(Dense(len(train_y[0]), activation='softmax'))
        '''Compile the model with Stochastic Gradient Descent with a learning rate and
           nesterov accelerated gradient descent'''
        sgd = SGD(learning_rate=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        # fit the model with training input and output sets
        hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=10, verbose=1)
        # save the model and words, classes which can be used for prediction
        model.save('chatbot_model.h5', hist)
        pickle.dump({'words': self.words, 'classes': self.classes, 'train_x': train_x, 'train_y': train_y},
                    open("training_data", "wb"))


# train the model
Training().build()
