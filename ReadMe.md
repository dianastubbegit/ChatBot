# ChatBot

A chatbot is an application that allows users to have a text-based or text-to-speech conversation using artificial intelligence (AI) and natural language processing (NLP). Chatbots come in an assortment of shapes and sizes. The primary types include rule-based chatbots and AI-based chatbots. 
The rule-based chatbots interpret inquiries in a tree-like pattern, also known as decision tree bots. It has a collection of predefined responses for each query. They have no idea what the user’s inquiry entails.

The AI-based chatbots employ NLP and machine learning (ML) to deliver a better response, comprehend the context and intent of a user’s question pattern, and make connections between specific questions asked in different ways.

the chatbot is an open-domain AI-based contextual chatbot that keeps track of the context or how the user poses a question. The project uses a Sequential model for the training sets of data using deep learning (DL) techniques. 
The chatbot trains using intents, patterns, and responses. The user’s question translates to the intents class with the help of neural networks (NN), which will ingest the context and then return a random response within the class.

The prerequisites used for the chatbot project include:

    1. PyCharm IDE: 2021.3.1 Community Edition.
    
    2. Python: 3.8.5.
    
    3. Modules: nltk 3.6.7, pickle, TensorFlow 2.7.0, Keras 2.7.0, NumPy 1.19.5, and sklearn 0.0 
       (Note: a dependency on Scipy’s package is also required).
    
    4. API: Open Weather API (OpenWeather, 2021) retrieves the forecast by country and returns it 
       in the language for the country requested.
    
The files included in this project are listed below:

    1. intents.json: JSON file that contains sets of tags, patterns, and responses. 
    
    2. countries.json – JSON file that contains a list of countries and their respective 
       country code to use while calling the Open Weather API.
    
    3. training.py: creates the model and trains the python chatbot.
    
    4. training_data.file: contains a list of words, patterns, and training 
       sets in a binary format that trains the chatbot model.
    
    5. chatbot_model.h5: stores the trained model neurons weights and the configurations for the model.
    
    6. testing.py: used to predict which tag (classes) the user’s query belongs to 
       and returns a random response from that tag.
    
    7. chatbot_gui.py: the GUI for the chatbot where users can interact with the bot and train the bot.
    
The project utilizes TensorFlow’s Keras function to generate the model. This model has three layers: an input layer, a hidden layer, and an output layer. The Dropout function works to avoid overfitting between layers. In addition, the ‘relu’ activation function has been used on input and hidden layers, while the ‘softmax’ activation function works on the output’s dense layer. The model trains for 200 epochs in a batch size of 10 using the SGD optimizer, a form of gradient descent, with a learning rate of 0.01.

For the interface, Python’s Tkinter package provides the necessary options to construct a graphical user interface (GUI). Because it comes with several valuable libraries, the Tkinter library is the quickest and most straightforward approach to creating GUI programs. The GUI accepts a user’s query and returns the query’s response upon clicking the send button on the first tab. The second tab allows the user to train the chatbot with new data to chat with the chatbot.

**Run the chatbot and sample conversations**

Usage Instructions (please see the figures below for explanations):

1. Please extract the zip file into a location in the folder system.
2. For the executable to run, please double-click on chatbot_gui.exe.
3. The files in the folder include must include the following:

   a. chatbot_gui.exe
   
   b. chatbot_model.h5
   
   c. countries.json
   
   d. intents.json
   
   e. training_data.file
   
   f. 3 Python files
   

