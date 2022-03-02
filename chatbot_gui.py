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

Command for the executable:  pyinstaller --onefile --hidden-import scipy chatbot_gui.py
"""

from tkinter import *
from tkinter import ttk
import tkinterpp
import json
# import the training.py
# and testing.py file
import requests
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import testing as testpy
import training as trainpy

BG_GRAY = "#ABB2B9"
BG_COLOR = "#808080"
TEXT_COLOR = "#FFF"
FONT = "Arial 12"
FONT_BOLD = "Arial 11 bold"
api_key = "e95abc1560983e41dc514771db837784"


class ChatBot:
    def __init__(self):
        # initialize the tkinter window
        self.weather = None
        self.window = Tk()
        self.main_window()
        self.test = testpy.Testing()

    # run the window
    def run(self):
        self.window.mainloop()

    def main_window(self):
        # add a title to the window and configure it
        self.window.title("ChatBot")
        pic = PhotoImage(file='icon.ico')
        self.window.iconphoto(False, pic)
        self.window.resizable(width=False, height=False)
        self.window.configure(width=520, height=520, bg=BG_COLOR)

        # add two tabs, for Chatbot and Train Bot in the Notebook frame
        self.tab = ttk.Notebook(self.window)
        self.tab.pack(expand=1, fill='both')
        self.bot_frame = ttk.Frame(self.tab, width=520, height=520)
        self.train_frame = ttk.Frame(self.tab, width=520, height=520)
        self.tab.add(self.bot_frame, text='Chat Bot'.center(100))
        self.tab.add(self.train_frame, text='Train Bot'.center(100))
        self.add_bot()
        self.add_train()

    def add_bot(self):
        # add a heading to the Chatbot window
        head_text = "Welcome to the CSC 525 Chatbot"
        head_label = Label(self.bot_frame, bg=BG_COLOR, fg=TEXT_COLOR, text=head_text,
                           font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)
        line = Label(self.bot_frame, width=450, bg=BG_COLOR)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        # create a text widget where conversation will be displayed
        introtext = "Hi, my name is CSUBot. I am a chatbot here to help with your questions! \nPlease " \
                    "ask me with one question at a time in the blue box. \n" \
                    "Or, you can train me by going to the - Train Bot - tab. \n\n" \
                    "You can ask me questions like: What type of help do you provide?\n" \
                    "Or: What is the weather in France today?\n" \
                    "How can I help you today? "
        self.text_widget = Text(self.bot_frame, width=20, height=2, bg="#fff", fg="#000", font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.insert(END, introtext + "\n\n")
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        # create a scrollbar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)

        # create a bottom label where the message widget will be
        bottom_label = Label(self.bot_frame, bg=BG_GRAY, fg=TEXT_COLOR, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        # this is for user to add a query
        self.msg_entry = Entry(bottom_label, bg="#6e9ecd", fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.788, relheight=0.06, rely=0.008, relx=0.008)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self.on_enter)

        # send button which will call on_enter function to send the query
        send_button = Button(bottom_label, text="Send", font=FONT_BOLD, fg=TEXT_COLOR, width=8, bg="Green",
                             command=lambda: self.on_enter(None))
        send_button.place(relx=0.80, rely=0.008, relheight=0.06, relwidth=0.20)

    def add_train(self):
        # add the heading to the Train Bot window
        head_label = Label(self.train_frame, bg=BG_COLOR, fg=TEXT_COLOR, text="Train Bot", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        # tag the label and entry for the intents tag
        taglabel = Label(self.train_frame, fg="#000", text="Tag", font=FONT)
        taglabel.place(relwidth=0.2, rely=0.14, relx=0.008)
        self.tag = Entry(self.train_frame, bg="#fff", fg="#000", font=FONT)
        self.tag.place(relwidth=0.7, relheight=0.06, rely=0.14, relx=0.22)

        # pattern label and entry for the pattern in the tag
        self.pattern = []
        for i in range(2):
            patternlabel = Label(self.train_frame, fg="#000", text="Pattern%d" % (i + 1), font=FONT)
            patternlabel.place(relwidth=0.2, rely=0.28 + 0.08 * i, relx=0.008)
            self.pattern.append(Entry(self.train_frame, bg="#fff", fg="#000", font=FONT))
            self.pattern[i].place(relwidth=0.7, relheight=0.06, rely=0.28 + 0.08 * i, relx=0.22)

        # response label and entry for response to the pattern
        self.response = []
        for i in range(2):
            responselabel = Label(self.train_frame, fg="#000", text="Response%d" % (i + 1), font=FONT)
            responselabel.place(relwidth=0.2, rely=0.50 + 0.08 * i, relx=0.008)
            self.response.append(Entry(self.train_frame, bg="#fff", fg="#000", font=FONT))
            self.response[i].place(relwidth=0.7, relheight=0.06, rely=0.50 + 0.08 * i, relx=0.22)

        # to train the bot, create Train Bot button which will call on_train function
        bottom_label = Label(self.train_frame, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)
        train_button = Button(bottom_label, text="Train Bot", font=FONT_BOLD, fg=TEXT_COLOR, width=6, bg="Green",
                              command=lambda: self.on_train(None))
        train_button.place(relx=0.20, rely=0.008, relheight=0.06, relwidth=0.60)

    def on_train(self, event):
        # read intent file and append created tag, pattern, and responses from the add_train function
        with open('intents.json', 'r+') as json_file:
            file_data = json.load(json_file)
            file_data['intents'].append({
                "tag": self.tag.get(),
                "patterns": [i.get() for i in self.pattern],
                "responses": [i.get() for i in self.response],
                "context": ""
            })
            json_file.seek(0)
            json.dump(file_data, json_file, indent=1)
        # run and compile the model from the training.py file
        train = trainpy.Training()
        train.build()
        print("Trained Successfully")
        self.test = testpy.Testing()

    def on_enter(self, event):
        # get user query and bot response
        msg = self.msg_entry.get()
        self.my_msg(msg, "You")
        self.bot_response(msg, "Bot")

    def bot_response(self, msg, sender):
        self.text_widget.configure(state=NORMAL)
        country = ""
        weather = ""
        # opening JSON file
        countries = open('countries.json')
        # returns JSON object as a dictionary
        data = json.load(countries)
        # iterating through the json list
        for i in data['countries']:
            if msg.__contains__(i['country']):
                country = i['country']
                countrycode = i['abbreviation']
                api_url = "http://api.openweathermap.org/data/2.5/weather?q={}&appid={}&lang={}".format(country,
                                                                                                        api_key,
                                                                                                        countrycode)
                print(api_url)
                response = requests.get(api_url)
                response_dict = response.json()
                weather = response_dict["weather"][0]["description"]

                if response.status_code == 200:
                    weather = weather
                else:
                    weather = ""

                # closing the json countries file
                countries.close()

        # get the response for the user's query from the testing.py file
        if weather == "":
            self.text_widget.insert(END, str(sender) + " : " + self.test.response(msg) + "\n\n")
        else:
            self.text_widget.insert(END, str(sender) + " : " + self.test.response(msg) + "\nThe weather app says that "
                                    + country + "'s weather is: " + weather + "\n\n")
        self.text_widget.configure(state=DISABLED)
        self.text_widget.see(END)

    def my_msg(self, msg, sender):
        # will display a user query and bot response in the text_widget
        if not msg:
            return
        self.msg_entry.delete(0, END)
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, str(sender) + " : " + str(msg) + "\n")
        self.text_widget.configure(state=DISABLED)


# run the program
if __name__ == "__main__":
    bot = ChatBot()
    bot.run()
