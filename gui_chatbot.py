import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.keras')
import json
import random

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    # Tokenize the pattern - splitting the words into an array
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatize every word - reducing to base form
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Return a bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bag_of_words(sentence, words, show_details=True):
    # Tokenize patterns
    sentence_words = clean_up_sentence(sentence)
    # Bag of words - vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                # Assigning 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("Found in bag: %s" % word)
    return np.array(bag)

def predict_class(sentence):
    # Filter below threshold predictions
    p = bag_of_words(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Sorting strength probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Create tkinter GUI
import tkinter
from tkinter import *

def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)
    
    if msg != '':
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "You: " + msg + '\n\n')
        ChatBox.config(foreground="#442265", font=("Verdana", 12))
        
        ints = predict_class(msg)
        res = get_response(ints, intents)
        
        ChatBox.insert(END, "Bot: " + res + '\n\n')
        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)

root = Tk()
root.title('Chatbot')
root.geometry('400x500')
root.resizable(width=FALSE, height=FALSE)

# Create Chat window
ChatBox = Text(root, bd=0, bg='white', height='8', width='50', font='Arial',)
ChatBox.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="heart")
ChatBox['yscrollcommand'] = scrollbar.set

# Create Button to send message
SendButton = Button(root, font=('Verdana', 12, 'bold'), text='Send', width='12', height=5, bd=0, bg='#fca602', activebackground='#3c9d9b', fg='#000000', command=send)

# Create the box to enter message
EntryBox = Text(root, bd=0, bg='white', width='29', height='5', font='Arial')
EntryBox.bind("<Return>", lambda event: send())

# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatBox.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=6, y=401, height=90, width=265)
SendButton.place(x=275, y=401, height=90)

root.mainloop()
