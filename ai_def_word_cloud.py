import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download("wordnet", download_dir = "./nltk_data")
nltk.download("punkt_tab", download_dir = "./nltk_data")
nltk.download("stopwords", download_dir = "./nltk_data")
nltk.data.path = ["./nltk_data"]

def clean_text(text):
    # Remove special characters from the text string.
    text = text.lower()
    text = re.sub(pattern = "-", repl = " ", string = text)
    text = re.sub(pattern = "\?", repl = " ", string = text)
    text = re.sub(pattern = "[^a-z A-Z 0-9]", repl = "", string = text)    

    # Tokenize the text string.
    text = nltk.word_tokenize(text)
    
    # Remove stop words from the text.
    stop_words = stopwords.words("english")
    additional_stop_words = ["artificial", "intelligence", "ai", "define", "defined"]
    stop_words.extend(additional_stop_words)
    stop_words = set(stop_words)
    
    text = [word for word in text if word not in stop_words]

    # Lemmatize the text.
    wnl = nltk.WordNetLemmatizer()
    text = [wnl.lemmatize(word) for word in text]
    
    # Create the cleaned text string.
    text = " ".join(text)

    return(text)

import pandas as pd

ai_def = pd.read_csv("./ai_definitions.csv")
ai_def_text = " ".join(definition for definition in ai_def.definition)

wc_text = clean_text(ai_def_text)

import wordcloud as wc
import matplotlib.pyplot as plt

wc_text_obj = wc.WordCloud(
    background_color="white", 
    width = 800, 
    height = 800, 
    random_state = 124)\
.generate(wc_text)

wc_text_obj.to_file("./images/ai_def_word_cloud.png")

