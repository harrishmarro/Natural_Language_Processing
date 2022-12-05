

import nltk

nltk.download()

pip install -U spacy

pip install textblob

from nltk.tokenize import sent_tokenize, word_tokenize

EXAMPLE_TEXT = "Whatzz upp??"

nltk.download('punkt')

print(sent_tokenize(EXAMPLE_TEXT))
from textblob import TextBlob
zen = TextBlob("Data is a new fuel. "
               "Explicit is better than implicit. "
               "Simple is better than complex. ")
               
zen.words

import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
for token in doc:
    print(token.text)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sent = "Hello, this is stopwords example."

nltk.download('stopwords')

stop_words = ['this','is','as','was']

word_tokens = word_tokenize(example_sent)

filtered_sentence = [w for w in word_tokens if not w in stop_words]

filtered_sentence = []

for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()
example_words = ["stresses","cries","rational"]
for w in example_words:
    print(ps.stem(w))

from nltk.stem import wordnet 
from nltk.tokenize import word_tokenize 
nltk.download('omw-1.4')

lemma = wordnet.WordNetLemmatizer()
nltk.download('wordnet')
# lemmatize string 
def lemmatize_word(text): 
    word_tokens = word_tokenize(text) 
    # provide context i.e. part-of-speech(pos)
    lemmas = [lemma.lemmatize(word, pos ='v') for word in word_tokens] 
    return lemmas 
    text = 'Data is the new revolution in the World, in a day one individual would generate terabytes of data.'
lemmatize_word(text)

from nltk.tokenize import word_tokenize 
from nltk import pos_tag 
nltk.download('averaged_perceptron_tagger')

def pos_tagg(text): 
    word_tokens = word_tokenize(text) 
    return pos_tag(word_tokens) 
  
pos_tagg('Hello this is amazing!!')

x = TextBlob("An apple a day keeps the doctor away.")
x.ngrams(n=4)

from spacy import displacy

NER = spacy.load("en_core_web_sm")
raw_text="The Indian Space Research Organisation or is the national space agency of India, headquartered in Bengaluru. It operates under Department of Space which is directly overseen by the Prime Minister of India while Chairman of ISRO acts as executive of DOS as well."
text1= NER(raw_text)
for word in text1.ents:
    print(word.text,word.label_)

pip install speechRecognition

pip install gtts

from gtts import gTTS

from google.colab import drive
drive.mount('/content/drive')

text_to_say="Hello!welcome to the class,how r u"

gTTS(text=text_to_say)

language="en"

gtts_object=gTTS(text=text_to_say,lang=language)
gtts_object.save('/content/drive/MyDrive/gtts.wav')

from IPython.display import Audio
Audio("/content/drive/MyDrive/gtts.wav")

from sklearn.feature_extraction.text import CountVectorizer

ents = ['coronavirus is a highly infectious disease',
   'coronavirus affects older people the most', 
   'older people are at high risk due to this disease']
cv = CountVectorizer()

X = cv.fit_transform(ents) 
X = X.toarray()

sorted(cv.vocabulary_.keys())

print(X)

from sklearn.feature_extraction.text import TfidfVectorizer

sents = ['coronavirus is a highly infectious disease',
   'coronavirus affects older people the most', 
   'older people are at high risk due to this disease','older people are prone to coronavirus']

tfidf = TfidfVectorizer()
transformed = tfidf.fit_transform(sents)

import pandas as pd
df = pd.DataFrame(transformed[0].T.todense(),
    	index=tfidf.get_feature_names(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)

print(df)

!pip install transformers

!pip install torch===1.7.1 torchvision===0.8.2 torchaudio===0.7.2 -f

!pip install bert-extractive-summarizer

from summarizer import Summarizer

text_content='''Avul Pakir Jainulabdeen Abdul Kalam was born on 15 October 1931, to a Tamil Muslim family in the pilgrimage centre of Rameswaram on Pamban Island, then in the Madras Presidency and now in the State of Tamil Nadu. His father Jainulabdeen Marakayar was a boat owner and imam of a local mosque;[9] his mother Ashiamma was a housewife.[10][11][12][13] His father owned a ferry that took Hindu pilgrims back and forth between Rameswaram and the now uninhabited Dhanushkodi.[14][15] Kalam'''

model = Summarizer()
result = model(text_content, min_length=60)

full = ''.join(result)
print(full)








