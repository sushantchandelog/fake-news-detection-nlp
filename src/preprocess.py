#text processing

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings("ignore")

nltk.download('stopwords')
nltk.download('punkt')

#Importing the Data
real = r"C:\Users\Sushant\Desktop\Name\unsupervised\fake news\data\True.csv"
fake = r"C:\Users\Sushant\Desktop\Name\unsupervised\fake news\data\Fake.csv"
#reading the df
real_df = pd.read_csv(real)
fake_df = pd.read_csv(fake)

#making the Labels
real_df['label'] = 1
fake_df['label'] = 0

#concatenating on the rows

df_One = pd.concat([real_df,fake_df], axis=0, ignore_index=False)

df_One = df_One.sample(frac=1, random_state=42).reset_index(drop=True)

#continuing the preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

#creating the function

def clean_text(text):
    #handle missinf value If any
    if not isinstance(text, str):
        return ""

    text = text.lower()

    #removing urls ()
    text = re.sub(r'http\S+|www\S+', '', text)

    #removing the special char
    text = re.sub(r'[^a-zA-Z\s]]','',text)

    #splitting the words #tokenize
    words = text.split()

    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

def preprocess_df(df):

    print("Cleaning the dataframe ")
    df = df.dropna(subset=['text','label'])
    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'].str.strip() != '']
    return df

print("Cleaning Complete")
df_cleaned = preprocess_df(df_One)

print(df_cleaned.head(20))
print("Cleaning Done --")
print('saving the File')
df_cleaned.to_csv('Cleaned_data.csv', index=False)
print('File Saved !!')