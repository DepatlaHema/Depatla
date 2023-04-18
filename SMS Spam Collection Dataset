import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
 import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno #Heatmap
from sklearn.preprocessing import LabelEncoder
import pickle

import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
import tensorflow as tf
import keras
from keras.models import Sequential


from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout
from tqdm.notebook import tqdm
from tokenizers import BertWordPieceTokenizer

from transformers import TFBertModel
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

import pickle
pd.set_option('display.max_rows', None)
[nltk_data] Downloading package punkt to /usr/share/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
[nltk_data] Downloading package stopwords to /usr/share/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
!pip3 install catboost
!pip3 install transformers
df = pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv", encoding="latin-1")
df.head()
print(df.shape)
print('==========')
print(df.info())
print('==========')
print(df.describe())
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
df.rename(columns={'v1':'target','v2':'text'},inplace=True) #drop columns
df = df.drop_duplicates(keep='first') #drop duplicates
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target']) #1 hot
df.head()
print(f'{df.isnull().sum()}\nNum duplicates: {df.duplicated().sum()}')
df.shape
print(df['target'].value_counts())
plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()
df['num_characters'] = df['text'].apply(len)
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
df.head()
df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()
df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()
sns.pairplot(df,hue='target')
sns.heatmap(data=df.corr(),
            annot=True,
            cmap="coolwarm",
            center=0,
            ax=plt.subplots(figsize=(8,6))[1]);
 def transform_text(text):
    # Convert the text to lowercase and tokenize it
    tokens = nltk.word_tokenize(text.lower())
    
    # Remove non-alphanumeric tokens (%^# )
    tokens = [t for t in tokens if t.isalnum()]
    
    # Remove stopwords and punctuation (I, how, u, are, is)
    stopwords_set = set(stopwords.words('english'))
    punctuation_set = set(string.punctuation)
    tokens = [t for t in tokens if t not in stopwords_set and t not in punctuation_set]
    
    # Stemming the remaining tokens using PorterStemmer (loving=love)
    ps = PorterStemmer()
    tokens = [ps.stem(t) for t in tokens]
    
    # Join the tokens back into a string and return it
    return " ".join(tokens)
print(f"{df['text'][10]}\n{transform_text(df['text'][10])}")
df['transformed_text'] = df['text'].apply(transform_text)
df.head()
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15,6))
plt.imshow(spam_wc)
ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15,6))
plt.imshow(ham_wc)
spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)

ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)
  
print(len(spam_corpus), len(ham_corpus))
from collections import Counter
word_freq = Counter(spam_corpus).most_common(30)
word_freq_df = pd.DataFrame(word_freq, columns=['word', 'count'])
sns.barplot(data=word_freq_df, x='word', y='count')
plt.xticks(rotation='vertical')
plt.show()
word_freq = Counter(ham_corpus).most_common(30)
word_freq_df = pd.DataFrame(word_freq, columns=['word', 'count'])
sns.barplot(data=word_freq_df, x='word', y='count')
plt.xticks(rotation='vertical')
plt.show()
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model_params = {
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'randomforestclassifier__n_estimators': [10]
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
  'params': {
            'logisticregression__C': [1, 5]
        }
    },
    'catboost': {
        'model': CatBoostClassifier(verbose=False),
        'params': {
            'catboostclassifier__iterations': [20],
            'catboostclassifier__learning_rate': [0.01],
            'catboostclassifier__depth': [4]
        }
    },
    'xgboost': {
        'model': XGBClassifier(),
        'params': {
            'xgbclassifier__n_estimators': [50],
            'xgbclassifier__max_depth': [4],
            'xgbclassifier__learning_rate': [0.01],
            'xgbclassifier__booster': ['gbtree']
        }
    },
    'gaussian_nb': {
        'model': GaussianNB(),
        'params': {}
         },
    'multinomial_nb': {
        'model': MultinomialNB(),
        'params': {}
    },
    'bernoulli_nb': {
        'model': BernoulliNB(),
        'params': {}
    },

    'stacking_model': {
        'model': StackingClassifier(
            estimators=[
                ('svc', SVC(kernel='linear', C=1, probability=True)),
                ('rf', RandomForestClassifier(n_estimators=10, random_state=42))
            ],
            final_estimator=LogisticRegression()
        ),
        'params': {
            'stackingclassifier__svc__kernel': ['rbf', 'linear'],
            'stackingclassifier__svc__C': [1, 10],
            'stackingclassifier__rf__n_estimators': [1, 5],
            'stackingclassifier__final_estimator__C': [1, 5]
             }
    }
}

# Run a grid search to find the best hyperparameters for each model
best_scores = []
best_estimators = {}
kfold = KFold(n_splits=5, shuffle=True, random_state=42)


for algo, mp in model_params.items():
    pipe = make_pipeline(MinMaxScaler(), mp['model'])
    clf =  RandomizedSearchCV(pipe, mp['params'], cv=kfold, return_train_score=False)
    clf.fit(X_train, y_train)
    best_scores.append({
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[algo] = clf.best_estimator_
    
df = pd.DataFrame(best_scores,columns=['model','best_score','best_params'])
df
for algo, estimator in best_estimators.items():
    print(f"Evaluating {algo}...")
    score = estimator.score(X_test, y_test)
    print(f"Test score: {score}\n")
