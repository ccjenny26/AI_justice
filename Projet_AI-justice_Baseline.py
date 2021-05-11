
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
# for lstm model
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
#from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
import os, re, csv, math, codecs
import tensorflow as tf
import tensorflow.keras
#from tensorflow.keras import optimizers
#from tensorflow.keras import backend as K
#from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
#from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
#from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping

# import data
#define a function for extraction information from the parts faits,motifs and decision for all files
def create_data(filenames):
    classes = []
    description = []
    for file_name in filenames:
        with open(file_name) as fp:
            soup = BeautifulSoup(fp, 'html.parser')
        factors = soup.find_all('div', {'class': ['faits','motifs']}) # can be any combination of (faits,motifs,jugement)
        #factors = soup.find_all('p')
        decision = soup.find('meta', attrs={'name': 'output'})['content']
    
        # extract the text from faits and motifs, transform into string and store in a list
        factors_text = []
        for i in factors:
            factors_text.append(str. rstrip(i.text))
            factors_text = [item.strip().replace("\n", " ") for item in factors_text if str(item)]
        # return faits_motifs_text, decision
        classes.append(decision)
        description.append(factors_text) # description is a list of list
    return description, classes


filenames = glob.glob("input_documents/*.html")
description, classes = create_data(filenames)



### SVM

# transform classes into numeric
le = LabelEncoder()
y = le.fit_transform(classes)

# check the label encoder mapping
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

# check the number of observations in each class
class_ids = np.unique(classes)
K = len(class_ids)
counts = np.bincount(y)
plt.bar(range(K), counts)
plt.ylabel('number of cases')
plt.title('dataset constitution')
plt.xlabel('decision')
plt.xticks(range(K), class_ids)
plt.show()

# show number of each class
count_mapping = dict(zip(le.classes_, counts))
print(count_mapping)


# convert each list of strings (represents one document) into one string for tfidf vectorization
inputs = []
for i in range(len(description)):
    document = " ".join(description[i])
    inputs.append(document)

# split the data into train and test
input_train, input_test, y_train, y_test = train_test_split(inputs, y, test_size=0.10, random_state=42)

# instantiate vectorizer
vectorizer = TfidfVectorizer(token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', analyzer="word", max_df=0.9)
vect_transform = vectorizer.fit_transform(input_train)
# vectorization of input_train
X_train = vectorizer.transform(input_train).toarray()
# vectorization of input_test
X_test = vectorizer.transform(input_test).toarray()

## show weights of tokens
#vect_score = np.asarray(vect_transform.mean(axis=0)).ravel().tolist()
#vect_array = pd.DataFrame({'term': vectorizer.get_feature_names(), 'weight': vect_score})
#vect_array.sort_values(by='weight',ascending=False,inplace=True)
#print(vect_array)

#print(X_train.shape)
#print(vectorizer.get_feature_names())


# instantiate model
svm = SVC()
# define grid of parameter
C_grid = np.logspace(-3,3,7)
gamma_grid = [0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10]
param_grid = [
    {'kernel' : ['linear'], 'C' : C_grid, 'decision_function_shape': ['ovr','ovo']},
    {'kernel' : ['rbf'], 'C' : C_grid, 'gamma' : gamma_grid, 'decision_function_shape': ['ovr','ovo']}
]

# optimise parameter
grid_search = GridSearchCV(svm, param_grid, cv = 5)
grid_search.fit(X_train, y_train)
print("The best parameters are %s with a score of %0.2f"
     % (grid_search.best_params_, grid_search.best_score_))

# performance on dataset train
pred_train = cross_val_predict(grid_search.best_estimator_, X_train, y_train, cv = 5)

print('*** global accuracy ***')
print(accuracy_score(y_train, pred_train))
print('*** classification report ***')
print(classification_report(y_train, pred_train, zero_division=1)) # to solve the problem of true positive + false positive == 0
print('*** confusion matrix ***')
print(confusion_matrix(y_train, pred_train))



#cmat = confusion_matrix(y_train, pred_train)
#sns.heatmap(cmat, square = True, annot = True, cbar = False)
#plt.title('train set : confusion matrix')
#plt.show()

# performance on dataset test
pred_test = grid_search.best_estimator_.predict(X_test)

print('*** global accuracy ***')
print(accuracy_score(y_test, pred_test))
print('*** classification report ***')
print(classification_report(y_test, pred_test, zero_division=1))
print('*** confusion matrix ***')
print(confusion_matrix(y_test, pred_test))

# check prediction on dataset test
pred_test_label = le.inverse_transform(pred_test)
y_test_label = le.inverse_transform(y_test)
print('******Dataset test******')
print('Prediction:',pred_test_label)
print('True label:',y_test_label)


### LSTM
sns.set_style("whitegrid")
np.random.seed(0)

MAX_NB_WORDS = 100000
#tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('french'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}',"'"])


print('loading word embeddings...')
embeddings_index = {}
f = codecs.open('cc.fr.300.vec')
for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('found %s word vectors' % len(embeddings_index))


# split the data into train and test
input_train, input_test, y_train, y_test = train_test_split(inputs, y, test_size=0.10, random_state=42)

# preprocessing train data
processed_docs_train = []
for doc in tqdm(input_train):
    tokens = nltk.word_tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    processed_docs_train.append(" ".join(filtered))

processed_docs_test = []
for doc in tqdm(input_test):
    tokens = nltk.word_tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    processed_docs_test.append(" ".join(filtered))

# tokenize input data
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
tokenizer.fit_on_texts(processed_docs_train + processed_docs_test)
word_seq_train = tokenizer.texts_to_sequences(processed_docs_train)
word_seq_test = tokenizer.texts_to_sequences(processed_docs_test)
word_index = tokenizer.word_index
print("dictionary size: ", len(word_index))


# check the length(number of words) of each document
doc_len = np.array([len(t.split()) for t in input_train])
max_seq_len = np.round(doc_len.mean() + doc_len.std()).astype(int)
# show plot
sns.distplot(doc_len, hist=True, kde=True, color='b', label='doc len')
plt.axvline(x=max_seq_len, color='k', linestyle='--', label='max len')
plt.title('document length')
plt.legend()
plt.show()

#pad sequences
word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)

# reshape y to vectors of length 3(number of classes)
y_train_reformat = tensorflow.keras.utils.to_categorical(y_train, num_classes=3)
y_test_reformat = tensorflow.keras.utils.to_categorical(y_test, num_classes=3)


#### Train with LSTM

# Converting all the words to index in number, to the embedding index in pre-trained model and converted all the missing words to 0
#embedding matrix

print('preparing embedding matrix...')

embed_dim = 300 
words_not_found = []
nb_words = min(MAX_NB_WORDS, len(word_index)+1)
embedding_matrix = np.zeros((nb_words, embed_dim))

for word, i in word_index.items():
    if i >= nb_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if (embedding_vector is not None) and len(embedding_vector) > 0:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        words_not_found.append(word)
print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

print("sample words not found: ", np.random.choice(words_not_found, 10))


print('embedding shape:',embedding_matrix.shape)


### Train the LSTM
model = Sequential()
model.add(Embedding(nb_words,embed_dim,input_length=max_seq_len, weights=[embedding_matrix],trainable=False))
model.add(LSTM(128))
model.add(Dense(3,activation='softmax'))  
model.summary()
# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# fit model
#early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, mode = 'auto')
#history = model.fit(word_seq_train, y_train_reformat, epochs = 100, validation_split = 0.1, callbacks = [early_stopping])
history = model.fit(word_seq_train, y_train_reformat, epochs = 100, validation_split = 0.1)

# plot of loss
plt.figure()
plt.plot(history.history['loss'], lw=2.0, color='b', label='train')
plt.plot(history.history['val_loss'], lw=2.0, color='r', label='val')
plt.title('LSTM classification')
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.legend(loc='upper right')
plt.show()

# plot of accuracy
plt.figure()
plt.plot(history.history['accuracy'], lw=2.0, color='b', label='train')
plt.plot(history.history['val_accuracy'], lw=2.0, color='r', label='val')
plt.title('LSTM classification')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()

# prediction on train
pred_test = model.predict_classes(word_seq_train, verbose=1)

print('*** global accuracy ***')
print(accuracy_score(y_train, pred_train))
print('*** classification report ***')
print(classification_report(y_train, pred_train, zero_division=1)) # to solve the problem of true positive + false positive == 0
print('*** confusion matrix ***')
print(confusion_matrix(y_train, pred_train))

# prediction on test
pred_test = model.predict_classes(word_seq_test, verbose=1) 

print('*** global accuracy ***')
print(accuracy_score(y_test, pred_test))
print('*** classification report ***')
print(classification_report(y_test, pred_test, zero_division=1)) # to solve the problem of true positive + false positive == 0
print('*** confusion matrix ***')
print(confusion_matrix(y_test, pred_test))

# check prediction on test
pred_test_label = le.inverse_transform(pred_test)
y_test_label = le.inverse_transform(y_test)
print('******Dataset test******')
print('Prediction:',pred_test_label)
print('True label:',y_test_label)





