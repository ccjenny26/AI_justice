import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import bs4
from bs4 import BeautifulSoup
import lxml
import glob
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
#from nltk.tokenize import RegexpTokenizer 
from nltk.tokenize import word_tokenize
import os, re, csv, math, codecs
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping




#### extraction and pre-process of data for both methods
# define a function for extraction of label
def get_label(filenames):
    labels = []
    for file_name in filenames:
        with open(file_name) as fp:
            soup = BeautifulSoup(fp, 'html.parser')
        decision = soup.find('meta', attrs={'name': 'output'})['content']
        labels.append(decision)
    return labels

# define a function for extraction input and output
def create_data(filenames):
    classes = []
    description = []
    for file_name in filenames:
        with open(file_name) as fp:
            soup = BeautifulSoup(fp, 'html.parser')
        #factors = soup.find_all('div', {'class': ['faits','motifs']}) # can be any combination of (faits,motifs,jugement) ## to extraction faits and motifs
        #factors = soup.find_all('div', {'class': ['faits','motifs','jugement']})  ## to extract faits,motifs and jugement
        factors = soup.find_all('p')  ## to use the whole file
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
    
## to extract the part after the last annotation, we use a different function as follow
#def create_data(filenames):
#    classes = []
#    description = []
#    for file_name in filenames:
#        with open(file_name) as fp:
#            soup = BeautifulSoup(fp, 'lxml')
#            body = soup.find("body")
#        all_sections = [t for t in body]   # chaque élément est soit du text soit une balise
#        text = []
#        for section in reversed(all_sections): # itération en partant de la fin
#            if type(section) == bs4.element.Tag: # (il faudra faire un "import bs4" au début du script)
#                if section.name == "div":   # on s'arrête au premier tag div
#                    break
#                else:    # on récupère le text des autres tags (<p>, etc)
#                    section_text = section.text
#                    if section_text.strip():
#                        text.append(section_text)
#            else:
#                text.append(str(section))
#        factors_text = list(reversed(text))
#        factors_text = [item.strip().replace("\n", " ") for item in factors_text if str(item)]
#        decision = soup.find('meta', attrs={'name': 'output'})['content']
#        # return faits_motifs_text, decision
#        classes.append(decision)
#        description.append(factors_text) # description is a list of list
#    return description, classes

# get labels
filenames = glob.glob("input_documents/*.html")
labels = get_label(filenames)
# get a list of filenames wanted
idx_keep = [idx for idx, value in enumerate(labels) if value in ['M','P']]
filenames_new = [filenames[i] for i in idx_keep]
# get input and output
description, classes = create_data(filenames_new)
print('number of files used:',len(filenames_new))


# transform classes into numeric
le = LabelEncoder()
y = le.fit_transform(classes)

# convert each list of strings (represents one document) into one string
inputs = []
for i in range(len(description)):
    document = " ".join(description[i])
    inputs.append(document)


#### preprocessing for SVM
# define grid of parameter
C_grid = np.logspace(-1,1,3)
gamma_grid = [0.5, 0.75, 1, 2]
param_grid = [
    {'kernel' : ['linear'], 'C' : C_grid,},
    {'kernel' : ['rbf'], 'C' : C_grid, 'gamma' : gamma_grid}
]



#### preprocessing for LSTM
MAX_NB_WORDS = 100000
stop_words = set(stopwords.words('french'))
#stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}',])
stop_words.update(['.', ',', '"', ':', ';', '(', ')', '[', ']', '{', '}',])

#print('loading word embeddings...')
embeddings_index = {}
f = codecs.open('cc.fr.300.vec')
for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
#print('found %s word vectors' % len(embeddings_index))

# define a document processing function
def processed_doc(input_doc):
    processed_docs = []
    for doc in tqdm(input_doc):
        tokens = nltk.word_tokenize(doc)
        filtered = [word for word in tokens if word not in stop_words]
        processed_docs.append(" ".join(filtered))
    return processed_docs

#### test the two models by a loop of 100 executions
# metrics of SVM model
acc_train_svm = []
f1_macro_train_svm = []
f1_weighted_train_svm = []
acc_val_svm = []
f1_macro_val_svm = []
f1_weighted_val_svm = []
acc_test_svm = []
f1_macro_test_svm = []
f1_weighted_test_svm = []
# metrics of LSTM model
acc_train_lstm_avg = []
acc_train_lstm_std = []
f1_macro_train_lstm_avg = []
f1_macro_train_lstm_std = []
f1_weighted_train_lstm_avg = []
f1_weighted_train_lstm_std = []

acc_val_lstm_avg = []
acc_val_lstm_std = []
f1_macro_val_lstm_avg = []
f1_macro_val_lstm_std = []
f1_weighted_val_lstm_avg = []
f1_weighted_val_lstm_std = []

acc_test_lstm_avg = []
acc_test_lstm_std = []
f1_macro_test_lstm_avg = []
f1_macro_test_lstm_std = []
f1_weighted_test_lstm_avg = []
f1_weighted_test_lstm_std = []

for i in range(5):
    # split the data into train and test
    input_train, input_test, y_train, y_test = train_test_split(inputs, y, test_size=0.20, stratify = y)
    
    #### SVM
    # instantiate vectorizer
    vectorizer = TfidfVectorizer(token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', analyzer="word", max_df=0.9, ngram_range = (1,3))
    vect_transform = vectorizer.fit_transform(input_train)
    # vectorization of input_train
    X_train = vectorizer.transform(input_train).toarray()
    # vectorization of input_test
    X_test = vectorizer.transform(input_test).toarray()
    
    # instantiate model
    svm = SVC()
    # optimise parameter
    grid_search = GridSearchCV(svm, param_grid, cv = 5)
    grid_search.fit(X_train, y_train)
    #print("The best parameters are %s with a score of %0.2f" % (grid_search.best_params_, grid_search.best_score_))
   
    # evaluation on dataset train
    pred_train=grid_search.best_estimator_.predict(X_train)
    acc_train_svm.append(round(accuracy_score(y_train, pred_train)*100, 2))
    f1_macro_train_svm.append(round(f1_score(y_train, pred_train, average='macro')*100,2))
    f1_weighted_train_svm.append(round(f1_score(y_train, pred_train, average='weighted')*100,2))
    # evaluation on dataset val
    pred_val = cross_val_predict(grid_search.best_estimator_, X_train, y_train, cv = 5)
    acc_val_svm.append(round(accuracy_score(y_train, pred_val)*100, 2))
    f1_macro_val_svm.append(round(f1_score(y_train, pred_val, average='macro')*100,2))
    f1_weighted_val_svm.append(round(f1_score(y_train, pred_val, average='weighted')*100,2))
    # evaluation on dataset test
    pred_test=grid_search.best_estimator_.predict(X_test)
    acc_test_svm.append(round(accuracy_score(y_test, pred_test)*100, 2))
    f1_macro_test_svm.append(round(f1_score(y_test, pred_test, average='macro')*100,2))
    f1_weighted_test_svm.append(round(f1_score(y_test, pred_test, average='weighted')*100,2))
    
    
    #### LSTM
    # split again training data into train and val
    input_train, input_val, y_train, y_val = train_test_split(input_train, y_train, test_size=0.25, stratify = y_train)
    # preprocessing train,val and test data
    processed_docs_train = processed_doc(input_train)
    processed_docs_val = processed_doc(input_val)
    processed_docs_test = processed_doc(input_test)

    # tokenize input data
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
    tokenizer.fit_on_texts(processed_docs_train + processed_docs_test + processed_docs_val)

    word_seq_train = tokenizer.texts_to_sequences(processed_docs_train)
    word_seq_val = tokenizer.texts_to_sequences(processed_docs_val)
    word_seq_test = tokenizer.texts_to_sequences(processed_docs_test)

    word_index = tokenizer.word_index
    #print("dictionary size: ", len(word_index))

    # check the length(number of words) of each document
    doc_len = np.array([len(t.split()) for t in input_train])
    max_seq_len = np.round(doc_len.mean() + doc_len.std()).astype(int)

    #pad sequences
    word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
    word_seq_val = sequence.pad_sequences(word_seq_val, maxlen=max_seq_len)
    word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)

    # reformat y to vector of 3
    y_train_reformat = tensorflow.keras.utils.to_categorical(y_train, num_classes=3)
    y_val_reformat = tensorflow.keras.utils.to_categorical(y_val, num_classes=3)
    y_test_reformat = tensorflow.keras.utils.to_categorical(y_test, num_classes=3)

    # Converting all the words to index in number, to the embedding index in pre-trained model and converted all the missing words to 0
    # embedding matrix
    #print('preparing embedding matrix...')
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
    #print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    #print("sample words not found: ", np.random.choice(words_not_found, 10))
    #print('embedding shape:',embedding_matrix.shape)
    
    # run lstm model for 5 times and store each time the mean and std
    acc_scores_train = []
    f1_macro_scores_train = []
    f1_weighted_scores_train = []
    acc_scores_val = []
    f1_macro_scores_val = []
    f1_weighted_scores_val = []
    acc_scores_test = []
    f1_macro_scores_test = []
    f1_weighted_scores_test = []
    
    
    for j in range(5):
        # build model
        model = Sequential()
        model.add(Embedding(nb_words,embed_dim,input_length=max_seq_len, weights=[embedding_matrix],trainable=False))
        model.add(LSTM(64))
        model.add(Dense(3,activation='softmax'))
        #model.summary()
        
        # compile model
        opt = tensorflow.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # train model
        #early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, mode = 'auto')
        #history = model.fit(word_seq_train, y_train_reformat, epochs = 100, validation_split = 0.1, callbacks = [early_stopping])
        model.fit(word_seq_train, y_train_reformat, epochs = 100, batch_size = 8, validation_data = (word_seq_val,y_val_reformat), verbose=0)
        
        # evaluation on dataset train
        pred_train = model.predict_classes(word_seq_train, verbose=0)
        acc_scores_train.append(round(accuracy_score(y_train, pred_train)*100, 2))
        f1_macro_scores_train.append(round(f1_score(y_train, pred_train, average='macro')*100,2))
        f1_weighted_scores_train.append(round(f1_score(y_train, pred_train, average='weighted')*100,2))
        # evaluation on dataset val
        pred_val = model.predict_classes(word_seq_val, verbose=0)
        acc_scores_val.append(round(accuracy_score(y_val, pred_val)*100, 2))
        f1_macro_scores_val.append(round(f1_score(y_val, pred_val, average='macro')*100,2))
        f1_weighted_scores_val.append(round(f1_score(y_val, pred_val, average='weighted')*100,2))
        # evaluation on dataset test
        pred_test = model.predict_classes(word_seq_test, verbose=0)
        acc_scores_test.append(round(accuracy_score(y_test, pred_test)*100, 2))
        f1_macro_scores_test.append(round(f1_score(y_test, pred_test, average='macro')*100,2))
        f1_weighted_scores_test.append(round(f1_score(y_test, pred_test, average='weighted')*100,2))
    # calculate the mean and std of the scores for each metric
    acc_train_lstm_avg.append(round(np.mean(acc_scores_train),2))
    acc_train_lstm_std.append(round(np.std(acc_scores_train),2))
    f1_macro_train_lstm_avg.append(round(np.mean(f1_macro_scores_train),2))
    f1_macro_train_lstm_std.append(round(np.std(f1_macro_scores_train),2))
    f1_weighted_train_lstm_avg.append(round(np.mean(f1_weighted_scores_train),2))
    f1_weighted_train_lstm_std.append(round(np.std(f1_weighted_scores_train),2))
    
    acc_val_lstm_avg.append(round(np.mean(acc_scores_val),2))
    acc_val_lstm_std.append(round(np.std(acc_scores_val),2))
    f1_macro_val_lstm_avg.append(round(np.mean(f1_macro_scores_val),2))
    f1_macro_val_lstm_std.append(round(np.std(f1_macro_scores_val),2))
    f1_weighted_val_lstm_avg.append(round(np.mean(f1_weighted_scores_val),2))
    f1_weighted_val_lstm_std.append(round(np.std(f1_weighted_scores_val),2))
    
    acc_test_lstm_avg.append(round(np.mean(acc_scores_test),2))
    acc_test_lstm_std.append(round(np.std(acc_scores_test),2))
    f1_macro_test_lstm_avg.append(round(np.mean(f1_macro_scores_test),2))
    f1_macro_test_lstm_std.append(round(np.std(f1_macro_scores_test),2))
    f1_weighted_test_lstm_avg.append(round(np.mean(f1_weighted_scores_test),2))
    f1_weighted_test_lstm_std.append(round(np.std(f1_weighted_scores_test),2))
    
print('****** SVM model results ******')
print('*** dataset train')
print('acc_train_svm =', acc_train_svm)
print('f1_macro_train_svm =', f1_macro_train_svm)
print('f1_weighted_train_svm =', f1_weighted_train_svm)
print('*** dataset val')
print('acc_val_svm =', acc_val_svm)
print('f1_macro_val_svm =', f1_macro_val_svm)
print('f1_weighted_val_svm =', f1_weighted_val_svm)
print('*** dataset test')
print('acc_test_svm =', acc_test_svm)
print('f1_macro_test_svm =', f1_macro_test_svm)
print('f1_weighted_test_svm =', f1_weighted_test_svm)

print('****** LSTM model results ******')
print('*** dataset train')
print('acc_train_lstm_avg =', acc_train_lstm_avg)
print('acc_train_lstm_std =', acc_train_lstm_std)
print('f1_macro_train_lstm_avg =', f1_macro_train_lstm_avg)
print('f1_macro_train_lstm_std =', f1_macro_train_lstm_std)
print('f1_weighted_train_lstm_avg =', f1_weighted_train_lstm_avg)
print('f1_weighted_train_lstm_std =', f1_weighted_train_lstm_std)
print('*** dataset val')
print('acc_val_lstm_avg =', acc_val_lstm_avg)
print('acc_val_lstm_std =', acc_val_lstm_std)
print('f1_macro_val_lstm_avg =', f1_macro_val_lstm_avg)
print('f1_macro_val_lstm_std =', f1_macro_val_lstm_std)
print('f1_weighted_val_lstm_avg =', f1_weighted_val_lstm_avg)
print('f1_weighted_val_lstm_std =', f1_weighted_val_lstm_std)
print('*** dataset test')
print('acc_test_lstm_avg =', acc_test_lstm_avg)
print('acc_test_lstm_std =', acc_test_lstm_std)
print('f1_macro_test_lstm_avg =', f1_macro_test_lstm_avg)
print('f1_macro_test_lstm_std =', f1_macro_test_lstm_std)
print('f1_weighted_test_lstm_avg =', f1_weighted_test_lstm_avg)
print('f1_weighted_test_lstm_std =', f1_weighted_test_lstm_std)

