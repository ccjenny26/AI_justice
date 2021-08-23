import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import argparse
import bs4
from bs4 import BeautifulSoup
import lxml
import glob
from sklearn.preprocessing import LabelEncoder
import spacy
from spacy.lang.fr.examples import sentences
import fr_core_news_sm
from sklearn.model_selection import train_test_split
import random
import torch
import transformers
from transformers import FlaubertTokenizer, FlaubertModel, FlaubertConfig
from torch.nn.functional import cross_entropy
from sklearn.metrics import f1_score
import time

print('flaubert_all_2classes')
### import data
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

# get a list of filenames wanted, here only 'M' and 'P'
idx_keep = [idx for idx, value in enumerate(labels) if value in ['M','P']]
filenames_new = [filenames[i] for i in idx_keep]
print('number of files used:',len(filenames_new))

# get input and output
description, classes = create_data(filenames_new)

## preprocession of the data
# transform classes into numeric
le = LabelEncoder()
y = le.fit_transform(classes)

# convert each list of strings (represents one document) into one string
inputs = []
for i in range(len(description)):
    document = " ".join(description[i])
    inputs.append(document)

# segmentation of sentences for each document
nlp = spacy.load("fr_core_news_sm")
doc_sent = []
for i in range(len(inputs)):
    doc = nlp(inputs[i])
    sentences = [sent.text for sent in doc.sents]
    doc_sent.append(sentences)


# define tokenizer of sentences
MAX_LEN = 512
tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased', padding=True, truncation=True)

def get_tokenized(documents,label):
    tokenizer_out = tokenizer(
        documents,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
        )
    label = torch.tensor(label, dtype=torch.long)
    # tokenizer_out est un dictionnaire qui contient 2 clés: input_ids et attention_mask
    return tokenizer_out, label # on renvoie un tuple à 2 éléments


# build the classification model
PRE_TRAINED_MODEL_NAME = 'flaubert/flaubert_base_cased'
flaubert = FlaubertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
FREEZE_PRETRAINED_MODEL = True

class FlaubertForCourtDecisionClassification(torch.nn.Module):
    def __init__(self, config, num_labels, freeze_encoder=False, lstm_hidden = 300):
        # instantiate Flaubert model
        #super().__init__(config)
        super(FlaubertForCourtDecisionClassification, self).__init__()
        # instantiate num. of classes
        self.num_labels = num_labels
        # instantiate and load a pretrained Flaubert model as encoder
        self.encoder = FlaubertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        # freeze the encoder parameters if required
        if freeze_encoder:
          for param in self.encoder.parameters():
              param.requires_grad = False
        # the classifier: a feed-forward layer attached to the encoder's head
        self.classifier = torch.nn.Linear(
            in_features=lstm_hidden*2, out_features=self.num_labels, bias=True)
        # instantiate a dropout function for the classifier's input
        self.dropout = torch.nn.Dropout(p=0.1)
        self.loss = torch.nn.CrossEntropyLoss()
        # apply a LSTM
        self.lstm = torch.nn.LSTM(self.encoder.config.hidden_size,lstm_hidden,1, batch_first = True, bidirectional=True)   #(input_size, hidden_size, num_layers)
        #self.lstm = torch.nn.LSTM(768,lstm_hidden,1, batch_first = True, bidirectional=True)   #(input_size, hidden_size, num_layers)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        label=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        output_dict = {}
        # encode a batch of sequences with FlaubertModel
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # extract the hidden representations from the encoder output
        hidden_state = encoder_output[0]  # (bs, seq_len, dim)
        # only select the encoding corresponding to the first token of each sequence in the batch
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        # apply aggregation
        _,(hn,cn) = self.lstm(pooled_output.unsqueeze(0))
        
        # apply dropout
        hn = self.dropout(hn.view(1,-1))  # (bs, dim)
        # feed into the classifier
        logits = self.classifier(hn)  # (bs, dim)
        output_dict["scores"] = logits
        output_dict["prediction"] = torch.argmax(logits)
    
        if label is not None:
            loss = self.loss(logits, label.view(-1))
            output_dict["loss"] = loss

        return output_dict

train_accuracy_list = []
train_f1_macro_list = []
train_f1_weighted_list = []
val_accuracy_list = []
val_f1_macro_list = []
val_f1_weighted_list = []
test_accuracy_list = []
test_f1_macro_list = []
test_f1_weighted_list = []

# repeat the training 5 times, each time re-split the data
for i in range(5):
    # split the data into train and test
    x_train, x_test, y_train, y_test = train_test_split(doc_sent, y, test_size=0.20, stratify = y)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.25, stratify = y_train)

    # tokenize all documents
    train_dataset = [get_tokenized(x_train[i], y_train[i]) for i in range(len(y_train))]
    val_dataset = [get_tokenized(x_val[i], y_val[i]) for i in range(len(y_val))]
    test_dataset = [get_tokenized(x_test[i], y_test[i]) for i in range(len(y_test))]

    # instantiate model
    num_labels = len(np.unique(classes))
    model = FlaubertForCourtDecisionClassification(
        config=flaubert.config, num_labels=num_labels,
        freeze_encoder = FREEZE_PRETRAINED_MODEL
        )

    ### @@@@: device = gpu (si disponible) ou cpu (sinon)
    if torch.cuda.is_available():
        print("Using CUDA")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("Warning: no GPU found, using CPUs")

    # on déplace les paramètres du modèle sur la mémoire du gpu
    model.to(device)
    ### @@@@

    # train the model
    num_training_epochs = 50
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_val_accuracy = 0

    for epoch in range(num_training_epochs):
        train_loss = 0
        train_accuracy = 0
            
        model.train()
            
        random.shuffle(train_dataset)

        for doc, label in train_dataset:
            
            optimizer.zero_grad()
            
            ### <<@@@@@ on doit également transférer les exemples d'entraînement sur le gpu
            
            # calculate the loss and prediction
            output_dict = model(input_ids=doc["input_ids"].to(device),
                                         attention_mask=doc["attention_mask"].to(device),
                                         label=label.to(device))
            ### @@@@@>>
            loss = output_dict["loss"]
            
            loss.backward()         # backpropagation
            optimizer.step()        # update the parameters
                
            # tester on each exemple if prediction equal to true label
            prediction = output_dict["prediction"]
            if prediction == label:
                train_accuracy += 1

            train_loss += loss.item()

        train_accuracy = train_accuracy/len(train_dataset) * 100
        train_loss = train_loss / len(train_dataset)

        ### @@@@ j'ai indenté ce bloc de code -> on veut évaluer sur le corpus de validation à chaque époque (de manière à sélectionner le meilleur modèle)
        # evaluate on test_dataset
        model.eval() # mode eval (va avoir un effet sur certaines couches, comme batchnorm ou dropout qui se comportent différemment en entraînement et en test)
        with torch.no_grad(): # dit à pytorch de ne pas réserver de mémoire pour le calcul de la backpropagation, ce qui rend l'inférence plus rapide
            val_loss = 0
            val_accuracy = 0
            val_pred_label = []
            val_true_label = []
            for doc, label in val_dataset:
                ### <<@@@@@ on doit également transférer les exemples d'entraînement sur le gpu
                output_dict = model(input_ids=doc["input_ids"].to(device),
                                             attention_mask=doc["attention_mask"].to(device),
                                             label=label.to(device))
                        
                val_loss += output_dict["loss"].item()
                prediction = output_dict["prediction"]
                
                if prediction == label:
                    val_accuracy += 1
                val_pred_label.append(prediction.cpu())
                val_true_label.append(label)

            # calculate loss, accuracy, f1 micro
            val_accuracy = round(val_accuracy / len(val_dataset) * 100, 2)
            val_loss = val_loss / len(val_dataset)
            
            print(f"Epoch {epoch}  train loss = {train_loss:.6f} train accuracy = {train_accuracy:.2f} val loss = {val_loss:.6f} val accuracy = {val_accuracy:.2f}")

            # save the best model
            if val_accuracy > best_val_accuracy:
                print("Saving model")
                ### @@@@ commentaire: dans une version précédente de pytorch, ça causait un bug si on ne faisait pas:
                ### @@@@ model.cpu() avant de sauver le modèle (ça ne devrait plus être le cas maintenant, mais si c'est le cas, il faudra décommenter ces lignes:
                
                # model.cpu()
                #model_name = f"bestmodel_flaubert_lstm_{round(time.time())}"
                model_name = "bestmodel_flaubert_all_2c"+str(i)
                torch.save(model, model_name)
                model.to(device)
                best_val_accuracy = val_accuracy


    # restore best parameters
    print("Training finished, restoring best parameters")
    model = torch.load(model_name)

    ### @@@@
    model.to(device)


    #print('****** evaluate best model on train,val and test ******')
    #print('****** dataset train ******')
    # evaluate the best model on dataset train
    train_loss = 0
    train_accuracy = 0
    train_pred_label = []
    train_true_label = []
    for doc, label in train_dataset:
        ###  @@@
        output_dict = model(input_ids=doc["input_ids"].to(device),
                                    attention_mask=doc["attention_mask"].to(device),
                                    label=label.to(device))
                    
        train_loss += output_dict["loss"].item()
        prediction = output_dict["prediction"]
        
        if prediction == label:
            train_accuracy += 1
        train_pred_label.append(prediction.cpu())
        train_true_label.append(label)
            
    # calculate loss, accuracy, f1 micro
    train_accuracy_list.append(round(train_accuracy / len(train_dataset) * 100, 2))
    train_f1_macro_list.append(round(f1_score(train_pred_label, train_true_label, average='macro') * 100, 2))
    train_f1_weighted_list.append(round(f1_score(train_pred_label, train_true_label, average='weighted') * 100, 2))

    # print out the result
    #print('train loss=',train_loss,',train_accuracy=',train_accuracy,',train_f1_micro=',train_f1_micro,'train_f1_macro=',train_f1_macro,'train_f1_weighted=',train_f1_weighted)


    #print('****** dataset val ******')
    # evaluate the best model on dataset train
    val_loss = 0
    val_accuracy = 0
    val_pred_label = []
    val_true_label = []
    for doc, label in val_dataset:
        ###  @@@
        output_dict = model(input_ids=doc["input_ids"].to(device),
                                    attention_mask=doc["attention_mask"].to(device),
                                    label=label.to(device))
                    
        val_loss += output_dict["loss"].item()
        prediction = output_dict["prediction"]
        
        if prediction == label:
            val_accuracy += 1
        val_pred_label.append(prediction.cpu())
        val_true_label.append(label)
            
    # calculate loss, accuracy, f1
    val_accuracy_list.append(round(val_accuracy / len(val_dataset) * 100, 2))
    val_f1_macro_list.append(round(f1_score(val_pred_label, val_true_label, average='macro') * 100, 2))
    val_f1_weighted_list.append(round(f1_score(val_pred_label, val_true_label, average='weighted') * 100, 2))

    # print out the result
    #print('val loss=',val_loss,',val_accuracy=',val_accuracy,',val_f1_micro=',val_f1_micro,'val_f1_macro=',val_f1_macro,'val_f1_weighted=',val_f1_weighted)


    #print('****** dataset test ******')
    # evaluate the best model on dataset test
    test_loss = 0
    test_accuracy = 0
    test_pred_label = []
    test_true_label = []

    for doc, label in test_dataset:
        ### @@@
        output_dict = model(input_ids=doc["input_ids"].to(device),
                                    attention_mask=doc["attention_mask"].to(device),
                                    label=label.to(device))
                    
        test_loss += output_dict["loss"].item()
        prediction = output_dict["prediction"]
        if prediction == label:
            test_accuracy += 1
        test_pred_label.append(prediction.cpu())
        test_true_label.append(label)

    # calculate loss, accuracy, f1 micro
    test_accuracy_list.append(round(test_accuracy / len(test_dataset) * 100, 2))
    test_f1_macro_list.append(round(f1_score(test_pred_label, test_true_label, average='macro') * 100, 2))
    test_f1_weighted_list.append(round(f1_score(test_pred_label, test_true_label, average='weighted') * 100, 2))

    # print out the result
    #print('test loss=',test_loss,',test_accuracy=',test_accuracy,',test_f1_micro=',test_f1_micro,'test_f1_macro=',test_f1_macro,'test_f1_weighted=',test_f1_weighted)

print('train_acc_list:', train_accuracy_list, 'train_acc_avg=', round(np.mean(train_accuracy_list),2), 'train_acc_std=', round(np.std(train_accuracy_list),2))
print('train_f1_macro_list:', train_f1_macro_list, 'train_f1_macro_avg=', round(np.mean(train_f1_macro_list),2), 'train_f1_macro_std=', round(np.std(train_f1_macro_list),2))
print('train_f1_weighted_list:', train_f1_weighted_list, 'train_f1_weighted_avg=', round(np.mean(train_f1_weighted_list),2), 'train_f1_weighted_std=', round(np.std(train_f1_weighted_list),2))

print('val_acc_list:', val_accuracy_list, 'val_acc_avg=', round(np.mean(val_accuracy_list),2), 'val_acc_std=', round(np.std(val_accuracy_list),2))
print('val_f1_macro_list:', val_f1_macro_list, 'val_f1_macro_avg=', round(np.mean(val_f1_macro_list),2), 'val_f1_macro_std=', round(np.std(val_f1_macro_list),2))
print('val_f1_weighted_list:', val_f1_weighted_list, 'val_f1_weighted_avg=', round(np.mean(val_f1_weighted_list),2), 'val_f1_weighted_std=', round(np.std(val_f1_weighted_list),2))

print('test_acc_list:', test_accuracy_list, 'test_acc_avg=', round(np.mean(test_accuracy_list),2), 'test_acc_std=', round(np.std(test_accuracy_list),2))
print('test_f1_macro_list:', test_f1_macro_list, 'test_f1_macro_avg=', round(np.mean(test_f1_macro_list),2), 'test_f1_macro_std=', round(np.std(test_f1_macro_list),2))
print('test_f1_weighted_list:', test_f1_weighted_list, 'test_f1_weighted_avg=', round(np.mean(test_f1_weighted_list),2), 'test_f1_weighted_std=', round(np.std(test_f1_weighted_list),2))

#print('****** input/predicted label/gold label on dataset test ******')
## extract input
#doc_ids = [doc_sent.index(x) for x in x_test]
#filenames_test = [filenames[i] for i in doc_ids]
## extract predicted label
#label_pred_test_num = [x.item() for x in test_pred_label]
#label_pred_test = le.inverse_transform(label_pred_test_num)
## extract gold label
#label_true_test_num = [x.item() for x in test_true_label]
#label_true_test = le.inverse_transform(label_true_test_num)
## print out the examples in form of input/predicted label/gold label
#print(list(zip(filenames_test, label_pred_test, label_true_test)))







