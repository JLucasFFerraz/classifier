import torch.nn as nn
import torch
from transformers import BatchEncoding
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import time
from sklearn.metrics import accuracy_score, classification_report

# Gerar embedding para cada seq
# Ficar com dataset embedding / viability


class PBRandomForestClassifier:
    def __init__(self, model, tokenizer, device, n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2):
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.classify = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split)
        self.status = "Raw"
        
    def set_status(self, new_status):
        self.status = new_status
        
    def get_embeddings_df(self, data):
        new_df = pd.DataFrame(columns=["Embedding", "Label"])
        for sequence, label in data:
            
            e = self.get_embedding(sequence)
            
            new_line = pd.DataFrame({"Embedding": [e], "Label": [label]})
            new_df = pd.concat([new_df, new_line], ignore_index=True)
        return new_df
    
    
    def fit(self, X_train, y_train):
        self.set_status("Forest fitting")
        self.model.eval()
        X_train = X_train.tolist()
        X_train = X_train[:5]
        y_train = y_train[:5]
        train_embeddings = []
        for seq in X_train:
            train_embeddings.append(self.generate_embeddings([seq]).cpu().detach())
        
        X_train = np.vstack(train_embeddings) 
        y_train = np.array(y_train)
        y_train=y_train.astype('int')
        
        self.classify.fit(X_train, y_train)
        
    def unfrozen_predict(self, sequences):
        self.model.train()
        embeddings = self.generate_embeddings(sequences)#.cpu().detach()
        embeddings_copy = embeddings.clone().cpu().detach()
        #print(embeddings)
        #embeddings = np.vstack(embeddings) 
        result = torch.Tensor(self.classify.predict(embeddings_copy)[0])
        print(result)
        return result
    
    def frozen_predict(self, sequences):
        self.model.eval()
        embeddings = self.generate_embeddings(sequences).cpu().detach()
        #print(embeddings)
        #embeddings = np.vstack(embeddings) 
        return self.classify.predict(embeddings)
    
    def evaluate(self, X_test, y_test):
        self.model.eval()
        y_test = np.array(y_test)
        y_test=y_test.astype('int')
        
        y_pred = []
        for seq in X_test:
            y_pred.append(self.frozen_predict([seq]))
        
        accuracy = accuracy_score(y_test, y_pred)
        #print("Acc =", accuracy)
        report = classification_report(y_test, y_pred)
        #print("Report =", report)
        return accuracy, report
    
    def get_tokens(self, sequences):
        #print(type(sequences))
        return self.tokenizer(sequences, return_tensors='pt', padding=True).to(self.device)
    
    def generate_embeddings(self, sequences):
        tokens = self.get_tokens(sequences)
        del tokens["token_type_ids"]
        return self.model(**tokens).pooler_output#.to(self.device)
        
    def fine_tuning_train(self, X_train, y_train, optimizer, epochs=1):
        
        self.set_status("Fine tuning")
        self.model.train()
        print(f"({self.status})")
        # Define a loss function and optimizer for fine-tuning
        criterion = nn.CosineSimilarity(dim=0)
        
        # Fine-tuning training loop
        for epoch in range(epochs):
            
            for i, elem in enumerate(X_train):
                #elem.requires_grad_()
                # Forward pass
                optimizer.zero_grad()
                print(elem)
                predicted_labels = self.unfrozen_predict([elem])
                #print("Prediction:", predicted_labels)
                predicted_labels_tensor = torch.Tensor([[0.0, 1.0] if x == 1 else [1.0, 0.0] for x in predicted_labels])
                predicted_labels_tensor.requires_grad_()
                
                #predicted_labels_tensor.requires_grad_()
                #print("Pred tensor:", predicted_labels_tensor)
                
                
                real_labels_tensor = torch.Tensor([y_train[i].tolist()])
                #real_labels_tensor.requires_grad_()
                #print("Real:", real_labels_tensor)
                
                # print("Calculating loss between", predicted_labels_tensor[0], "and", real_labels_tensor[0])
                l = criterion(predicted_labels_tensor[0], real_labels_tensor[0])
                #print("Result of cosine sim function between them:", l)
                
                loss = 1-l
                
                #print("Loss:")
                #print(loss.item())
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                
    # TO DO FIX
    def fine_tuning_test(self, X_test, y_test):
        self.model.eval()
        total_loss = 0.0
        
        predictions = []
        true_labels = []
        criterion = nn.CosineSimilarity(dim=0)
        with torch.no_grad():
            for i, input in enumerate(X_test):
                real_label = y_test[i].tolist()
                
                predicted_labels = self.predict([input])
                #print("Prediction:", predicted_labels)
                
                predicted_labels_tensor = torch.Tensor([[0.0, 1.0] if x == 1 else [1.0, 0.0] for x in predicted_labels])
                #predicted_labels_tensor.requires_grad_()
                #print("Pred tensor:", predicted_labels_tensor)
                
                
                real_labels_tensor = torch.Tensor([real_label])
                #real_labels_tensor.requires_grad_()
                #print("Real:", real_labels_tensor)
                
                l = criterion(predicted_labels_tensor[0], real_labels_tensor[0])
                #print("Result of cosine sim function between:", l)
                
                loss = 1-l
                total_loss += loss
                true_labels.append(real_label[0])
                predictions.append(predicted_labels[0])
                #print("Loss:")
                #print(loss.item())
                time.sleep(1)
        average_loss = total_loss / len(X_test)
        accuracy = accuracy_score(true_labels, predictions)
        #print("Total loss =", total_loss)
        #print("Average loss =", average_loss)
        
        #print("Acc =", accuracy)
        