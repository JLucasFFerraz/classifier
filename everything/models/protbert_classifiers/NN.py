import torch.nn as nn
import torch
from transformers import BatchEncoding
import pandas as pd
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,roc_auc_score


class PBLinearClassifier(torch.nn.Module):
    def __init__(self, model, tokenizer, device):
        super(PBLinearClassifier, self).__init__()
        self.model = model
        self.classify = nn.Linear(self.model.config.hidden_size, 2).to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.status = "raw"
        
    def train(self):
        self.model.train()
        self.classify.train()
        
    def eval(self):
        self.model.eval()
        self.classify.eval()
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        # Get BERT output
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output.to(self.device)
        
        # Get classification layer output
        logits = self.classify(pooled_output).to(self.device)
        
        # Turn that output into desired format
        return torch.softmax(logits, dim=1)
    
    def get_tokens(self, sequences):
        
        return self.tokenizer(sequences, return_tensors='pt', padding=True).to(self.device)
    
    def generate_embeddings(self, sequences):
        
        tokens = self.get_tokens(sequences)
        #del tokens["token_type_ids"]
        return self.model(**tokens).pooler_output

    def set_status(self, new_status):
        self.status = new_status

    def train(self, train_dataloader, optimizer, criterion):
        self.model.train() 
        # Defines start of training mode
        
        loss_list = []
        total_loss = 0
        i = 0
        
        for batched_sequences, labels in train_dataloader:
            # Reset optimizer gradient
            optimizer.zero_grad()
            
            #print("Labels:", labels)

            # Get the tokens
            tokens = self.get_tokens(batched_sequences)
            
            # Get predictions
            outputs = self(**tokens)
            
            labels =  torch.eye(2)[labels]
            real = labels.to(device=self.device, dtype=torch.float32)
            
            # print(outputs)
            # print(real)

            # TO DO IT ONLY WORKS FOR BATCH_SIZE = 1 RIGHT NOW
            loss = criterion(outputs, real)
            
            # THIS IS ONLY FOR ICS BS > 1
            loss = 1-torch.mean(loss)
            
            # print(loss)
            # print("\n")
            # print("Loss:", outputs, "vs", real, "=", loss)
            # print("\n")
            #print("Loss =", loss.item())
            #print("---------------------------------------------")
            
            # Propagate the loss
            loss.backward()
            
            optimizer.step()
            
            
            loss_list.append(loss.item())
            
            i += 1
            
        return sum(loss_list)/len(train_dataloader), loss_list

    def test(self, test_dataloader, criterion):
        every_loss = []
        i = 0
        self.model.eval()
        
        every_output = []
        every_label = []
        # Without affecting the weights of the model
        with torch.no_grad():
            for batched_sequences, labels in test_dataloader:

                #print(str(i), "/", str(len(test_dataloader)))
                
                # Get the tokens
                tokens = self.get_tokens(batched_sequences)
                #del tokens["token_type_ids"]
                
                # Get predictions
                outputs = self(**tokens)
                
                labels =  torch.eye(2)[labels]
                
                #labels = torch.Tensor([[0.0, 1.0] if x == 1 else torch.Tensor([0.0, 1.0]) for x in labels])
                
                real = labels.to(device=self.device, dtype=torch.float32)#[0].to(model.device)
                #real.requires_grad_()
                #print("Labels after", real)
                
                
                loss = criterion(outputs, real)
                
                # THIS IS ONLY FOR ICS BS > 1
                loss = 1-torch.mean(loss)
                #print(loss)
                every_output.append(outputs)
                every_label.append(real)
                # Accumulate loss
                
                every_loss.append(loss.item())
                
                i += 1
        
        true_labels = []
        for x in every_label:
            if not isinstance(x, list):
                for case in x:
                    if (case[0] < case[1]):
                        true_labels.append(1)
                    else:
                        true_labels.append(0)
            else:
                if (x[0] < x[1]):
                    true_labels.append(1)
                else:
                    true_labels.append(0)
        true_outputs = []
        
        for x in every_output:
            if not isinstance(x, list):
                for case in x:
                    if (case[0] < case[1]):
                        true_outputs.append(1)
                    else:
                        true_outputs.append(0)
            else:
                if (x[0] < x[1]):
                    true_outputs.append(1)
                else:
                    true_outputs.append(0)
        #print(true_outputs)
        accuracy = accuracy_score(y_true=true_labels, y_pred=true_outputs)
        precision = precision_score(y_true=true_labels, y_pred=true_outputs, average="macro")
        recall = recall_score(y_true=true_labels, y_pred=true_outputs, average="macro")
        try:
            roc= roc_auc_score(y_true=true_labels, y_score=true_outputs, average='macro') 
        except:
            roc = None
            print("Error in roc")
        f1 = f1_score(y_true=true_labels, y_pred=true_outputs, average="macro")
        try:
            average_loss = sum(every_loss)/len(every_loss)
        except:
            average_loss = None
        return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1, "ROC AUC": roc}, average_loss, every_loss



