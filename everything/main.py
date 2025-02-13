import pickle
import random
from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import os


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.protbert_classifiers.NN import PBLinearClassifier
from models.protbert_classifiers.RandomForest import PBRandomForestClassifier
from models.pure_classifiers.NN import BinaryClassifier
from torch.utils.data import DataLoader
import os
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedShuffleSplit


import pandas as pd

import datetime
import numpy as np
import re
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR, LambdaLR

torch.cuda.empty_cache()
# Check for GPU availability and allow GPU use if available
print("Loading")
print("GPU available:", torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get ProtBERT and BertTokenizer from HuggingFace



pre = "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMIT"
pre_after_process = re.sub(r"[UZOB]", "X", " ".join([x for x in pre]))
post = "QAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL"
post_after_process = re.sub(r"[UZOB]", "X", " ".join([x for x in post]))

# Deep Diversification data preprocessing

class SequenceViabilityDataset:
  def __init__(self, sequences, labels):
    self.sequences = sequences
    self.labels = labels

  # Fetches a pair of sequence and corresponding viability information
  def __getitem__(self, i):
    return self.sequences[i], self.labels[i]

  # Returns the number of pairs stored
  def __len__(self):
    return len(self.sequences)

  # Separates the data into train, test and validation partitions
  # (Returns a dict with the names of the partitions as keys and 
  # the indexes of data belonging to that partition as values)
  def get_idx_split(self):
    all_ids = range(len(self.sequences)) #random.sample(range(len(self.sequences)), 100) 
    train_ids, test_ids = train_test_split(all_ids, train_size = 0.8, random_state=32)
    test_ids, val_ids = train_test_split(test_ids, train_size = 0.5, random_state=32)
    return {"train": train_ids, "test": test_ids, "valid": val_ids}


  def get_stratified_ttv_split(self):
      stratified_splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.8)
      first_split_result = stratified_splitter.split(self.sequences, self.labels)
      
      ab =  [x for x in first_split_result][0]
      train_indexes= ab[0]
      temporary_indexes = ab[1]
      
      train_data = [self[i] for i in train_indexes]
      stratified_splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.5)
      
      remaining_indexes = [i for i in temporary_indexes]
      remaining_sequences, remaining_labels = [self.sequences[i] for i in remaining_indexes], [self.labels[i] for i in remaining_indexes]
      
      second_split_result = stratified_splitter.split(remaining_sequences, remaining_labels)
      ab =  [x for x in second_split_result][0]
      test_indexes= ab[0]
      val_indexes = ab[1]
      
      
      test_data, val_data = [(remaining_sequences[i], remaining_labels[i]) for i in test_indexes], [(remaining_sequences[i], remaining_labels[i]) for i in val_indexes]
      
      return {"train": train_data, "test": test_data, "valid": val_data}
  
  def get_kfold_splits(self, k):
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=32)
    return kf.split(np.array(self.sequences), np.array(self.labels))
  

pre = "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMIT"
pre_after_process = re.sub(r"[UZOB]", "X", " ".join([x for x in pre]))
post = "QAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL"
post_after_process = re.sub(r"[UZOB]", "X", " ".join([x for x in post]))

# Deep Diversification data preprocessing
def deep_diversification_data_preprocess(partitions_to_consider, full_sequences=False):
  # Read the csv containing the data
  df = pd.read_csv("./datasets/protein_mutation_viability.csv", delimiter=";")
  print("Should do full:", full_sequences)
  # Preprocess the data
  all_data = []
 
  # For row in the dataset
  for index, row in df.iterrows():
    
    p = row["partition"]
    
    # If the partition is one of the ones we want
    if p in partitions_to_consider:
      s = row["sequence"]
      l = row["is_viable"]
      
      
      # Turn TRUE and FALSE into 1s and 0s
      try:
        l = int(l)
      except:
        print("ERROR IN COLUMN")
      
      # Make AA sequence uppercase and separate each AA with blankspaces for ProtBERT
      s = " ".join(list(s.upper()))
      
      # Replace instances of the U,Z,O,B symbols with X for ProtBERT
      s = re.sub(r"[UZOB]", "X", s)
      if full_sequences:
        s = "[CLS] "+pre_after_process + " " + s + " " + post_after_process+" [SEP]"
      else:
        s = "[CLS] "+s+" [SEP]"
      all_data.append((s, l))
  
  return all_data

def generate_completely_random_sequences(number_to_generate, sequence_size):
    every_aa = 'VDSHIATLYWCRQNFMGKEP'
    every_sequence_edge = []
    every_sequence_no_edge = []
    random.seed(32)
    for n in range(number_to_generate):
        generated_sequence = random.choices(every_aa, k=sequence_size)
        
        generated_sequence = " ".join(generated_sequence)
        generated_sequence = re.sub(r"[UZOB]", "X", generated_sequence)
        every_sequence_no_edge.append("[CLS] "+generated_sequence+" [SEP]")
    
        generated_sequence_edge = "[CLS] "+pre_after_process + " " + generated_sequence + " " + post_after_process+" [SEP]"
        every_sequence_edge.append(generated_sequence_edge)
        
    return every_sequence_edge, every_sequence_no_edge
    
def get_realistic_random_sequences(number_to_get, largest_sequence_size):    
    with open("random_sequences.txt", "r+") as f:
        
        unique = f.readlines()
    f.close()
    only_fragments = []
    for seq in unique:
        before, mid, after = seq[:len(pre)], seq[len(pre):len(pre)+largest_sequence_size], seq[len(pre)+largest_sequence_size:]
        only_fragments.append(mid)
    unique_newline_removed = [x.replace("\n", "") for x in unique]
    processed_unique = ["[CLS] "+re.sub(r"[UZOB]", "X", " ".join([x for x in y]))+" [SEP]" for y in unique_newline_removed]
    return processed_unique[:number_to_get], only_fragments[:number_to_get]

def get_deep_cached_data():
    with open("embeddings_and_labels.pickle", "rb") as file:
            deep_diversification_data = pickle.load(file)
    file.close()
    
    #print(deep_diversification_data)
    return deep_diversification_data

def get_random_data(nr_to_get, completely_random=False, with_edges=False, largest_sequence_size=28):
    if completely_random:
        random_sequences_edges, random_sequences_no_edges = generate_completely_random_sequences(nr_to_get, largest_sequence_size)
        
    else:
        random_sequences_edges, random_sequences_no_edges = get_realistic_random_sequences(nr_to_get, largest_sequence_size)
    return random_sequences_edges, random_sequences_no_edges
    
def deep_vs_random_pca_tnse(deep_diversification_embeddings, random_embeddings, name_to_give, appendix):
    
    fixed = pd.Series([np.array(x[0]) for x in deep_diversification_embeddings["Embedding"].values])
    
    
    non_random_flag = pd.Series(0, index=deep_diversification_embeddings.index, name="Random")
    
    #print(non_random_flag)
    deep_diversification_embeddings = pd.DataFrame({"Embedding": fixed, "Random": non_random_flag})
    # print("Deep")
    # print(deep_diversification_embeddings)

    random_embeddings = random_embeddings["Embedding"]
    random_flag = pd.Series(1, index=random_embeddings.index, name="Random")
    random_embeddings =  pd.DataFrame({"Embedding": random_embeddings, "Random": random_flag})
    # print("Random")
    # print(random_embeddings)

    data_subset = pd.concat([random_embeddings, deep_diversification_embeddings], ignore_index=True)#.values
    
    res_arr = []
    res_labels = []
    for i, row in data_subset.iterrows():
        arr = np.array(row["Embedding"])
        
        label = row["Random"]
        
        res_arr.append(arr)
        res_labels.append(label)
    
    res_arr = np.array(res_arr, dtype=object)
    res_labels = np.array(res_labels, dtype=object)
    
    pca = PCA(n_components=2)
    pca_embeddings_2d = pca.fit_transform(res_arr)
    df_pca = pd.DataFrame(data=pca_embeddings_2d, columns=['Dimension 1', 'Dimension 2'])
    #print(df_pca)
    df_pca['Label'] = res_labels
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df_pca['Dimension 1'], df_pca['Dimension 2'], c=df_pca['Label'], cmap='viridis')
    plt.colorbar(scatter, label='Label')
    if appendix:
        extra_name = f"{appendix}_"
    else:
        extra_name = ""
    plt.title(extra_name+name_to_give.split(".")[0]+ ' PCA Plot')
    
    
    
    plt.savefig(f"./{extra_name}pca_results/{name_to_give}.png")
    plt.close()
    
    tsne = TSNE(random_state=42)
    embeddings_2d = tsne.fit_transform(res_arr)
    df_tsne = pd.DataFrame(data=embeddings_2d, columns=['Dimension 1', 'Dimension 2'])
    df_tsne['Label'] = res_labels

    # Plot the t-SNE plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df_tsne['Dimension 1'], df_tsne['Dimension 2'], c=df_tsne['Label'], cmap='viridis')
    plt.colorbar(scatter, label='Label')
    plt.title(extra_name+name_to_give.split(".")[0]+ ' t-SNE Plot')
    
    plt.savefig(f"./{extra_name}tsne_results/{name_to_give}.png")
    plt.close()

def viability_deep_vs_random_tnse(deep_diversification_embeddings, random_embeddings, name_to_give, result_folder):
    fixed = pd.Series([np.array(x[0]) for x in deep_diversification_embeddings["Embedding"].values])
    
    non_random_flag = pd.Series([np.array(x) for x in deep_diversification_embeddings["Label"].values])
    
    deep_diversification_embeddings = pd.DataFrame({"Embedding": fixed, "Random": non_random_flag})
    #print(random_embeddings)
    random_flag = pd.Series(2, index=range(len(random_embeddings)), name="Random")
    random_embeddings =  pd.DataFrame({"Embedding": random_embeddings, "Random": random_flag})

    data_subset = pd.concat([random_embeddings, deep_diversification_embeddings], ignore_index=True)

    res_arr = []
    res_labels = []
    for i, row in data_subset.iterrows():
        arr = np.array(row["Embedding"])
        
        label = row["Random"]
        
        res_arr.append(arr)
        res_labels.append(label)
    res_arr = np.array(res_arr)
    
    
    tsne = TSNE(random_state=42)
    embeddings_2d = tsne.fit_transform(res_arr)
    df_tsne = pd.DataFrame(data=embeddings_2d, columns=['Dimension 1', 'Dimension 2'])
    df_tsne['Label'] = res_labels

    # Plot the t-SNE plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df_tsne['Dimension 1'], df_tsne['Dimension 2'], c=df_tsne['Label'], cmap='viridis')
    plt.colorbar(scatter, label='Label')
    plt.title(name_to_give.split(".")[0]+' viability t-SNE Plot')
    result_file = os.path.join(result_folder, name_to_give)
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    plt.savefig(result_file)
    plt.close()

def save_data_pickle(res_arr, res_labels):

    arr_and_labels = [res_arr, res_labels]
    with open("full_embeddings_and_labels_arrays_for_pca.pickle", "wb") as file:
        pickle.dump(arr_and_labels, file)
    file.close()

def load_data_pickle():
    with open("full_embeddings_and_labels_arrays_for_pca.pickle", "rb") as file:
        arr_and_labels = pickle.load(file)
    file.close()
    return arr_and_labels

def generate_embeddings_pickles_rf(model, partitions, nr_embeddings_to_fetch, with_edges=False, save_folder=None):
    read = pd.read_csv("./datasets/protein_mutation_viability.csv", delimiter=";")
    read = read.sample(frac=1, random_state=32).reset_index(drop=True)
    every_partition_dict = {x[0]: pd.DataFrame(columns=["Embedding", "Label"]) for x in partitions}
    counters = {x[0]: 0 for x in partitions}
    if save_folder:
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
    for i, row in read.iterrows():
            
            p = row["partition"]
            
            for attributed_name, partition_set in partitions:
                if counters[attributed_name] < nr_embeddings_to_fetch:
                    
                    if p in partition_set:
                        
                        
                        sequence = row["sequence"]
                        
                        sequence = sequence.upper()
                        sequence = re.sub(r"[UZOB]", "X", " ".join([x for x in sequence]))
                        if with_edges:
                            sequence = pre_after_process + " " + sequence + " " + post_after_process
                        sequence = "[CLS] " + sequence+" [SEP]"
                        
                        
                        
                        e = model.generate_embeddings([sequence]).cpu().detach()
                        
                        l = 1 if row["is_viable"] else 0
                        new_line = pd.DataFrame({"Embedding": [e], "Label": [l]})
                        
                        every_partition_dict[attributed_name] = pd.concat([every_partition_dict[attributed_name], new_line], ignore_index=True)
                        counters[attributed_name] += 1
                        
                        
                        if i % 100 == 0:
                            print(attributed_name, counters[attributed_name],"/",nr_embeddings_to_fetch)
                        
                        
                        
                        #print("After add:", all_embeddings_labels)

        
        # print("writing pickle")
        # with open("every_walked.pickle", "wb") as file:
        #     pickle.dump(all_embeddings_labels, file)
        # file.close()
    
    for name in every_partition_dict:
        p = os.path.join(save_folder, name+".pickle")
        with open(p, "wb") as file:
            pickle.dump(every_partition_dict[name], file)
        file.close()
    
def generate_embeddings_pickles_nn(model, partitions, nr_embeddings_to_fetch, with_edges=False, save_folder=None, special_name=None):
    read = pd.read_csv("./datasets/protein_mutation_viability.csv", delimiter=";")
    read = read.sample(frac=1, random_state=32).reset_index(drop=True)
    every_partition_dict = {special_name: pd.DataFrame(columns=["Embedding", "Label"])}
    counters = {special_name: 0}
    if save_folder:
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
    for i, row in read.iterrows():
            
            p = row["partition"]
            
            for attributed_name, partition_set in partitions:
                if counters[special_name] < nr_embeddings_to_fetch:
                    
                    if p in partition_set:
                        
                        
                        sequence = row["sequence"]
                        sequence = sequence.upper()
                        sequence = re.sub(r"[UZOB]", "X", " ".join([x for x in sequence]))
                        
                        if with_edges:
                            sequence = pre_after_process + " " + sequence + " " + post_after_process
                        sequence = "[CLS] "+sequence+" [SEP]"
                        e = model.generate_embeddings([sequence]).cpu().detach()
                        
                        l = 1 if row["is_viable"] else 0
                        
                        new_line = pd.DataFrame({"Embedding": [e], "Label": [l]})
                        
                        every_partition_dict[special_name] = pd.concat([every_partition_dict[special_name], new_line], ignore_index=True)
                        counters[special_name] += 1
                        
                        
                        if i % 100 == 0:
                            print(attributed_name, counters[special_name],"/",nr_embeddings_to_fetch)
                        
                        
                        
                        #print("After add:", all_embeddings_labels)

        
        # print("writing pickle")
        # with open("every_walked.pickle", "wb") as file:
        #     pickle.dump(all_embeddings_labels, file)
        # file.close()
    print("Every partition dict", every_partition_dict)
    
    p = os.path.join(save_folder, special_name+".pickle")
    print("Going for save folder", save_folder, special_name)
    with open(p, "wb") as file:
        pickle.dump(every_partition_dict[special_name], file)
    file.close()
    
def fine_tune(model, partitions, tokenizer, device):
    return train_test_val_full_run_nn(model, partitions, tokenizer, device)

def generate_tsne_comparison_images(viab=""):
    
    for file_name in os.listdir(f"./raw_{viab}tsne_results"):
        before = os.path.join(f"./raw_{viab}tsne_results", file_name)
        after = os.path.join(f"./fine_tuned_{viab}tsne_results", file_name)
        arrow = Image.open("arrow.png")
        before_image = Image.open(before)
        after_image = Image.open(after)
        # Get the size of the first image
        width1, height1 = before_image.size

        # Get the size of the second image
        width2, height2 = after_image.size

        result_width = width1 + width2
        result_height = max(height1, height2)

        result_image = Image.new("RGB", (result_width, result_height))

        # Paste the first image onto the new image at (0, 0)
        result_image.paste(before_image, (0, 0))

        # Paste the second image onto the new image at (width1, 0)
        #arrow = arrow.resize(size=(100, 100))
        #result_image.paste(arrow, (0, 0))
        
        result_image.paste(after_image, (width1, 0))
        new_name = file_name.replace("raw_", "")[:-4]
        result_image.save(f"./{viab}tsne_comparison/{new_name}.jpg")
    
                
def generate_comparison_images_with_viability():
    #generate_pca_comparison_images("viability_")
    generate_tsne_comparison_images("viability_")

# Always with viability
def generate_tsne_comparison_images_rf(folder_names=[], result_folder=""):
    
    raw_folder, forest_fit_folder, fine_tuned_folder = folder_names 
    
    for filename in os.listdir(raw_folder):
        raw_image = Image.open(os.path.join(raw_folder, filename))
        forest_image = Image.open(os.path.join(forest_fit_folder, filename))
        fine_tuned_image = Image.open(os.path.join(fine_tuned_folder, filename))
        
        width1, height1 = raw_image.size
        width2, height2 = forest_image.size
        width3, height3 = fine_tuned_image.size

        result_width = width1 + width2 + width3
        result_height = max(height1, height2, height3)

        result_image = Image.new("RGB", (result_width, result_height))

        result_image.paste(raw_image, (0, 0))
        result_image.paste(forest_image, (width1, 0))
        result_image.paste(fine_tuned_image, (width1+width2, 0))
        
        result_image.save(os.path.join(result_folder, filename))

def generate_tsne_comparison_images_nn(folder_names=[], result_folder=""):
    
    raw_folder, fine_tuned_folder = folder_names 
    
    for filename in os.listdir(raw_folder):
        if filename.endswith(".png"):
            try:
                raw_image = Image.open(os.path.join(raw_folder, filename))
                
                fine_tuned_image = Image.open(os.path.join(fine_tuned_folder, filename))
                
                width1, height1 = raw_image.size
                
                width3, height3 = fine_tuned_image.size

                result_width = width1 +  width3
                result_height = max(height1, height3)

                result_image = Image.new("RGB", (result_width, result_height))

                result_image.paste(raw_image, (0, 0))
                
                result_image.paste(fine_tuned_image, (width1, 0))
                
                result_image.save(os.path.join(result_folder, filename))
            except:
                print("Skipped", filename)

def generate_tsne_plots_rf(model, partitions, random_sequences, nr_embeddings_to_generate, embedding_folder_name, result_folder_name, should_do_full_sequence):
    
        # Check which embeddings need to be generated
        partitions_to_generate = []
        for possible, _ in partitions:
            embed_file = os.path.join(embedding_folder_name, possible+".pickle")
            print("Checking"+embed_file)
            #if not os.path.exists(embed_file):
            partitions_to_generate.append((possible, _))
    
        #if len(partitions_to_generate) > 0:
        
        generate_embeddings_pickles_rf(model, partitions_to_generate, nr_embeddings_to_generate, should_do_full_sequence, embedding_folder_name)

        random_embeddings = []
        for s in random_sequences:
            random_embeddings.append(model.generate_embeddings([s]).cpu().detach()[0])
        
        
        for name, _ in partitions:
            embed_file = os.path.join(embedding_folder_name, name+".pickle")
            
            with open(embed_file, "rb") as f:
                data_embeddings = pickle.load(f)
            f.close()
            extra_name = model.status.replace(" ", "_")
            viability_deep_vs_random_tnse(data_embeddings, random_embeddings, f"{extra_name}_{name}.png", result_folder_name)
            print(f"Generated {result_folder_name}/{extra_name}_{name}.png")
            
def generate_tsne_plots_nn(model, partitions, random_sequences, nr_embeddings_to_generate, embedding_folder_name, result_folder_name, should_do_full_sequence, special_name=None):
       
    # Check which embeddings need to be generated
    partitions_to_generate = []
    for possible, _ in partitions:
        embed_file = os.path.join(embedding_folder_name, possible+".pickle")
        print("Checking"+embed_file)
        #if not os.path.exists(embed_file):
        partitions_to_generate.append((possible, _))

    #if len(partitions_to_generate) > 0:
    #print("Generating embeddings for", partitions_to_generate)
    print("Embedding folder name is", embedding_folder_name)
    generate_embeddings_pickles_nn(model, partitions_to_generate, nr_embeddings_to_generate, should_do_full_sequence, embedding_folder_name, special_name=special_name)

    #print("Generating embeddings for random sequences")
    
    random_embeddings = []
    for seq in random_sequences:
        res = model.generate_embeddings([seq]).cpu().detach()
        
        random_embeddings.append(model.generate_embeddings([seq]).cpu().detach()[0])
    
    #print("Generating plots")
    for name, _ in partitions:
        embed_file = os.path.join(embedding_folder_name, special_name+".pickle")
        #print("Generating plot using", embed_file)
        with open(embed_file, "rb") as f:
            data_embeddings = pickle.load(f)
        f.close()
        
        viability_deep_vs_random_tnse(data_embeddings, random_embeddings, f"{special_name}.png", result_folder_name)
        print("sAVED")
        print(name, "done")

def stratified_k_fold_rf_classification_layer_fit(model, partitions, k=10, full_sequences=False):
  all_data = deep_diversification_data_preprocess(partitions, full_sequences)
  results = {}
  i = 0
  
  kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=32)
  X = [x[0] for x in all_data]
  y = [x[1] for x in all_data]
  X = np.array(X)
  y = np.array(y)
  #print([z for z in y])
  for i, (train_index, test_index) in enumerate(kf.split(X, y)):
    print(f"Starting fold {i+1}")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    y_train = [torch.Tensor([0.0,1.0]) if x == 1 else torch.Tensor([1.0,0.0]) for x in y_train]
    y_test = [torch.Tensor([0.0,1.0]) if x == 1 else torch.Tensor([1.0,0.0]) for x in y_test]
    y_train_integer = [torch.argmax(tensor).item() for tensor in y_train]
    y_test_integer = [torch.argmax(tensor).item() for tensor in y_test]

    #print("Starting fit")
    model.fit(X_train, y_train_integer)
    #print("Fit done, evaluating")
    #acc, report = model.evaluate(X_test, y_test_integer)
    #results[i] = (acc, report)
    # print("Saving model")
    # with open(f"rf_classifier_pre_trained_{i}.pickle", "wb") as f:
    #   pickle.dump(model, f)
    # f.close()
    #print(f"Fold {i+1} done")
  #print("Final results:")
  #print(results)
  return model

def stratified_k_fold_rf_fine_tuning(model, partitions, k=10, full_sequences=False):
  optimizer = torch.optim.Adam(model.model.parameters(), lr=0.01)
  all_data = deep_diversification_data_preprocess(partitions, full_sequences)
  kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=32)
  X = [x[0] for x in all_data]
  y = [x[1] for x in all_data]
  X = np.array(X)
  y = np.array(y)
  #print([z for z in y])
  for i, (train_index, test_index) in enumerate(kf.split(X, y)):
    print(f"RF fold {i+1}")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    y_train = [torch.Tensor([0.0,1.0]) if x == 1 else torch.Tensor([1.0,0.0]) for x in y_train]
    y_test = [torch.Tensor([0.0,1.0]) if x == 1 else torch.Tensor([1.0,0.0]) for x in y_test]
    
    model.fine_tuning_train(X_train, y_train, optimizer)
    model.fine_tuning_test(X_test, y_test)
    
  with open("fine_tuned_model.pickle", "wb") as f:
    pickle.dump(model, f)
  f.close()  
  return model

def stratified_k_fold_nn_fine_tuning(model, partitions, k=10, full_sequences=False, loss_plot_save_path=None, stats_save_path=None):
  
  optimizer = torch.optim.Adam(model.model.parameters(), lr=0.00001)
  all_data = deep_diversification_data_preprocess(partitions, full_sequences)
  dataset = SequenceViabilityDataset([x[0] for x in all_data], [x[1] for x in all_data])
  
  total_training_loss = 0
  total_test_loss = 0
  total_val_loss = 0
  k_fold_splits = dataset.get_kfold_splits(k=k)
  stats = []
  
  for fold_nr, (train_indexes, test_indexes) in enumerate(k_fold_splits):
    print("NN", fold_nr+1)
    
    train_data = [dataset[i] for i in train_indexes]
    test_data = [dataset[i] for i in test_indexes]
    #print(train_data[0])
    # Define batch_size
    BATCH_SIZE = 1

    # Get each partition into a DataLoader object (to allow the use of batches)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    train_loss, loss_list = model.train(train_loader, optimizer, criterion)
    
    #print("Training loss", train_loss)
    stats.append(model.test(test_loader, criterion))
    iterations = np.arange(len(loss_list))
    
    plt.plot(loss_list, label='Average Training Loss')
    plt.xlabel('Sequence Nr')
    plt.ylabel('Loss')
    plt.title(f'Training loss evolution: Fold {fold_nr+1}')
    plt.xticks(iterations, [int(x) for x in iterations])
    plt.legend()
    plt.savefig(loss_plot_save_path.replace(".png", "")+f"_{fold_nr+1}.png")
    plt.close()

    with open(stats_save_path, "wb") as f:
      pickle.dump(stats, f)
    f.close()
  return model
    
def plot_loss_train(loss_list, fold_nr, loss_plot_save_path):
    
    iterations = np.arange(len(loss_list))
    plt.plot(loss_list, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    if not fold_nr and fold_nr != 0:
        fold_nr = "final"
        plt.title(f'Training loss evolution: Overview')
    else:
        fold_nr += 1
        plt.title(f'Training loss evolution: Epoch {fold_nr}')
    
    plt.xticks(iterations, [int(x) for x in iterations])
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.set_ylim(0, 1)
    plt.legend()
    plt.savefig(loss_plot_save_path.replace(".png", "")+f"_{fold_nr}.png")
    plt.close()
    
def plot_loss_test(loss_list, fold_nr, loss_plot_save_path):
    iterations = np.arange(len(loss_list))
    plt.plot(loss_list, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    if not fold_nr and fold_nr != 0:
        fold_nr = "final"
        plt.title(f'Test loss evolution: Overview')
    else:
        fold_nr += 1
        plt.title(f'Test loss evolution: Epoch {fold_nr}')
    
    plt.xticks(iterations, [int(x) for x in iterations])
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.set_ylim(0, 1)
    plt.legend()
    plt.savefig(loss_plot_save_path.replace(".png", "")+f"_{fold_nr}.png")
    plt.close()

def plot_loss_val(loss_list, fold_nr, loss_plot_save_path):
    iterations = np.arange(len(loss_list))
    plt.plot(loss_list, label='Validation Loss')
    plt.xlabel('Sequence Nr')
    plt.ylabel('Loss')
    if not fold_nr and fold_nr != 0:
        fold_nr = "final"
        plt.title(f'Validation loss evolution')
    else:
        plt.title = "ERROR"
    
    plt.xticks(iterations, [int(x) for x in iterations])
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.set_ylim(0, 1)
    plt.legend()
    plt.savefig(loss_plot_save_path.replace(".png", "")+f"_{fold_nr}.png")
    plt.close()

from datetime import datetime

def dynamic_lr_stratified_train_test_val_fine_tuning(model, partitions, full_sequences=False, train_loss_plot_save_path=None, test_loss_plot_save_path= None, val_loss_plot_save_path=None, stats_save_path=None, lr_strategy="step", epochs=1, batch_size=1, lr=0.00001, draw_every_epoch=False):   
    print("Starting dynamic")
    optimizer = torch.optim.Adam(model.model.parameters(), lr=lr)
    
    BATCH_SIZE = batch_size
    if partitions:
        all_data = deep_diversification_data_preprocess(partitions, full_sequences)#[300:450]#[3000:3050]
    else:
        #print("Dummy test")
        all_data = []
        for i in range(500):
            all_data.append(("[CLS] "+"A "*735+"[SEP]", 0))
            all_data.append(("[CLS] "+"B "*735+"[SEP]", 1))
         
    dataset = SequenceViabilityDataset([x[0] for x in all_data], [x[1] for x in all_data])
    
    if lr_strategy == "step":
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    elif lr_strategy == "lambda":
        lambda1 = lambda epoch: 0.65 ** epoch
        scheduler = LambdaLR(optimizer, lr_lambda=lambda1)
    
    data_dict = {"valid": []}
    data_dict = dataset.get_stratified_ttv_split()
    train_data = data_dict["train"]
    test_data = data_dict["test"]
    data_dict["valid"] = all_data
    val_data = data_dict["valid"]
    
    averages = []
    test_averages = []
    early_stopping_patience = 3  # Number of epochs to wait for improvement
    early_stopping_counter = 0
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    #print(data_dict)
    print("Start of test:", datetime.now())
    start_val_stats, start_avg_val_loss, _ = model.test(val_loader, criterion)
    print("End of test:", datetime.now())
    start_stats_save_path = stats_save_path.replace("NN", "START_NN")
    with open(start_stats_save_path, "wb") as f:
        pickle.dump(start_val_stats, f)
    f.close()
    for epoch in range(epochs):
        
        print("Full:", full_sequences, datetime.now(), "| Epoch", epoch+1)
        # Get each partition into a DataLoader object (to allow the use of batches)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
        
        avg_train_loss, current_loss_list = model.train(train_loader, optimizer, criterion)
        
        print(train_loss_plot_save_path.split("_train")[0],datetime.now(), epoch+1, "/", epochs," | Average train loss =", avg_train_loss)
        test_stats, avg_test_loss, _ = model.test(test_loader, criterion)
        print(train_loss_plot_save_path.split("_train")[0], datetime.now(), epoch+1, "/", epochs, " | Test stats =", test_stats, "\n\t\t\t| Average test loss =", avg_test_loss)   
        scheduler.step()
        
        if epoch > 0 and avg_test_loss >= np.min(test_averages):
            early_stopping_counter += 1
        else:
            early_stopping_counter = 0
        
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping! No improvement in test loss for", early_stopping_patience, "consecutive epochs.")
            with open("./stops.txt", "a") as f:
                f.write(f"{stats_save_path}: Epoch {epoch}"+"\n")
            f.close()
            break
        
        # Plot epoch loss
        if draw_every_epoch:
            plot_loss_train(current_loss_list, epoch, loss_plot_save_path=train_loss_plot_save_path)
        averages.append(avg_train_loss)
        test_averages.append(avg_test_loss)
        
        # Plot overalls
        plot_loss_train(averages, None, loss_plot_save_path=train_loss_plot_save_path)
        plot_loss_test(test_averages, None, loss_plot_save_path=test_loss_plot_save_path)
        
        
    # new_every = []
    # for l_list in every_loss:
    #     for l in l_list:
    #         new_every.append(l)
            
    # Plot overall losses
    
    
    val_stats, avg_val_loss, every_val_loss = model.test(val_loader, criterion)
    print(datetime.now(), " | Val stats =",val_stats, "\n\t| Total val loss =",sum(every_val_loss))   
    plot_loss_val(every_val_loss, None, loss_plot_save_path=val_loss_plot_save_path)
    
    with open(stats_save_path, "wb") as f:
        pickle.dump(val_stats, f)
    f.close()
    return model

if __name__ == "__main__":
    with open("errors.txt", "w") as f:
        f.close()
    with open("stops.txt", "w") as f:
        f.close()
    nr_random_sequences = 100
    completely_random_sequences = True
    random_sequences_edges, random_sequences_no_edges = get_random_data(nr_random_sequences, completely_random_sequences)
    nr_embeddings_to_generate = 100

    #(False, "fragment", random_sequences_no_edges),
    nr_epochs = 1
    full = ['designed', 'rand', 'single', 'stop', 'cnn_designed_plus_rand_train_walked',
    'lr_rand_doubles_plus_single_walked', 'cnn_rand_doubles_plus_single_seed',
    'cnn_standard_walked', 'rnn_rand_doubles_plus_singles_seed',
    'lr_standard_walked', 'lr_rand_doubles_plus_single_seed',
    'lr_designed_plus_rand_train_seed', 'cnn_rand_doubles_plus_single_walked',
    'previous_chip_nonviable', 'cnn_designed_plus_rand_train_seed',
    'lr_standard_seed', 'rnn_designed_plus_rand_train_walked',
    'lr_designed_plus_rand_train_walked', 'random_doubles', 'rnn_standard_seed',
    'previous_chip_viable', 'rnn_standard_walked',
    'rnn_designed_plus_rand_train_seed', 'cnn_standard_seed',
    'rnn_rand_doubles_plus_singles_walked', 'singles', 'wild_type']
    previous = ["previous_chip_nonviable", "previous_chip_viable"]
    singles = ["single", "singles"]
    rand = ["rand"]
    named = [("Full", full), ("Previous Chip", previous), ("Rand", rand), ("Singles", singles)]
    # counts = {n: {0: 0, 1:0} for n, _ in named}
        
    # for name, parts in named:
    #     d = deep_diversification_data_preprocess(parts, True)
        
    #     for seq, lab in d:
    #         counts[name][lab] += 1
        
    
    # for n in counts:
    #     print(n, "\t", counts[n][0],  "\t",counts[n][1],  "\t",counts[n][0]+counts[n][1], counts[n][0]/(counts[n][0]+counts[n][1]), counts[n][1]/(counts[n][0]+counts[n][1]))
    
    
    designed = ["designed"]
    criterion = nn.CosineSimilarity(dim=1) #nn.BCEWithLogitsLoss()#nn.CosineSimilarity(dim=1)#nn.BCEWithLogitsLoss()
    cases = [(False, "ML fragment LLM", random_sequences_no_edges), (True, "ML complete LLM", random_sequences_edges)]
    previous_chip_names = ["previous_chip_nonviable", "previous_chip_viable"]
    ml_designed_names = [f for f in full if f not in previous_chip_names and f not in ["wild_type", "stop"]]
    partition_cases = [("Non-ML-designed LLM", previous_chip_names), ("ML-designed LLM", ml_designed_names)]#("BATCHED_ICS_previous_chip", previous)]#("BATCHED_ICS_dynamic_full", full)]#("BATCHED_BCE_dynamic_singles", singles),("BATCHED_BCE_previous_chip", previous), ("BATCHED_BCE_dynamic_rand", rand)]#("BATCHED_dynamic_full", full)]#("BATCHED_BCE_dynamic_singles", singles),("BATCHED_DUMMY_TEST", None),("BATCHED_BCE_previous_chip", previous), ("BATCHED_BCE_dynamic_rand", rand)] # 
    models = [("NN", PBLinearClassifier)]
    
    for batch_size in [5,4,3,2]:
        for lr in [0.000001]:
            for should_do_full_sequence, base_case_name, random_sequences in cases:
                for partition_case_name, partitions in partition_cases:
                    for i, (model_type, model) in enumerate(models):
                        try:
                            if not (batch_size == 5 and partition_case_name == "Non-ML-designed LLM") and not (batch_size == 5 and partition_case_name == "ML-designed LLM" and should_do_full_sequence == False):
                                print("DOING", model_type, base_case_name, partition_case_name, batch_size)
                                tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
                                bert_model = BertModel.from_pretrained("Rostlab/prot_bert")

                                bert_model = bert_model.to(device)
                                plot_base_path = f"./{model_type}_plots"
                                if not os.path.exists(plot_base_path):
                                    os.mkdir(plot_base_path)
                                plot_case_path = os.path.join(plot_base_path, f"{base_case_name}")
                                if not os.path.exists(plot_case_path):
                                    os.mkdir(plot_case_path)
                                pickles_base_path = f"./{model_type}_pickles"
                                if not os.path.exists(pickles_base_path):
                                    os.mkdir(pickles_base_path)
                                pickles_case_path = os.path.join(pickles_base_path, f"{base_case_name}")
                                if not os.path.exists(pickles_case_path):
                                    os.mkdir(pickles_case_path)
                                    
                                raw_model = model(bert_model, tokenizer, device)
                            
                                raw_plot_path = os.path.join(plot_case_path, "raw")
                                raw_pickles_path = os.path.join(pickles_case_path, "raw")
                                
                                if not os.path.exists(raw_plot_path):
                                    os.mkdir(raw_plot_path)
                                if not os.path.exists(raw_pickles_path):
                                    os.mkdir(raw_pickles_path)
                                special_name = f"{model_type}_{base_case_name}_{partition_case_name}_BS={str(batch_size)}_LR={str(lr)}"
                                if partitions:
                                    #pass
                                    generate_tsne_plots_nn(raw_model,  [(partition_case_name, partitions)], random_sequences, nr_embeddings_to_generate, raw_pickles_path, raw_plot_path, should_do_full_sequence, special_name=special_name)
                                
                                stats_save_path = f"./stats/{model_type}_{base_case_name}_{partition_case_name}_BS={str(batch_size)}_LR={str(lr)}_tuned_stats.pickle"
                                
                                tuned_plot_path = os.path.join(plot_case_path, "tuned")
                                tuned_pickles_path = os.path.join(pickles_case_path, "tuned")
                                if not os.path.exists(tuned_plot_path):
                                    os.mkdir(tuned_plot_path)
                                if not os.path.exists(tuned_pickles_path):
                                    os.mkdir(tuned_pickles_path)
                                fine_tuned_nn_model = dynamic_lr_stratified_train_test_val_fine_tuning(raw_model, partitions, full_sequences=should_do_full_sequence, train_loss_plot_save_path=tuned_plot_path+f"/{model_type}_{base_case_name}_{partition_case_name}_BS={str(batch_size)}_LR={str(lr)}_train_loss.png",test_loss_plot_save_path=tuned_plot_path+f"/{model_type}_{base_case_name}_{partition_case_name}_BS={str(batch_size)}_LR={str(lr)}_test_loss.png",val_loss_plot_save_path=tuned_plot_path+f"/{model_type}_{base_case_name}_{partition_case_name}_BS={str(batch_size)}_LR={str(lr)}_val_loss.png", stats_save_path=stats_save_path, epochs=nr_epochs, batch_size=batch_size, lr=lr)
                                
                                if partitions:
                                    generate_tsne_plots_nn(fine_tuned_nn_model,  [(partition_case_name, partitions)], random_sequences, nr_embeddings_to_generate, tuned_pickles_path, tuned_plot_path, should_do_full_sequence, special_name=special_name)
                                
                                if not os.path.exists("./comparisons"):
                                    os.mkdir("./comparisons")
                                    
                                comparisons_path = f"./comparisons/{model_type}"
                                if not os.path.exists(comparisons_path):
                                    os.mkdir(comparisons_path)
                                if partitions:
                                    generate_tsne_comparison_images_nn([raw_plot_path, tuned_plot_path], comparisons_path)
                                    print("Compared", f"./model_instances/{partition_case_name+'_BS='+str(batch_size)+'_LR='+str(lr)+'_'+tuned_plot_path.replace('.', '').replace('/', '_')}")
                                    with open("done.txt", "a") as f:
                                        f.write(f"./model_instances/{partition_case_name+'_BS='+str(batch_size)+'_LR='+str(lr)+'_'+tuned_plot_path.replace('.', '').replace('/', '_')}\n")
                                    f.close()
                                    
                                print("Saving model")
                                with open(f"./model_instances/{partition_case_name+'_BS='+str(batch_size)+'_LR='+str(lr)+'_'+tuned_plot_path.replace('.', '').replace('/', '_')}.pickle", "wb") as f:
                                    pickle.dump(fine_tuned_nn_model, f)
                                f.close()
                                print("Model saved")
                            else:
                                print("Skipped!")
                        except Exception as e:
                            import traceback
                            with open("./errors.txt", "a") as f:
                                f.write(f"{stats_save_path}: {e}"+"\n")
                            f.close()
                            print("Killed", stats_save_path, traceback.format_exception(e))
                            
                                
                    
    # print("Done")
    
        
    import numpy



    RESIDUES = tuple('ACDEFGHIKLMNPQRSTVWYX')

  
    # One hot encoding of sequences
    #tf.io.parse_single_example("ACD")
    #ex = {"sequence": "ACD", "is_viable": True}
    
    def ohe_encoding(sequence_list, maior_tamanho):
        
        complete = []
        for sequence in sequence_list:
            encoded = []
            for k in range(maior_tamanho):
                lista_de_zeros = [0 for i in range(len(RESIDUES))]
                if k < len(sequence):
                    aa = sequence[k]
                    if aa in RESIDUES:
                        #print(sequence)
                        #print("Looking for", aa, "in", RESIDUES)
                        pos_to_target = RESIDUES.index(aa)
                        #print("Index is", pos_to_target)
                        #print("Code has length", len(lista_de_zeros))
                        lista_de_zeros[pos_to_target] = 1
                encoded.append(lista_de_zeros)
            complete.append(encoded)
        return complete
    
    def get_ohe_embeddings(partitions, complete, maior_tamanho=None):
        preprocessed = deep_diversification_data_preprocess(partitions, complete)
        #print(preprocessed[0])
        formatted = []
        for x in preprocessed:
            #print("Before:", x[0])
            revert_pb_format = "".join(x[0].split(" "))
            revert_pb_format = revert_pb_format[5:len(revert_pb_format)-5]
            #print("Reverted:", revert_pb_format)
            #print("real size:", len(revert_pb_format))
            #correct_format = {"sequence": revert_pb_format, "is_viable": x[1]}
            formatted.append(revert_pb_format)
        labels = [x[1] for x in preprocessed]
        #print(labels)
        if not maior_tamanho:
            maior_tamanho = max(len(seq) for seq in formatted)
        embeddings = ohe_encoding(formatted, maior_tamanho)
        return embeddings, maior_tamanho, labels
        
# full = ['designed', 'rand', 'single', 'stop', 'cnn_designed_plus_rand_train_walked',
#     'lr_rand_doubles_plus_single_walked', 'cnn_rand_doubles_plus_single_seed',
#     'cnn_standard_walked', 'rnn_rand_doubles_plus_singles_seed',
#     'lr_standard_walked', 'lr_rand_doubles_plus_single_seed',
#     'lr_designed_plus_rand_train_seed', 'cnn_rand_doubles_plus_single_walked',
#     'previous_chip_nonviable', 'cnn_designed_plus_rand_train_seed',
#     'lr_standard_seed', 'rnn_designed_plus_rand_train_walked',
#     'lr_designed_plus_rand_train_walked', 'random_doubles', 'rnn_standard_seed',
#     'previous_chip_viable', 'rnn_standard_walked',
#     'rnn_designed_plus_rand_train_seed', 'cnn_standard_seed',
#     'rnn_rand_doubles_plus_singles_walked', 'singles', 'wild_type']
# previous_chip_names = ["previous_chip_nonviable", "previous_chip_viable"]
# ml_designed_names = [f for f in full if f not in previous_chip_names and f not in ["wild_type", "stop"]]

# import random

# random.seed(42)

# # Define the range and the number of random integers

# n = 500  # Number of random integers

def pure_deep_vs_random_tnse(real_embeddings, real_labels, random_embeddings, extra=""):
    fixed = pd.Series(real_embeddings)
    
    non_random_flag = pd.Series(real_labels)
    
    real_embeddings = pd.DataFrame({"Embedding": fixed, "Random": non_random_flag})
    #print(random_embeddings)
    random_flag = pd.Series(2, index=range(len(random_embeddings)), name="Random")
    random_embeddings =  pd.DataFrame({"Embedding": random_embeddings, "Random": random_flag})

    data_subset = pd.concat([random_embeddings, real_embeddings], ignore_index=True)

    res_arr = []
    res_labels = []
    for i, row in data_subset.iterrows():
        arr = np.array(row["Embedding"])
        
        label = row["Random"]
        
        res_arr.append(arr)
        res_labels.append(label)
    for x in data_subset["Embedding"]:
        print(len(x))
    #print(data_subset)
    res_arr = np.array(res_arr)
    res_arr = np.array([arr.flatten() for arr in res_arr])
    #print(data_subset)
    
    tsne = TSNE(random_state=42)
    embeddings_2d = tsne.fit_transform(res_arr)
    print("End of transform")
    df_tsne = pd.DataFrame(data=embeddings_2d, columns=['Dimension 1', 'Dimension 2'])
    df_tsne['Label'] = res_labels

    
    # Plot the t-SNE plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df_tsne['Dimension 1'], df_tsne['Dimension 2'], c=df_tsne['Label'], cmap='viridis')
    plt.colorbar(scatter, label='Label')
    plt.title(length_name+" "+case_name + ' t-SNE Plot')
    
    plt.savefig(f"./ai4sc-LLM/{file_name}{extra}.png")
    plt.close()
    
    

# Define a function to create directories if they don't exist
def create_directories_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Define a function to save the trained model
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Define a function to save the training loss plot
def save_loss_plot(loss_values, filename):
    plt.plot(loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Plot')
    plt.savefig(filename)
    plt.close()

# Define a function to save evaluation metrics
def save_evaluation_metrics(accuracy, precision, recall, f1, roc_auc, filename):
    with open(filename, 'w') as file:
        file.write(f'Accuracy: {accuracy}\n')
        file.write(f'Precision: {precision}\n')
        file.write(f'Recall: {recall}\n')
        file.write(f'F1 Score: {f1}\n')
        file.write(f'ROC AUC: {roc_auc}\n')

# # Define directory names
# output_directory = 'ai4sc-LLM'
# model_directory = os.path.join(output_directory, 'models')
# plot_directory = os.path.join(output_directory, 'plots')
# metrics_directory = os.path.join(output_directory, 'metrics')

# # Create directories if they don't exist
# create_directories_if_not_exist(output_directory)
# create_directories_if_not_exist(model_directory)
# create_directories_if_not_exist(plot_directory)
# create_directories_if_not_exist(metrics_directory)

# for batch_size in [1]:#2,3,4,5,6][::-1]:
#         for lr in [0.000001]:
#             for should_do_full_sequence, base_case_name, random_sequences in cases:
#                 for partition_case_name, partitions in partition_cases:
#                     for i, (model_type, model) in enumerate(models):
#                         try:
#                             print("Doing", model_type, base_case_name, partition_case_name)
#                             tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
#                             bert_model = BertModel.from_pretrained("Rostlab/prot_bert")

#                             bert_model = bert_model.to(device)
#                             plot_base_path = f"./{model_type}_plots"
#                             if not os.path.exists(plot_base_path):
#                                 os.mkdir(plot_base_path)
#                             plot_case_path = os.path.join(plot_base_path, f"{base_case_name}")
#                             if not os.path.exists(plot_case_path):
#                                 os.mkdir(plot_case_path)
#                             pickles_base_path = f"./{model_type}_pickles"
#                             if not os.path.exists(pickles_base_path):
#                                 os.mkdir(pickles_base_path)
#                             pickles_case_path = os.path.join(pickles_base_path, f"{base_case_name}")
#                             if not os.path.exists(pickles_case_path):
#                                 os.mkdir(pickles_case_path)
                                
#                             raw_model = model(bert_model, tokenizer, device)
                        
#                             raw_plot_path = os.path.join(plot_case_path, "raw")
#                             raw_pickles_path = os.path.join(pickles_case_path, "raw")
                            
#                             if not os.path.exists(raw_plot_path):
#                                 os.mkdir(raw_plot_path)
#                             if not os.path.exists(raw_pickles_path):
#                                 os.mkdir(raw_pickles_path)
#                             special_name = f"{model_type}_{base_case_name}_{partition_case_name}_BS={str(batch_size)}_LR={str(lr)}"
#                             if partitions:
#                                 pass
#                                 #generate_tsne_plots_nn(raw_model,  [(partition_case_name, partitions)], random_sequences, nr_embeddings_to_generate, raw_pickles_path, raw_plot_path, should_do_full_sequence, special_name=special_name)
                            
#                             stats_save_path = f"./stats/{model_type}_{base_case_name}_{partition_case_name}_BS={str(batch_size)}_LR={str(lr)}_tuned_stats.pickle"
                            
#                             tuned_plot_path = os.path.join(plot_case_path, "tuned")
#                             tuned_pickles_path = os.path.join(pickles_case_path, "tuned")
#                             if not os.path.exists(tuned_plot_path):
#                                 os.mkdir(tuned_plot_path)
#                             if not os.path.exists(tuned_pickles_path):
#                                 os.mkdir(tuned_pickles_path)
#                             fine_tuned_nn_model = dynamic_lr_stratified_train_test_val_fine_tuning(raw_model, partitions, full_sequences=should_do_full_sequence, train_loss_plot_save_path=tuned_plot_path+f"/{model_type}_{base_case_name}_{partition_case_name}_BS={str(batch_size)}_LR={str(lr)}_train_loss.png",test_loss_plot_save_path=tuned_plot_path+f"/{model_type}_{base_case_name}_{partition_case_name}_BS={str(batch_size)}_LR={str(lr)}_test_loss.png",val_loss_plot_save_path=tuned_plot_path+f"/{model_type}_{base_case_name}_{partition_case_name}_BS={str(batch_size)}_LR={str(lr)}_val_loss.png", stats_save_path=stats_save_path, epochs=nr_epochs, batch_size=batch_size, lr=lr)
                            
#                             # if partitions:
#                             #     generate_tsne_plots_nn(fine_tuned_nn_model,  [(partition_case_name, partitions)], random_sequences, nr_embeddings_to_generate, tuned_pickles_path, tuned_plot_path, should_do_full_sequence, special_name=special_name)
                            
#                             # if not os.path.exists("./comparisons"):
#                             #     os.mkdir("./comparisons")
                                
#                             # comparisons_path = f"./comparisons/{model_type}"
#                             # if not os.path.exists(comparisons_path):
#                             #     os.mkdir(comparisons_path)
#                             # if partitions:
#                             #     generate_tsne_comparison_images_nn([raw_plot_path, tuned_plot_path], comparisons_path)
                                
                                
#                             # print("Saving model")
#                             # with open(f"./model_instances/{partition_case_name+'_BS='+str(batch_size)+'_LR='+str(lr)+'_'+tuned_plot_path.replace('.', '').replace('/', '_')}.pickle", "wb") as f:
#                             #     pickle.dump(fine_tuned_nn_model, f)
#                             # f.close()
#                             # print("Model saved")
#                         except Exception as e:
#                             import traceback
#                             with open("./errors.txt", "a") as f:
#                                 f.write(f"{stats_save_path}: {e}"+"\n")
#                             f.close()
#                             print("Killed", stats_save_path, traceback.format_exception(e))
                            