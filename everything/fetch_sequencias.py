import pandas as pd
pre_sequence = "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMIT"
post_sequence = "QAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL"
def build_and_save_table(data_list):
    # Create an empty DataFrame
    df = pd.DataFrame(columns=['Case', 'Number of Sequences', 'Viability Percentage'])

    # Iterate over each case and its corresponding sequences and labels
    for case_name, sequences_and_labels in data_list:
        # Calculate the number of sequences
        num_sequences = len(sequences_and_labels)
        # Calculate the percentage of viability
        viability_percentage = sum(label for _, label in sequences_and_labels) / num_sequences * 100

        # Append the data to the DataFrame
        df = df.append({'Case': case_name,
                        'Number of Sequences': num_sequences,
                        'Viability Percentage': viability_percentage},
                        ignore_index=True)

    # Save the DataFrame as a CSV file
    df.to_csv('sequences_summary.csv', index=False)
def get_partition_data(file_name, partitions_to_get, get_full_sequences=False):
  # Read the csv containing the data
  df = pd.read_csv(file_name, delimiter=";")
  
  print("Should do full:", get_full_sequences)
  
  all_data = []
 
  # For row in the dataset
  for index, row in df.iterrows():
    
    p = row["partition"]
    
    # If the partition is one of the ones we want
    if p in partitions_to_get:
      s = row["sequence"]
      l = row["is_viable"]
      
      
      # Turn TRUE and FALSE into 1s and 0s
      try:
        l = int(l)
      except:
        print("ERROR IN COLUMN")

      if get_full_sequences:
        s = pre_sequence + s + post_sequence

      # Add the sequence and binary label to list of sequences to return
      all_data.append((s, l))
  
  return all_data
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
previous_chip_names = ["previous_chip_nonviable", "previous_chip_viable"]
ml_designed_names = [f for f in full if f not in previous_chip_names and f not in ["wild_type", "stop"]]
cases = [("Non-ML-designed", previous_chip_names), ("ML-designed", ml_designed_names)]
folder = "./datasets/protein_mutation_viability.csv"
all_sequences_and_labels = []
# Deverá devolver todos os pares (seq, label) presentes nas partições designed, rand e single, com as sequências completas
for (case_name, partitions) in cases:
  row_name = case_name
  sequences_and_labels = get_partition_data(folder, partitions, True)
  all_sequences_and_labels.append((case_name, sequences_and_labels))
  
build_and_save_table(all_sequences_and_labels)  
