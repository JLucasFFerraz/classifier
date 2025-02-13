import pandas as pd 

print("1st order fitness landascape")
print("Notebook Filipa")
reconstructed_csv = pd.read_csv("./outputs/reconstructed_sequences_viability_f.csv")

nr_viable_CMV_recons = reconstructed_csv["viability_CMV"].value_counts()[1]
nr_viable_Rep_recons = reconstructed_csv["viability_Rep"].value_counts()[1]
total_recons = len(reconstructed_csv)

print("CMV:", nr_viable_CMV_recons, "/", total_recons, "=", str(round(nr_viable_CMV_recons/total_recons*100, 1))+"%")
print("Rep:", nr_viable_Rep_recons, "/", total_recons, "=", str(round(nr_viable_Rep_recons/total_recons*100, 1))+"%")
print()
print("Notebook Lucas")
reconstructed_csv = pd.read_csv("./outputs/reconstructed_sequences_viability_l.csv")

nr_viable_CMV_recons = reconstructed_csv["viability_CMV"].value_counts()[1]
nr_viable_Rep_recons = reconstructed_csv["viability_Rep"].value_counts()[1]
total_recons = len(reconstructed_csv)

print("CMV:", nr_viable_CMV_recons, "/", total_recons, "=", str(round(nr_viable_CMV_recons/total_recons*100, 1))+"%")
print("Rep:", nr_viable_Rep_recons, "/", total_recons, "=", str(round(nr_viable_Rep_recons/total_recons*100, 1))+"%")
print()
print("Deep diversification")
csv = pd.read_csv("./datasets/protein_mutation_viability.csv", delimiter=";")

nr_viable = csv["is_viable"].value_counts()[True]
total = len(csv)
print("Viable:", nr_viable, "/", total, "=", str(round(nr_viable/total*100, 1))+"%")