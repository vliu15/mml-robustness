import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


mtl_similar_tasks = {}
mtl_dissimilar_tasks = {}

stl_similar_tasks = {
    "Bangs:Wearing_Hat": [(95.93, (95.77,96.08)), (63.16, (50.64,75.68)), (95.55, (95.38,95.71)), (77.36, (66.71,88.01))],
    "Blond_Hair:Wearing_Hat":
        [(95.86, (95.7,96.02)), (35.71, (20.72,50.71)), (94.93, (94.76,95.11)), (48.72, (33.07,64.36))],
    "Big_Nose:Wearing_Lipstick":
        [(84.05, (83.76,84.35)), (32.28, (30.66,33.9)), (78.98, (78.66,79.31)), (47.23, (45.51,48.94))],
    "High_Cheekbones:Smiling": [(87.39, (87.12,87.65)), (38.24, (36.79,39.69)), (84.3, (84.01,84.6)), (50.03, (48.6,51.47))],
    "Big_Lips:Goatee":
        [(70.02, (69.66,70.39)), (11.43, (10.97,11.89)), (70.15, (69.78,70.52)), (39.65, (38.94,40.35))],
    "Wearing_Lipstick:Male":
        [(93.49, (93.3,93.69)), (36.72, (28.78,44.65)), (92.68, (92.47,92.89)), (49.66, (41.63,57.7))],
    "Bags_Under_Eyes:Double_Chin":
        [(85.43, (85.15,85.72)), (53.45, (52.49,54.4)), (82.2, (81.89,82.51)), (60.66, (59.73,61.6))],
    "High_Cheekbones:Rosy_Cheeks": [(87.49, (87.23,87.76)), (71.34, (65.62,77.05)), (85.78, (85.5,86.06)), (69.29, (63.46,75.12))],
    "Brown_Hair:Wearing_Hat":
        [(88.68, (88.42,88.93)), (22.77, (13.49,32.05)), (86.49, (86.21,86.76)), (38.46, (27.66,49.26))]
}

stl_dissimilar_tasks = {}

stl_dissimilar_tasks["Big_Lips:Chubby"] = [
    (69.94, (69.57,70.3)), (11.01, (10.56,11.46)), (68.32, (67.94,68.69)), (41.24, (40.53,41.95))
]

stl_dissimilar_tasks["Bushy_Eyebrows:Blond_Hair"] = [
    (69.94, (69.57,70.3)), (11.01, (10.56,11.46)), (68.32, (67.94,68.69)), (41.24, (40.53,41.95))
]

stl_dissimilar_tasks["Wearing_Lipstick:Male"] = [
    (93.49, (93.3,93.69)), (36.72, (28.78,44.65)), (92.68, (92.47,92.89)), (49.66, (41.63,57.7))
]

stl_dissimilar_tasks["Gray_Hair:Young"] = [
    (98.16, (98.05,98.26)), (25.0, (15.0,35.0)), (97.95, (97.84,98.06)), (15.32, (7.09,23.55))
]

stl_dissimilar_tasks["High_Cheekbones:Smiling"] = [
    (87.39, (87.12,87.65)), (38.24, (36.79,39.69)), (84.3, (84.01,84.6)), (50.03, (48.6,51.47))
]

stl_dissimilar_tasks["Brown_Hair:Wearing_Hat"] = [
   (88.68, (88.42,88.93)), (22.77, (13.49,32.05)), (86.49, (86.21,86.76)), (38.46, (27.66,49.26))
]

stl_dissimilar_tasks["Wearing_Earrings:Male"] = [
    (90.38, (90.14,90.62)), (32.67, (28.74,36.6)), (88.66, (88.41,88.92)), (42.28, (38.14,46.42))
]

stl_dissimilar_tasks["Attractive:Male"] = [
    (82.18, (81.87,82.48)),
    (66.11, (65.21,67.02)),
    (79.76, (79.44,80.08)),
    (65.62, (64.39,66.85)),
]

stl_dissimilar_tasks["No_Beard:Heavy_Makeup"] = [
    (96.05, (95.9,96.21)),
    (28.99, (12.0,45.98)),
    (89.7, (89.45,89.94)),
    (0.0, (0.0,0.0)),
]

stl_dissimilar_tasks["Pointy_Nose:Heavy_Makeup"] = [
    (77.34, (77.01,77.68)),
    (20.21, (19.25,21.18)),
    (72.46, (72.1,72.81)),
    (35.24, (34.09,36.39)),
]

mtl_dissimilar_tasks["Pairing_1"] = {
    "Big_Lips:Chubby": [(70.6, (70.24,70.97)), (15.35, (14.84,15.87)), (66.92, (66.54,67.29)), (59.01, (58.4,59.61))],
    "Bushy_Eyebrows:Blond_Hair":
        [(92.76, (92.55,92.97)), (11.75, (4.0,19.49)), (88.63, (88.37,88.88)), (39.17, (28.13,50.2))]
}

mtl_dissimilar_tasks["Pairing_2"] = {
    "Wearing_Lipstick:Male": [
        (93.64, (93.44,93.83)), (31.19, (23.54,38.83)), (93.6, (93.4,93.79)), (45.95, (37.78,54.13))],
    "Gray_Hair:Young": [(98.17, (98.06,98.27)), (26.3, (16.14,36.46)), (96.93, (96.79,97.07)), (40.88, (29.78,51.99))]
}

mtl_dissimilar_tasks["Pairing_3"] = {
    "High_Cheekbones:Smiling":
        [(87.77, (87.51,88.04)), (38.94, (37.48,40.4)), (85.54, (85.26,85.82)), (51.7, (50.21,53.2))],
    "Brown_Hair:Wearing_Hat": [(88.96, (88.71,89.21)), (18.83, (10.18,27.48)), (82.91, (82.61,83.21)), (51.52, (40.71,62.33))]
}

mtl_dissimilar_tasks["Pairing_4"] = {
    "Wearing_Earrings:Male":
        [(90.63, (90.4,90.87)), (35.98, (31.96,40.0)), (87.95, (87.69,88.21)), (54.76, (50.59,58.94))],
    "Attractive:Male": [(82.55, (82.25,82.85)), (66.38, (65.36,67.41)), (80.1, (79.78,80.42)), (70.21, (69.03,71.39))]
}

mtl_dissimilar_tasks["Pairing_5"] = {
    "No_Beard:Heavy_Makeup":
        [(96.12, (95.97,96.28)), (28.99, (12.0,45.98)), (95.58, (95.42,95.75)), (35.19, (17.65,52.72))],
    "Pointy_Nose:Heavy_Makeup":
        [(77.58, (77.25,77.92)), (21.39, (20.41,22.37)), (71.77, (71.41,72.13)), (43.82, (42.63,45.0))]
}

mtl_similar_tasks["Pairing_1"] = { 
    "Bangs:Wearing_Hat": [(95.99, (95.83,96.14)), (57.89, (45.08,70.71)), (95.22, (95.05,95.39)), (91.51, (90.53,92.48))],
    "Blond_Hair:Wearing_Hat": [(95.88, (95.72,96.04)), (30.77, (16.28,45.25)), (95.01, (94.83,95.18)), (34.45, (19.86,49.04))]
}

mtl_similar_tasks["Pairing_2"] = {
    "Big_Nose:Wearing_Lipstick": [(84.27, (83.98,84.56)), (34.52, (32.88,36.17)), (76.76, (76.42,77.1)), (62.33, (61.3,63.36))],
    "High_Cheekbones:Smiling": [(87.68, (87.42,87.95)), (38.16, (36.71,39.62)), (84.74, (84.45,85.03)), (51.32, (49.94,52.7))]
}

mtl_similar_tasks["Pairing_3"] = {
    "Big_Lips:Goatee":[(70.56, (70.19,70.92)), (12.48, (12.0,12.95)), (68.51, (68.14,68.88)), (49.48, (48.76,50.2))],
    "Wearing_Lipstick:Male": [(93.75, (93.55,93.94)), (38.71, (30.72,46.7)), (93.42, (93.23,93.62)), (49.65, (41.41,57.89))]
}

mtl_similar_tasks["Pairing_4"] = {
    "Bags_Under_Eyes:Double_Chin":[(85.62, (85.34,85.9)), (56.75, (55.81,57.7)), (82.14, (81.84,82.45)), (63.36, (62.26,64.46))],
    "High_Cheekbones:Rosy_Cheeks": [(87.86, (87.6,88.12)), (70.5, (64.74,76.26)), (85.58, (85.3,85.86)), (72.12, (66.47,77.76))]
}

mtl_similar_tasks["Pairing_5"] = {
    "Blond_Hair:Wearing_Hat": [(95.94, (95.78,96.1)), (35.71, (20.72,50.71)), (95.17, (95.0,95.34)), (45.6, (30.49,60.71))],
    "Brown_Hair:Wearing_Hat": [(88.81, (88.55,89.06)), (21.68, (12.54,30.81)), (85.68, (85.4,85.96)), (47.43, (36.36,58.5))]
}

mtl_similar_disjoint = ["Pairing_2", "Pairing_3", "Pairing_4"]
mtl_similar_nondisjoint = ["Pairing_1", "Pairing_5"]

mtl_dissimilar_disjoint = ["Pairing_1", "Pairing_2", "Pairing_3"]
mtl_dissimilar_nondisjoint = ["Pairing_4","Pairing_5" ]


def avg_gain_over_stl(stl_dict, mtl_dict, use_group_val_labels=True, use_pairings = ["Pairing_1", "Pairing_2", "Pairing_3", "Pairing_4", "Pairing_5"]):
    avg_acc_gain = 0
    avg_worst_group_gain = 0

    for pairing_num in use_pairings:
        for task_name in mtl_dict[pairing_num]:

            mtl_results = mtl_dict[pairing_num][task_name]
            stl_results = stl_dict[task_name]

            if use_group_val_labels:
                avg_acc_gain += mtl_results[2][0] - stl_results[2][0]
                avg_worst_group_gain += mtl_results[3][0] - stl_results[3][0]

            else:
                avg_acc_gain += mtl_results[0][0] - stl_results[0][0]
                avg_worst_group_gain += mtl_results[1][0] - stl_results[1][0]

    avg_acc_gain /= len(use_pairings)*2
    avg_worst_group_gain /= len(use_pairings)*2

    return avg_acc_gain, avg_worst_group_gain

def wg_average_in_pairings(stl_dict, mtl_dict, use_group_val_labels=True):
    pairing_to_res = {}

    for pairing_num in mtl_dict.keys():
        avg_mtl_wg = 0
        avg_stl_wg = 0

        combined_mtl_se = 0
        combined_stl_se = 0

        for task_name in mtl_dict[pairing_num]:

            mtl_results = mtl_dict[pairing_num][task_name]
            stl_results = stl_dict[task_name]

            if use_group_val_labels:
                avg_mtl_wg += mtl_results[3][0]
                avg_stl_wg += stl_results[3][0]

                mtl_se = mtl_results[3][1][1] - mtl_results[3][1][0]
                mtl_se = mtl_se/2

                stl_se = stl_results[3][1][1] - stl_results[3][1][0]
                stl_se = stl_se/2

                combined_mtl_se += mtl_se**2
                combined_stl_se += stl_se**2


            else:
                avg_mtl_wg += mtl_results[1][0]
                avg_stl_wg += stl_results[1][0]
                
                mtl_se = mtl_results[1][1][1] - mtl_results[1][1][0]
                mtl_se = mtl_se/2

                stl_se = stl_results[1][1][1] - stl_results[1][1][0]
                stl_se = stl_se/2

                combined_mtl_se += mtl_se**2
                combined_stl_se += stl_se**2

        avg_mtl_wg /= len(mtl_dict[pairing_num])
        avg_stl_wg /= len(mtl_dict[pairing_num])

        combined_mtl_se = np.sqrt(combined_mtl_se)
        combined_stl_se = np.sqrt(combined_stl_se)

        pairing_to_res[pairing_num] = [avg_stl_wg, combined_stl_se, avg_mtl_wg, combined_mtl_se]

    return pairing_to_res

def make_gain_over_stl_plot(use_group_val_labels = True, pairings_to_use = None):

    dissimilar_pairings = ["Pairing_1", "Pairing_2", "Pairing_3", "Pairing_4", "Pairing_5"]
    similar_pairings = ["Pairing_1", "Pairing_2", "Pairing_3", "Pairing_4", "Pairing_5"]

    if pairings_to_use == "Disjoint":
        dissimilar_pairings = mtl_dissimilar_disjoint
        similar_pairings = mtl_similar_disjoint
    elif pairings_to_use == "Nondisjoint":
        dissimilar_pairings = mtl_dissimilar_nondisjoint
        similar_pairings = mtl_similar_nondisjoint

    dissimilar_gains = avg_gain_over_stl(stl_dissimilar_tasks, mtl_dissimilar_tasks, use_group_val_labels, dissimilar_pairings)
    similar_gains= avg_gain_over_stl(stl_similar_tasks, mtl_similar_tasks, use_group_val_labels, similar_pairings)

    gains_data = [[similar_gains[0], 'Average Acc', 'Similar Pairings'], [similar_gains[1], 'Worst Group Acc', 'Similar Pairings'], [dissimilar_gains[0], 'Average Acc', 'Dissimilar Pairings'], [dissimilar_gains[1], 'Worst Group Acc', 'Dissimilar Pairings']]
    gains_df = pd.DataFrame(gains_data, columns=['Average Gain over STL', 'Metric', 'MTL Pairing Type'])
    plt.figure(figsize=(10, 10))
    sns.barplot(x = 'MTL Pairing Type',
            y = 'Average Gain over STL',
            hue = 'Metric',
            data = gains_df)

    ckpt_type = 'WG' if use_group_val_labels else "Avg"

    if pairings_to_use is None:
        plt.title(f"MTL when using Similar vs Dissimilar Pairings [{ckpt_type} Ckpt]")
        plt.savefig(f"./plots/similar_vs_dissimilar_group_val_labels_{str(use_group_val_labels)}.png")
    else:
        plt.title(f"MTL when using Similar vs Dissimilar Pairings for {pairings_to_use} [{ckpt_type} Ckpt]")
        plt.savefig(f"./plots/similar_vs_dissimilar_{pairings_to_use.lower()}_group_val_labels_{str(use_group_val_labels)}.png")
    plt.close()


def make_avg_in_pairing_plot(use_group_val_labels = True):

    dissimilar_gains = wg_average_in_pairings(stl_dissimilar_tasks, mtl_dissimilar_tasks, use_group_val_labels)
    similar_gains = wg_average_in_pairings(stl_similar_tasks, mtl_similar_tasks, use_group_val_labels)

    avg_pairing_data = []
    for pairing_num in dissimilar_gains:
        res = dissimilar_gains[pairing_num]
        avg_pairing_data.append([res[0], res[1], "STL", f"DS {pairing_num}"])
        avg_pairing_data.append([res[2], res[3], "MTL", f"DS {pairing_num}"])
        
    for pairing_num in similar_gains:
        res = similar_gains[pairing_num]
        avg_pairing_data.append([res[0], res[1], "STL", f"S {pairing_num}"])
        avg_pairing_data.append([res[2], res[3], "MTL", f"S {pairing_num}"])

    plt.figure(figsize=(20, 10))
    avg_pairing_df = pd.DataFrame(avg_pairing_data, columns=['Avg Worst Group Accuracy', "Worst Group SE", 'Learning Type', 'Pairing Number'])

    sns.barplot(x = 'Pairing Number',
            y = 'Avg Worst Group Accuracy',
            hue = 'Learning Type',
            data = avg_pairing_df)


    bar_indices = np.array([i for i in range(int(len(avg_pairing_data)/2))])
    bar_indices = np.repeat(bar_indices, 2)

    width = .25
    add = np.array([-1*width, width])
    add = np.tile(add, int(len(avg_pairing_data)/2))
    x = bar_indices + add

    plt.errorbar(x = x, y = avg_pairing_df['Avg Worst Group Accuracy'],
                yerr=avg_pairing_df['Worst Group SE'], fmt='none', c= 'black', capsize = 2)

    ckpt_type = 'WG' if use_group_val_labels else "Avg"
    plt.title(f"Average Worst Group Accuracy Across Pairings [{ckpt_type} Ckpt]")
    plt.savefig(f"./plots/similar_vs_dissimilar_pairings_group_val_labels_{str(use_group_val_labels)}.png")
    plt.close()
  
 
make_gain_over_stl_plot(use_group_val_labels = True, pairings_to_use = "Disjoint")
make_gain_over_stl_plot(use_group_val_labels = False, pairings_to_use = "Disjoint")

make_avg_in_pairing_plot(use_group_val_labels = True)
make_avg_in_pairing_plot(use_group_val_labels = False)