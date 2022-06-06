import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


mtl_strong_spur_corr_tasks = {}
mtl_weak_spur_corr_tasks = {}

stl_strong_spur_corr_tasks = {
    "Wearing_Earrings:Male": [(90.38, (90.14,90.62)), (32.67, (28.74,36.6)), (88.66, (88.41,88.92)), (42.28, (38.14,46.42))],
    "Attractive:Male":
        [(82.18, (81.87,82.48)),
        (66.11, (65.21,67.02)),
        (79.76, (79.44,80.08)),
        (65.62, (64.39,66.85))],
    "Attractive:Gray_Hair":
        [(82.2, (81.9,82.51)),
        (38.69, (25.76,51.63)),
        (80.67, (80.35,80.98)),
        (50.0, (36.72,63.28))],
    "Big_Nose:Gray_Hair": [(84.0, (83.71,84.29)),
        (53.86, (52.77,54.95)),
        (80.52, (80.21,80.84)),
        (56.93, (55.48,58.39))],
    "Wearing_Lipstick:Male":
        [(93.49, (93.3,93.69)), (36.72, (28.78,44.65)), (92.68, (92.47,92.89)), (49.66, (41.63,57.7))],
    "Gray_Hair:Young":
        [(98.16, (98.05,98.26)), (25.0, (15.0,35.0)), (97.95, (97.84,98.06)), (15.32, (7.09,23.55))],
    "High_Cheekbones:Smiling":
        [(87.39, (87.12,87.65)), (38.24, (36.79,39.69)), (84.3, (84.01,84.6)), (50.03, (48.6,51.47))],
    "Heavy_Makeup:Male": [(91.37, (91.15,91.6)), (24.12, (13.81,34.43)), (91.18, (90.95,91.41)), (24.12, (13.81,34.43))],
}

stl_weak_spur_corr_tasks = {}

stl_weak_spur_corr_tasks["Bags_Under_Eyes:Double_Chin"] = [(85.43, (85.15,85.72)), (53.45, (52.49,54.4)), (82.2, (81.89,82.51)), (60.66, (59.73,61.6))]

stl_weak_spur_corr_tasks["High_Cheekbones:Rosy_Cheeks"] = [
   (87.49, (87.23,87.76)), (71.34, (65.62,77.05)), (85.78, (85.5,86.06)), (69.29, (63.46,75.12))
]

stl_weak_spur_corr_tasks["Bangs:Wearing_Hat"] = [
    (95.93, (95.77,96.08)), (63.16, (50.64,75.68)), (95.55, (95.38,95.71)), (77.36, (66.71,88.01))
]

stl_weak_spur_corr_tasks["Blond_Hair:Wearing_Hat"] = [
    (95.86, (95.7,96.02)), (35.71, (20.72,50.71)), (94.93, (94.76,95.11)), (48.72, (33.07,64.36))
]

stl_weak_spur_corr_tasks["No_Beard:Wearing_Lipstick"] = [
    (96.05, (95.9,96.21)),
    (20.0, (0,40.24)),
    (94.65, (94.47,94.83)),
    (0.0, (0.0,0.0)),
]

stl_weak_spur_corr_tasks["Young:Chubby"] = [
    (88.01, (87.75,88.27)), (63.12, (62.26,63.98)), (85.97, (85.69,86.25)), (66.57, (65.56,67.58))
]

stl_weak_spur_corr_tasks["Big_Lips:Chubby"] = [
   (69.94, (69.57,70.3)), (11.01, (10.56,11.46)), (68.32, (67.94,68.69)), (41.24, (40.53,41.95))
]

stl_weak_spur_corr_tasks["High_Cheekbones:Rosy_Cheeks"] = [
     (87.49, (87.23,87.76)), (71.34, (65.62,77.05)), (85.78, (85.5,86.06)), (69.29, (63.46,75.12))
]

stl_weak_spur_corr_tasks["Brown_Hair:Wearing_Hat"] = [
  (88.68, (88.42,88.93)), (22.77, (13.49,32.05)), (86.49, (86.21,86.76)), (38.46, (27.66,49.26))
]


mtl_weak_spur_corr_tasks["Pairing_1"] = {
    "Bags_Under_Eyes:Double_Chin":[(85.62, (85.34,85.9)), (56.75, (55.81,57.7)), (82.14, (81.84,82.45)), (63.36, (62.26,64.46))],
    "High_Cheekbones:Rosy_Cheeks": [(87.86, (87.6,88.12)), (70.5, (64.74,76.26)), (85.58, (85.3,85.86)), (72.12, (66.47,77.76))]
}

mtl_weak_spur_corr_tasks["Pairing_2"] = {
    "Bangs:Wearing_Hat": [(95.99, (95.83,96.14)), (57.89, (45.08,70.71)), (95.22, (95.05,95.39)), (91.51, (90.53,92.48))],
    "Blond_Hair:Wearing_Hat": [(95.88, (95.72,96.04)), (30.77, (16.28,45.25)), (95.01, (94.83,95.18)), (34.45, (19.86,49.04))]
}

mtl_weak_spur_corr_tasks["Pairing_3"] = {
    "No_Beard:Wearing_Lipstick": [(96.18, (96.02,96.33)), (20.0, (-0.24,40.24)), (95.64, (95.48,95.8)), (0.0, (0.0,0.0))],
    "Young:Chubby": [(88.49, (88.23,88.74)), (64.44, (63.58,65.29)), (85.75, (85.47,86.03)), (70.31, (68.99,71.64))]
}

mtl_weak_spur_corr_tasks["Pairing_4"] = {
    "Big_Lips:Chubby": [(71.12, (70.76,71.48)), (17.09, (16.55,17.63)), (66.82, (66.45,67.2)), (56.23, (55.52,56.94))],
    "Young:Chubby": [(88.13, (87.87,88.39)), (61.82, (60.96,62.69)), (85.91, (85.63,86.19)), (69.14, (68.15,70.13))]
}

mtl_weak_spur_corr_tasks["Pairing_5"] = {
    "High_Cheekbones:Rosy_Cheeks": [(87.82, (87.56,88.08)), (71.27, (65.55,76.99)), (86.29, (86.02,86.57)), (77.59, (72.32,82.86))],
    "Brown_Hair:Wearing_Hat": [(88.97, (88.71,89.22)), (18.83, (10.18,27.48)), (84.98, (84.69,85.27)), (54.22, (43.32,65.12))]
}

mtl_strong_spur_corr_tasks["Pairing_1"] = {
     "Wearing_Earrings:Male": [(90.63, (90.4,90.87)), (35.98, (31.96,40.0)), (87.95, (87.69,88.21)), (54.76, (50.59,58.94))],
    "Attractive:Male": [(82.55, (82.25,82.85)), (66.38, (65.36,67.41)), (80.1, (79.78,80.42)), (70.21, (69.03,71.39))]
}

mtl_strong_spur_corr_tasks["Pairing_2"] = {
   "Attractive:Gray_Hair": [(82.72, (82.42,83.03)), (40.42, (27.46,53.38)), (77.9, (77.57,78.23)), (62.8, (62.11,63.49))],
    "Big_Nose:Gray_Hair": [(84.53, (84.24,84.82)), (53.95, (52.86,55.04), (81.47, (81.16,81.78)), (57.33, (53.99,60.66))]
}

mtl_strong_spur_corr_tasks["Pairing_3"] = {
   "Wearing_Lipstick:Male": [(93.64, (93.44,93.83)), (31.19, (23.54,38.83)), (93.6, (93.4,93.79)), (45.95, (37.78,54.13))],
    "Gray_Hair:Young": [(98.17, (98.06,98.27)), (26.3, (16.14,36.46)), (96.93, (96.79,97.07)), (40.88, (29.78,51.99))]
}

mtl_strong_spur_corr_tasks["Pairing_4"] = {
    "Wearing_Lipstick:Male": [(94.03, (93.84,94.22)), (36.72, (28.78,44.65)), (93.52, (93.32,93.72)), (42.38, (34.27,50.49))],
    "High_Cheekbones:Smiling": [(87.92, (87.66,88.18)), (37.33, (35.88,38.77)), (84.96, (84.67,85.24)), (48.03, (46.66,49.41))]
}

mtl_strong_spur_corr_tasks["Pairing_5"] = {
    "Heavy_Makeup:Male": [(91.31, (91.09,91.54)), (14.75, (6.31,23.19)), (90.19, (89.95,90.43)), (21.04, (11.22,30.86))],
    "Wearing_Earrings:Male":
        [(90.7, (90.47,90.93)), (34.01, (30.06,37.96)), (88.59, (88.33,88.84)), (51.61, (47.46,55.75))]
}

mtl_strong_spur_corr_disjoint = ["Pairing_3", "Pairing_4"]
mtl_strong_spur_corr_nondisjoint = ["Pairing_1", "Pairing_2", "Pairing_5"]

mtl_weak_spur_corr_disjoint = ["Pairing_1", "Pairing_3", "Pairing_5"]
mtl_weak_spur_corr_nondisjoint = ["Pairing_2", "Pairing_4"]

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

    weak_pairings = ["Pairing_1", "Pairing_2", "Pairing_3", "Pairing_4", "Pairing_5"]
    strong_pairings = ["Pairing_1", "Pairing_2", "Pairing_3", "Pairing_4", "Pairing_5"]

    if pairings_to_use == "Disjoint":
        weak_pairings = mtl_weak_spur_corr_disjoint
        strong_pairings = mtl_strong_spur_corr_disjoint
    elif pairings_to_use == "Nondisjoint":
        weak_pairings = mtl_weak_spur_corr_nondisjoint
        strong_pairings = mtl_strong_spur_corr_nondisjoint


    weak_spur_corr_gains = avg_gain_over_stl(stl_weak_spur_corr_tasks, mtl_weak_spur_corr_tasks, use_group_val_labels, weak_pairings)
    strong_spur_corr_gains= avg_gain_over_stl(stl_strong_spur_corr_tasks, mtl_strong_spur_corr_tasks, use_group_val_labels, strong_pairings)

    gains_data = [[strong_spur_corr_gains[0], 'Average Acc', 'Strong Pairings'], [strong_spur_corr_gains[1], 'Worst Group Acc', 'Strong Pairings'], [weak_spur_corr_gains[0], 'Average Acc', 'Weak Pairings'], [weak_spur_corr_gains[1], 'Worst Group Acc', 'Weak Pairings']]
    gains_df = pd.DataFrame(gains_data, columns=['Average Gain over STL', 'Metric', 'MTL Pairing Type'])
    plt.figure(figsize=(10, 10))
    sns.barplot(x = 'MTL Pairing Type',
            y = 'Average Gain over STL',
            hue = 'Metric',
            data = gains_df)

    ckpt_type = 'WG' if use_group_val_labels else "Avg"

    if pairings_to_use is None:
        plt.title(f"MTL when using Strong vs Weak Pairings [{ckpt_type} Ckpt]")
        plt.savefig(f"./plots/strong_vs_weak_group_val_labels_{str(use_group_val_labels)}.png")
    else:
        plt.title(f"MTL when using Strong vs Weak Pairings for {pairings_to_use} [{ckpt_type} Ckpt]")
        plt.savefig(f"./plots/strong_vs_weak_{pairings_to_use.lower()}_group_val_labels_{str(use_group_val_labels)}.png")

    plt.close()


def make_avg_in_pairing_plot(use_group_val_labels = True):

    weak_spur_corr_gains = wg_average_in_pairings(stl_weak_spur_corr_tasks, mtl_weak_spur_corr_tasks, use_group_val_labels)
    strong_spur_corr_gains = wg_average_in_pairings(stl_strong_spur_corr_tasks, mtl_strong_spur_corr_tasks, use_group_val_labels)

    avg_pairing_data = []
    for pairing_num in weak_spur_corr_gains:
        res = weak_spur_corr_gains[pairing_num]
        avg_pairing_data.append([res[0], res[1], "STL", f"W {pairing_num}"])
        avg_pairing_data.append([res[2], res[3], "MTL", f"W {pairing_num}"])
        
    for pairing_num in strong_spur_corr_gains:
        res = strong_spur_corr_gains[pairing_num]
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
    plt.savefig(f"./plots/strong_vs_weak_pairings_group_val_labels_{str(use_group_val_labels)}.png")
    plt.close()
  
 
make_gain_over_stl_plot(use_group_val_labels = True, pairings_to_use = "Nondisjoint")
make_gain_over_stl_plot(use_group_val_labels = False, pairings_to_use = "Nondisjoint")

make_avg_in_pairing_plot(use_group_val_labels = True)
make_avg_in_pairing_plot(use_group_val_labels = False)