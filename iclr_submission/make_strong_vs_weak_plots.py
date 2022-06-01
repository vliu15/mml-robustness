import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


mtl_strong_spur_corr_tasks = {}
mtl_weak_spur_corr_tasks = {}

stl_strong_spur_corr_tasks = {
    "Wearing_Earrings:Male": [(90.38, (90.14, 90.61)), (32.9, (28.98, 36.83)), (88.6, (88.35, 88.86)), (42.36, (38.23, 46.49))],
    "Attractive:Male":
        [(82.17, (81.87, 82.48)), (66.11, (65.2, 67.01)), (79.74, (79.42, 80.06)), (65.61, (64.38, 66.84))],
    "Attractive:Gray_Hair":
        [(82.2, (81.89, 82.51)), (39.63, (27.02, 52.23)), (80.6, (80.29, 80.92)), (50.0, (37.11, 62.89))],
    "Big_Nose:Gray_Hair": [(84.0, (83.7, 84.29)), (53.86, (52.77, 54.95)), (80.52, (80.2, 80.83)), (56.91, (55.46, 58.37))],
    "Wearing_Lipstick:Male":
        [(93.49, (93.29, 93.69)), (37.23, (29.35, 45.10)), (92.67, (92.46, 92.88)), (49.65, (41.51, 57.80))],
    "Gray_Hair:Young":
        [(98.15, (98.05, 98.26)), (26.27, (16.36, 36.17)), (97.94, (97.83, 98.06)), (18.36, (9.64, 27.07))],
    "High_Cheekbones:Smiling":
        [(87.39, (87.12, 87.65)), (38.26, (36.81, 39.72)), (84.29, (84.00, 84.58)), (50.03, (48.6, 51.47))],
    "Heavy_Makeup:Male": [(91.35, (91.12,91.57)), (24.23, (14.18,34.28)), (89.01, (88.76,89.26)), (34.25, (23.12,45.38))],
}

stl_weak_spur_corr_tasks = {}

stl_weak_spur_corr_tasks["Bags_Under_Eyes:Double_Chin"] = [(85.43, (85.15,85.71)), (53.44, (52.49,54.4)), (82.2, (81.89,82.5)), (60.65, (59.72,61.58))]

stl_weak_spur_corr_tasks["High_Cheekbones:Rosy_Cheeks"] = [
    (87.49, (87.23,87.76)), (70.92, (65.21,76.62)), (60.65, (59.72,61.58)), (68.86, (63.05,74.68))
]

stl_weak_spur_corr_tasks["Bangs:Wearing_Hat"] = [
    (95.92, (95.76, 96.08)), (62.33, (50.15, 74.5)), (95.53, (95.36, 95.69)), (73.83, (62.79, 84.88))
]

stl_weak_spur_corr_tasks["Blond_Hair:Wearing_Hat"] = [
    (95.86, (95.7, 96.02)), (37.16, (22.69, 51.63)), (94.93, (94.75, 95.1)), (48.83, (33.86, 63.8))
]

stl_weak_spur_corr_tasks["No_Beard:Wearing_Lipstick"] = [
(96.06, (95.91, 96.22)), (26.12, (6.28, 45.95)), (94.68, (94.5, 94.86)), (10.19, (-3.47, 23.86))
]

stl_weak_spur_corr_tasks["Young:Chubby"] = [
    (88.01, (87.75, 88.27)), (63.12, (62.26, 63.97)), (85.94, (85.66, 86.21)), (66.56, (65.55, 67.57))
]

stl_weak_spur_corr_tasks["Big_Lips:Chubby"] = [
    (69.93, (69.57, 70.30)), (11.27, (10.81, 11.72)), (68.32, (67.94, 68.69)), (41.26, (40.55, 41.97))
]

stl_weak_spur_corr_tasks["High_Cheekbones:Rosy_Cheeks"] = [
     (87.49, (87.23,87.76)), (70.92, (65.21,76.62)), (60.65, (59.72,61.58)), (68.86, (63.05,74.68))
]

stl_weak_spur_corr_tasks["Brown_Hair:Wearing_Hat"] = [
   (88.67, (88.42, 88.93)), (24.34, (15.04, 33.64)), (86.48, (86.21, 86.75)), (39.0, (28.44, 49.57))
]


mtl_weak_spur_corr_tasks["Pairing_1"] = {
    "Bags_Under_Eyes:Double_Chin":[(85.67, (85.39,85.95)), (56.86, (55.91,57.8)), (83.68, (83.38,83.98)), (61.15, (60.03,62.26))],
    "High_Cheekbones:Rosy_Cheeks": [(87.84, (87.58,88.1)), (68.45, (62.62,74.29)), (86.19, (85.91,86.47)), (77.07, (71.79,82.34))]
}

mtl_weak_spur_corr_tasks["Pairing_2"] = {
    "Bangs:Wearing_Hat": [(95.98, (95.83, 96.14)), (57.4, (44.97, 69.82)), (95.22, (95.05, 95.39)), (91.41, (90.43, 92.39))],
    "Blond_Hair:Wearing_Hat":
        [(95.87, (95.71, 96.03)), (32.49, (18.47, 46.52)), (94.99, (94.82, 95.17)), (37.16, (22.69, 51.63))]
}

mtl_weak_spur_corr_tasks["Pairing_3"] = {
    "No_Beard:Wearing_Lipstick": [(96.17, (96.02,96.33)), (26.12, (6.28,45.95)), (95.63, (95.47,95.79)), (26.12, (6.28,45.95))],
    "Young:Chubby": [(88.48, (88.23,88.74)), (64.36, (63.51,65.21)), (85.73, (85.45,86.01)), (70.27, (68.94,71.6))]
}

mtl_weak_spur_corr_tasks["Pairing_4"] = {
    "Big_Lips:Chubby": [(71.23, (70.87,71.59)), (17.9, (17.34,18.45)), (66.09, (65.71,66.47)), (54.09, (53.47,54.71))],
    "Young:Chubby": [(88.37, (88.11,88.63)), (63.42, (62.56,64.28)), (85.44, (85.16,85.72)), (70.77, (69.45,72.09))]
}

mtl_weak_spur_corr_tasks["Pairing_5"] = {
    "High_Cheekbones:Rosy_Cheeks": [(87.68, (87.42,87.94)), (70.51, (64.78,76.23)), (86.04, (85.77,86.32)), (80.26, (79.41,81.11))],
    "Brown_Hair:Wearing_Hat": [(88.77, (88.52,89.02)), (20.68, (11.9,29.45)), (83.02, (82.72,83.32)), (74.95, (74.27,75.63))]
}

mtl_strong_spur_corr_tasks["Pairing_1"] = {
    "Wearing_Earrings:Male":
        [(90.63, (90.4, 90.86)), (36.18, (32.16, 40.19)), (87.91, (87.65, 88.17)), (54.73, (50.57, 58.89))],
    "Attractive:Male": [(82.55, (82.24, 82.85)), (66.37, (65.35, 67.4)), (80.07, (79.75, 80.39)), (70.1, (68.92, 71.29))]
}

mtl_strong_spur_corr_tasks["Pairing_2"] = {
    "Attractive:Gray_Hair": [
        (82.72, (82.42, 83.02)), (41.36, (28.66, 54.05)), (77.9, (77.57, 78.23)), (62.79, (62.11, 63.48))
    ],
    "Big_Nose:Gray_Hair": [(84.53, (84.24, 84.81)), (53.94, (52.86, 55.03)), (81.43, (81.12, 81.74)), (57.23, (53.89, 60.57))]
}

mtl_strong_spur_corr_tasks["Pairing_3"] = {
    "Wearing_Lipstick:Male": [
        (93.64, (93.44, 93.83)), (31.7, (24.13, 39.28)), (93.59, (93.39, 93.78)), (46.2, (38.08, 54.32))
    ],
    "Gray_Hair:Young": [(98.16, (98.05, 98.27)), (27.58, (17.53, 37.64)), (96.79, (96.65, 96.94)), (42.09, (30.98, 53.2))]
}

mtl_strong_spur_corr_tasks["Pairing_4"] = {
    "Wearing_Lipstick:Male": [(94.05, (93.86,94.24)), (36.54, (28.69,44.38)), (93.39, (93.2,93.59)), (44.82, (36.72,52.92))],
    "High_Cheekbones:Smiling": [(87.89, (87.63,88.15)), (37.42, (35.98,38.87)), (85.39, (85.11,85.67)), (48.79, (47.35,50.22))]
}

mtl_strong_spur_corr_tasks["Pairing_5"] = {
    "Heavy_Makeup:Male": [(91.42, (91.19,91.64)), (18.5, (9.39,27.61)), (90.02, (89.78,90.26)), (22.8, (12.96,32.63))],
    "Wearing_Earrings:Male":
        [(90.79, (90.55,91.02)), (33.81, (29.86,37.77)), (88.39, (88.13,88.64)), (51.45, (47.28,55.63))]
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