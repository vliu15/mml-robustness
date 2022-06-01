import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


mtl_similar_tasks = {}
mtl_dissimilar_tasks = {}

stl_similar_tasks = {
    "Bangs:Wearing_Hat": [(95.92, (95.76, 96.08)), (62.33, (50.15, 74.5)), (95.53, (95.36, 95.69)), (73.83, (62.79, 84.88))],
    "Blond_Hair:Wearing_Hat":
        [(95.86, (95.7, 96.02)), (37.16, (22.69, 51.63)), (94.93, (94.75, 95.1)), (48.83, (33.86, 63.8))],
    "Big_Nose:Wearing_Lipstick":
        [(84.05, (83.76,84.34)), (32.45, (30.83,34.07)), (78.93, (78.6,79.25)), (47.28, (45.55,49.01))],
    "High_Cheekbones:Smiling": [(87.39, (87.12, 87.65)), (38.26, (36.81, 39.72)), (84.29, (84.00, 84.58)), (50.03, (48.6, 51.47))],
    "Big_Lips:Goatee":
        [(70.02, (69.65,70.39)), (11.46, (11.0,11.92)), (70.12, (69.75,70.48)), (39.67, (38.96,40.37))],
    "Wearing_Lipstick:Male":
        [(93.49, (93.29, 93.69)), (37.23, (29.35, 45.10)), (92.67, (92.46, 92.88)), (49.65, (41.51, 57.80))],
    "Bags_Under_Eyes:Double_Chin":
        [(85.43, (85.15,85.71)), (53.44, (52.49,54.4)), (82.2, (81.89,82.5)), (60.65, (59.72,61.58))],
    "High_Cheekbones:Rosy_Cheeks": [(87.49, (87.23,87.76)), (70.92, (65.21,76.62)), (60.65, (59.72,61.58)), (68.86, (63.05,74.68))],
    "Brown_Hair:Wearing_Hat":
        [(88.67, (88.42, 88.93)), (24.34, (15.04, 33.64)), (86.48, (86.21, 86.75)), (39.0, (28.44, 49.57))]
}

stl_dissimilar_tasks = {}

stl_dissimilar_tasks["Big_Lips:Chubby"] = [
    (69.93, (69.57, 70.30)), (11.27, (10.81, 11.72)), (68.32, (67.94, 68.69)), (41.26, (40.55, 41.97))
]

stl_dissimilar_tasks["Bushy_Eyebrows:Blond_Hair"] = [
    (92.60, (92.39, 92.81)), (15.64, (7.12, 24.15)), (90.62, (90.39, 90.85)), (32.82, (21.81, 43.83))
]

stl_dissimilar_tasks["Wearing_Lipstick:Male"] = [
    (93.49, (93.29, 93.69)), (37.23, (29.35, 45.10)), (92.67, (92.46, 92.88)), (49.65, (41.51, 57.80))
]

stl_dissimilar_tasks["Gray_Hair:Young"] = [
    (98.15, (98.05, 98.26)), (26.27, (16.36, 36.17)), (97.94, (97.83, 98.06)), (18.36, (9.64, 27.07))
]

stl_dissimilar_tasks["High_Cheekbones:Smiling"] = [
    (87.39, (87.12, 87.65)), (38.26, (36.81, 39.72)), (84.29, (84.00, 84.58)), (50.03, (48.6, 51.47))
]

stl_dissimilar_tasks["Brown_Hair:Wearing_Hat"] = [
   (88.67, (88.42, 88.93)), (24.34, (15.04, 33.64)), (86.48, (86.21, 86.75)), (39.0, (28.44, 49.57))
]

stl_dissimilar_tasks["Wearing_Earrings:Male"] = [
    (90.38, (90.14, 90.61)), (32.9, (28.98, 36.83)), (88.6, (88.35, 88.86)), (42.36, (38.23, 46.49))
]

stl_dissimilar_tasks["Attractive:Male"] = [
    (82.17, (81.87, 82.48)), (66.11, (65.2, 67.01)), (79.74, (79.42, 80.06)), (65.61, (64.38, 66.84))
]

stl_dissimilar_tasks["No_Beard:Heavy_Makeup"] = [
    (96.05, (95.89, 96.2)), (26.12, (6.28, 45.95)), (89.68, (89.44, 89.93)), (6.23, (-2.3, 14.76))
]

stl_dissimilar_tasks["Pointy_Nose:Heavy_Makeup"] = [
    (77.34, (77.0, 77.68)), (20.26, (19.3, 21.23)), (72.44, (72.08, 72.8)), (35.3, (34.15, 36.44))
]

mtl_dissimilar_tasks["Pairing_1"] = {
    "Big_Lips:Chubby": [(70.6, (70.24, 70.97)), (15.49, (14.97, 16.02)), (66.9, (66.53, 67.28)), (58.84, (58.23, 59.45))],
    "Bushy_Eyebrows:Blond_Hair":
        [(92.76, (92.55, 92.96)), (14.2, (6.02, 22.39)), (88.52, (88.27, 88.78)), (41.41, (29.86, 52.96))]
}

mtl_dissimilar_tasks["Pairing_2"] = {
    "Wearing_Lipstick:Male": [
        (93.64, (93.44, 93.83)), (31.7, (24.13, 39.28)), (93.59, (93.39, 93.78)), (46.2, (38.08, 54.32))
    ],
    "Gray_Hair:Young": [(98.16, (98.05, 98.27)), (27.58, (17.53, 37.64)), (96.79, (96.65, 96.94)), (42.09, (30.98, 53.2))]
}

mtl_dissimilar_tasks["Pairing_3"] = {
    "High_Cheekbones:Smiling":
        [(87.77, (87.51, 88.03)), (38.96, (37.5, 40.42)), (85.53, (85.25, 85.81)), (51.7, (50.21, 53.19))],
    "Brown_Hair:Wearing_Hat": [
        (88.95, (88.7, 89.2)), (20.68, (11.9, 29.45)), (82.79, (82.48, 83.09)), (51.22, (40.39, 62.05))
    ]
}

mtl_dissimilar_tasks["Pairing_4"] = {
    "Wearing_Earrings:Male":
        [(90.63, (90.4, 90.86)), (36.18, (32.16, 40.19)), (87.91, (87.65, 88.17)), (54.73, (50.57, 58.89))],
    "Attractive:Male": [(82.55, (82.24, 82.85)), (66.37, (65.35, 67.4)), (80.07, (79.75, 80.39)), (70.1, (68.92, 71.29))]
}

mtl_dissimilar_tasks["Pairing_5"] = {
    "No_Beard:Heavy_Makeup":
        [(96.12, (95.96, 96.27)), (32.17, (15.68, 48.65)), (95.58, (95.41, 95.74)), (38.65, (21.47, 55.84))],
    "Pointy_Nose:Heavy_Makeup":
        [(77.58, (77.25, 77.92)), (21.75, (20.76, 22.74)), (71.76, (71.4, 72.12)), (43.92, (42.72, 45.11))]
}

mtl_similar_tasks["Pairing_1"] = {
    "Bangs:Wearing_Hat": [(95.98, (95.83, 96.14)), (57.4, (44.97, 69.82)), (95.22, (95.05, 95.39)), (91.41, (90.43, 92.39))],
    "Blond_Hair:Wearing_Hat":
        [(95.87, (95.71, 96.03)), (32.49, (18.47, 46.52)), (94.99, (94.82, 95.17)), (37.16, (22.69, 51.63))]
}

mtl_similar_tasks["Pairing_2"] = {
    "Big_Nose:Wearing_Lipstick": [(84.35, (84.06,84.64)), (31.98, (30.36,33.59)), (78.65, (78.33,78.98)), (55.59, (53.87,57.31))],
    "High_Cheekbones:Smiling": [(87.77, (87.51,88.04)), (36.12, (34.68,37.56)), (83.63, (83.33,83.92)), (48.55, (47.21,49.88))]
}

mtl_similar_tasks["Pairing_3"] = {
    "Big_Lips:Goatee":[(70.79, (70.43,71.16)), (14.27, (13.77,14.78)), (67.51, (67.13,67.88)), (54.16, (53.44,54.87))],
    "Wearing_Lipstick:Male": [(93.9, (93.7,94.09)), (39.99, (32.01,47.97)), (93.24, (93.04,93.44)), (45.51, (37.4,53.62))]
}

mtl_similar_tasks["Pairing_4"] = {
    "Bags_Under_Eyes:Double_Chin":[(85.67, (85.39,85.95)), (56.86, (55.91,57.8)), (83.68, (83.38,83.98)), (61.15, (60.03,62.26))],
    "High_Cheekbones:Rosy_Cheeks": [(87.84, (87.58,88.1)), (68.45, (62.62,74.29)), (86.19, (85.91,86.47)), (77.07, (71.79,82.34))]
}

mtl_similar_tasks["Pairing_5"] = {
    "Blond_Hair:Wearing_Hat": [(95.91, (95.75,96.07)), (37.16, (22.69,51.63)), (95.24, (95.07,95.41)), (51.17, (36.2,66.14))],
    "Brown_Hair:Wearing_Hat": [(88.74, (88.49,89.0)), (23.12, (13.98,32.25)), (84.11, (83.82,84.4)), (47.56, (36.74,58.38))]
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