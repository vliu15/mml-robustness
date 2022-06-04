import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np 

mtl_disjoint_tasks = {}
mtl_nondisjoint_tasks = {}

stl_disjoint_tasks = {
    "Big_Lips:Chubby": [(69.94, (69.57,70.3)), (11.01, (10.56,11.46)), (68.32, (67.94,68.69)), (41.24, (40.53,41.95))],
    "Bushy_Eyebrows:Blond_Hair":
        [(92.6, (92.39,92.81)), (13.64, (5.36,21.92)), (90.8, (90.57,91.03)), (27.04, (16.94,37.15))],
    "Wearing_Lipstick:Male":
        [(93.49, (93.3,93.69)), (36.72, (28.78,44.65)), (92.68, (92.47,92.89)), (49.66, (41.63,57.7))],
    "Gray_Hair:Young": [(98.16, (98.05,98.26)), (25.0, (15.0,35.0)), (97.95, (97.84,98.06)), (15.32, (7.09,23.55))],
    "High_Cheekbones:Smiling":
        [(87.39, (87.12,87.65)), (38.24, (36.79,39.69)), (84.3, (84.01,84.6)), (50.03, (48.6,51.47))],
    "Brown_Hair:Wearing_Hat":
        [(88.68, (88.42,88.93)), (22.77, (13.49,32.05)), (86.49, (86.21,86.76)), (38.46, (27.66,49.26))],
    "No_Beard:Wearing_Lipstick":
        [(96.05, (95.9,96.21), (20.0, (0.0,40.24)), (94.65, (94.47,94.83)), (0.0, (0.0,0.0))],
    "Young:Chubby": [(88.01, (87.75,88.27)), (63.12, (62.26,63.98)), (85.97, (85.69,86.25)), (66.57, (65.56,67.58))],
    "Bangs:Wearing_Hat": [(95.93, (95.77,96.08)), (63.16, (50.64,75.68)), (95.55, (95.38,95.71)), (77.36, (66.71,88.01))],
    "Pointy_Nose:Heavy_Makeup":
        [(77.34, (77.01,77.68)), (20.21, (19.25,21.18)), (72.46, (72.1,72.81)), (35.24, (34.09,36.39))]
}

stl_nondisjoint_tasks = {
    "Wearing_Earrings:Male": [
        (90.38, (90.14,90.62)), (32.67, (28.74,36.6)), (88.66, (88.41,88.92)), (42.28, (38.14,46.42)) 
    ],
    "Attractive:Male": [
        (82.18, (81.87,82.48)),
        (66.11, (65.21,67.02)),
        (79.76, (79.44,80.08)),
        (65.62, (64.39,66.85)),
    ],
    "No_Beard:Heavy_Makeup": [
        (96.05, (95.9,96.21)),
        (28.99, (12.0,45.98)),
        (89.7, (89.45,89.94)),
        (0.0, (0.0,0.0)),
    ],
    "Pointy_Nose:Heavy_Makeup":
        [
            (77.34, (77.01,77.68)),
            (20.21, (19.25,21.18)),
            (72.46, (72.1,72.81)),
            (35.24, (34.09,36.39)),
        ],
    "Attractive:Gray_Hair": [
        (82.2, (81.9,82.51)),
        (38.69, (25.76,51.63)),
        (80.67, (80.35,80.98)),
        (50.0, (36.72,63.28)),
    ],
    "Big_Nose:Gray_Hair": [
        (84.0, (83.71,84.29)),
        (53.86, (52.77,54.95)),
        (80.52, (80.21,80.84)),
        (56.93, (55.48,58.39)),
    ],
    "Heavy_Makeup:Wearing_Lipstick":
        [(91.34, (91.11,91.56)), (31.19, (26.76,35.62)), (90.58, (90.34,90.81)), (33.26, (28.76,37.77))],
    "No_Beard:Wearing_Lipstick":
        [
            (96.05, (95.9,96.21)),
            (20.0, (0,40.24)),
            (94.65, (94.47,94.83)),
            (0.0, (0.0,0.0)),
        ],
    "Bangs:Wearing_Hat": [
        (95.93, (95.77,96.08)),
        (63.16, (50.64,75.68)),
        (95.55, (95.38,95.71)),
        (77.36, (66.71,88.01)),
    ],
    "Blond_Hair:Wearing_Hat": [
        (95.86, (95.7,96.02)), (35.71, (20.72,50.71)), (94.93, (94.76,95.11)), (48.72, (33.07,64.36))
    ]
}

mtl_nondisjoint_tasks["Pairing_1"] = {
    "Wearing_Earrings:Male":
        [(90.63, (90.4, 90.86)), (36.18, (32.16, 40.19)), (87.91, (87.65, 88.17)), (54.73, (50.57, 58.89))],
    "Attractive:Male": [(82.55, (82.24, 82.85)), (66.37, (65.35, 67.4)), (80.07, (79.75, 80.39)), (70.1, (68.92, 71.29))]
}

mtl_nondisjoint_tasks["Pairing_2"] = {
    "No_Beard:Heavy_Makeup":
        [(96.12, (95.96, 96.27)), (32.17, (15.68, 48.65)), (95.58, (95.41, 95.74)), (38.65, (21.47, 55.84))],
    "Pointy_Nose:Heavy_Makeup":
        [(77.58, (77.25, 77.92)), (21.75, (20.76, 22.74)), (71.76, (71.4, 72.12)), (43.92, (42.72, 45.11))]
}

mtl_nondisjoint_tasks["Pairing_3"] = {
    "Attractive:Gray_Hair": [
        (82.72, (82.42, 83.02)), (41.36, (28.66, 54.05)), (77.9, (77.57, 78.23)), (62.79, (62.11, 63.48))
    ],
    "Big_Nose:Gray_Hair": [(84.53, (84.24, 84.81)), (53.94, (52.86, 55.03)), (81.43, (81.12, 81.74)), (57.23, (53.89, 60.57))]
}

mtl_nondisjoint_tasks["Pairing_4"] = {
    "Heavy_Makeup:Wearing_Lipstick":
        [(91.49, (91.26, 91.71)), (32.07, (27.63, 36.51)), (90.54, (90.31, 90.77)), (35.61, (31.05, 40.17))],
    "No_Beard:Wearing_Lipstick":
        [(96.13, (95.98, 96.28)), (26.12, (6.28, 45.95)), (95.36, (95.19, 95.53)), (26.12, (6.28, 45.95))]
}

mtl_nondisjoint_tasks["Pairing_5"] = {
    "Bangs:Wearing_Hat": [(95.98, (95.83, 96.14)), (57.4, (44.97, 69.82)), (95.22, (95.05, 95.39)), (91.41, (90.43, 92.39))],
    "Blond_Hair:Wearing_Hat":
        [(95.87, (95.71, 96.03)), (32.49, (18.47, 46.52)), (94.99, (94.82, 95.17)), (37.16, (22.69, 51.63))]
}

mtl_disjoint_tasks["Pairing_1"] = {
    "Big_Lips:Chubby": [(70.6, (70.24, 70.97)), (15.49, (14.97, 16.02)), (66.9, (66.53, 67.28)), (58.84, (58.23, 59.45))],
    "Bushy_Eyebrows:Blond_Hair":
        [(92.76, (92.55, 92.96)), (14.2, (6.02, 22.39)), (88.52, (88.27, 88.78)), (41.41, (29.86, 52.96))]
}

mtl_disjoint_tasks["Pairing_2"] = {
    "Wearing_Lipstick:Male": [
        (93.64, (93.44, 93.83)), (31.7, (24.13, 39.28)), (93.59, (93.39, 93.78)), (46.2, (38.08, 54.32))
    ],
    "Gray_Hair:Young": [(98.16, (98.05, 98.27)), (27.58, (17.53, 37.64)), (96.79, (96.65, 96.94)), (42.09, (30.98, 53.2))]
}

mtl_disjoint_tasks["Pairing_3"] = {
    "High_Cheekbones:Smiling":
        [(87.77, (87.51, 88.03)), (38.96, (37.5, 40.42)), (85.53, (85.25, 85.81)), (51.7, (50.21, 53.19))],
    "Brown_Hair:Wearing_Hat": [
        (88.95, (88.7, 89.2)), (20.68, (11.9, 29.45)), (82.79, (82.48, 83.09)), (51.22, (40.39, 62.05))
    ]
}

mtl_disjoint_tasks["Pairing_4"] = {
    "No_Beard:Wearing_Lipstick":
        [(96.17, (96.02, 96.33)), (26.12, (6.28, 45.95)), (95.63, (95.47, 95.79)), (26.12, (6.28, 45.95))],
    "Young:Chubby": [(88.48, (88.23, 88.74)), (64.36, (63.51, 65.21)), (85.73, (85.45, 86.01)), (70.27, (68.94, 71.6))]
}

mtl_disjoint_tasks["Pairing_5"] = {
    "Bangs:Wearing_Hat": [(95.98, (95.82, 96.14)), (55.75, (43.27, 68.23)), (94.84, (94.66, 95.01)), (77.12, (66.56, 87.67))],
    "Pointy_Nose:Heavy_Makeup":
        [(77.47, (77.13, 77.8)), (20.95, (19.97, 21.93)), (71.28, (70.92, 71.64)), (47.18, (45.98, 48.38))]
}


def avg_gain_over_stl(stl_dict, mtl_dict, use_group_val_labels=True):
    avg_acc_gain = 0
    avg_worst_group_gain = 0

    for pairing_num in mtl_dict.keys():
        for task_name in mtl_dict[pairing_num]:

            mtl_results = mtl_dict[pairing_num][task_name]
            stl_results = stl_dict[task_name]

            if use_group_val_labels:
                avg_acc_gain += mtl_results[2][0] - stl_results[2][0]
                avg_worst_group_gain += mtl_results[3][0] - stl_results[3][0]

            else:
                avg_acc_gain += mtl_results[0][0] - stl_results[0][0]
                avg_worst_group_gain += mtl_results[1][0] - stl_results[1][0]

    avg_acc_gain /= len(stl_dict.keys())
    avg_worst_group_gain /= len(stl_dict.keys())

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


def avg_difference_between_avg_acc_and_worst_group_acc(stl_dict, mtl_dict, use_group_val_labels=True):
    avg_diff_mtl = 0
    avg_diff_stl = 0

    for task_name in stl_dict.keys():

        for pairing_num in mtl_dict.keys():
            if task_name in mtl_dict[pairing_num]:
                mtl_results = mtl_dict[pairing_num][task_name]
                stl_results = stl_dict[task_name]

                if use_group_val_labels:
                    #print(f"Task: {task_name}")
                    #print(f"MTL avg: {mtl_results[2][0]}")
                    #print(f"MTL group: {mtl_results[3][0]}")
                    avg_diff_mtl += mtl_results[2][0] - mtl_results[3][0]
                    avg_diff_stl += stl_results[2][0] - stl_results[3][0]
                else:
                    avg_diff_mtl += mtl_results[0][0] - mtl_results[1][0]
                    avg_diff_stl += stl_results[0][0] - stl_results[1][0]

    avg_diff_mtl /= len(stl_dict.keys())
    avg_diff_stl /= len(stl_dict.keys())

    return avg_diff_mtl, avg_diff_stl


def make_gain_over_stl_plot(use_group_val_labels = True):
    nondisjoint_gains = avg_gain_over_stl(stl_nondisjoint_tasks, mtl_nondisjoint_tasks, use_group_val_labels)
    disjoint_gains= avg_gain_over_stl(stl_disjoint_tasks, mtl_disjoint_tasks, use_group_val_labels)

    gains_data = [[disjoint_gains[0], 'Average Acc', 'Disjoint Pairings'], [disjoint_gains[1], 'Worst Group Acc', 'Disjoint Pairings'], [nondisjoint_gains[0], 'Average Acc', 'Nondisjoint Pairings'], [nondisjoint_gains[1], 'Worst Group Acc', 'Nondisjoint Pairings']]
    gains_df = pd.DataFrame(gains_data, columns=['Average Gain over STL', 'Metric', 'MTL Pairing Type'])
    plt.figure(figsize=(10, 10))
    sns.barplot(x = 'MTL Pairing Type',
            y = 'Average Gain over STL',
            hue = 'Metric',
            data = gains_df)

    ckpt_type = 'WG' if use_group_val_labels else "Avg"
    plt.title(f"MTL when using Disjoint vs Nondisjoint Pairings [{ckpt_type} Ckpt]")
    plt.savefig(f"./plots/disjoint_vs_nondisjoint_group_val_labels_{str(use_group_val_labels)}.png")
    plt.close()


def make_avg_in_pairing_plot(use_group_val_labels = True):

    nondisjoint_gains = wg_average_in_pairings(stl_nondisjoint_tasks, mtl_nondisjoint_tasks, use_group_val_labels)
    disjoint_gains = wg_average_in_pairings(stl_disjoint_tasks, mtl_disjoint_tasks, use_group_val_labels)

    avg_pairing_data = []
    for pairing_num in nondisjoint_gains:
        res = nondisjoint_gains[pairing_num]
        avg_pairing_data.append([res[0], res[1], "STL", f"ND {pairing_num}"])
        avg_pairing_data.append([res[2], res[3], "MTL", f"ND {pairing_num}"])
        
    for pairing_num in disjoint_gains:
        res = disjoint_gains[pairing_num]
        avg_pairing_data.append([res[0], res[1], "STL", f"D {pairing_num}"])
        avg_pairing_data.append([res[2], res[3], "MTL", f"D {pairing_num}"])

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
    plt.savefig(f"./plots/disjoint_vs_nondisjoint_pairings_group_val_labels_{str(use_group_val_labels)}.png")
    plt.close()
  
 
make_gain_over_stl_plot(use_group_val_labels = True)
make_gain_over_stl_plot(use_group_val_labels = False)

make_avg_in_pairing_plot(use_group_val_labels = True)
make_avg_in_pairing_plot(use_group_val_labels = False)

'''
nondisjoint_gains_group_val = avg_gain_over_stl(stl_nondisjoint_tasks, mtl_nondisjoint_tasks, True)
nondisjoint_gains_no_group_val = avg_gain_over_stl(stl_nondisjoint_tasks, mtl_nondisjoint_tasks, False)

disjoint_gains_group_val = avg_gain_over_stl(stl_disjoint_tasks, mtl_disjoint_tasks, True)
disjoint_gains_no_group_val = avg_gain_over_stl(stl_disjoint_tasks, mtl_disjoint_tasks, False)

print(
    f"Nondisjoint not using group val labels saw an average increase of {nondisjoint_gains_no_group_val[0]} in average accuracy and {nondisjoint_gains_no_group_val[1]} in worst group accuracy"
)
print(
    f"Nondisjoint using group val labels saw an average increase of {nondisjoint_gains_group_val[0]} in average accuracy and {nondisjoint_gains_group_val[1]} in worst group accuracy"
)

print(
    f"Disjoint not using group val labels saw an average increase of {disjoint_gains_no_group_val[0]} in average accuracy and {disjoint_gains_no_group_val[1]} in worst group accuracy"
)
print(
    f"Disjoint using group val labels saw an average increase of {disjoint_gains_group_val[0]} in average accuracy and {disjoint_gains_group_val[1]} in worst group accuracy \n"
)

nondisjoint_diff_group_val = avg_difference_between_avg_acc_and_worst_group_acc(
    stl_nondisjoint_tasks, mtl_nondisjoint_tasks, True
)
nondisjoint_diff_no_group_val = avg_difference_between_avg_acc_and_worst_group_acc(
    stl_nondisjoint_tasks, mtl_nondisjoint_tasks, False
)

disjoint_diff_group_val = avg_difference_between_avg_acc_and_worst_group_acc(stl_disjoint_tasks, mtl_disjoint_tasks, True)
disjoint_diff_no_group_val = avg_difference_between_avg_acc_and_worst_group_acc(stl_disjoint_tasks, mtl_disjoint_tasks, False)

print(
    f"MTL Nondisjoint not using group val labels had an average difference between avg accuracy and worst group of {nondisjoint_diff_no_group_val[0]} whereas STL not using group val labels had: {nondisjoint_diff_no_group_val[1]}"
)
print(
    f"MTL Nondisjoint using group val labels had an average difference between avg accuracy and worst group of {nondisjoint_diff_group_val[0]} whereas STL using group val labels had: {nondisjoint_diff_group_val[1]}"
)

print(
    f"MTL disjoint not using group val labels had an average difference between avg accuracy and worst group of {disjoint_diff_no_group_val[0]} whereas STL not using group val labels had: {disjoint_diff_no_group_val[1]}"
)
print(
    f"MTL Disjoint using group val labels had an average difference between avg accuracy and worst group of {disjoint_diff_group_val[0]} whereas STL using group val labels had: {disjoint_diff_group_val[1]}"
)

#### how to factor in CI for differences? - clarify with tatsu and double check

#### all Tasks bar plot - ask Tatsu if needed
'''