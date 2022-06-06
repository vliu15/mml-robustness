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
        [(90.63, (90.4,90.87)), (35.98, (31.96,40.0)), (87.95, (87.69,88.21)), (54.76, (50.59,58.94))],
    "Attractive:Male": [(82.55, (82.25,82.85)), (66.38, (65.36,67.41)), (80.1, (79.78,80.42)), (70.21, (69.03,71.39))]
}

mtl_nondisjoint_tasks["Pairing_2"] = {
    "No_Beard:Heavy_Makeup":
        [(96.12, (95.97,96.28)), (28.99, (12.0,45.98)), (95.58, (95.42,95.75)), (35.19, (17.65,52.72))],
    "Pointy_Nose:Heavy_Makeup":
        [(77.58, (77.25,77.92)), (21.39, (20.41,22.37)), (71.77, (71.41,72.13)), (43.82, (42.63,45.0))]
}

mtl_nondisjoint_tasks["Pairing_3"] = {
    "Attractive:Gray_Hair": [
        (82.72, (82.42,83.03)), (40.42, (27.46,53.38)), (77.9, (77.57,78.23)), (62.8, (62.11,63.49))
    ],
    "Big_Nose:Gray_Hair": [(84.53, (84.24,84.82)), (53.95, (52.86,55.04), (81.47, (81.16,81.78)), (57.33, (53.99,60.66))]
}

mtl_nondisjoint_tasks["Pairing_4"] = {
    "Heavy_Makeup:Wearing_Lipstick":
        [(91.49, (91.27,91.71)), (31.89, (27.43,36.35)), (90.55, (90.31,90.78)), (35.47, (30.9,40.05))],
    "No_Beard:Wearing_Lipstick":
        [(96.13, (95.98,96.29)), (20.0, (0.0,40.24)), (95.38, (95.21,95.55)), (20.0, (0.0,40.24))]
}

mtl_nondisjoint_tasks["Pairing_5"] = {
    "Bangs:Wearing_Hat": [(95.99, (95.83,96.14)), (57.89, (45.08,70.71)), (95.22, (95.05,95.39)), (91.51, (90.53,92.48))],
    "Blond_Hair:Wearing_Hat":
        [(95.88, (95.72,96.04)), (30.77, (16.28,45.25)), (95.01, (94.83,95.18)), (34.45, (19.86,49.04))]
}

mtl_disjoint_tasks["Pairing_1"] = {
    "Big_Lips:Chubby": [(70.6, (70.24,70.97)), (15.35, (14.84,15.87)), (66.92, (66.54,67.29)), (59.01, (58.4,59.61))],
    "Bushy_Eyebrows:Blond_Hair":
        [(92.76, (92.55,92.97)), (11.75, (4.0,19.49)), (88.63, (88.37,88.88)), (39.17, (28.13,50.2))]
}

mtl_disjoint_tasks["Pairing_2"] = {
    "Wearing_Lipstick:Male": [
        (93.64, (93.44,93.83)), (31.19, (23.54,38.83)), (93.6, (93.4,93.79)), (45.95, (37.78,54.13))],
    "Gray_Hair:Young": [(98.17, (98.06,98.27)), (26.3, (16.14,36.46)), (96.93, (96.79,97.07)), (40.88, (29.78,51.99))]
}

mtl_disjoint_tasks["Pairing_3"] = {
    "High_Cheekbones:Smiling":
        [(87.77, (87.51,88.04)), (38.94, (37.48,40.4)), (85.54, (85.26,85.82)), (51.7, (50.21,53.2))],
    "Brown_Hair:Wearing_Hat": [(88.96, (88.71,89.21)), (18.83, (10.18,27.48)), (82.91, (82.61,83.21)), (51.52, (40.71,62.33))]
}

mtl_disjoint_tasks["Pairing_4"] = {
    "No_Beard:Wearing_Lipstick":
        [(96.18, (96.02,96.33)), (20.0, (-0.24,40.24)), (95.64, (95.48,95.8)), (0.0, (0.0,0.0))],
    "Young:Chubby": [(88.49, (88.23,88.74)), (64.44, (63.58,65.29)), (85.75, (85.47,86.03)), (70.31, (68.99,71.64))]
}

mtl_disjoint_tasks["Pairing_5"] = {
    "Bangs:Wearing_Hat": [(95.98, (95.83,96.14)), (56.32, (43.55,69.08)), (94.85, (94.67,95.03)), (79.61, (69.21,90.01))],
    "Pointy_Nose:Heavy_Makeup":
        [(77.47, (77.13,77.8)), (77.47, (77.13,77.8)), (71.29, (70.93,71.66)), (47.16, (45.96,48.36))]
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