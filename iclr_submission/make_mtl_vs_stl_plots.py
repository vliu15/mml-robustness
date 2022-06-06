import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


mtl_results = {}
stl_results = {}

stl_results["ERM"] = {"Big_Lips:Chubby": [(69.94, (69.57,70.3)), (11.01, (10.56,11.46)), (68.32, (67.94,68.69)), (41.24, (40.53,41.95))],
    "Bushy_Eyebrows:Blond_Hair":
        [(92.6, (92.39,92.81)), (13.64, (5.36,21.92)), (90.8, (90.57,91.03)), (27.04, (16.94,37.15))]
        "Wearing_Lipstick:Male":
        [(93.49, (93.3,93.69)), (36.72, (28.78,44.65)), (92.68, (92.47,92.89)), (49.66, (41.63,57.7))],
        "Gray_Hair:Young": [(98.16, (98.05,98.26)), (25.0, (15.0,35.0)), (97.95, (97.84,98.06)), (15.32, (7.09,23.55))],
        "High_Cheekbones:Smiling":
        [(87.39, (87.12,87.65)), (38.24, (36.79,39.69)), (84.3, (84.01,84.6)), (50.03, (48.6,51.47))]}

stl_results["RWY"] = {"Big_Lips:Chubby": [(69.58, (69.22,69.95)), (49.31, (48.59,50.03)), (67.44, (67.07,67.82)), (57.85, (57.14,58.57))],
    "Bushy_Eyebrows:Blond_Hair": [(91.57, (91.35,91.79)), (30.0, (18.99,41.01)), (75.64, (75.3,75.98)), (67.32, (66.79,67.86))],
    "Wearing_Lipstick:Male":
        [(93.59, (93.39,93.78)), (33.87, (26.08,41.66)), (92.68, (92.47,92.89)), (34.0, (26.19,41.81))],
        "Gray_Hair:Young": [(97.1, (96.96,97.23)), (41.34, (30.07,52.6)), (92.38, (92.17,92.59)), (75.25, (74.33,76.17))],
        "High_Cheekbones:Smiling":
        [(86.7, (86.43,86.97)), (32.34, (30.94,33.73)), (85.31, (85.02,85.59)), (49.57, (48.19,50.95))]}

stl_results["SUBY"] = {"Big_Lips:Chubby": [(69.46, (69.09,69.83)), (28.6, (27.95,29.25)), (68.69, (68.32,69.06)), (60.39, (59.69,61.1))],
     "Bushy_Eyebrows:Blond_Hair": [(91.31, (91.08,91.54)), (16.17, (7.54,24.8)), (77.0, (76.66,77.34)), (71.64, (71.22,72.06))],
     "Wearing_Lipstick:Male":
        [(93.61, (93.42,93.81)), (35.23, (27.37,43.08)), (93.15, (92.95,93.36)), (38.88, (30.86,46.9))],
        "Gray_Hair:Young": [(95.93, (95.78,96.09)), (41.15, (30.53,51.76)), (93.0, (92.79,93.2)), (74.1, (73.17,75.03))],
        "High_Cheekbones:Smiling":
        [(87.67, (87.4,87.93)), (38.49, (37.04,39.95)), (85.13, (84.84,85.41)), (51.95, (50.57,53.33))]}

stl_results["JTT"] = {"Big_Lips:Chubby": [(32.7, (32.33,33.08)), (0.0, (0.0,0.0)), (32.7, (32.33,33.08)), (0.0, (0.0,0.0))],
    "Bushy_Eyebrows:Blond_Hair": [(91.94, (91.72,92.16)), (21.04, (11.22,30.86)), (71.45, (71.09,71.81)), (64.15, (63.71,64.6))],
    "Wearing_Lipstick:Male":
        [(91.27, (91.05,91.5)), (42.45, (34.32,50.57)), (72.96, (72.6,73.32)), (51.52, (49.32,53.73))],
        "Gray_Hair:Young": [(97.81, (97.69,97.93)), (22.31, (12.82,31.8)), (90.83, (90.6,91.06)), (68.34, (67.53,69.14))],
        "High_Cheekbones:Smiling":
        [(TODO, (TODO)), (TODO, (TODO)), (TODO, (TODO), (TODO, (TODO))]}


mtl_results["ERM"] = {
    "Big_Lips:Chubby": [(70.6, (70.24,70.97)), (15.35, (14.84,15.87)), (66.92, (66.54,67.29)), (59.01, (58.4,59.61))],
    "Bushy_Eyebrows:Blond_Hair":
        [(92.76, (92.55,92.97)), (11.75, (4.0,19.49)), (88.63, (88.37,88.88)), (39.17, (28.13,50.2))]
}

mtl_results["RWY"] = {
    "Big_Lips:Chubby": [(69.65, (69.28,70.02)), (28.98, (28.33,29.63)), (66.06, (65.68,66.44)), (58.87, (58.16,59.58))],
    "Bushy_Eyebrows:Blond_Hair":
        [(88.79, (88.53,89.04)), (41.49, (30.25,52.73)), (80.75, (80.43,81.06)), (73.53, (72.82,74.24))]
}

mtl_results["SUBY"] = {
    "Big_Lips:Chubby": [(67.33, (66.96,67.71)), (47.29, (46.58,48.01)), (64.07, (63.68,64.45)), (61.45, (60.84,62.05))],
    "Bushy_Eyebrows:Blond_Hair":
        [(84.37, (84.08,84.66)), (44.82, (33.07,56.57)), (79.56, (79.24,79.88)), (72.23, (71.51,72.96))]
}

mtl_results["JTT"] = {
    "Big_Lips:Chubby": [(61.14, (60.75,61.53)), (54.02, (53.48,54.57)), (60.03, (59.63,60.42)), (50.47, (49.93,51.02))],
    "Bushy_Eyebrows:Blond_Hair":
        [(72.96, (72.6,73.31)), (67.3, (66.86,67.74)), (77.42, (77.09,77.76)), (68.85, (68.11,69.6))]
}


def make_plot(wg_acc = True):

    plt.figure(figsize=(12, 7))
    avg_opt_data = []

    y_label = "Mean Worst Group Accuracy Across Tasks" if wg_acc else "Mean Average Accuracy Across Tasks"

    for opt_type in mtl_results.keys():
        
        avg_mtl = 0
        avg_stl = 0

        for task_name in mtl_results[opt_type]:

            mtl_res = mtl_results[opt_type][task_name]
            stl_res = stl_results[opt_type][task_name]

            if wg_acc:
                avg_mtl += mtl_res[3][0]
                avg_stl += stl_res[3][0]
            else:
                avg_mtl += mtl_res[2][0]
                avg_stl += stl_res[2][0]
                

        avg_mtl /= len(mtl_results[opt_type])
        avg_stl /= len(mtl_results[opt_type])

        avg_opt_data.append([avg_stl, "STL", opt_type])
        avg_opt_data.append([avg_mtl, "MTL", opt_type])


    avg_opt_df = pd.DataFrame(avg_opt_data, columns=[y_label, 'Learning Type', 'Optimization Procedure'])


    sns.barplot(x = 'Optimization Procedure',
            y = y_label,
            hue = 'Learning Type',
            data = avg_opt_df)

            
    plt.title(f"STL vs MTL Comparison")
    plt.savefig(f"./plots/stl_mtl_comparison_wg_{wg_acc}.png")
    plt.close()

make_plot(wg_acc = True)
make_plot(wg_acc = False)