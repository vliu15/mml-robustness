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

stl_weak_spur_corr_tasks["Big_Lips:Chubby"] = [
   (69.93, (69.57, 70.30)), (11.27, (10.81, 11.72)), (68.32, (67.94, 68.69)), (41.26, (40.55, 41.97))
]

stl_weak_spur_corr_tasks["Bushy_Eyebrows:Blond_Hair"] = [
    (92.60, (92.39, 92.81)), (15.64, (7.12, 24.15)), (90.62, (90.39, 90.85)), (32.82, (21.81, 43.83))
]

stl_weak_spur_corr_tasks["Wearing_Lipstick:Male"] = [
    (93.49, (93.29, 93.69)), (37.23, (29.35, 45.10)), (92.67, (92.46, 92.88)), (49.65, (41.51, 57.80))
]

stl_weak_spur_corr_tasks["Gray_Hair:Young"] = [
    (98.15, (98.05, 98.26)), (26.27, (16.36, 36.17)), (97.94, (97.83, 98.06)), (18.36, (9.64, 27.07))
]

stl_weak_spur_corr_tasks["High_Cheekbones:Smiling] = [
    (87.39, (87.12, 87.65)), (38.26, (36.81, 39.72)), (84.29, (84.00, 84.58)), (50.03, (48.6, 51.47))
]

stl_weak_spur_corr_tasks["Brown_Hair:Wearing_Hat"] = [
    (88.67, (88.42, 88.93)), (24.34, (15.04, 33.64)), (86.48, (86.21, 86.75)), (39.0, (28.44, 49.57))
]

stl_weak_spur_corr_tasks["Wearing_Earrings:Male"] = [
    (90.38, (90.14, 90.61)), (32.9, (28.98, 36.83)), (88.6, (88.35, 88.86)), (42.36, (38.23, 46.49))
]

stl_weak_spur_corr_tasks["Attractive:Male"] = [
     (82.17, (81.87, 82.48)), (66.11, (65.2, 67.01)), (79.74, (79.42, 80.06)), (65.61, (64.38, 66.84))
]

stl_weak_spur_corr_tasks["No_Beard:Heavy_Makeup"] = [
   (96.05, (95.89, 96.2)), (26.12, (6.28, 45.95)), (89.68, (89.44, 89.93)), (6.23, (-2.3, 14.76))
]

stl_weak_spur_corr_tasks["Pointy_Nose:Heavy_Makeup"] = [
     (77.34, (77.0, 77.68)), (20.26, (19.3, 21.23)), (72.44, (72.08, 72.8)), (35.3, (34.15, 36.44))
]

mtl_weak_spur_corr_tasks["Pairing_1"] = {
    "Bags_Under_Eyes:Double_Chin": [],
    "High_Cheekbones:Rosy_Cheeks": []
}

mtl_weak_spur_corr_tasks["Pairing_2"] = {
    "Bangs:Wearing_Hat": [(95.98, (95.83, 96.14)), (57.4, (44.97, 69.82)), (95.22, (95.05, 95.39)), (91.41, (90.43, 92.39))],
    "Blond_Hair:Wearing_Hat":
        [(95.87, (95.71, 96.03)), (32.49, (18.47, 46.52)), (94.99, (94.82, 95.17)), (37.16, (22.69, 51.63))]
}

mtl_weak_spur_corr_tasks["Pairing_3"] = {
    "No_Beard:Wearing_Lipstick": [],
    "Young:Chubby": []
}

mtl_weak_spur_corr_tasks["Pairing_4"] = {
    "Big_Lips:Chubby": [],
    "Young:Chubby": []
}

mtl_weak_spur_corr_tasks["Pairing_5"] = {
    "High_Cheekbones:Rosy_Cheeks": [],
    "Brown_Hair:Wearing_Hat": []
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
    "Wearing_Lipstick:Male": [],
    "High_Cheekbones:Smiling": []
}

mtl_strong_spur_corr_tasks["Pairing_5"] = {
    "Heavy_Makeup:Male": [],
    "Wearing_Earrings:Male":
        []
}