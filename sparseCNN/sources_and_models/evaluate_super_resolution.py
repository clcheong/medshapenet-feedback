import os
import nibabel as nib
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error

# Define root directories for each type of volume
output_sr_root = "C://Users//User//utm//postgrad//research//codes//medshapenet-feedback//sparseCNN//predictions"
incomplete_root = "C://Users//User//utm//postgrad//research//codes//medshapenet-feedback//anatomy-completor//completor_dataset//dataset//test//incomplete"
output_root = "C://Users//User//utm//postgrad//research//codes//medshapenet-feedback//anatomy-completor//output_multiclass"
ground_truth_root = "C://Users//User//utm//postgrad//research//codes//medshapenet-feedback//anatomy-completor//completor_dataset//dataset//test//complete"

# Define functions for Dice Similarity Coefficient and metrics calculations
def dice_coefficient(vol1, vol2):
    vol1 = vol1 > 0  # Binary threshold
    vol2 = vol2 > 0
    intersection = np.sum(vol1 * vol2)
    return 2 * intersection / (np.sum(vol1) + np.sum(vol2))

def calculate_metrics(vol1, vol2):
    dsc = dice_coefficient(vol1, vol2)
    mse = mean_squared_error(vol1.flatten(), vol2.flatten())
    rmse = np.sqrt(mse)
    return dsc, mse, rmse

# Initialize lists to store metric values for T-Tests
dsc_output_list, mse_output_list, rmse_output_list = [], [], []
dsc_sr_list, mse_sr_list, rmse_sr_list = [], [], []

print("output_sr_root = ", output_sr_root)

# Loop through each file in output_sr_root
for folder in os.listdir(output_sr_root):

    output_sr_folder = os.path.join(output_sr_root,folder)

    for filename in os.listdir(output_sr_folder):
        if not filename.endswith("-SR.nii.gz"):
            continue

        # Extract dataset code and index from filename
        dataset_code, index = filename.split("-")[0].split("_")

        # Define paths for each corresponding file
        output_sr_path = os.path.join(output_sr_folder, filename)
        output_path = os.path.join(output_root, dataset_code, f"{dataset_code}_{index}.nii.gz")
        ground_truth_path = os.path.join(ground_truth_root, f"{dataset_code}_full.nii.gz")

        print(f"output_sr_path: ", output_sr_path)
        print(f"output_path: ", output_path)
        print(f"ground_truth_path: ", ground_truth_path)

        # Check if corresponding files exist
        if not os.path.exists(output_path) or not os.path.exists(ground_truth_path):
            print(f"Missing corresponding files for {filename}, skipping.")
            continue

        print("Processing Evaluation for ", output_sr_path)

        # Load Ground Truth volume
        ground_truth_data = nib.load(ground_truth_path).get_fdata()

        # Load Output (before SR) volume
        output_data = nib.load(output_path).get_fdata()

        # Load Output Super Resolution volume
        output_sr_data = nib.load(output_sr_path).get_fdata()

        # Calculate metrics for Output vs Ground Truth
        dsc_output, mse_output, rmse_output = calculate_metrics(output_data, ground_truth_data)
        dsc_output_list.append(dsc_output)
        mse_output_list.append(mse_output)
        rmse_output_list.append(rmse_output)

        # Calculate metrics for Output Super Resolution vs Ground Truth
        dsc_sr, mse_sr, rmse_sr = calculate_metrics(output_sr_data, ground_truth_data)
        dsc_sr_list.append(dsc_sr)
        mse_sr_list.append(mse_sr)
        rmse_sr_list.append(rmse_sr)


# Calculate and print average metrics for both sets
avg_dsc_output = np.mean(dsc_output_list)
avg_mse_output = np.mean(mse_output_list)
avg_rmse_output = np.mean(rmse_output_list)

avg_dsc_sr = np.mean(dsc_sr_list)
avg_mse_sr = np.mean(mse_sr_list)
avg_rmse_sr = np.mean(rmse_sr_list)

print("\nAverage Metrics:")
print("Original Output vs Ground Truth:")
print(f"  Average Dice Similarity Coefficient: {avg_dsc_output}")
print(f"  Average Mean Squared Error: {avg_mse_output}")
print(f"  Average Root Mean Squared Error: {avg_rmse_output}")

print("\nOutput Super Resolution vs Ground Truth:")
print(f"  Average Dice Similarity Coefficient: {avg_dsc_sr}")
print(f"  Average Mean Squared Error: {avg_mse_sr}")
print(f"  Average Root Mean Squared Error: {avg_rmse_sr}")


# Perform paired T-Tests for significance
dsc_t_stat, dsc_p_value = stats.ttest_rel(dsc_output_list, dsc_sr_list)
mse_t_stat, mse_p_value = stats.ttest_rel(mse_output_list, mse_sr_list)
rmse_t_stat, rmse_p_value = stats.ttest_rel(rmse_output_list, rmse_sr_list)

print("\nPaired T-Tests for significance of improvement after super resolution:")
print(f"Dice Similarity Coefficient - T-Statistic: {dsc_t_stat}, P-Value: {dsc_p_value}")
print(f"Mean Squared Error - T-Statistic: {mse_t_stat}, P-Value: {mse_p_value}")
print(f"Root Mean Squared Error - T-Statistic: {rmse_t_stat}, P-Value: {rmse_p_value}")

# Interpretation
significance_level = 0.05
if dsc_p_value < significance_level:
    print("\nDice Similarity Coefficient shows significant improvement after super resolution.")
else:
    print("\nNo significant improvement in Dice Similarity Coefficient after super resolution.")

if mse_p_value < significance_level:
    print("Mean Squared Error shows significant improvement after super resolution.")
else:
    print("No significant improvement in Mean Squared Error after super resolution.")

if rmse_p_value < significance_level:
    print("Root Mean Squared Error shows significant improvement after super resolution.")
else:
    print("No significant improvement in Root Mean Squared Error after super resolution.")
