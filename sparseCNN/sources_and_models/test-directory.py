import os

train_recon_dir = "../super_resolution_dataset/train/reconstructed"
train_gt_dir = "../super_resolution_dataset/train/ground_truth"
valid_recon_dir = "../super_resolution_dataset/test/reconstruction"
valid_gt_dir = "../super_resolution_dataset/test/ground_truth"



print(os.path.exists(train_recon_dir))  # Should print True if the directory exists
print(os.path.exists(train_gt_dir))     # Should print True if the directory exists
print(os.path.exists(valid_recon_dir))  # Should print True if the directory exists
print(os.path.exists(valid_gt_dir))     # Should print True if the directory exists
