import os
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import MinkowskiEngine as ME
import numpy as np
import matplotlib.pyplot as plt

# Utility functions for data_skull module
def create_sparse_tensor(features, coordinates, device):
    return ME.SparseTensor(
        features=features.to(device),
        coordinates=coordinates.to(device)  # Ensure this is on the right device
    )

def filter_implant(dense_out, dense_in):
    """
    Applies implant filtering logic. This can be customized based on application needs.
    For simplicity, here it filters dense_out values greater than dense_in.
    """
    return np.where(dense_out > dense_in, dense_out, 0)


def plot_slice_in_out_truth(dense_in, in_len, dense_out, out_len, dense_truth, truth_len, slice_idx, epoch, show=False):
    """
    Plots slices of input, output, and ground truth for visualization.
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(dense_in[slice_idx], cmap="gray")
    axs[0].set_title(f"Input ({in_len})")
    axs[1].imshow(dense_out[slice_idx], cmap="gray")
    axs[1].set_title(f"Output ({out_len})")
    axs[2].imshow(dense_truth[slice_idx], cmap="gray")
    axs[2].set_title(f"Ground Truth ({truth_len})")

    plt.suptitle(f"Epoch {epoch} - Slice {slice_idx}")
    if show:
        plt.show()
    else:
        plt.close(fig)


class SuperResolutionDataset(Dataset):
    def __init__(self, recon_dir, gt_dir, transform=None):
        self.recon_dir = recon_dir
        self.gt_dir = gt_dir
        self.transform = transform
        
        # Generate a list of pairs of (reconstruction_path, ground_truth_path)
        self.data_pairs = []
        for subj in os.listdir(recon_dir):
            recon_subdir = os.path.join(recon_dir, subj)
            gt_file = os.path.join(gt_dir, f"{subj}_full.nii.gz")

            print("GROUND TRUTH FILE: ", gt_file)

            if not os.path.isfile(gt_file):
                print("GROUND TRUTH FILE NOT FOUND")
                continue  # Skip if no matching ground truth file
            
            # Match each reconstruction file to its ground truth
            for recon_file in sorted(os.listdir(recon_subdir)):
                recon_path = os.path.join(recon_subdir, recon_file)
                print("RECONSTRUCTED FILE:", recon_path)
                self.data_pairs.append((recon_path, gt_file))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        recon_path, gt_path = self.data_pairs[idx]

        # Load the .nii.gz files for reconstruction (input) and ground truth
        recon_img = nib.load(recon_path).get_fdata()
        gt_img = nib.load(gt_path).get_fdata()

        # Ensure both images are in the correct resolution
        if recon_img.shape != gt_img.shape:
            raise ValueError("Incompatible input & output image resolution")

        # Convert to torch tensors and add a channel dimension
        recon_tensor = torch.tensor(recon_img, dtype=torch.float32)
        gt_tensor = torch.tensor(gt_img, dtype=torch.float32)

        # shape = [recon_img.shape, gt_img.shape]

        # Apply any optional transformations
        # if self.transform:
        #     recon_tensor = self.transform(recon_tensor)
        #     gt_tensor = self.transform(gt_tensor)

        return {"defective": recon_tensor, "complete": gt_tensor}


def sparse_collate_fn(batch):

    print("In Sparse Collate Fn")

    defective_coords = []
    complete_coords = []

    for batch_index, data in enumerate(batch):
        defective_tensor = data['defective']
        complete_tensor = data['complete']

        # Extract non-zero elements for defective tensor
        defective_nz_coords = torch.nonzero(defective_tensor, as_tuple=False)
        
        # Add batch dimension to the coordinates for defective tensor
        defective_batch_coords = torch.cat(
            [torch.full((defective_nz_coords.shape[0], 1), batch_index, dtype=torch.int32), defective_nz_coords.int()], 
            dim=1
        )
        defective_coords.append(defective_batch_coords)

        # Extract non-zero elements for complete tensor (only x, y, z coordinates, no batch dimension)
        complete_nz_coords = torch.nonzero(complete_tensor, as_tuple=False).float()  # Convert to float as required
        complete_coords.append(complete_nz_coords)

    # Concatenate all defective coordinates into a single tensor
    batched_defective_coords = torch.cat(defective_coords, dim=0)

    print("End Sparse Collate Fn")

    # Return the defective coordinates directly as a tensor and complete coordinates as a list of tensors
    return {
        "defective": batched_defective_coords,  # Shape: [num_non_zero, 4]
        "complete": complete_coords,  # List of tensors, each with shape [num_non_zero, 3]
        "shape": [batched_defective_coords.shape, [c.shape for c in complete_coords]],  # Shape info for debugging if needed
        "defectiveTensor": defective_tensor,
        "completeTensor": complete_tensor
    }



# Define paths to the train and validation directories
train_recon_dir = "../super_resolution_dataset/train/reconstructed"
train_gt_dir = "../super_resolution_dataset/train/ground_truth"
valid_recon_dir = "../super_resolution_dataset/test/reconstruction"
valid_gt_dir = "../super_resolution_dataset/test/ground_truth"

# Initialize dataset and dataloaders
train_dataset = SuperResolutionDataset(recon_dir=train_recon_dir, gt_dir=train_gt_dir)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=sparse_collate_fn)

valid_dataset = SuperResolutionDataset(recon_dir=valid_recon_dir, gt_dir=valid_gt_dir)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=sparse_collate_fn)


# data_skull as a module-like object with methods and dataset references for train, eval, test functions
class DataSkull:
    def __init__(self, train_dataloader, valid_dataloader):
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

    def plot_slice_in_out_truth(self, *args, **kwargs):
        plot_slice_in_out_truth(*args, **kwargs)

    def filter_implant(self, dense_out, dense_in):
        return filter_implant(dense_out, dense_in)


# Instantiate the data_skull with train and validation dataloaders
data_skull = DataSkull(train_dataloader, valid_dataloader)
