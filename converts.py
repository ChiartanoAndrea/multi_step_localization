import os
import numpy as np
from tqdm import tqdm

src_path = 'data/features/perception_encoder/npz_features/'
dst_path = 'data/features/perception_encoder/npy_features/'
os.makedirs(dst_path, exist_ok=True)

for file_name in tqdm(os.listdir(src_path)):
    if file_name.endswith('.npz'):
        # Load the archive
        data = np.load(os.path.join(src_path, file_name))
        
        # Extract the single feature matrix
        feature_array = data['arr_0'] # Based on your output
        
        # Save as standard .npy
        new_name = file_name.replace('.npz', '.npy')
        np.save(os.path.join(dst_path, new_name), feature_array)

print("Unpacking complete!")