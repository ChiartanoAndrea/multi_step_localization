import yaml
import os
import glob
import subprocess

def run_loo_pipeline():
    # 1. Path to your base YAML and the folder containing LOO JSONs
    base_yaml_path = "./multi_step_localization/actionformer/configs/ego4d_pe.yaml"
    loo_json_folder = "./captaincook_actionformer_annotations/normal/recordings_LOO"
    
    # Get all the generated LOO json files
    json_files = glob.glob(os.path.join(loo_json_folder, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {loo_json_folder}")
        return

    # Load the base configuration
    with open(base_yaml_path, 'r') as file:
        base_config = yaml.safe_load(file)

    for json_path in json_files:
        # Extract the recording ID from the file name (e.g., recordings_loo_001.json -> 001)
        filename = os.path.basename(json_path)
        video_id = filename.replace("recordings_loo_", "").replace(".json", "")
        
        print(f"\n{'='*50}")
        print(f"Starting LOO fold for Video ID: {video_id}")
        print(f"{'='*50}")

        # 2. Update the YAML configuration for this specific run
        current_config = base_config.copy()
        current_config['dataset']['json_file'] = json_path
        
        # VERY IMPORTANT: Change output folder so checkpoints don't overwrite!
        current_config['output_folder'] = f"L1O_output_actionFormer/LOO_{video_id}"

        # 3. Save a temporary YAML file for ActionFormer to use
        temp_yaml_path = f"temp_loo_config_{video_id}.yaml"
        with open(temp_yaml_path, 'w') as file:
            yaml.dump(current_config, file, default_flow_style=False)

        # 4. Run ActionFormer Training
        # Replace 'train.py' with the actual path to your ActionFormer training script if different
        train_cmd = [
            "python", 
            "./multi_step_localization/train.py", 
            temp_yaml_path, 
            "--output", "reproduce", 
            "--backbone", "perception_encoder"
        ]
        print(f"Running Training: {' '.join(train_cmd)}")
        
        subprocess.run(train_cmd, check=True)

        # 5. Run ActionFormer Evaluation
        # Replace 'eval.py' with the actual inference script used by ActionFormer
        # You may need to pass the checkpoint path depending on ActionFormer's specific eval arguments
        eval_cmd = [
            "python", 
            "./multi_step_localization/eval.py", 
            temp_yaml_path, 
            "reproduce", 
            "--backbone", "perception_encoder", 
            "--saveonly"
        ]
        print(f"Running Evaluation: {' '.join(eval_cmd)}")
        subprocess.run(eval_cmd, check=True)
        
        # Optional: Clean up the temp yaml to keep the folder clean
        os.remove(temp_yaml_path)

if __name__ == '__main__':
    run_loo_pipeline()