import yaml
import os
import glob
import subprocess
import shutil

def run_kfold_pipeline():
    base_yaml_path = "./multi_step_localization/actionformer/configs/ego4d_pe.yaml"
    # Cambiato il percorso per puntare alla cartella KFold
    #json_folder = "./captaincook_actionformer_annotations/normal/recordings_KFold"
    json_folder = "./captaincook_actionformer_annotations/combined/recordings_KFold"
    
    json_files = glob.glob(os.path.join(json_folder, "*.json"))
    with open(base_yaml_path, 'r') as file:
        base_config = yaml.safe_load(file)

    for json_path in json_files:
        filename = os.path.basename(json_path)
        fold_id = filename.replace("recordings_fold_", "").replace(".json", "")
        
        print(f"\n>>> Elaborazione FOLD: {fold_id}")

        current_config = base_config.copy()
        current_config['dataset']['json_file'] = json_path
        # Cartella di output temporanea per questo fold
        # Aggiungo prova per vedere i valori del train
        output_dir = f"Output_KFold/out_fold_{fold_id}"
        current_config['output_folder'] = output_dir

        temp_yaml = f"temp_fold_{fold_id}.yaml"
        with open(temp_yaml, 'w') as file:
            yaml.dump(current_config, file)

        # 1. Training
        subprocess.run(["python", "./multi_step_localization/train.py", temp_yaml, "--output", "reproduce", "--backbone", "perception_encoder"], check=True)

        # 2. Evaluation
        subprocess.run(["python", "./multi_step_localization/eval.py", temp_yaml, "reproduce", "--backbone", "perception_encoder", "--saveonly"], check=True)
        #for result mAP = 57% circa
        #subprocess.run(["python", "./multi_step_localization/eval.py", temp_yaml, "reproduce", "--backbone", "perception_encoder",], check=True)

        # 3. CLEANUP: Salviamo il risultato e cancelliamo i modelli pesanti
        res_src = os.path.join(output_dir, "ego4d", "perception_encoder_recordings_reproduce", "eval_results.pkl")
        if os.path.exists(res_src):
            shutil.copy(res_src, f"ActionFormer_eval/final_results_fold_{fold_id}.pkl")
            print(f"Risultati salvati in ActionFormer_eval/final_results_fold_{fold_id}.pkl")

        print(f"Pulizia cartella {output_dir} per risparmiare spazio...")
        shutil.rmtree(output_dir) 
        os.remove(temp_yaml)

if __name__ == '__main__':
    run_kfold_pipeline()