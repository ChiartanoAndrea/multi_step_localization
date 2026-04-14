import json
import os
import random
import numpy as np
import pandas as pd
from collections import defaultdict


def csv_to_json_kfold(split_type, video_category, val_size=10, seed=42):
    """
    Genera fold in cui ogni validation set contiene ~val_size recording_id,
    scelti in modo random ma stratificato per ricetta.
    Ogni recording_id appare in validation esattamente una volta.
    Numero fold automatico: tot_ids // val_size.
    """
    steps_count = 353
    fps = 29.97
    captaincook_dataset_version = f"CaptainCookDataset-{fps}fps-{steps_count}Steps"

    dataset_dir = "./multi_step_localization/captaincook"
    annotations_json_dir = os.path.join(dataset_dir, "annotation_json")
    data_splits_dir = os.path.join(dataset_dir, "data_splits")

    data_split_json_files = [
        f for f in os.listdir(data_splits_dir)
        if split_type in f and video_category in f
    ]
    assert len(data_split_json_files) == 1, f"Expected 1 {split_type} data split file"

    step_annotations_json = json.load(open(os.path.join(annotations_json_dir, "step_annotations.json"), 'r'))
    video_durations = pd.read_csv(os.path.join(dataset_dir, "metadata", "video_information.csv"))
    data_split_json = json.load(open(os.path.join(data_splits_dir, data_split_json_files[0]), 'r'))

    # Tutti i recording_id presenti nelle annotation
    all_ids = []
    for s in ["train", "val", "test"]:
        if s in data_split_json:
            all_ids.extend(data_split_json[s])

    total = len(all_ids)
    n_folds = total // val_size
    leftover = total % val_size

    print(f"Totale recording_id: {total}")
    print(f"Val size per fold:   {val_size}")
    print(f"Numero fold:         {n_folds}")
    if leftover > 0:
        print(f"Resto (non assegnati a nessun fold): {leftover} ID")

    # ------------------------------------------------------------------
    # Shuffle stratificato per ricetta
    # recipe_id = tutto prima dell'ultimo "_" (es. "1" da "1_7")
    # Dentro ogni ricetta, mescoliamo i video casualmente.
    # Poi distribuiamo round-robin sui fold → ogni fold ha video
    # di tutte le ricette in proporzione.
    # ------------------------------------------------------------------
    recipe_to_ids = defaultdict(list)
    for rid in all_ids:
        parts = rid.rsplit("_", 1)
        recipe_id = parts[0] if len(parts) > 1 else rid
        recipe_to_ids[recipe_id].append(rid)

    print(f"\nRicette trovate: {len(recipe_to_ids)}")
    for rid, vids in sorted(recipe_to_ids.items()):
        print(f"  Ricetta {rid}: {len(vids)} video")

    rng = random.Random(seed)
    slots = [[] for _ in range(n_folds)]
    for recipe_id, ids in sorted(recipe_to_ids.items()):
        shuffled = ids[:]
        rng.shuffle(shuffled)
        for i, vid in enumerate(shuffled):
            slots[i % n_folds].append(vid)

    # Appiattimento in ordine round-robin e taglio del leftover
    ordered = []
    for slot in slots:
        ordered.extend(slot)
    ordered = ordered[:n_folds * val_size]

    folds = [ordered[i * val_size:(i + 1) * val_size] for i in range(n_folds)]

    # Verifica: ogni ID compare esattamente una volta
    all_val_ids = [vid for fold in folds for vid in fold]
    assert len(all_val_ids) == len(set(all_val_ids)), "ERRORE: duplicati nei fold!"
    print(f"\nVerifica OK: ogni recording_id appare esattamente una volta in validation.")

    print(f"\nDistribuzione fold:")
    for i, fold in enumerate(folds):
        print(f"  Fold {i:02d}: {len(fold)} video in validation")

    # ------------------------------------------------------------------
    # Generazione JSON
    # ------------------------------------------------------------------
    output_base_dir = os.path.join(
        "captaincook_actionformer_annotations", video_category, f"{split_type}_KFold"
    )
    os.makedirs(output_base_dir, exist_ok=True)

    for idx, val_chunk in enumerate(folds):
        val_set = set(val_chunk)
        captaincook_dataset = {}

        for recording_id in all_ids:
            duration_val = video_durations[
                video_durations["recording_id"] == recording_id
            ]["duration(sec)"]
            if duration_val.empty:
                continue

            steps = step_annotations_json.get(recording_id, {}).get("steps", [])
            annotations = [{
                "label": s["description"],
                "segment": [float(s["start_time"]), float(s["end_time"])],
                "segment(frames)": [
                    np.floor(float(s["start_time"]) * fps),
                    np.ceil(float(s["end_time"]) * fps)
                ],
                "label_id": int(s["step_id"]),
                "has_error": bool(s["has_errors"])
            } for s in steps]

            subset = "validation" if recording_id in val_set else "training"

            captaincook_dataset[recording_id] = {
                "subset": subset,
                "duration": duration_val.values[0],
                "fps": fps,
                "annotations": annotations,
                "has_error": False,
            }

        json_path = os.path.join(output_base_dir, f"{split_type}_fold_{idx:02d}.json")
        with open(json_path, "w") as f:
            json.dump(
                {"version": captaincook_dataset_version, "database": captaincook_dataset},
                f, indent=4
            )

    print(f"\nGenerati {n_folds} fold in '{output_base_dir}'")


def generate_jsons():
    csv_to_json_kfold("recordings", "combined", val_size=10, seed=42)


if __name__ == '__main__':
    generate_jsons()