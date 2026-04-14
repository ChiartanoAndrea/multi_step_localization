import argparse
import pickle
import pprint
from pathlib import Path
import numpy as np
from collections import defaultdict
import glob
import os

# - video_ids: list of video ids
# - t-starts: list of start times for each step
# - t-ends: list of end times for each step
# - label : listo of the step-id associated with video_ids, t-starts and t-ends
# - scores: list of confidence scores for each step prediction

def load_all_pkls(pkl_dir: Path):
    """
    Scansiona la directory, carica tutti i file .pkl e unisce le predizioni.
    """
    # Cerca file .pkl o .pkl.tar (per compatibilità)
    pkl_files = glob.glob(str(pkl_dir / "*.pkl"))
    
    if not pkl_files:
        raise FileNotFoundError(f"Nessun file .pkl trovato in {pkl_dir}")
    
    print(f"Trovati {len(pkl_files)} file di risultati. Inizio accorpamento...")
    
    # Usiamo defaultdict per unire tutto
    aggregated_predictions = defaultdict(list)
    
    for pkl_path in pkl_files:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Verifica chiavi necessarie
        required_keys = ['video-id', 't-start', 't-end', 'label', 'score']
        for key in required_keys:
            if key not in data:
                print(f"Attenzione: Chiave '{key}' mancante nel file {pkl_path}. Salto il file.")
                continue
        
        # Unione dei dati
        for video_id, t_start, t_end, label, score in zip(
            data['video-id'], data['t-start'], data['t-end'], data['label'], data['score']
        ):
            aggregated_predictions[str(video_id)].append({
                't_start': t_start,
                't_end': t_end,
                'label': label,
                'score': score
            })
            
    print(f"Accorpamento completato. Totale video processati: {len(aggregated_predictions)}")
    return aggregated_predictions



def filter_prediction(row: list, score_threshold: float, min_step_duration: float) -> list:
    filtered_steps = []
    for step in row:
        st, end, label, score = step['t_start'], step['t_end'], step['label'], step['score']
        if score >= score_threshold and (end - st) >= min_step_duration:
            filtered_steps.append((st, end, label, score))
    ordered_step = sorted(filtered_steps, key=lambda x: (x[0], x[1]))
    return ordered_step

def load_npz_features(feature_dir: Path, video_id: str) -> np.ndarray:
    # Nota: Usiamo il pattern del file di feature caricato precedentemente
    npz_path = feature_dir / f"{video_id}_360p.mp4_1s_1s.npz"
    if not npz_path.exists():
        print(f"Feature non trovate per video: {video_id} al percorso {npz_path}")
        return None

    with np.load(npz_path) as data:
        if "arr_0" in data:
            return data["arr_0"]
        return None

def compute_step_embedding(features: np.ndarray, t_start: float, t_end: float, segment_sec: float) -> np.ndarray:
    # Calcolo indici basato sul tempo
    start_i = int(np.floor(t_start / segment_sec))
    end_i = int(np.ceil(t_end / segment_sec))
    
    T = features.shape[0]
    start_i = max(0, min(start_i, T - 1))
    end_i = max(start_i + 1, min(end_i, T))
    
    step_features = features[start_i:end_i]
    return np.mean(step_features, axis=0)

def save_step_embeddings(output_dir: Path, step_embeddings: np.ndarray, video_id: str, segments: list, labels: list, scores: list):
    output_path = output_dir / f"{video_id}.npz"
    np.savez(output_path, 
             step_embedding=step_embeddings,
             segments=segments,
             label=labels,
             score=scores)

def main(args):
    # Setup percorsi
    pkl_dir = Path(args.pkl_dir)
    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Caricamento accorpato (MODIFICATO)
    all_predictions = load_all_pkls(pkl_dir)
    
    # 2. Filtraggio
    ordered_prediction = {}
    for video_id, row in all_predictions.items():
        filtered = filter_prediction(row, args.score_threshold, args.min_step_duration)
        if filtered:
            ordered_prediction[video_id] = filtered

    # 3. Estrazione Embedding
    for video_id, steps in ordered_prediction.items():
        print(f"Processing features for video: {video_id}")
        features = load_npz_features(features_dir, video_id)
        
        if features is None:
            continue
            
        embeddings, segments, labels, scores = [], [], [], []
        
        for st, end, label, score in steps:
            step_emb = compute_step_embedding(features, st, end, args.segment_sec)
            embeddings.append(step_emb)
            segments.append((st, end))
            labels.append(label)
            scores.append(score)
            
        if embeddings:
            save_step_embeddings(output_dir, np.array(embeddings), video_id, segments, labels, scores)
            print(f"Salvati {len(embeddings)} embedding per {video_id}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Step Localization Filtering and Embedding Extraction (K-Fold version)')
    
    # Cambiato da --pkl_path a --pkl_dir
    parser.add_argument('--pkl_dir', type=str, required=True,
                        help='Directory contenente i file eval_results.pkl del K-Fold')
    parser.add_argument('--features_dir', type=str, 
                        default=r".\data\features\perception_encoder\npz_features")
    parser.add_argument('--output_dir', type=str, 
                        default=r".\output_step_embeddings\ego4d\perception_encoder")
    
    # Iperparametri
    parser.add_argument('--score_threshold', type=float, default=0.03)
    parser.add_argument('--min_step_duration', type=float, default=1.0)
    parser.add_argument('--segment_sec', type=float, default=1/1.875)
    
    args = parser.parse_args()
    main(args)