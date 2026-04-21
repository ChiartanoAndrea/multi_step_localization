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
    pkl_files = glob.glob(str(pkl_dir / "*.pkl"))
    
    if not pkl_files:
        raise FileNotFoundError(f"Nessun file .pkl trovato in {pkl_dir}")
    
    print(f"Trovati {len(pkl_files)} file di risultati. Inizio accorpamento...")
    
    aggregated_predictions = defaultdict(list)
    
    for pkl_path in pkl_files:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        required_keys = ['video-id', 't-start', 't-end', 'label', 'score']
        for key in required_keys:
            if key not in data:
                print(f"Attenzione: Chiave '{key}' mancante nel file {pkl_path}. Salto il file.")
                continue
        
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


def non_max_suppression(steps: list, iou_threshold: float) -> list:
    """
    Rimuove step sovrapposti tenendo solo quello con score più alto.
    steps: lista di tuple (t_start, t_end, label, score).
    iou_threshold: soglia IoU sopra la quale due step si considerano sovrapposti.
                   0.0 = qualsiasi sovrapposizione viene soppressa,
                   1.0 = NMS di fatto disabilitata.

    Logica: ogni candidato (ordinato per score desc) viene accettato solo se
    NON si sovrappone oltre la soglia con nessuno degli step già tenuti.
    """
    if not steps:
        return []

    # Ordina per score decrescente: il migliore viene valutato per primo
    steps_by_score = sorted(steps, key=lambda x: x[3], reverse=True)

    kept = []

    for st_i, end_i, label_i, score_i in steps_by_score:
        overlaps = False
        for st_k, end_k, _, _ in kept:
            inter = max(0.0, min(end_i, end_k) - max(st_i, st_k))
            union = (end_i - st_i) + (end_k - st_k) - inter
            iou   = inter / union if union > 0 else 0.0
            if iou >= iou_threshold:
                overlaps = True
                break

        if not overlaps:
            kept.append((st_i, end_i, label_i, score_i))

    # Riordina per tempo crescente prima di restituire
    return sorted(kept, key=lambda x: (x[0], x[1]))


def load_npz_features(feature_dir: Path, video_id: str) -> np.ndarray:
    npz_path = feature_dir / f"{video_id}_360p.mp4_1s_1s.npz"
    if not npz_path.exists():
        print(f"Feature non trovate per video: {video_id} al percorso {npz_path}")
        return None

    with np.load(npz_path) as data:
        if "arr_0" in data:
            return data["arr_0"]
        return None


def compute_step_embedding(features: np.ndarray, t_start: float, t_end: float, segment_sec: float) -> np.ndarray:
    start_i = int(np.floor(t_start / segment_sec))
    end_i   = int(np.ceil(t_end   / segment_sec))
    
    T = features.shape[0]
    start_i = max(0, min(start_i, T - 1))
    end_i   = max(start_i + 1, min(end_i, T))
    
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
    pkl_dir      = Path(args.pkl_dir)
    features_dir = Path(args.features_dir)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Caricamento accorpato
    all_predictions = load_all_pkls(pkl_dir)
    
    # 2. Filtraggio + NMS
    ordered_prediction = {}
    for video_id, row in all_predictions.items():
        filtered = filter_prediction(row, args.score_threshold, args.min_step_duration)
        if filtered:
            filtered = non_max_suppression(filtered, iou_threshold=args.nms_iou_threshold)
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
    
    parser.add_argument('--pkl_dir', type=str, required=True,
                        help='Directory contenente i file eval_results.pkl del K-Fold')
    parser.add_argument('--features_dir', type=str, 
                        default=r".\data\features\perception_encoder\npz_features")
    parser.add_argument('--output_dir', type=str, 
                        default=r".\output_step_embeddings\ego4d\perception_encoder")
    
    # Iperparametri di filtraggio
    parser.add_argument('--score_threshold', type=float, default=0.03,
                        help='Score minimo per mantenere una predizione')
    parser.add_argument('--min_step_duration', type=float, default=1.0,
                        help='Durata minima in secondi per uno step')
    parser.add_argument('--segment_sec', type=float, default=1.0,
                        help='Durata in secondi di ogni segmento di feature')

    # NMS
    parser.add_argument('--nms_iou_threshold', type=float, default=0.5,
                        help=(
                            'Soglia IoU per la Non-Maximum Suppression. '
                            'Step sovrapposti oltre questa soglia vengono eliminati '
                            'tenendo solo quello con score più alto. '
                            '0.0 = sopprime qualsiasi sovrapposizione, '
                            '1.0 = NMS disabilitata (default: 0.5)'
                        ))
    
    args = parser.parse_args()
    main(args)