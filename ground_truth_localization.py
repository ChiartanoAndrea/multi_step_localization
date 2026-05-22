import argparse
import json
from pathlib import Path
import numpy as np


def load_annotations(json_path: Path) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


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
    ann_path = Path(args.annotations_json)
    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_annotations(ann_path)

    for video_id, rec in annotations.items():
        steps = rec.get('steps', [])
        if not steps:
            continue

        features = load_npz_features(features_dir, video_id)
        if features is None:
            continue

        embeddings, segments, labels, scores = [], [], [], []

        for s in steps:
            st = float(s.get('start_time', -1.0))
            end = float(s.get('end_time', -1.0))
            step_id = s.get('step_id')

            # skip invalid timestamps
            if st < 0 or end < 0 or end <= st:
                continue

            emb = compute_step_embedding(features, st, end, args.segment_sec)
            embeddings.append(emb)
            segments.append((st, end))
            labels.append(int(step_id) if step_id is not None else -1)
            scores.append(1.0)

        if embeddings:
            save_step_embeddings(output_dir, np.array(embeddings), video_id, segments, labels, scores)
            print(f"Salvati {len(embeddings)} embedding ground-truth per {video_id}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ground-truth Step Localization -> embeddings (.npz)')
    parser.add_argument('--annotations_json', type=str,
                        default=r"annotations/annotation_json/step_annotations.json")
    parser.add_argument('--features_dir', type=str,
                        default=r"./data/features/perception_encoder/npz_features")
    parser.add_argument('--output_dir', type=str,
                        default=r".\output_Kfold_groundTruth")
    parser.add_argument('--segment_sec', type=float, default=1.0,
                        help='Durata in secondi di ogni segmento di feature')

    args = parser.parse_args()
    main(args)