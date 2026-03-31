import argparse
import pickle
import pprint
from pathlib import Path
import numpy as np
from collections import defaultdict # Corretto da sklearn.base a collections


# label del eval_results.pkl:
# - video_ids: list of video ids
# - t-starts: list of start times for each step
# - t-ends: list of end times for each step
# - label : listo of the step-id associated with video_ids, t-starts and t-ends
# - scores: list of confidence scores for each step prediction

def load_pkl(pkl_path: Path):
    """Load the evaluation results from a pickle file and organize them into a dictionary format."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    required_keys = ['video-id', 't-start', 't-end', 'label', 'score']
    for key in required_keys:
        if key not in data:
            print(f"Key '{key}' is missing from the data.")
    
    # Use defaultdict to automatically create a list when a new video_id is seen
    prediction = defaultdict(list)
    
    for video_id, t_start, t_end, label, score in zip(data['video-id'], data['t-start'], data['t-end'], data['label'], data['score']):
        prediction[str(video_id)].append({
            't_start': t_start,
            't_end': t_end,
            'label': label,
            'score': score
        })

    return prediction

def filter_prediction(row: list, score_threshold: float, min_step_duration: float) -> list:
    """
    Filter the prediction for a single video based on score threshold and minimum step duration.
    """
    filtered_steps = []
    for step in row:
        st, end, label, score = step['t_start'], step['t_end'], step['label'], step['score']
        if score >= score_threshold and (end - st) >= min_step_duration:
            filtered_steps.append((st, end, label, score))
    ordered_step = sorted(filtered_steps, key=lambda x: (x[0], x[1]))
    return ordered_step

def load_npz_features(feature_dir: Path, video_id: str) -> np.ndarray:
    """Load the features for a specific video from a .npz file."""
    npz_path = feature_dir / f"{video_id}_360p.mp4_1s_1s.npz"

    with np.load(npz_path) as data:
        if "arr_0" in data:
            print(f"Loaded features for video ID '{video_id}' with shape {data['arr_0'].shape}.")
            return data["arr_0"]
        else:
            print(f"Video ID '{video_id}' not found in the .npz file.")
            return None

def feature_index_from_time(start:float, end:float, segment_sec: float)-> tuple:
    start_index = int(np.floor(start / segment_sec)) #round down to the left, so that we don't miss any step that starts in the middle of a segment
    end_index = int(np.ceil(end / segment_sec)) # round up to the right, so that we don't miss any step that ends in the middle of a segment
    return start_index, end_index

def compute_step_embedding(features: np.ndarray, t_start: float, t_end: float, segment_sec: float) -> np.ndarray:
    start_i, end_i = feature_index_from_time(t_start, t_end, segment_sec)

    # total number of segments in the video
    T= features.shape[0]
    #print(f"Total number of segments in the video: {T}")

    # Ensure that the calculated indices are within the bounds of the feature array
    start_i = max(0, min(start_i, T - 1)) # control that start_i is not negative and does not exceed the last index of the feature array
    end_i = max(start_i + 1, min(end_i,T)) # control that end_i is greater than start_i and does not exceed the length of the feature array
    
    #Taking the features in the range
    step_features = features[start_i:end_i] # (K,D) with K number of segments in the step

    #compute the mean for the step
    step_embedding = np.mean(step_features, axis=0) # (D,)
    return step_embedding

def save_step_embeddings(output_dir: Path, step_embeddings: np.ndarray, video_id: str, segments: list, labels: list, scores: list):
    """Save the computed step embeddings to a .npz file."""
    output_path = output_dir / f"{video_id}.npz"
    np.savez(output_path, 
             step_embedding=step_embeddings, #(N,D) number of steps, dimension of the embedding
             segments=segments, #(N,2) start and end time of each step
             label=labels, #(N,) label of each step
             score=scores) #(N,) confidence score of each step


def main(args):
    # Cast paths to pathlib.Path objects
    pkl_path = Path(args.pkl_path)
    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    pred = load_pkl(pkl_path)
    ordered_prediction = defaultdict(list)
    
    for video_id, row in pred.items():
        print(f"Processing video: {video_id}")
        filtered_steps = filter_prediction(
            row, 
            score_threshold=args.score_threshold, 
            min_step_duration=args.min_step_duration
        )
        ordered_prediction[video_id] = filtered_steps

    feature_dict = defaultdict(list)
    for video_id in ordered_prediction.keys():
        features = load_npz_features(features_dir, video_id)
        # Salviamo le feature solo se sono state trovate
        if features is not None:
            feature_dict[video_id] = features

    for video_id, steps in ordered_prediction.items():
        # Saltiamo il video se non abbiamo trovato le sue features nel passaggio precedente
        if video_id not in feature_dict:
            continue
            
        features = feature_dict[video_id]
        segments = []
        labels = []
        scores = []
        embeddings = []
        
        for step in steps:
            st, end, label, score = step
            step_embedding = compute_step_embedding(features, st, end, args.segment_sec)
            embeddings.append(step_embedding)
            segments.append((st, end))
            labels.append(label)
            scores.append(score)
            
        if embeddings: # Salva solo se c'è almeno un embedding
            save_step_embeddings(output_dir, np.array(embeddings), video_id, segments, labels, scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Step Localization Filtering and Embedding Extraction')
    
    # Percorsi file e cartelle
    parser.add_argument('--pkl_path', type=str, 
                        default=r".\output_actionFormer\ego4d\perception_encoder_recordings_reproduce\eval_results.pkl",
                        help='Percorso del file eval_results.pkl')
    parser.add_argument('--features_dir', type=str, 
                        default=r".\data\features\perception_encoder\npz_features",
                        help='Cartella contenente i file di feature .npz')
    parser.add_argument('--output_dir', type=str, 
                        default=r".\output_step_embeddings\ego4d\perception_encoder",
                        help='Cartella di destinazione per gli embedding calcolati')
    
    # Iperparametri
    parser.add_argument('--score_threshold', type=float, default=0.03, 
                        help='Score di confidenza minimo per mantenere lo step')
    parser.add_argument('--min_step_duration', type=float, default=1.0, 
                        help='Durata minima (in secondi) per mantenere lo step')
    parser.add_argument('--segment_sec', type=float, default=1/1.875, 
                        help='Durata in secondi di un singolo segmento feature')
    parser.add_argument('--stride', default=30, type=int, 
                        help='Parametro stride')

    args = parser.parse_args()
    main(args)