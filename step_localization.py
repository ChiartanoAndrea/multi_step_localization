import argparse
import pickle
import pprint
from pathlib import Path
import numpy as np
from sklearn.base import defaultdict


# label del eval_results.pkl:
# - video_ids: list of video ids
# - t-starts: list of start times for each step
# - t-ends: list of end times for each step
# - label : listo of the step-id associated with video_ids, t-starts and t-ends
# - scores: list of confidence scores for each step prediction



def load_pkl(pkl_path:Path):
    """Load the evaluation results from a pickle file and organize them into a dictionary format."""

    with open(pkl_path, 'rb') as f:
        data= pickle.load(f)
    """
    print("Type of data:", type(data))

    if isinstance(data, dict):
        print("Keys in the dictionary:", data.keys())
        for k,v in data.items():
            print(f"Key: {k}, Value Type: {type(v)}")
            if isinstance(v, dict):
                print(f"  Sub-keys in '{k}': {v.keys()}")
                for sub_k, sub_v in v.items():
                    print(f"    Sub-key: {sub_k}, Sub-value Type: {type(sub_v)}")
                    if isinstance(sub_v, dict):
                        print(f"      Sub-sub-keys in '{sub_k}': {sub_v.keys()}")

    for k,v in data.items():
        print(f"KEY: {k}")
        if isinstance(v, np.ndarray):
            print(f"Value is a numpy array with shape: {v.shape}")
            print(f"Data type of elements: {v.dtype}")
            print(f"First 5 elements: {v[:5]}")

        elif isinstance(v, list):
            print(f"Value is a list with length: {len(v)}")
            print(f"Elements type: {type(v[0]) if len(v) > 0 else 'N/A'}")
            print(f"First 5 elements: {v[:5]}")
    """

    required_keys = ['video-id', 't-start', 't-end', 'label', 'score']
    for key in required_keys:
        if key not in data:
            print(f"Key '{key}' is missing from the data.")
    
    # Use defaultdict to automatically create a list when a new video_id is seen
    prediction = defaultdict(list)
    
    for video_id, t_start, t_end, label, score in zip(data['video-id'], data['t-start'], data['t-end'], data['label'], data['score']):
        
        # create a row for each video id, and append the corresponding t_start, t_end, label and score
        #  for each step in the video to the row
        prediction[str(video_id)].append({
            't_start': t_start,
            't_end': t_end,
            'label': label,
            'score': score
        })


    #print(prediction["1_20"])
    return prediction

def filter_prediction(row: list, score_threshold: float, min_step_duration: float)-> list:
    """
    Filter the prediction for a single video based on score threshold and minimum step duration.
    
    Args:
        row (list): A list of dictionaries, where each dictionary contains 't_start', 't_end', 'label', and 'score' for a step.
        score_threshold (float): The minimum confidence score required to keep a step.
        min_step_duration (float): The minimum duration (in seconds) required to keep a step.
    """
    filtered_steps = []
    for step in row:
        st, end, label, score = step['t_start'], step['t_end'], step['label'], step['score']
        #print(f"Processing step with start time {st}, end time {end}, label {label}, and score {score}")
        if score >= score_threshold and (end - st) >= min_step_duration:
            filtered_steps.append((st, end, label, score))
    ordered_step=sorted(filtered_steps, key=lambda x: (x[0], x[1]))
    return ordered_step

def load_npz_features(feature_dir: Path, video_id: str)-> np.ndarray:
    """Load the features for a specific video from a .npz file."""
    npz_path = feature_dir / f"{video_id}_360p.mp4_1s_1s.npz"
    print(f"Loading features from: {npz_path}")

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

def compute_step_embedding(features: np.ndarray, t_start: float, t_end: float, segment_sec: float)-> np.ndarray:
    # Calculate the start and end indices for the feature segment based on the provided time range and segment duration
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

def save_step_embeddings( output_dir: Path, step_embeddings: np.ndarray, video_id:str, segments: list, labels: list, scores: list):
    """Save the computed step embeddings to a .npz file."""
    output_path = output_dir / f"{video_id}.npz"
    print(f"Saving step embeddings to: {output_path}")
    np.savez(output_path, 
             step_embedding=step_embeddings, #(N,D) number of steps, dimension of the embedding
             segments=segments, #(N,2) start and end time of each step
             label=labels, #(N,) label of each step
             score=scores) #(N,) confidence score of each step


def main(args):

    path= ".\output_actionFormer\ego4d\perception_encoder_recordings_reproduce\eval_results.pkl"
    output_dir = ".\output_step_embeddings\ego4d\perception_encoder"
    segment_sec = 1/ 1.875
    pred= load_pkl(path)
    ordered_prediction = defaultdict(list)
    for video_id, row in pred.items():
        #print(video_id)
        filtered_steps = filter_prediction(row, score_threshold=0.0, min_step_duration=2.0)
        ordered_prediction[video_id] = filtered_steps
    #print(ordered_prediction["1_20"])
    feature_dict= defaultdict(list)
    for video_id in ordered_prediction.keys():
        features=load_npz_features(Path(r".\data\features\perception_encoder\npz_features"), video_id)
        feature_dict[video_id] = features
    #print(feature_dict["1_20"])

    for video_id, steps in ordered_prediction.items():
        features= feature_dict[video_id]
        segments=[]
        labels=[]
        scores=[]
        embeddings=[]
        for step in steps:
            st, end, label, score = step
            step_embedding = compute_step_embedding(features, st, end, segment_sec)
            #print(f"Video ID: {video_id}, Step Label: {label}, Step Embedding Shape: {step_embedding.shape}")
            embeddings.append(step_embedding)
            segments.append((st, end))
            labels.append(label)
            scores.append(score)
        save_step_embeddings(Path(output_dir), np.array(embeddings), video_id, segments, labels, scores)



if __name__ == '__main__':
    """Entry Point"""
    # the arg parser Path pkl, score_threshold, min_step_duration, path features, path output , segment_sec default=1/1.875
    parser = argparse.ArgumentParser(
        description='')
   
    parser.add_argument('--stride', default=30, type=int, )
    args = parser.parse_args()
    main(args)