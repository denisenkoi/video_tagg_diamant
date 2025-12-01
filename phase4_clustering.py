"""
Post-clustering for face detections using Agglomerative Clustering.

Reads embeddings from database, performs clustering, and saves results
to person_id_postclustering column.

Usage:
    python post_clustering.py [video_name]
    python post_clustering.py --all
"""

import sys
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from typing import Dict, List
from db_manager_postgres import PostgresDBManager


def run_postclustering(
    db: PostgresDBManager,
    video_name: str,
    distance_threshold: float = 0.3
) -> Dict[str, int]:
    """
    Run post-clustering on video face detections.

    Args:
        db: Database manager
        video_name: Video to process
        distance_threshold: Cosine distance threshold (0.3 = similarity 0.7)

    Returns:
        Stats dict with cluster counts
    """
    print(f"\n=== Post-clustering for {video_name} ===")

    # Get embeddings
    data = db.get_embeddings_for_clustering(video_name)
    if not data:
        print(f"No embeddings found for {video_name}")
        return {'clusters': 0, 'detections': 0}

    print(f"Loaded {len(data)} embeddings")

    # Extract IDs and embeddings
    detection_ids = [d['id'] for d in data]
    embeddings = np.array([d['embedding'] for d in data], dtype=np.float64)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = embeddings / norms

    # Compute cosine distances
    print("Computing distance matrix...")
    distances = cosine_distances(normalized)

    # Agglomerative clustering
    print(f"Running Agglomerative clustering (threshold={distance_threshold})...")
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='precomputed',
        linkage='average'
    )
    labels = clusterer.fit_predict(distances)

    n_clusters = len(set(labels))
    print(f"Found {n_clusters} clusters")

    # Build labels dict
    labels_dict: Dict[int, str] = {}
    for det_id, label in zip(detection_ids, labels):
        cluster_label = f"post_{label:02d}"
        labels_dict[det_id] = cluster_label

    # Save to database
    print("Saving to database...")
    updated = db.update_postclustering_labels(video_name, labels_dict)
    print(f"Updated {updated} records")

    return {
        'clusters': n_clusters,
        'detections': len(data)
    }


def main():
    """Main entry point"""
    db = PostgresDBManager()

    if len(sys.argv) < 2:
        print("Usage: python post_clustering.py <video_name>")
        print("       python post_clustering.py --all")
        sys.exit(1)

    if sys.argv[1] == '--all':
        # Process all videos
        videos = db.list_videos()
        total_clusters = 0
        total_detections = 0

        for video in videos:
            video_name = video['video_name']
            stats = run_postclustering(db, video_name)
            total_clusters += stats['clusters']
            total_detections += stats['detections']

        print(f"\n=== Summary ===")
        print(f"Videos processed: {len(videos)}")
        print(f"Total clusters: {total_clusters}")
        print(f"Total detections: {total_detections}")
    else:
        # Process single video
        video_name = sys.argv[1]
        run_postclustering(db, video_name)


if __name__ == "__main__":
    main()
