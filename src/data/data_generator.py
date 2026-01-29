import os
import numpy as np
import pandas as pd

from src.config import DATA_DIR, ADOPTER_BIAS_SHIFT


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_dataset(
    n_samples: int = 8000,
    random_state: int = 42,
    save_csv: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates a local synthetic dataset with realistic signal.

    Returns:
    - base_df: base features plus adopter label (0/1)
    - delta_df: delta features plus engagement target for regression
    """
    rng = np.random.default_rng(random_state)

    songs = rng.integers(0, 8000, n_samples)
    loved = rng.integers(0, 1500, n_samples)
    playlists = rng.integers(0, 150, n_samples)
    posts = rng.integers(0, 400, n_samples)
    shouts = rng.integers(0, 250, n_samples)
    tenure = rng.integers(1, 3650, n_samples)
    friends = rng.integers(10, 700, n_samples)

    score = (
        0.0004 * songs
        + 0.0012 * loved
        + 0.0020 * playlists
        + 0.0010 * posts
        + 0.0008 * shouts
        + 0.0003 * tenure
        + 0.0004 * friends
        - ADOPTER_BIAS_SHIFT
    )

    probability = _sigmoid(score)
    adopter = (rng.random(n_samples) < probability).astype(int)

    base_df = pd.DataFrame(
        {
            "songsListened": songs,
            "lovedTracks": loved,
            "playlists": playlists,
            "posts": posts,
            "shouts": shouts,
            "tenure": tenure,
            "friends": friends,
            "adopter": adopter,
        }
    )

    adopter_shift = adopter.astype(float)

    delta_df = pd.DataFrame(
        {
            "delta_friend_cnt": (
                friends
                + rng.normal(0, 30, n_samples)
                + 25 * adopter_shift
            ).clip(0),
            "delta_songs": (
                songs
                + rng.normal(0, 500, n_samples)
                + 250 * adopter_shift
            ).clip(0),
            "delta_loved": (
                loved
                + rng.normal(0, 200, n_samples)
                + 100 * adopter_shift
            ).clip(0),
            "delta_posts": (
                posts
                + rng.normal(0, 50, n_samples)
                + 20 * adopter_shift
            ).clip(0),
            "delta_playlists": (
                playlists
                + rng.normal(0, 30, n_samples)
                + 10 * adopter_shift
            ).clip(0),
            "delta_shouts": (
                shouts
                + rng.normal(0, 40, n_samples)
                + 10 * adopter_shift
            ).clip(0),
        }
    )

    delta_df["engagement"] = (
        delta_df["delta_songs"]
        + delta_df["delta_loved"]
        + delta_df["delta_posts"]
        + delta_df["delta_playlists"]
        + delta_df["delta_shouts"]
        + 0.25 * delta_df["delta_friend_cnt"]
    ).astype(float)

    if save_csv:
        os.makedirs(DATA_DIR, exist_ok=True)
        merged_df = base_df.join(delta_df)
        merged_df.to_csv(
            os.path.join(DATA_DIR, "generated_data.csv"),
            index=False,
        )

    return base_df, delta_df
