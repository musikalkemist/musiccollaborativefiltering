from pathlib import Path

import implicit

from musiccollaborativefiltering.data import load_user_artists


class Recommender:
    """The Recommender class computes recommendations for a given user."""

    def __init__(self):
        self._model = None

    def fit(self, user_artists_matrix: implicit.als.AlternatingLeastSquares):
        """Fit the model to the user artists matrix."""
        self._model = user_artists_matrix

    def recommend(self, user_id: int, n: int = 10) -> list:
        """Return the top n recommendations for the given user."""
        recommendations = self._model.recommend(user_id, user_artists_matrix)
        return recommendations


if __name__ == "__main__":
    user_artists_sparse_matrix = load_user_artists(Path(
        "../lastfmdata/user_artists.dat"))

    model = implicit.als.AlternatingLeastSquares(factors=100, iterations=15)
    model.fit(user_artists_sparse_matrix)
    user_id = 1002
    ids, scores = model.recommend(user_id, user_artists_sparse_matrix[user_id])

    print(ids)
    print(scores)