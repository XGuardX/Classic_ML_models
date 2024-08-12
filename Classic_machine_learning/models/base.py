from abc import ABC, abstractmethod


class BaseEstimator(ABC):
    @abstractmethod
    def fit(self, X, y):
        """Обучение модели на данных X и y"""
        pass

    def predict(self, X):
        """Делает предсказания на данных X"""
        pass

    def predict_proba(self, X):
        """Делает предсказания 0 или 1 на данных X"""
        pass
