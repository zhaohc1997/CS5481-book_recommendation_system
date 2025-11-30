import pandas as pd
import numpy as np
from typing import Tuple, Dict


class DataLoader:
    def __init__(self, data_path: str = "../data/raw/"):
        self.data_path = data_path

    def load_books_data(self) -> pd.DataFrame:
        """加载图书数据"""
        books = pd.read_csv(f"{self.data_path}/Books.csv")
        return books

    def load_users_data(self) -> pd.DataFrame:
        """加载用户数据"""
        users = pd.read_csv(f"{self.data_path}/Users.csv")
        return users

    def load_ratings_data(self) -> pd.DataFrame:
        """加载评分数据"""
        ratings = pd.read_csv(f"{self.data_path}/Ratings.csv")
        return ratings

    def get_dataset_info(self) -> Dict:
        """获取数据集基本信息"""
        books = self.load_books_data()
        users = self.load_users_data()
        ratings = self.load_ratings_data()

        info = {
            "books_shape": books.shape,
            "users_shape": users.shape,
            "ratings_shape": ratings.shape,
            "books_columns": books.columns.tolist(),
            "users_columns": users.columns.tolist(),
            "ratings_columns": ratings.columns.tolist()
        }
        return info

    def check_missing_values(self) -> pd.DataFrame:
        """检查缺失值"""
        books = self.load_books_data()
        users = self.load_users_data()
        ratings = self.load_ratings_data()

        missing_info = {}
        for name, df in [("books", books), ("users", users), ("ratings", ratings)]:
            missing_info[name] = df.isnull().sum()

        return pd.DataFrame(missing_info)