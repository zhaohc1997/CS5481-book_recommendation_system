import pandas as pd
import numpy as np


class DataCleaner:
    def __init__(self):
        pass

    def clean_books_data(self, books: pd.DataFrame) -> pd.DataFrame:
        """清洗图书数据"""
        books_clean = books.copy()

        # 处理出版年份异常值
        books_clean['Year-Of-Publication'] = pd.to_numeric(books_clean['Year-Of-Publication'], errors='coerce')
        books_clean = books_clean[books_clean['Year-Of-Publication'] <= 2023]
        books_clean = books_clean[books_clean['Year-Of-Publication'] >= 1800]

        # 处理出版社缺失值 - 使用loc避免SettingWithCopyWarning
        books_clean.loc[:, 'Publisher'] = books_clean['Publisher'].fillna('Unknown')

        # 处理作者缺失值
        books_clean.loc[:, 'Book-Author'] = books_clean['Book-Author'].fillna('Unknown')

        # 处理ISBN格式问题
        books_clean = books_clean[books_clean['ISBN'].notna()]
        books_clean = books_clean[books_clean['ISBN'].str.len() >= 10]

        return books_clean

    def clean_users_data(self, users: pd.DataFrame) -> pd.DataFrame:
        """清洗用户数据"""
        users_clean = users.copy()

        # 处理年龄异常值
        users_clean = users_clean[(users_clean['Age'] >= 5) & (users_clean['Age'] <= 100)]

        # 处理位置信息
        users_clean.loc[:, 'Location'] = users_clean['Location'].fillna('Unknown')

        return users_clean

    def clean_ratings_data(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """清洗评分数据"""
        ratings_clean = ratings.copy()

        # 移除评分为0的数据（隐式反馈）
        ratings_clean = ratings_clean[ratings_clean['Book-Rating'] > 0]

        # 移除缺失用户ID或ISBN的数据
        ratings_clean = ratings_clean[ratings_clean['User-ID'].notna()]
        ratings_clean = ratings_clean[ratings_clean['ISBN'].notna()]

        return ratings_clean

    def filter_sparse_data(self, ratings: pd.DataFrame,
                           min_user_ratings: int = 5,
                           min_book_ratings: int = 10) -> pd.DataFrame:
        """过滤稀疏数据"""
        # 过滤评分数量少的用户
        user_rating_counts = ratings['User-ID'].value_counts()
        active_users = user_rating_counts[user_rating_counts >= min_user_ratings].index
        ratings = ratings[ratings['User-ID'].isin(active_users)]

        # 过滤评分数量少的图书
        book_rating_counts = ratings['ISBN'].value_counts()
        popular_books = book_rating_counts[book_rating_counts >= min_book_ratings].index
        ratings = ratings[ratings['ISBN'].isin(popular_books)]

        return ratings