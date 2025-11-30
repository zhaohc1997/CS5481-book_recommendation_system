import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


class FeatureEngineer:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.label_encoders = {}

    def create_book_features(self, books: pd.DataFrame) -> pd.DataFrame:
        """创建图书特征"""
        # TF-IDF特征（书名）
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english'
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(
                books['Book-Title'].fillna('')
            )
        else:
            tfidf_features = self.tfidf_vectorizer.transform(
                books['Book-Title'].fillna('')
            )

        # 将TF-IDF特征转换为DataFrame
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'title_tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        tfidf_df.index = books.index

        # 编码分类特征
        books_encoded = books.copy()
        categorical_columns = ['Book-Author', 'Publisher']

        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                books_encoded[col] = self.label_encoders[col].fit_transform(
                    books[col].fillna('Unknown')
                )
            else:
                books_encoded[col] = self.label_encoders[col].transform(
                    books[col].fillna('Unknown')
                )

        # 合并所有特征
        book_features = pd.concat([books_encoded, tfidf_df], axis=1)
        return book_features

    def create_user_features(self, users: pd.DataFrame) -> pd.DataFrame:
        """创建用户特征"""
        user_features = users.copy()

        # 从位置信息提取国家
        user_features['Country'] = user_features['Location'].str.split(',').str[-1].str.strip()

        # 编码国家特征
        if 'Country' not in self.label_encoders:
            self.label_encoders['Country'] = LabelEncoder()
            user_features['Country_encoded'] = self.label_encoders['Country'].fit_transform(
                user_features['Country'].fillna('Unknown')
            )
        else:
            user_features['Country_encoded'] = self.label_encoders['Country'].transform(
                user_features['Country'].fillna('Unknown')
            )

        return user_features