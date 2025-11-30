# src/models/collaborative_filtering.py

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import pickle


class ItemBasedCF:
    def __init__(self, k=20):
        self.k = k
        self.item_similarity = None
        self.ratings_matrix = None
        self.books_df = None
        self.isbn_to_title = {}
        self.title_to_isbn = {}
        
    def train(self, ratings_df, books_df, users_df=None):
        """训练物品协同过滤模型"""
        print("开始训练物品协同过滤模型...")
        self.books_df = books_df.copy()
        
        # 创建用户-物品评分矩阵
        user_item_matrix = ratings_df.pivot_table(
            index='User-ID',
            columns='ISBN',
            values='Book-Rating',
            fill_value=0
        )
        
        self.ratings_matrix = user_item_matrix
        print(f"评分矩阵形状: {user_item_matrix.shape}")
        
        # 计算物品相似度
        print("计算物品相似度...")
        item_matrix = user_item_matrix.T
        self.item_similarity = cosine_similarity(item_matrix)
        
        # 创建ISBN和标题的映射
        for _, row in books_df.iterrows():
            isbn = row['ISBN']
            title = row['Book-Title']
            self.isbn_to_title[isbn] = title
            self.title_to_isbn[title] = isbn
        
        print(f"标题映射数量: {len(self.title_to_isbn)}")
        print(f"ISBN映射数量: {len(self.isbn_to_title)}")
        print("物品协同过滤模型训练完成")
    
    def recommend(self, book_title, n=5, **kwargs):
        """推荐相似图书"""
        # 查找图书ISBN
        if book_title not in self.title_to_isbn:
            # 尝试模糊匹配
            matched = self.books_df[
                self.books_df['Book-Title'].str.contains(book_title, case=False, na=False, regex=False)
            ]
            if matched.empty:
                # 降级：返回热门图书
                # print(f"[ItemBasedCF] 图书 '{book_title}' 不在评分矩阵中，返回热门图书")
                return self._get_popular_books(n)
            book_title = matched.iloc[0]['Book-Title']
        
        target_isbn = self.title_to_isbn.get(book_title)
        if not target_isbn:
            return self._get_popular_books(n)
        
        # 获取物品索引
        if target_isbn not in self.ratings_matrix.columns:
            # print(f"[ItemBasedCF] ISBN '{target_isbn}' 不在评分矩阵中，返回热门图书")
            return self._get_popular_books(n)
        
        item_idx = self.ratings_matrix.columns.get_loc(target_isbn)
        
        # 获取相似物品
        similarities = self.item_similarity[item_idx]
        similar_indices = np.argsort(similarities)[::-1][1:n+11]  # 多取一些
        
        # 生成推荐列表
        recommendations = []
        for idx in similar_indices:
            if len(recommendations) >= n:
                break
                
            isbn = self.ratings_matrix.columns[idx]
            if isbn in self.isbn_to_title:
                title = self.isbn_to_title[isbn]
                book_info = self.books_df[self.books_df['ISBN'] == isbn]
                
                if not book_info.empty:
                    recommendations.append({
                        'title': title,
                        'author': book_info.iloc[0].get('Book-Author', 'Unknown'),
                        'score': float(similarities[idx])
                    })
        
        return recommendations
    
    def _get_popular_books(self, n=5):
        """降级：返回热门图书"""
        if not hasattr(self, 'ratings_matrix') or self.ratings_matrix is None:
            return []
        
        # 计算每本书的评分数
        rating_counts = self.ratings_matrix.astype(bool).sum(axis=0).sort_values(ascending=False)
        
        recommendations = []
        for isbn in rating_counts.index[:n*2]:  # 多取一些
            if len(recommendations) >= n:
                break
                
            if isbn in self.isbn_to_title:
                title = self.isbn_to_title[isbn]
                book_info = self.books_df[self.books_df['ISBN'] == isbn]
                
                if not book_info.empty:
                    recommendations.append({
                        'title': title,
                        'author': book_info.iloc[0].get('Book-Author', 'Unknown'),
                        'score': float(rating_counts[isbn]) / float(rating_counts.max()) if rating_counts.max() > 0 else 0.5
                    })
        
        return recommendations
    
    def save_model(self, filepath):
        """保存模型"""
        model_data = {
            'k': self.k,
            'item_similarity': self.item_similarity,
            'ratings_matrix': self.ratings_matrix,
            'books_df': self.books_df,
            'isbn_to_title': self.isbn_to_title,
            'title_to_isbn': self.title_to_isbn
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"ItemBasedCF 模型已保存到 {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.k = model_data['k']
        self.item_similarity = model_data['item_similarity']
        self.ratings_matrix = model_data['ratings_matrix']
        self.books_df = model_data['books_df']
        self.isbn_to_title = model_data['isbn_to_title']
        self.title_to_isbn = model_data['title_to_isbn']
        print(f"ItemBasedCF 模型已从 {filepath} 加载")