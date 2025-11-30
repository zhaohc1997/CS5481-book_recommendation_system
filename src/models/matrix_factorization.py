# src/models/matrix_factorization.py

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import pickle


class MatrixFactorization:
    """矩阵分解推荐算法（SVD）"""
    
    def __init__(self, n_factors=50):
        self.n_factors = n_factors
        self.model = None
        self.item_factors = None  # 图书的隐向量
        self.user_factors = None  # 用户的隐向量
        self.books_df = None
        self.isbn_to_idx = {}
        self.idx_to_isbn = {}
        self.ratings_matrix = None
        
    def train(self, ratings_df, books_df, users_df=None):
        """训练矩阵分解模型"""
        print("="*60)
        print("训练 Matrix Factorization 模型")
        print("="*60)
        
        self.books_df = books_df.copy()
        
        # 构建用户-图书评分矩阵
        print("  构建评分矩阵...")
        user_item_matrix = ratings_df.pivot_table(
            index='User-ID', 
            columns='ISBN', 
            values='Book-Rating', 
            fill_value=0
        )
        
        self.ratings_matrix = user_item_matrix
        
        print(f"  矩阵形状: {user_item_matrix.shape}")
        print(f"  稀疏度: {(user_item_matrix == 0).sum().sum() / user_item_matrix.size:.2%}")
        
        # SVD 分解
        print(f"  执行 SVD 分解 (n_factors={self.n_factors})...")
        self.model = TruncatedSVD(n_components=self.n_factors, random_state=42)
        
        # 分解：评分矩阵 ≈ 用户矩阵 × 图书矩阵
        self.user_factors = self.model.fit_transform(user_item_matrix)
        self.item_factors = self.model.components_.T  # (n_items, n_factors)
        
        # 构建 ISBN 索引
        for idx, isbn in enumerate(user_item_matrix.columns):
            self.isbn_to_idx[isbn] = idx
            self.idx_to_isbn[idx] = isbn
        
        print(f"✓ 训练完成:")
        print(f"  - 图书数: {len(self.idx_to_isbn):,}")
        print(f"  - 因子数: {self.n_factors}")
        print(f"  - 图书向量形状: {self.item_factors.shape}")
        print(f"  - 解释方差比: {self.model.explained_variance_ratio_.sum():.2%}")
        
        print("="*60)
    
    # ✅ 新增：获取所有图书的 Embedding
    def get_all_embeddings(self):
        """
        返回所有图书的 Embedding 字典
        
        返回:
            {isbn: vector} - 每个 ISBN 对应一个 n_factors 维的向量
        """
        embeddings = {}
        
        if self.item_factors is None:
            print("  ⚠️  模型未训练，无法获取 Embedding")
            return embeddings
        
        for isbn, idx in self.isbn_to_idx.items():
            embeddings[isbn] = self.item_factors[idx]
        
        return embeddings
    
    # ✅ 新增：获取单本书的 Embedding
    def get_embedding(self, isbn):
        """获取单本书的 Embedding 向量"""
        if isbn not in self.isbn_to_idx:
            return None
        
        idx = self.isbn_to_idx[isbn]
        return self.item_factors[idx]
    
    def recommend(self, book_title, n=10):
        """基于矩阵分解的推荐"""
        if self.item_factors is None:
            return []
        
        # 查找图书
        book_data = self.books_df[self.books_df['Book-Title'] == book_title]
        if book_data.empty:
            return self._get_popular_books(n)
        
        target_isbn = book_data.iloc[0]['ISBN']
        
        if target_isbn not in self.isbn_to_idx:
            return self._get_popular_books(n)
        
        # 获取目标图书的向量
        target_idx = self.isbn_to_idx[target_isbn]
        target_vector = self.item_factors[target_idx].reshape(1, -1)
        
        # 计算与所有图书的相似度
        similarities = cosine_similarity(target_vector, self.item_factors)[0]
        
        # 排序（排除自己）
        similar_indices = np.argsort(similarities)[::-1][1:n+1]
        
        # 构建推荐列表
        recommendations = []
        for idx in similar_indices:
            isbn = self.idx_to_isbn[idx]
            book_info = self.books_df[self.books_df['ISBN'] == isbn]
            
            if not book_info.empty:
                recommendations.append({
                    'title': book_info.iloc[0]['Book-Title'],
                    'author': book_info.iloc[0].get('Book-Author', 'Unknown'),
                    'score': float(similarities[idx])
                })
        
        return recommendations
    
    def _get_popular_books(self, n):
        """降级：返回热门图书"""
        if self.ratings_matrix is None:
            return []
        
        # 计算每本书的评分数
        popularity = self.ratings_matrix.sum(axis=0).sort_values(ascending=False)
        
        recommendations = []
        for isbn in popularity.head(n).index:
            book_info = self.books_df[self.books_df['ISBN'] == isbn]
            if not book_info.empty:
                recommendations.append({
                    'title': book_info.iloc[0]['Book-Title'],
                    'author': book_info.iloc[0].get('Book-Author', 'Unknown'),
                    'score': 0.0
                })
        
        return recommendations
    
    def save_model(self, filepath):
        """保存模型"""
        model_data = {
            'model': self.model,
            'item_factors': self.item_factors,
            'user_factors': self.user_factors,
            'isbn_to_idx': self.isbn_to_idx,
            'idx_to_isbn': self.idx_to_isbn,
            'ratings_matrix': self.ratings_matrix,
            'books_df': self.books_df,
            'n_factors': self.n_factors
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"MatrixFactorization 模型已保存到 {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.item_factors = model_data['item_factors']
        self.user_factors = model_data.get('user_factors')
        self.isbn_to_idx = model_data['isbn_to_idx']
        self.idx_to_isbn = model_data['idx_to_isbn']
        self.ratings_matrix = model_data.get('ratings_matrix')
        self.books_df = model_data['books_df']
        self.n_factors = model_data.get('n_factors', 50)
        
        print(f"MatrixFactorization 模型已从 {filepath} 加载")