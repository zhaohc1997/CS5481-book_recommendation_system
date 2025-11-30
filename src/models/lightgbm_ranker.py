# src/models/lightgbm_ranker.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
from collections import defaultdict


class LightGBMRanker:
    """LightGBM 排序模型 - 支持 Embedding 特征与全量打分（增强版）"""
    
    def __init__(self, n_estimators=100, learning_rate=0.05):
        self.model = None
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.books_df = None
        self.ratings_df = None
        
        # 特征缓存
        self.book_features = {}
        self.isbn_stats = {}
        self.cooccurrence = {}
        
        # 作者和出版商编码
        self.author_to_id = {}
        self.publisher_to_id = {}
        
        # Embedding 向量缓存
        self.embeddings = {} 
    
    def set_embeddings(self, embedding_dict):
        """
        注入预训练的 Embedding
        参数: embedding_dict: {isbn: vector} 字典
        """
        self.embeddings = embedding_dict
        print(f"  ✓ LightGBM 已加载 {len(self.embeddings):,} 个 Embedding 向量")
        
        if self.embeddings:
            # 检查向量维度
            sample_isbn = list(self.embeddings.keys())[0]
            sample_vector = self.embeddings[sample_isbn]
            vector_dim = len(sample_vector)
            print(f"  ✓ Embedding 维度: {vector_dim}")
    
    def _calculate_embedding_similarity(self, isbn_a, isbn_b):
        """
        计算两本书的向量相似度 (Cosine Similarity)
        返回: float 相似度分数 [-1, 1]
        """
        if not self.embeddings:
            return 0.0
        
        vec_a = self.embeddings.get(isbn_a)
        vec_b = self.embeddings.get(isbn_b)
        
        if vec_a is None or vec_b is None:
            return 0.0
        
        try:
            # Cosine Similarity: (A . B) / (|A| * |B|)
            dot_product = np.dot(vec_a, vec_b)
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return dot_product / (norm_a * norm_b)
        except:
            return 0.0

    def _compute_book_features(self, books_df, ratings_df):
        """为每本书计算统计特征"""
        print("  计算图书特征...")
        
        # 评分统计
        rating_stats = ratings_df[ratings_df['Book-Rating'] > 0].groupby('ISBN').agg({
            'Book-Rating': ['mean', 'count', 'std', 'min', 'max'],
            'User-ID': 'nunique'
        }).reset_index()
        
        rating_stats.columns = ['ISBN', 'avg_rating', 'rating_count', 'rating_std', 
                                'min_rating', 'max_rating', 'unique_users']
        rating_stats['rating_std'] = rating_stats['rating_std'].fillna(0)
        
        # 合并到图书数据
        books_with_stats = books_df.merge(rating_stats, on='ISBN', how='left')
        
        # 填充缺失值
        global_avg_rating = ratings_df[ratings_df['Book-Rating'] > 0]['Book-Rating'].mean()
        books_with_stats['avg_rating'] = books_with_stats['avg_rating'].fillna(global_avg_rating)
        books_with_stats['rating_count'] = books_with_stats['rating_count'].fillna(0)
        books_with_stats['rating_std'] = books_with_stats['rating_std'].fillna(0)
        books_with_stats['unique_users'] = books_with_stats['unique_users'].fillna(0)
        books_with_stats['min_rating'] = books_with_stats['min_rating'].fillna(0)
        books_with_stats['max_rating'] = books_with_stats['max_rating'].fillna(0)
        
        # 构建作者和出版商编码
        unique_authors = books_with_stats['Book-Author'].unique()
        unique_publishers = books_with_stats['Publisher'].unique()
        
        self.author_to_id = {author: idx for idx, author in enumerate(unique_authors)}
        self.publisher_to_id = {pub: idx for idx, pub in enumerate(unique_publishers)}
        
        # 存储特征
        for _, row in books_with_stats.iterrows():
            isbn = row['ISBN']
            
            author = row.get('Book-Author', 'Unknown')
            publisher = row.get('Publisher', 'Unknown')
            year = row.get('Year-Of-Publication', 2000)
            
            # 处理异常年份
            try:
                year = int(year)
                if year < 1900 or year > 2025:
                    year = 2000
            except:
                year = 2000
            
            # book_features
            self.book_features[isbn] = {
                'avg_rating': row['avg_rating'],
                'rating_count': row['rating_count'],
                'rating_std': row['rating_std'],
                'min_rating': row['min_rating'],
                'max_rating': row['max_rating'],
                'unique_users': row['unique_users'],
                'popularity': np.log1p(row['rating_count']),
                'author': author,
                'publisher': publisher,
                'year': year,
                'author_id': self.author_to_id.get(author, 0),
                'publisher_id': self.publisher_to_id.get(publisher, 0),
            }
            
            # isbn_stats (辅助)
            self.isbn_stats[isbn] = {
                'rating_count': row['rating_count'],
                'avg_rating': row['avg_rating'],
                'rating_std': row['rating_std'],
                'user_count': row['unique_users']
            }
        
        print(f"  ✓ 计算了 {len(self.book_features)} 本书的特征")
        print(f"  ✓ 唯一作者数: {len(self.author_to_id)}")
        print(f"  ✓ 唯一出版商数: {len(self.publisher_to_id)}")
    
    def _compute_cooccurrence(self, ratings_df):
        """计算图书共现矩阵"""
        print("  计算图书共现特征...")
        
        user_books = ratings_df[ratings_df['Book-Rating'] >= 7].groupby('User-ID')['ISBN'].apply(list)
        
        cooccurrence = {}
        for user_id, books in tqdm(user_books.items(), desc="  计算共现"):
            if len(books) < 2:
                continue
            
            # 仅在一定窗口内或全量计算
            for i in range(len(books)):
                for j in range(i + 1, len(books)):
                    book_a, book_b = books[i], books[j]
                    
                    # 排序 key 保证一致性
                    key = tuple(sorted((book_a, book_b)))
                    cooccurrence[key] = cooccurrence.get(key, 0) + 1
        
        self.cooccurrence = cooccurrence
        print(f"  ✓ 计算了 {len(cooccurrence):,} 对图书共现")
    
    def _extract_pairwise_features(self, isbn_a, isbn_b):
        """
        提取成对特征（增强版 - 16个特征）
        包含：属性匹配、目标书统计、对比特征、共现特征、交叉特征、Embedding相似度
        """
        features = []
        
        feat_a = self.book_features.get(isbn_a, {})
        feat_b = self.book_features.get(isbn_b, {})
        
        # === 1-3. 属性匹配特征 ===
        same_author = 1 if feat_a.get('author') == feat_b.get('author') else 0
        features.append(same_author)
        
        same_publisher = 1 if feat_a.get('publisher') == feat_b.get('publisher') else 0
        features.append(same_publisher)
        
        year_diff = abs(feat_a.get('year', 2000) - feat_b.get('year', 2000))
        features.append(min(year_diff / 10.0, 10.0))  # 归一化到 [0, 10]
        
        # === 4-7. 候选书统计特征 ===
        features.append(feat_b.get('avg_rating', 0))                    # 4. 平均评分
        features.append(feat_b.get('popularity', 0))                    # 5. 流行度（对数）
        features.append(feat_b.get('rating_std', 0))                    # 6. 评分标准差
        
        rating_range = feat_b.get('max_rating', 0) - feat_b.get('min_rating', 0)
        features.append(rating_range)                                   # 7. 评分范围
        
        # === 8-10. 成对对比特征 ===
        rating_diff = abs(feat_a.get('avg_rating', 0) - feat_b.get('avg_rating', 0))
        features.append(rating_diff)                                    # 8. 评分差异
        
        pop_a = feat_a.get('popularity', 0) + 1
        pop_b = feat_b.get('popularity', 0) + 1
        features.append(pop_b / pop_a)                                  # 9. 流行度比例
        
        rating_sim = 1.0 - min(rating_diff / 10.0, 1.0)                 # 10. 评分相似度
        features.append(rating_sim)
        
        # === 11-13. 共现特征 ===
        # 注意：key 需要排序
        co_key = tuple(sorted((isbn_a, isbn_b)))
        cooccur_count = self.cooccurrence.get(co_key, 0)
        
        features.append(cooccur_count)                                  # 11. 共现次数
        features.append(np.log1p(cooccur_count))                        # 12. 共现对数
        
        if feat_a.get('unique_users', 0) > 0:
            cooccur_ratio = cooccur_count / feat_a.get('unique_users', 1)
        else:
            cooccur_ratio = 0
        features.append(cooccur_ratio)                                  # 13. 共现比例
        
        # === 14-15. 交叉特征 ===
        pop_product = np.log1p(
            feat_a.get('rating_count', 0) * feat_b.get('rating_count', 0)
        )
        features.append(pop_product)                                    # 14. 流行度乘积
        
        hotness_a = feat_a.get('avg_rating', 0) * np.log1p(feat_a.get('rating_count', 0))
        hotness_b = feat_b.get('avg_rating', 0) * np.log1p(feat_b.get('rating_count', 0))
        features.append(hotness_b / max(hotness_a, 1))                  # 15. 热门度比例
        
        # === 16. Embedding 相似度 ===
        emb_sim = self._calculate_embedding_similarity(isbn_a, isbn_b)
        features.append(emb_sim)                                        # 16. 向量相似度
        
        return features
    
    def train(self, ratings_df, books_df, users_df=None):
        """训练 LightGBM 排序模型（增强版）"""
        print("="*60)
        print("训练 LightGBM 排序模型（精排阶段 - Embedding 增强版）")
        print("="*60)
        
        self.books_df = books_df.copy()
        self.ratings_df = ratings_df.copy()
        
        # 计算图书特征
        self._compute_book_features(books_df, ratings_df)
        
        # 计算共现矩阵
        self._compute_cooccurrence(ratings_df)
        
        # === 构造训练样本 ===
        print("\n构造训练样本...")
        X_train = []
        y_train = []
        
        user_book_map = ratings_df[ratings_df['Book-Rating'] >= 7].groupby('User-ID')['ISBN'].apply(list)
        
        sample_count = 0
        max_samples = 50000
        
        for user_id, liked_books in tqdm(user_book_map.items(), desc="  生成样本"):
            if sample_count >= max_samples:
                break
            
            if len(liked_books) < 2:
                continue
            
            liked_books_set = set(liked_books)
            
            # 为每个用户生成样本
            for i in range(min(len(liked_books), 15)):
                if sample_count >= max_samples:
                    break
                
                book_a = liked_books[i]
                
                # 正样本：用户也喜欢的其他书
                for j in range(min(len(liked_books), 15)):
                    if i == j:
                        continue
                    
                    book_b = liked_books[j]
                    features = self._extract_pairwise_features(book_a, book_b)
                    
                    if features and len(features) == 16:  # ✅ 确保16个特征
                        X_train.append(features)
                        y_train.append(1)
                        sample_count += 1
                    
                    if sample_count >= max_samples:
                        break
                
                # 负样本：随机选择用户没评分的书
                neg_samples = 3
                all_books = list(self.book_features.keys())
                
                # 简单随机采样
                for _ in range(neg_samples):
                    book_b = np.random.choice(all_books)
                    if book_b in liked_books_set:
                        continue
                        
                    features = self._extract_pairwise_features(book_a, book_b)
                    
                    if features and len(features) == 16:
                        X_train.append(features)
                        y_train.append(0)
                        sample_count += 1
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        print(f"\n✓ 训练样本: {len(X_train):,}")
        print(f"✓ 特征维度: {X_train.shape[1]}")
        print(f"✓ 正样本比例: {y_train.mean():.2%}")
        
        # 划分训练集和验证集
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # 训练 LightGBM
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': self.learning_rate,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'min_data_in_leaf': 20,
        }
        
        print("\n训练 LightGBM...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(20),
                lgb.log_evaluation(20)
            ]
        )
        
        # 特征重要性
        feature_names = [
            'same_author', 'same_publisher', 'year_diff',
            'b_avg_rating', 'b_popularity', 'b_rating_std', 'b_rating_range',
            'rating_diff', 'popularity_ratio', 'rating_similarity',
            'cooccur_count', 'cooccur_log', 'cooccur_ratio',
            'popularity_product', 'hotness_ratio',
            'embedding_similarity'  # ✅ 特征16
        ]
        
        print(f"\n📊 特征重要性 Top 10:")
        importance = self.model.feature_importance()
        importances = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
        for name, imp in importances[:10]:
            print(f"  {name:25s}: {imp:8.1f}")
        
        print("\n✅ LightGBM 排序模型训练完成（增强版）")
    
    def recommend(self, book_title, n=10, candidate_pool=None):
        """
        推荐图书（增强版：返回完整预测列表，不截断）
        """
        if self.model is None:
            return []
        
        if self.books_df is None:
            return []
        
        # 获取查询图书的 ISBN
        query_book = self.books_df[self.books_df['Book-Title'] == book_title]
        if query_book.empty:
            return []
        
        query_isbn = query_book.iloc[0]['ISBN']
        
        # 确定候选集
        if candidate_pool:
            candidates = [isbn for isbn in candidate_pool if isbn != query_isbn]
        else:
            # 如果没有提供候选池，理论上不应该发生（因为是 Ranker），但可以兜底
            candidates = []
        
        if not candidates:
            return []
        
        # === 构建特征 ===
        X = []
        valid_candidates = []
        
        for candidate_isbn in candidates:
            features = self._extract_pairwise_features(query_isbn, candidate_isbn)
            
            if features is not None and len(features) == 16:
                X.append(features)
                valid_candidates.append(candidate_isbn)
        
        if not X:
            return []
        
        X = np.array(X)
        
        # === 预测 ===
        try:
            scores = self.model.predict(X)
        except Exception as e:
            print(f"  ❌ 预测失败: {e}")
            return []
        
        # === 返回完整结果 (交给 TwoStage 进行截断和混合) ===
        recommendations = []
        
        # 我们不在这里做截断，而是返回所有有效候选的预测分
        # 排序可以在这里做，也可以在 TwoStage 做，但这里做一下比较方便
        sorted_indices = np.argsort(scores)[::-1]
        
        # 即使请求了 n，如果提供了 candidate_pool，我们最好也返回更多结果
        # 但为了 API 兼容性，我们至少返回 n 个，或者全部
        # 鉴于 TwoStage 的优化逻辑，我们返回 *所有* 计算了分数的候选
        
        for idx in sorted_indices:
            candidate_isbn = valid_candidates[idx]
            book_info = self.books_df[self.books_df['ISBN'] == candidate_isbn]
            
            if not book_info.empty:
                recommendations.append({
                    'title': book_info.iloc[0]['Book-Title'],
                    'author': book_info.iloc[0].get('Book-Author', 'Unknown'),
                    'score': float(scores[idx])
                })
        
        return recommendations

    def save_model(self, filepath):
        """保存模型（完整版）"""
        model_data = {
            'model': self.model,
            'cooccurrence': getattr(self, 'cooccurrence', {}),
            'isbn_stats': getattr(self, 'isbn_stats', {}),
            'book_features': getattr(self, 'book_features', {}),
            'author_to_id': getattr(self, 'author_to_id', {}),
            'publisher_to_id': getattr(self, 'publisher_to_id', {}),
            'books_df': self.books_df,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'embeddings': self.embeddings 
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"LightGBM 模型已保存到 {filepath}")

    def load_model(self, filepath):
        """加载模型（完整版）"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        if isinstance(model_data, dict):
            self.model = model_data.get('model')
            self.cooccurrence = model_data.get('cooccurrence', {})
            self.isbn_stats = model_data.get('isbn_stats', {})
            self.book_features = model_data.get('book_features', {})
            self.author_to_id = model_data.get('author_to_id', {})
            self.publisher_to_id = model_data.get('publisher_to_id', {})
            self.books_df = model_data.get('books_df')
            self.n_estimators = model_data.get('n_estimators', 100)
            self.learning_rate = model_data.get('learning_rate', 0.05)
            self.embeddings = model_data.get('embeddings', {})
            
            print(f"LightGBM 模型已从 {filepath} 加载")
            print(f"  ✓ Embedding: {len(self.embeddings):,}")
        else:
            self.model = model_data
            print("⚠️ 加载了旧格式模型")