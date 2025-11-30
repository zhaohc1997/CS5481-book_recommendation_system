# src/models/two_stage_recommender.py

import numpy as np
import pickle
from collections import defaultdict

class TwoStageRecommender:
    """两阶段推荐系统"""
    
    def __init__(self, recall_models, ranking_model, model_weights=None):
        if isinstance(recall_models, dict):
            self.recall_models = recall_models
        else:
            self.recall_models = {}
            for model in recall_models:
                name = model.__class__.__name__
                self.recall_models[name] = model
        
        self.ranking_model = ranking_model
        self.books_df = None
        self.model_weights = model_weights or {}
        if not self.model_weights:
            self._set_default_weights()
    
    def _set_default_weights(self):
        default_weights = {
            'ItemBasedCF': 0.6,          # CF 在图书领域通常最强
            'MatrixFactorization': 0.2,
            'LightFM': 0.2
        }
        for name in self.recall_models.keys():
            self.model_weights[name] = default_weights.get(name, 0.25)
    
    def train(self, ratings_df, books_df, users_df=None):
        """训练流程（保持不变，重点是 Embedding 传递）"""
        print(f"\n{'='*60}\n两阶段系统训练\n{'='*60}")
        self.books_df = books_df.copy()
        
        # 1. 训练召回模型 & 收集 Embedding
        collected_embeddings = {}
        for name, model in self.recall_models.items():
            print(f"训练召回模型: {name}...")
            try:
                model.train(ratings_df, books_df, users_df)
                if hasattr(model, 'get_all_embeddings'):
                    emb = model.get_all_embeddings()
                    if emb: collected_embeddings.update(emb)
            except Exception as e:
                print(f"❌ {name} 训练失败: {e}")

        # 2. 训练排序模型
        if self.ranking_model:
            print(f"训练排序模型: {self.ranking_model.__class__.__name__}...")
            if collected_embeddings and hasattr(self.ranking_model, 'set_embeddings'):
                self.ranking_model.set_embeddings(collected_embeddings)
            try:
                self.ranking_model.train(ratings_df, books_df, users_df)
            except Exception as e:
                print(f"❌ 排序模型训练失败: {e}")

    def recommend(self, book_title, n=10, recall_size=500, use_ranking=True):
        """
        推荐流程（核心优化：归一化混合）
        Args:
            recall_size: 默认扩大到 500，防止漏掉好书
        """
        # === 阶段1: 召回 ===
        candidate_scores = defaultdict(float)
        
        for name, model in self.recall_models.items():
            weight = self.model_weights.get(name, 0.3)
            # 动态调整每个模型的召回量
            k = max(20, int(recall_size * weight))
            try:
                recs = model.recommend(book_title, k)
                for i, rec in enumerate(recs):
                    # 解析结果
                    if isinstance(rec, dict):
                        t, s = rec.get('title', ''), rec.get('score', 0)
                    else:
                        t, s = str(rec), 0
                    
                    if not t: continue
                    
                    # 基础分 + 位置分
                    # 位置分很重要：即使分数不可靠，排名也是可靠的
                    rank_boost = 1.0 / (i + 5.0)  # 平滑位置衰减
                    candidate_scores[t] += weight * (s * 0.5 + rank_boost * 0.5)
            except:
                pass

        # 筛选 Top K 候选进入精排
        candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:recall_size]
        if not candidates: return []
        
        top_titles = [t for t, s in candidates]
        
        # === 阶段2: 排序 ===
        final_results = []
        lgb_success = False
        
        if use_ranking and self.ranking_model:
            try:
                # 转换 ISBN
                candidate_isbns = []
                title_map = {}
                if self.books_df is not None:
                    # 批量查找优化性能
                    mask = self.books_df['Book-Title'].isin(top_titles)
                    found_books = self.books_df[mask][['Book-Title', 'ISBN']]
                    # 去重，每本书只取一个 ISBN
                    found_books = found_books.drop_duplicates('Book-Title')
                    
                    candidate_isbns = found_books['ISBN'].tolist()
                    title_map = dict(zip(found_books['ISBN'], found_books['Book-Title']))

                if candidate_isbns:
                    # 全量打分 (不要只打 Top N)
                    lgb_recs = self.ranking_model.recommend(book_title, n=len(candidate_isbns), candidate_pool=candidate_isbns)
                    
                    if lgb_recs:
                        # --- 核心优化：分数归一化与混合 ---
                        
                        # 1. 提取分数数组
                        lgb_scores = np.array([r['score'] for r in lgb_recs])
                        rec_scores = np.array([candidate_scores[r['title']] for r in lgb_recs])
                        
                        # 2. Min-Max 归一化 (防止量纲不一致)
                        def normalize(arr):
                            if len(arr) < 2 or arr.max() == arr.min():
                                return np.zeros_like(arr) + 0.5 # 无法归一化则取中值
                            return (arr - arr.min()) / (arr.max() - arr.min())
                        
                        lgb_norm = normalize(lgb_scores)
                        rec_norm = normalize(rec_scores)
                        
                        # 3. 动态权重决策
                        lgb_std = np.std(lgb_scores)
                        # 如果 LightGBM 区分度高(>0.01)，给它 60% 权重；否则只给 20%（防止捣乱）
                        lgb_weight = 0.6 if lgb_std > 0.01 else 0.2
                        rec_weight = 1.0 - lgb_weight
                        
                        # 4. 融合
                        final_scores = lgb_weight * lgb_norm + rec_weight * rec_norm
                        
                        # 5. 组装结果
                        for i, rec in enumerate(lgb_recs):
                            final_results.append({
                                'title': rec['title'],
                                'score': final_scores[i], # 使用融合分
                                'author': rec.get('author', ''),
                                'debug': f"LGB:{lgb_scores[i]:.2f}, Rec:{rec_scores[i]:.2f}"
                            })
                        
                        lgb_success = True
            except Exception as e:
                print(f"排序异常: {e}")
        
        # 降级处理
        if not lgb_success:
            for t, s in candidates:
                info = self.books_df[self.books_df['Book-Title'] == t].iloc[0] if self.books_df is not None else {}
                final_results.append({
                    'title': t, 'score': s, 'author': info.get('Book-Author', '')
                })
        
        # 最终排序截断
        final_results.sort(key=lambda x: x['score'], reverse=True)
        return final_results[:n]

    # (save_model, load_model 等保持原样即可)
    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'recall_models': self.recall_models, 'ranking_model': self.ranking_model, 
                         'books_df': self.books_df, 'model_weights': self.model_weights}, f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.__dict__.update(data)