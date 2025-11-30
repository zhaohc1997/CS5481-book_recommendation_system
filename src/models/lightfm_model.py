import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from scipy.sparse import csr_matrix
import pickle

class LightFMModel:
    """LightFM 推荐模型"""
    
    def __init__(self, no_components=50, loss='warp', learning_rate=0.05, epochs=30):
        """
        初始化 LightFM 模型
        
        参数:
            no_components: 嵌入维度
            loss: 损失函数 ('warp', 'bpr', 'logistic', 'warp-kos')
            learning_rate: 学习率
            epochs: 训练轮数
        """
        self.model = LightFM(
            no_components=no_components,
            loss=loss,
            learning_rate=learning_rate,
            random_state=42
        )
        self.epochs = epochs
        self.dataset = None
        self.user_id_map = {}
        self.item_id_map = {}
        self.item_id_to_isbn = {}  # 正确的反向映射：internal_id -> ISBN
        self.interaction_matrix = None
        self.books_df = None
        self.isbn_to_title = {}
        
    def train(self, ratings_df, books_df, users_df=None):
        """
        训练 LightFM 模型
        
        参数:
            ratings_df: 评分数据
            books_df: 图书数据
            users_df: 用户数据（可选）
        """
        print("训练 LightFM 模型...")
        self.books_df = books_df.copy()
        
        # 创建 ISBN 到标题的映射
        self.isbn_to_title = dict(zip(books_df['ISBN'], books_df['Book-Title']))
        
        # 创建 Dataset 对象
        self.dataset = Dataset()
        
        # 构建用户和物品映射
        unique_users = ratings_df['User-ID'].unique()
        unique_items = ratings_df['ISBN'].unique()
        
        self.dataset.fit(
            users=unique_users,
            items=unique_items
        )
        
        # 获取映射
        user_id_map, user_features_map, item_id_map, item_features_map = self.dataset.mapping()
        
        # 保存映射
        self.user_id_map = user_id_map
        self.item_id_map = item_id_map  # ISBN -> internal_id
        
        # 创建正确的反向映射：internal_id -> ISBN
        self.item_id_to_isbn = {internal_id: isbn for isbn, internal_id in item_id_map.items()}
        
        print(f"映射信息:")
        print(f"  - item_id_map: {len(self.item_id_map)} (ISBN -> internal_id)")
        print(f"  - item_id_to_isbn: {len(self.item_id_to_isbn)} (internal_id -> ISBN)")
        
        # 构建交互矩阵
        interactions = []
        for _, row in ratings_df.iterrows():
            user_id = row['User-ID']
            item_id = row['ISBN']
            rating = row['Book-Rating']
            
            # 只使用显式正反馈 (评分 > 0)
            if rating > 0:
                interactions.append((user_id, item_id, rating))
        
        # 转换为 LightFM 格式
        (self.interaction_matrix, _) = self.dataset.build_interactions(interactions)
        
        # 训练模型
        print(f"训练样本数: {len(interactions)}")
        print(f"用户数: {len(unique_users)}, 物品数: {len(unique_items)}")
        
        self.model.fit(
            self.interaction_matrix,
            epochs=self.epochs,
            num_threads=4,
            verbose=True
        )
        
        print("LightFM 模型训练完成")
        
    def recommend(self, book_title, n=5):
        """
        基于给定图书推荐相似图书
        
        参数:
            book_title: 图书标题
            n: 推荐数量
            
        返回:
            推荐图书列表
        """
        if self.books_df is None or self.model is None:
            return []
        
        try:
            # 查找图书 ISBN - 支持模糊匹配
            book_match = self.books_df[self.books_df['Book-Title'].str.contains(
                book_title, case=False, na=False, regex=False
            )]
            
            if book_match.empty:
                # 尝试精确匹配
                book_match = self.books_df[self.books_df['Book-Title'] == book_title]
                if book_match.empty:
                    return []
            
            target_isbn = book_match.iloc[0]['ISBN']
            
            # 检查 ISBN 是否在训练数据中
            if target_isbn not in self.item_id_map:
                return []
            
            target_item_id = self.item_id_map[target_isbn]
            
            # 获取物品嵌入向量
            item_embeddings = self.model.item_embeddings
            item_biases = self.model.item_biases
            
            # 检查索引是否有效
            num_items = len(item_embeddings)
            if target_item_id >= num_items:
                return []
            
            target_embedding = item_embeddings[target_item_id]
            
            # 计算所有物品的相似度
            similarities = np.dot(item_embeddings, target_embedding)
            
            # 添加偏置项
            if len(item_biases) == num_items:
                similarities = similarities + item_biases
            
            # 创建候选列表（排除目标物品本身）
            candidates = []
            for internal_id in range(num_items):
                if internal_id != target_item_id:
                    # 使用正确的反向映射
                    if internal_id in self.item_id_to_isbn:
                        isbn = self.item_id_to_isbn[internal_id]
                        if isbn in self.isbn_to_title:
                            candidates.append({
                                'idx': internal_id,
                                'isbn': isbn,
                                'score': float(similarities[internal_id])
                            })
            
            # 按相似度排序
            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # 生成推荐列表
            recommendations = []
            for candidate in candidates[:n]:
                isbn = candidate['isbn']
                book_info = self.books_df[self.books_df['ISBN'] == isbn]
                
                if not book_info.empty:
                    recommendations.append({
                        'title': book_info.iloc[0]['Book-Title'],
                        'author': book_info.iloc[0].get('Book-Author', 'Unknown'),
                        'score': candidate['score']
                    })
            
            return recommendations
            
        except Exception as e:
            print(f"LightFM 推荐出错: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def predict_rating(self, user_id, isbn):
        """
        预测用户对图书的评分
        
        参数:
            user_id: 用户 ID
            isbn: 图书 ISBN
            
        返回:
            预测评分
        """
        if user_id not in self.user_id_map or isbn not in self.item_id_map:
            return 0.0
        
        user_idx = self.user_id_map[user_id]
        item_idx = self.item_id_map[isbn]
        
        try:
            prediction = self.model.predict(user_idx, item_idx)
            return float(prediction)
        except Exception as e:
            return 0.0
    
    def save_model(self, filepath):
        """保存模型"""
        model_data = {
            'model': self.model,
            'dataset': self.dataset,
            'user_id_map': self.user_id_map,
            'item_id_map': self.item_id_map,
            'item_id_to_isbn': self.item_id_to_isbn,
            'interaction_matrix': self.interaction_matrix,
            'books_df': self.books_df,
            'isbn_to_title': self.isbn_to_title,
            'epochs': self.epochs
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"LightFM 模型已保存到 {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.dataset = model_data['dataset']
        self.user_id_map = model_data['user_id_map']
        self.item_id_map = model_data['item_id_map']
        self.item_id_to_isbn = model_data['item_id_to_isbn']
        self.interaction_matrix = model_data['interaction_matrix']
        self.books_df = model_data['books_df']
        self.isbn_to_title = model_data.get('isbn_to_title', {})
        self.epochs = model_data.get('epochs', 30)
        print(f"LightFM 模型已从 {filepath} 加载")