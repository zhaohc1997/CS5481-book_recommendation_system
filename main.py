# main.py

"""
主程序：一键训练并保存所有模型（TwoStage + 独立子模型）
"""
import pandas as pd
import sys
import os

# 添加 src 目录到路径
sys.path.append('src')

from models.collaborative_filtering import ItemBasedCF
from models.matrix_factorization import MatrixFactorization
from models.lightfm_model import LightFMModel
from models.lightgbm_ranker import LightGBMRanker
from models.two_stage_recommender import TwoStageRecommender

def load_data():
    """加载并清洗数据"""
    print("⏳ 正在加载数据...")
    books = pd.read_csv('data/processed/books_clean.csv')
    ratings = pd.read_csv('data/processed/ratings_clean.csv')
    
    # 基础清洗：过滤交互过少的数据，加速训练并提高质量
    min_book_ratings = 5
    min_user_ratings = 5
    
    ratings = ratings[ratings.groupby('ISBN')['ISBN'].transform('count') >= min_book_ratings]
    ratings = ratings[ratings.groupby('User-ID')['User-ID'].transform('count') >= min_user_ratings]
    
    print(f"✓ 加载完成: 图书 {len(books):,} 本, 评分 {len(ratings):,} 条")
    return books, ratings

def ensure_dir(file_path):
    """确保目录存在"""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    # 1. 准备数据
    books_df, ratings_df = load_data()
    
    # 2. 初始化各个子模型
    print("\n🛠️  初始化各个子模型...")
    
    # ItemBasedCF
    item_cf = ItemBasedCF()
    
    # MatrixFactorization (设置 n_factors=50 效果较好)
    mf_model = MatrixFactorization(n_factors=50)
    
    # LightFM
    lightfm = LightFMModel()
    
    # LightGBM Ranker (排序器)
    lgb_ranker = LightGBMRanker()
    
    # 3. 组装 Two-Stage 模型
    # 将所有召回模型打包放入 Two-Stage
    recall_models = [item_cf, mf_model, lightfm]
    
    print("\n📦 组装 Two-Stage 系统...")
    two_stage = TwoStageRecommender(
        recall_models=recall_models,
        ranking_model=lgb_ranker
    )
    
    # 4. 统一训练 (One-Click Training)
    # 调用 two_stage.train() 会自动依次训练列表中的所有召回模型，
    # 并处理 Embedding 传递，最后训练 LightGBM。
    # 这样我们不需要手动一个个调用 train()。
    two_stage.train(ratings_df, books_df)
    
    # 5. 保存所有模型文件
    print("\n💾 正在保存所有模型文件...")
    
    # 确保保存目录存在
    model_dir = 'data/models/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # --- (A) 保存 Two-Stage 整体模型 ---
    ts_path = os.path.join(model_dir, 'two_stage_model.pkl')
    two_stage.save_model(ts_path)
    
    # --- (B) 保存独立的 ItemBasedCF ---
    # 从 two_stage 实例中提取已经训练好的 CF 模型
    # 注意：字典的 key 通常是类名，如 'ItemBasedCF'
    if 'ItemBasedCF' in two_stage.recall_models:
        cf_path = os.path.join(model_dir, 'itembasedcf.pkl')
        print(f"  正在保存 ItemBasedCF -> {cf_path}")
        two_stage.recall_models['ItemBasedCF'].save_model(cf_path)
    
    # --- (C) 保存独立的 MatrixFactorization ---
    if 'MatrixFactorization' in two_stage.recall_models:
        mf_path = os.path.join(model_dir, 'matrixfactorization.pkl')
        print(f"  正在保存 MatrixFactorization -> {mf_path}")
        two_stage.recall_models['MatrixFactorization'].save_model(mf_path)
        
    # --- (D) 保存独立的 LightFM ---
    # 类名可能是 LightFMModel
    lfm_key = 'LightFMModel'
    if lfm_key in two_stage.recall_models:
        lfm_path = os.path.join(model_dir, 'lightfm.pkl')
        print(f"  正在保存 LightFM -> {lfm_path}")
        two_stage.recall_models[lfm_key].save_model(lfm_path)

    print("✅ 所有模型保存完成！")
    
    # 6. 简单冒烟测试
    print("\n🧪 执行冒烟测试 (Smoke Test)...")
    test_book = "Harry Potter and the Sorcerer's Stone (Book 1)"
    print(f"  测试书名: {test_book}")
    
    try:
        recs = two_stage.recommend(test_book, n=5)
        print(f"\n针对 '{test_book}' 的推荐结果:")
        for i, rec in enumerate(recs, 1):
            print(f"  {i}. {rec['title'][:50]} (Score: {rec['score']:.4f})")
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        # 如果找不到书，尝试用第一本书测试
        if not books_df.empty:
            fallback_book = books_df.iloc[0]['Book-Title']
            print(f"  尝试使用第一本书测试: {fallback_book}")
            recs = two_stage.recommend(fallback_book, n=5)
            for i, rec in enumerate(recs, 1):
                print(f"  {i}. {rec['title'][:50]} (Score: {rec['score']:.4f})")

if __name__ == "__main__":
    main()