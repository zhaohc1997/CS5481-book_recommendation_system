# src/evaluation/compare_four_models.py

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# 添加 src 目录到路径
sys.path.append('src')

from models.collaborative_filtering import ItemBasedCF
from models.matrix_factorization import MatrixFactorization
from models.lightfm_model import LightFMModel
from models.lightgbm_ranker import LightGBMRanker
from models.two_stage_recommender import TwoStageRecommender

CUSTOM_COLORS = {
    'ItemBasedCF': "#72b4fb",          
    'MatrixFactorization': "#ffa74f",  
    'LightFM': "#61c481",              
    'TwoStage (LightGBM)': '#e15759'  
}
sns.set_style("whitegrid")

def load_and_split_data(test_ratio=0.2):
    """加载数据并统一分割"""
    print("⏳ 正在加载数据...")
    books_df = pd.read_csv('data/processed/books_clean.csv')
    ratings_df = pd.read_csv('data/processed/ratings_clean.csv')
    
    # 过滤交互过少的数据
    min_book_ratings = 5
    min_user_ratings = 5
    
    ratings_df = ratings_df[ratings_df.groupby('ISBN')['ISBN'].transform('count') >= min_book_ratings]
    ratings_df = ratings_df[ratings_df.groupby('User-ID')['User-ID'].transform('count') >= min_user_ratings]
    
    shuffled = ratings_df.sample(frac=1, random_state=42)
    test_size = int(len(shuffled) * test_ratio)
    
    test_df = shuffled.iloc[:test_size]
    train_df = shuffled.iloc[test_size:]
    
    print(f"✓ 数据集分割完成: 训练集 {len(train_df)} 条, 测试集 {len(test_df)} 条")
    return train_df, test_df, books_df

def evaluate_model(model, model_name, test_df, books_df, k=10, sample_size=500):
    """
    全方位评估模型
    新增指标: Latency_ms (平均每次推荐耗时)
    """
    print(f"\n📊 正在评估: {model_name} ...")
    
    hits = 0
    ndcg_sum = 0
    precision_sum = 0
    recall_sum = 0
    total_cases = 0
    
    all_recommended_isbns = set()
    total_popularity = 0
    total_recs_count = 0
    
    book_pop_map = test_df.groupby('ISBN').size().to_dict()
    
    test_users = test_df['User-ID'].unique()
    if len(test_users) > sample_size:
        np.random.seed(42)
        test_users = np.random.choice(test_users, sample_size, replace=False)
        
    user_groups = test_df[test_df['User-ID'].isin(test_users)].groupby('User-ID')
    
    # ✅ 计时开始
    start_time = time.time()
    
    for user_id, group in tqdm(user_groups, desc=f"Testing {model_name}"):
        user_books = group['ISBN'].tolist()
        if len(user_books) < 2:
            continue
            
        input_isbn = user_books[0]
        target_isbns = set(user_books[1:])
        
        input_book_row = books_df[books_df['ISBN'] == input_isbn]
        if input_book_row.empty: continue
        input_title = input_book_row.iloc[0]['Book-Title']
        
        try:
            # 核心推理步骤
            recommendations = model.recommend(input_title, n=k)
            
            rec_isbns = []
            for rec in recommendations:
                t = rec['title'] if isinstance(rec, dict) else str(rec)
                row = books_df[books_df['Book-Title'] == t]
                if not row.empty:
                    isbn = row.iloc[0]['ISBN']
                    rec_isbns.append(isbn)
                    
                    all_recommended_isbns.add(isbn)
                    total_popularity += book_pop_map.get(isbn, 0)
                    total_recs_count += 1
            
            # 指标计算
            hit_count = sum(1 for isbn in rec_isbns if isbn in target_isbns)
            
            if hit_count > 0: hits += 1
            if len(rec_isbns) > 0: precision_sum += hit_count / len(rec_isbns)
            if len(target_isbns) > 0: recall_sum += hit_count / len(target_isbns)

            dcg = 0; idcg = 0
            for i, isbn in enumerate(rec_isbns):
                if isbn in target_isbns: dcg += 1 / np.log2(i + 2)
            for i in range(min(len(target_isbns), k)): idcg += 1 / np.log2(i + 2)
            if idcg > 0: ndcg_sum += dcg / idcg
                
            total_cases += 1
            
        except Exception:
            continue

    # ✅ 计时结束 & 计算延迟
    total_time_sec = time.time() - start_time
    latency_ms = (total_time_sec / total_cases * 1000) if total_cases > 0 else 0
    
    metrics = {
        'Hit_Rate@10': hits / total_cases if total_cases > 0 else 0,
        'NDCG@10': ndcg_sum / total_cases if total_cases > 0 else 0,
        'Precision@10': precision_sum / total_cases if total_cases > 0 else 0,
        'Recall@10': recall_sum / total_cases if total_cases > 0 else 0,
        'Coverage': len(all_recommended_isbns) / len(books_df),
        'Avg_Popularity': total_popularity / total_recs_count if total_recs_count > 0 else 0,
        'Latency_ms': latency_ms  
    }
    
    return metrics

def plot_all_metrics(results_df, save_dir):
    """生成包含速度对比的高清图表"""
    
    # ✅ 7个指标，使用 2行 x 4列 的布局 (留空一个位置)
    metrics_to_plot = [
        ('Hit_Rate@10', 'Hit Rate@10 (Higher is Better)'),
        ('NDCG@10', 'NDCG@10 (Higher is Better)'),
        ('Precision@10', 'Precision@10 (Higher is Better)'),
        ('Recall@10', 'Recall@10 (Higher is Better)'),
        ('Coverage', 'Catalog Coverage (Higher is Better)'),
        ('Avg_Popularity', 'Avg Popularity (Lower is Better)'),
        ('Latency_ms', 'Latency (ms) - Lower is Better') # ✅ 新增图表
    ]
    
    # 调整画布大小以适应 2x4
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    axes = axes.flatten()
    
    palette = [CUSTOM_COLORS.get(m, '#888888') for m in results_df['Model']]

    for i, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[i]
        
        bars = sns.barplot(x='Model', y=metric, data=results_df, ax=ax, palette=palette)
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='x', rotation=15)
        
        # 添加数值标注
        for container in bars.containers:
            labels = []
            for v in container.datavalues:
                if metric == 'Coverage':
                    labels.append(f"{v:.1%}")
                elif metric == 'Avg_Popularity' or metric == 'Latency_ms':
                    labels.append(f"{v:.1f}")
                else:
                    labels.append(f"{v:.3f}")
            ax.bar_label(container, labels=labels, padding=3, fontsize=10)
    
    # 隐藏第8个多余的子图
    if len(metrics_to_plot) < 8:
        axes[7].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'all_metrics_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 图表已保存至: {save_path}")

def main():
    # 1. 准备数据
    train_df, test_df, books_df = load_and_split_data()
    results = []
    
    # 2. ItemBased CF
    print("\n[1/4] ItemBased CF")
    cf = ItemBasedCF()
    cf.train(train_df, books_df)
    res = evaluate_model(cf, "ItemBasedCF", test_df, books_df)
    results.append({**res, 'Model': 'ItemBasedCF'})
    
    # 3. Matrix Factorization
    print("\n[2/4] Matrix Factorization")
    mf = MatrixFactorization()
    mf.train(train_df, books_df)
    res = evaluate_model(mf, "MatrixFactorization", test_df, books_df)
    results.append({**res, 'Model': 'MatrixFactorization'})
    
    # 4. LightFM
    print("\n[3/4] LightFM")
    lfm = LightFMModel()
    lfm.train(train_df, books_df)
    res = evaluate_model(lfm, "LightFM", test_df, books_df)
    results.append({**res, 'Model': 'LightFM'})
    
    # 5. Two-Stage
    print("\n[4/4] Two-Stage (LightGBM)")
    lgb_ranker = LightGBMRanker()
    two_stage = TwoStageRecommender(recall_models=[cf, mf, lfm], ranking_model=lgb_ranker)
    two_stage.train(train_df, books_df)
    res = evaluate_model(two_stage, "TwoStage (LightGBM)", test_df, books_df)
    results.append({**res, 'Model': 'TwoStage (LightGBM)'})
    
    # 6. 保存与展示
    results_df = pd.DataFrame(results)
    save_dir = 'results/four_models_comparison'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    csv_path = os.path.join(save_dir, 'summary.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ 详细数据已保存至: {csv_path}")
    
    # 打印表格 (新增 Latency 列)
    print("\n" + "="*95)
    print(f"{'Model':<25} {'HR@10':<8} {'NDCG@10':<8} {'Cov':<8} {'Pop':<8} {'Latency(ms)':<10}")
    print("-" * 95)
    for _, row in results_df.iterrows():
        print(f"{row['Model']:<25} {row['Hit_Rate@10']:.4f}   {row['NDCG@10']:.4f}   {row['Coverage']:.2%}   {row['Avg_Popularity']:.1f}    {row['Latency_ms']:.2f}")
    print("="*95)
    
    plot_all_metrics(results_df, save_dir)

if __name__ == "__main__":
    main()