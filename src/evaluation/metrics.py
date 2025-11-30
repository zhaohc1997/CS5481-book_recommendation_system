# src/evaluation/metrics.py

import numpy as np
from collections import defaultdict
import random

def calculate_hit_rate(recommendations, test_data, k=10):
    """计算命中率 (Hit Rate@K)"""
    hits = 0
    total = 0
    
    for user_id, true_items in test_data.items():
        if user_id in recommendations:
            recommended_items = recommendations[user_id][:k]
            if any(item in true_items for item in recommended_items):
                hits += 1
            total += 1
    
    return hits / total if total > 0 else 0.0

def calculate_precision(recommendations, test_data, k=10):
    """计算精确率 (Precision@K)"""
    precisions = []
    
    for user_id, true_items in test_data.items():
        if user_id in recommendations:
            recommended_items = recommendations[user_id][:k]
            hits = len(set(recommended_items) & set(true_items))
            precisions.append(hits / k if k > 0 else 0)
    
    return np.mean(precisions) if precisions else 0.0

def calculate_recall(recommendations, test_data, k=10):
    """计算召回率 (Recall@K)"""
    recalls = []
    
    for user_id, true_items in test_data.items():
        if user_id in recommendations and len(true_items) > 0:
            recommended_items = recommendations[user_id][:k]
            hits = len(set(recommended_items) & set(true_items))
            recalls.append(hits / len(true_items))
    
    return np.mean(recalls) if recalls else 0.0

def calculate_f1(recommendations, test_data, k=10):
    """计算 F1 分数"""
    precision = calculate_precision(recommendations, test_data, k)
    recall = calculate_recall(recommendations, test_data, k)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

def calculate_mrr(recommendations, test_data, k=10):
    """计算平均倒数排名 (Mean Reciprocal Rank)"""
    reciprocal_ranks = []
    
    for user_id, true_items in test_data.items():
        if user_id in recommendations:
            recommended_items = recommendations[user_id][:k]
            for i, item in enumerate(recommended_items):
                if item in true_items:
                    reciprocal_ranks.append(1.0 / (i + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

def calculate_ndcg(recommendations, test_data, k=10):
    """计算归一化折损累积增益 (NDCG@K)"""
    ndcgs = []
    
    for user_id, true_items in test_data.items():
        if user_id in recommendations:
            recommended_items = recommendations[user_id][:k]
            
            # 计算 DCG
            dcg = 0.0
            for i, item in enumerate(recommended_items):
                if item in true_items:
                    dcg += 1.0 / np.log2(i + 2)
            
            # 计算 IDCG
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), k)))
            
            # 计算 NDCG
            if idcg > 0:
                ndcgs.append(dcg / idcg)
            else:
                ndcgs.append(0.0)
    
    return np.mean(ndcgs) if ndcgs else 0.0

def calculate_auc(model, test_data, all_items, n_samples=5000, seed=42):
    """
    计算 AUC (Area Under ROC Curve)
    
    使用成对采样方法：
    - 随机采样 (正样本, 负样本) 对
    - 计算正样本得分高于负样本的概率
    
    Args:
        model: 推荐模型（需要有 recommend 方法）
        test_data: 测试数据 {user_id: [item_ids]}
        all_items: 所有物品列表
        n_samples: 采样对数
        seed: 随机种子
    
    Returns:
        auc: AUC 分数 (0.5-1.0)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    correct = 0
    total = 0
    
    # 获取所有用户
    users = list(test_data.keys())
    
    # 转换为集合以加速查找
    all_items_set = set(all_items)
    
    for _ in range(n_samples):
        # 随机选择一个用户
        user_id = random.choice(users)
        pos_items = test_data[user_id]
        
        if not pos_items:
            continue
        
        # 随机选择一个正样本
        pos_item = random.choice(pos_items)
        
        # 随机选择一个负样本（未在测试集中的物品）
        neg_items = list(all_items_set - set(pos_items))
        
        if not neg_items:
            continue
        
        neg_item = random.choice(neg_items)
        
        try:
            # 获取推荐列表（Top-100，足够大）
            recommendations = model.recommend(user_id, n=100)
            
            # 获取正负样本的排名
            pos_rank = recommendations.index(pos_item) if pos_item in recommendations else 999
            neg_rank = recommendations.index(neg_item) if neg_item in recommendations else 999
            
            # 比较排名
            if pos_rank < neg_rank:
                correct += 1
            elif pos_rank == neg_rank:
                correct += 0.5  # 平局算 0.5
            
            total += 1
            
        except Exception as e:
            # 模型推荐失败，跳过
            continue
    
    return correct / total if total > 0 else 0.5


def calculate_auc_fast(model, test_data, all_items, sample_users=200, sample_pairs_per_user=25, seed=42):
    """
    快速 AUC 计算（适合大数据集）
    
    Args:
        model: 推荐模型
        test_data: 测试数据 {user_id: [item_ids]}
        all_items: 所有物品列表
        sample_users: 采样用户数
        sample_pairs_per_user: 每个用户采样的正负对数
        seed: 随机种子
    
    Returns:
        auc: AUC 分数
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 采样用户
    users = list(test_data.keys())
    sampled_users = random.sample(users, min(sample_users, len(users)))
    
    all_items_set = set(all_items)
    
    user_aucs = []
    
    for user_id in sampled_users:
        pos_items = test_data[user_id]
        
        if not pos_items:
            continue
        
        neg_items = list(all_items_set - set(pos_items))
        
        if not neg_items:
            continue
        
        try:
            # 获取推荐列表
            recommendations = model.recommend(user_id, n=100)
            
            correct = 0
            total = 0
            
            # 对该用户采样多个正负对
            for _ in range(sample_pairs_per_user):
                pos_item = random.choice(pos_items)
                neg_item = random.choice(neg_items)
                
                pos_rank = recommendations.index(pos_item) if pos_item in recommendations else 999
                neg_rank = recommendations.index(neg_item) if neg_item in recommendations else 999
                
                if pos_rank < neg_rank:
                    correct += 1
                elif pos_rank == neg_rank:
                    correct += 0.5
                
                total += 1
            
            if total > 0:
                user_auc = correct / total
                user_aucs.append(user_auc)
        
        except Exception as e:
            continue
    
    return np.mean(user_aucs) if user_aucs else 0.5


def evaluate_model(model, test_data, all_items, k_values=[5, 10, 20], use_fast_auc=True):
    """
    综合评估模型性能
    
    Args:
        model: 推荐模型
        test_data: 测试数据 {user_id: [item_ids]}
        all_items: 所有物品列表
        k_values: K 值列表
        use_fast_auc: 是否使用快速 AUC 计算
    
    Returns:
        results: 评估结果字典
    """
    print("\n生成推荐列表...")
    
    # 生成推荐列表
    recommendations = {}
    max_k = max(k_values)
    
    for user_id in test_data.keys():
        try:
            recommendations[user_id] = model.recommend(user_id, n=max_k)
        except Exception as e:
            recommendations[user_id] = []
    
    results = {
        'hit_rate': {},
        'precision': {},
        'recall': {},
        'f1': {},
        'mrr': {},
        'ndcg': {},
        'auc': {}
    }
    
    # 计算各个 K 值的指标
    for k in k_values:
        print(f"\n计算 K={k} 的指标...")
        
        results['hit_rate'][k] = calculate_hit_rate(recommendations, test_data, k)
        results['precision'][k] = calculate_precision(recommendations, test_data, k)
        results['recall'][k] = calculate_recall(recommendations, test_data, k)
        results['f1'][k] = calculate_f1(recommendations, test_data, k)
        results['mrr'][k] = calculate_mrr(recommendations, test_data, k)
        results['ndcg'][k] = calculate_ndcg(recommendations, test_data, k)
        
        print(f"  Hit Rate@{k}: {results['hit_rate'][k]:.4f}")
        print(f"  Precision@{k}: {results['precision'][k]:.4f}")
        print(f"  Recall@{k}: {results['recall'][k]:.4f}")
        print(f"  F1@{k}: {results['f1'][k]:.4f}")
        print(f"  MRR@{k}: {results['mrr'][k]:.4f}")
        print(f"  NDCG@{k}: {results['ndcg'][k]:.4f}")
    
    # 计算 AUC（只计算一次，不依赖 K）
    print(f"\n计算 AUC...")
    
    if use_fast_auc:
        auc = calculate_auc_fast(model, test_data, all_items, 
                                 sample_users=200, 
                                 sample_pairs_per_user=25)
    else:
        auc = calculate_auc(model, test_data, all_items, n_samples=5000)
    
    # AUC 对所有 K 值相同
    for k in k_values:
        results['auc'][k] = auc
    
    print(f"  AUC: {auc:.4f}")
    
    return results