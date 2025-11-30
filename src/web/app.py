# src/web/app.py

"""
智能图书推荐系统 Web 应用
支持 4 种推荐模型的在线演示
"""
from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
import sys
import html

sys.path.append('src')

from models.collaborative_filtering import ItemBasedCF
from models.matrix_factorization import MatrixFactorization
from models.lightfm_model import LightFMModel
from models.lightgbm_ranker import LightGBMRanker
from models.two_stage_recommender import TwoStageRecommender

app = Flask(__name__)
app.config['SECRET_KEY'] = 'book-recommendation-system-2025'

# 全局变量声明
models = {}
books_df = None
ratings_df = None
popular_titles = []


def load_data():
    """加载数据和模型（修复版）"""
    global books_df, ratings_df, models, popular_titles
    
    print("="*60)
    print("加载数据和模型")
    print("="*60)
    
    # ============================================================
    # 加载数据
    # ============================================================
    try:
        books_df = pd.read_csv('data/processed/books_clean.csv')
        ratings_df = pd.read_csv('data/processed/ratings_clean.csv')
        print(f"✓ 图书数据: {len(books_df):,}")
        print(f"✓ 评分数据: {len(ratings_df):,}")
        
        # 缓存热门图书标题
        popular_isbns = ratings_df['ISBN'].value_counts().head(100).index
        for isbn in popular_isbns:
            book = books_df[books_df['ISBN'] == isbn]
            if not book.empty:
                popular_titles.append(book.iloc[0]['Book-Title'])
        print(f"✓ 缓存热门图书: {len(popular_titles)} 本")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False
    
    # ============================================================
    # 加载普通模型（ItemBasedCF, MatrixFactorization, LightFM）
    # ============================================================
    print("\n加载模型...")
    
    model_configs = [
        ('ItemBasedCF', ['data/models/itembasedcf.pkl', 'data/models/ItemBasedCF.pkl'], ItemBasedCF),
        ('MatrixFactorization', ['data/models/matrixfactorization.pkl', 'data/models/MatrixFactorization.pkl'], MatrixFactorization),
        ('LightFM', ['data/models/lightfm.pkl', 'data/models/LightFM.pkl'], LightFMModel),
    ]
    
    for name, filepaths, model_class in model_configs:
        loaded = False
        for filepath in filepaths:
            if not os.path.exists(filepath):
                continue
            
            try:
                model = model_class()
                model.load_model(filepath)
                models[name] = model
                print(f"✓ {name} (从 {filepath})")
                loaded = True
                break
            except Exception as e:
                print(f"⚠️  {name} 从 {filepath} 加载失败: {e}")
        
        if not loaded:
            print(f"❌ {name}: 未找到可用文件")
    
    # ============================================================
    # 特殊处理：TwoStage 模型
    # ============================================================
    print("\n加载 TwoStage 模型...")
    
    two_stage_paths = [
        'data/models/two_stage_model.pkl',
        'data/models/TwoStage_System.pkl'
    ]
    
    two_stage_loaded = False
    
    for filepath in two_stage_paths:
        print(f"  尝试: {filepath}")
        
        if not os.path.exists(filepath):
            print(f"    ⚠️  文件不存在")
            continue
        
        try:
            # ✅ 关键修复：先加载 pickle 数据，再手动重建实例
            print(f"    📂 文件存在，开始加载...")
            
            with open(filepath, 'rb') as f:
                saved_data = pickle.load(f)
            
            print(f"    ✓ Pickle 数据加载成功")
            print(f"    数据键: {list(saved_data.keys())}")
            
            # 提取保存的数据
            recall_models = saved_data.get('recall_models', {})
            ranking_model = saved_data.get('ranking_model')
            saved_books_df = saved_data.get('books_df')
            model_weights = saved_data.get('model_weights', {})
            
            # 创建 TwoStageRecommender 实例
            two_stage = TwoStageRecommender(
                recall_models=recall_models,
                ranking_model=ranking_model,
                model_weights=model_weights
            )
            
            # 设置 books_df
            if saved_books_df is not None:
                two_stage.books_df = saved_books_df
                print(f"    ✓ 使用保存的 books_df")
            else:
                two_stage.books_df = books_df
                print(f"    ✓ 使用当前 books_df")
            
            # 验证模型
            if hasattr(two_stage, 'recall_models'):
                if isinstance(two_stage.recall_models, dict):
                    print(f"    ✓ 召回模型: {list(two_stage.recall_models.keys())}")
                elif isinstance(two_stage.recall_models, list):
                    print(f"    ✓ 召回模型数量: {len(two_stage.recall_models)}")
            
            if hasattr(two_stage, 'ranking_model') and two_stage.ranking_model:
                print(f"    ✓ 排序模型: {two_stage.ranking_model.__class__.__name__}")
            
            if hasattr(two_stage, 'model_weights'):
                print(f"    ✓ 模型权重: {two_stage.model_weights}")
            
            models['TwoStage'] = two_stage
            print(f"✅ TwoStage 加载成功 (从 {filepath})")
            two_stage_loaded = True
            break
            
        except Exception as e:
            print(f"    ❌ 加载失败: {e}")
            import traceback
            traceback.print_exc()
    
    if not two_stage_loaded:
        print(f"❌ TwoStage 加载失败")
        print(f"\n💡 解决方法:")
        print(f"  1. 运行: python main.py")
        print(f"  2. 确保看到 '✓ TwoStage_System' 保存成功")
    
    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "="*60)
    if not models:
        print("❌ 没有可用的模型！")
        return False
    
    print(f"✅ 成功加载 {len(models)} 个模型:")
    for name in models.keys():
        print(f"  ✓ {name}")
    print("="*60)
    
    return True
 
def find_book_title(query):
    """智能查找图书标题（多关键词匹配版）"""
    global books_df, ratings_df, popular_titles
    
    # HTML 解码
    query = html.unescape(query.strip())
    
    if not query:
        return None, False
    
    print(f"[查找图书] 输入: '{query}'")
    
    # ============================================================
    # 方法1: 精确匹配
    # ============================================================
    exact_match = books_df[books_df['Book-Title'] == query]
    if not exact_match.empty:
        print(f"  → 精确匹配")
        return exact_match.iloc[0]['Book-Title'], True
    
    # ============================================================
    # 方法2: 使用多个关键词组合匹配
    # ============================================================
    # 提取前几个有意义的词
    import re
    
    # 分割成词组（按逗号、冒号）
    parts = re.split('[,:;]', query)
    
    # 取前两个部分
    if len(parts) >= 2:
        # "Hobbits, Elves, and Wizards" → 取前两个部分
        search_phrase = parts[0].strip() + ',' + parts[1].strip()
        print(f"  → 使用多关键词: '{search_phrase}'")
        
        contains_match = books_df[
            books_df['Book-Title'].str.contains(
                re.escape(search_phrase),  # ✅ 转义特殊字符
                case=False,
                na=False,
                regex=True
            )
        ]
        
        if not contains_match.empty:
            title = contains_match.iloc[0]['Book-Title']
            print(f"  → 多关键词匹配: '{title}'")
            return title, False
    
    # ============================================================
    # 方法3: 使用前3个单词
    # ============================================================
    words = query.split()[:3]  # "Hobbits", "Elves", "and"
    if len(words) >= 2:
        # 去掉停用词
        meaningful_words = [w for w in words if w.lower() not in {'and', 'the', 'a', 'an', 'of'}]
        
        if len(meaningful_words) >= 2:
            # 要求同时包含这些词
            search_phrase = ' '.join(meaningful_words[:2])
            print(f"  → 使用前两个关键词: '{search_phrase}'")
            
            # 检查是否同时包含这两个词
            mask = books_df['Book-Title'].str.contains(meaningful_words[0], case=False, na=False, regex=False)
            mask &= books_df['Book-Title'].str.contains(meaningful_words[1], case=False, na=False, regex=False)
            
            contains_match = books_df[mask]
            
            if not contains_match.empty:
                title = contains_match.iloc[0]['Book-Title']
                print(f"  → 双关键词匹配: '{title}'")
                return title, False
    
    # ============================================================
    # 方法4: 降级到单个关键词
    # ============================================================
    first_word = query.split(',')[0].strip()
    print(f"  → 降级到单关键词: '{first_word}'")
    
    contains_match = books_df[
        books_df['Book-Title'].str.contains(
            first_word,
            case=False,
            na=False,
            regex=False
        )
    ]
    
    if not contains_match.empty:
        title = contains_match.iloc[0]['Book-Title']
        print(f"  → 单关键词匹配: '{title}'")
        return title, False
    
    print(f"  → 未找到匹配")
    return None, False

def get_book_info(isbn):
    """获取图书详细信息"""
    global books_df, ratings_df
    
    book = books_df[books_df['ISBN'] == isbn]
    if book.empty:
        return None
    
    book = book.iloc[0]
    book_ratings = ratings_df[ratings_df['ISBN'] == isbn]
    
    return {
        'title': book.get('Book-Title', 'Unknown'),
        'author': book.get('Book-Author', 'Unknown'),
        'year': book.get('Year-Of-Publication', 'N/A'),
        'publisher': book.get('Publisher', 'Unknown'),
        'isbn': isbn,
        'image_url': book.get('Image-URL-M', '/static/images/book-placeholder.png'),
        'avg_rating': book_ratings['Book-Rating'].mean() if len(book_ratings) > 0 else 0,
        'rating_count': len(book_ratings)
    }


def search_books(query, limit=20):
    """搜索图书"""
    global books_df, ratings_df
    
    if not query:
        popular_isbns = ratings_df['ISBN'].value_counts().head(limit).index
        results = []
        for isbn in popular_isbns:
            info = get_book_info(isbn)
            if info:
                results.append(info)
        return results
    
    mask = books_df['Book-Title'].str.contains(query, case=False, na=False, regex=False)
    matched_books = books_df[mask].head(limit)
    
    results = []
    for _, book in matched_books.iterrows():
        info = get_book_info(book['ISBN'])
        if info:
            results.append(info)
    
    return results


@app.route('/')
def index():
    """首页"""
    global ratings_df, books_df, models
    
    popular_isbns = ratings_df['ISBN'].value_counts().head(12).index
    popular_books = []
    for isbn in popular_isbns:
        info = get_book_info(isbn)
        if info:
            popular_books.append(info)
    
    stats = {
        'total_books': len(books_df),
        'total_ratings': len(ratings_df),
        'total_users': ratings_df['User-ID'].nunique(),
        'models_available': len(models)
    }
    
    return render_template('index.html', 
                         popular_books=popular_books,
                         stats=stats,
                         models=list(models.keys()))


@app.route('/search')
def search():
    """搜索页面"""
    query = request.args.get('q', '')
    results = search_books(query, limit=50)
    
    return render_template('search.html', 
                         query=query,
                         results=results)


@app.route('/api/search_suggestions', methods=['GET'])
def search_suggestions():
    """API: 搜索建议（用于自动完成）"""
    global popular_titles
    
    query = request.args.get('q', '').strip()
    
    if not query or len(query) < 2:
        return jsonify({
            'suggestions': popular_titles[:10]
        })
    
    suggestions = []
    query_lower = query.lower()
    
    for title in popular_titles:
        if query_lower in title.lower():
            suggestions.append(title)
            if len(suggestions) >= 10:
                break
    
    return jsonify({
        'suggestions': suggestions
    })


@app.route('/book/<isbn>')
def book_detail(isbn):
    """图书详情页"""
    global books_df, models
    
    book_info = get_book_info(isbn)
    
    if not book_info:
        return "图书未找到", 404
    
    recommendations = []
    if models:
        model_name = list(models.keys())[0]
        model = models[model_name]
        
        try:
            recs = model.recommend(book_info['title'], n=6)
            for rec in recs:
                if isinstance(rec, dict):
                    rec_title = rec.get('title', '')
                else:
                    rec_title = str(rec)
                
                rec_book = books_df[books_df['Book-Title'] == rec_title]
                if not rec_book.empty:
                    rec_isbn = rec_book.iloc[0]['ISBN']
                    rec_info = get_book_info(rec_isbn)
                    if rec_info:
                        rec_info['score'] = rec.get('score', 0) if isinstance(rec, dict) else 0
                        recommendations.append(rec_info)
        except Exception as e:
            print(f"推荐失败: {e}")
            import traceback
            traceback.print_exc()
    
    return render_template('book_detail.html',
                         book=book_info,
                         recommendations=recommendations,
                         model_name=list(models.keys())[0] if models else 'None')


@app.route('/recommend', methods=['POST'])
def recommend():
    """API: 获取推荐"""
    global books_df, models, popular_titles
    
    data = request.json
    book_title = data.get('book_title', '').strip()
    model_name = data.get('model', 'ItemBasedCF')
    n_recommendations = data.get('n', 10)
    
    print(f"\n[推荐请求] 输入: '{book_title}', 模型: {model_name}")
    
    if model_name not in models:
        return jsonify({'error': f'模型 {model_name} 不可用'}), 400
    
    # 智能查找图书
    actual_title, is_exact = find_book_title(book_title)
    
    if not actual_title:
        return jsonify({
            'success': False,
            'error': f'未找到图书: {book_title}',
            'suggestions': popular_titles[:5]
        }), 404
    
    print(f"  找到图书: '{actual_title}' (精确匹配: {is_exact})")
    
    try:
        model = models[model_name]
        recs = model.recommend(actual_title, n=n_recommendations)
        print(f"  获得 {len(recs)} 条推荐")
        
        results = []
        for rec in recs:
            if isinstance(rec, dict):
                rec_title = rec.get('title', '')
                score = rec.get('score', 0)
            else:
                rec_title = str(rec)
                score = 0
            
            rec_book = books_df[books_df['Book-Title'] == rec_title]
            if not rec_book.empty:
                isbn = rec_book.iloc[0]['ISBN']
                info = get_book_info(isbn)
                if info:
                    info['score'] = float(score)
                    results.append(info)
        
        return jsonify({
            'success': True,
            'model': model_name,
            'query': book_title,
            'actual_title': actual_title,
            'is_exact_match': is_exact,
            'count': len(results),
            'recommendations': results
        })
    
    except Exception as e:
        print(f"❌ 推荐错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/compare')
def compare():
    """模型对比页面"""
    global models, popular_titles
    
    return render_template('compare.html', 
                         models=list(models.keys()),
                         popular_books=popular_titles[:6])


@app.route('/api/compare', methods=['POST'])
def api_compare():
    """API: 对比多个模型"""
    global books_df, models, popular_titles
    
    data = request.json
    book_title = data.get('book_title', '').strip()
    n_recommendations = data.get('n', 5)
    
    print(f"\n[对比请求] 输入: '{book_title}'")
    
    if not book_title:
        return jsonify({'error': '请提供图书标题'}), 400
    
    # 智能查找图书
    actual_title, is_exact = find_book_title(book_title)
    
    if not actual_title:
        return jsonify({
            'success': False,
            'error': f'未找到图书: {book_title}',
            'suggestions': popular_titles[:10]
        }), 404
    
    print(f"  找到图书: '{actual_title}' (精确匹配: {is_exact})")
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n  测试 {model_name}...")
        try:
            recs = model.recommend(actual_title, n=n_recommendations)
            print(f"    获得 {len(recs)} 条推荐")
            
            model_results = []
            for rec in recs:
                if isinstance(rec, dict):
                    rec_title = rec.get('title', '')
                    score = rec.get('score', 0)
                else:
                    rec_title = str(rec)
                    score = 0
                
                rec_book = books_df[books_df['Book-Title'] == rec_title]
                if not rec_book.empty:
                    isbn = rec_book.iloc[0]['ISBN']
                    info = get_book_info(isbn)
                    if info:
                        info['score'] = float(score)
                        model_results.append(info)
            
            results[model_name] = {
                'success': True,
                'recommendations': model_results
            }
            print(f"    ✓ 成功返回 {len(model_results)} 条")
            
        except Exception as e:
            print(f"    ❌ 失败: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {
                'success': False,
                'error': str(e)
            }
    
    return jsonify({
        'success': True,
        'query': book_title,
        'actual_title': actual_title,
        'is_exact_match': is_exact,
        'results': results
    })


@app.route('/about')
def about():
    """关于页面"""
    performance = {}
    try:
        summary_df = pd.read_csv('results/four_models_comparison/summary.csv')
        for _, row in summary_df.iterrows():
            model_name = row['Model']
            # 处理 "TwoStage (Optimized)" 格式
            if 'TwoStage' in model_name:
                model_key = 'TwoStage'
            else:
                model_key = model_name
            
            performance[model_key] = {
                'hit_rate': f"{row['Hit_Rate@10']*100:.2f}%",
                'precision': f"{row['Precision@10']*100:.2f}%",
                'f1': f"{row['F1@10']*100:.2f}%",
                'mrr': f"{row['MRR@10']*100:.2f}%",
                'ndcg': f"{row['NDCG@10']*100:.2f}%",
                'coverage': f"{row['Coverage']*100:.2f}%"
            }
    except Exception as e:
        print(f"加载性能数据失败: {e}")
    
    return render_template('about.html', performance=performance)


@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(e):
    return render_template('500.html'), 500


if __name__ == '__main__':
    if load_data():
        print("\n" + "="*60)
        print("🚀 启动 Web 服务器")
        print("="*60)
        print("\n📍 访问地址:")
        print("   http://127.0.0.1:5000")
        print("   http://localhost:5000")
        print("\n📚 可用功能:")
        print("   - 首页: /")
        print("   - 搜索: /search")
        print("   - 模型对比: /compare")
        print("   - 关于: /about")
        print("\n按 Ctrl+C 停止服务器\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ 启动失败")
        print("\n💡 解决方案:")
        print("  1. 先运行: python main.py (训练模型)")
        print("  2. 确保文件存在:")
        print("     - data/processed/books_clean.csv")
        print("     - data/processed/ratings_clean.csv")
        print("     - data/models/itembasedcf.pkl")
        print("     - data/models/matrixfactorization.pkl")
        print("     - data/models/lightfm.pkl")
        print("     - data/models/two_stage_model.pkl")