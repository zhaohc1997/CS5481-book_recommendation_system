# 📚 智能图书推荐系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.2-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个基于多种机器学习算法的智能图书推荐系统，提供个性化的图书推荐服务。项目实现了协同过滤、矩阵分解、LightFM 和 LightGBM 四种推荐算法，并采用两阶段架构（召回+排序）进行模型集成，提供 Web 界面进行在线演示。

## 🎯 项目特性

- **🤖 多算法集成**: 实现了 4 种经典推荐算法
  - Item-Based Collaborative Filtering (物品协同过滤)
  - Matrix Factorization (矩阵分解)
  - LightFM (混合推荐模型)
  - LightGBM (梯度提升排序)
  
- **🏗️ 两阶段架构**: 召回阶段 + 排序阶段的模型融合策略
- **🌐 Web 交互界面**: 基于 Flask 的友好用户界面，支持实时推荐
- **📊 全面评估**: 多维度评估指标（Precision、Recall、NDCG、Coverage 等）
- **🔧 可扩展架构**: 模块化设计，易于添加新算法和功能

## 🏗️ 项目架构

```
book_recommendation_system/
├── data/                          # 数据目录
│   ├── raw/                       # 原始数据（Books.csv, Ratings.csv, Users.csv）
│   ├── processed/                 # 清洗后的数据
│   └── models/                    # 训练好的模型文件
├── src/                           # 源代码
│   ├── data_processing/           # 数据处理模块
│   │   ├── data_loader.py         # 数据加载
│   │   ├── data_cleaner.py        # 数据清洗
│   │   └── feature_engineer.py    # 特征工程
│   ├── models/                    # 推荐模型
│   │   ├── collaborative_filtering.py   # 协同过滤
│   │   ├── matrix_factorization.py      # 矩阵分解
│   │   ├── lightfm_model.py             # LightFM
│   │   ├── lightgbm_ranker.py           # LightGBM 排序
│   │   └── two_stage_recommender.py     # 两阶段系统
│   ├── evaluation/                # 评估模块
│   │   ├── metrics.py             # 评估指标
│   │   └── compare_four_models.py # 模型对比
│   ├── web/                       # Web 应用
│   │   ├── app.py                 # Flask 应用
│   │   └── templates/             # HTML 模板
│   └── utils/                     # 工具函数
│       └── config.py              # 配置管理
├── notebooks/                     # Jupyter 笔记本
│   ├── 01_data_exploration.ipynb  # 数据探索
│   └── 02_data_analysis.ipynb     # 数据分析
├── results/                       # 实验结果
│   ├── figures/                   # 可视化图表
│   └── four_models_comparison/    # 模型对比结果
├── tests/                         # 测试文件
├── config.yaml                    # 系统配置文件
├── requirements.txt               # Python 依赖
├── main.py                        # 主程序（训练所有模型）
└── README.md                      # 项目文档
```

## 🚀 快速开始

### 环境要求

- **Python**: 3.8 或更高版本
- **操作系统**: Windows / macOS / Linux
- **内存**: 建议 4GB 以上（用于模型训练）

### 安装步骤

#### 1. 克隆项目

```bash
git clone https://github.com/YourUsername/book-recommendation-system.git
cd book-recommendation-system
```

#### 2. 创建虚拟环境

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

#### 4. 准备数据集

本项目使用 [Book-Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)：

- **方式一**: 从 Kaggle 下载
  ```bash
  # 下载后解压到 data/raw/ 目录
  # 需要的文件：Books.csv, Ratings.csv, Users.csv
  ```

- **方式二**: 手动下载并放置
  ```
  data/
  └── raw/
      ├── Books.csv
      ├── Ratings.csv
      └── Users.csv
  ```

#### 5. 数据预处理（可选）

如果需要重新清洗数据：

```bash
python src/data_processing/data_cleaner.py
```

处理后的数据会保存到 `data/processed/` 目录。

#### 6. 训练模型

运行主程序训练所有模型：

```bash
python main.py
```

训练过程包括：
- ✅ Item-Based Collaborative Filtering
- ✅ Matrix Factorization (SVD)
- ✅ LightFM
- ✅ LightGBM Ranker
- ✅ Two-Stage Recommender System

训练完成后，模型会保存到 `data/models/` 目录。

#### 7. 启动 Web 应用

```bash
python src/web/app.py
```

打开浏览器访问：**http://localhost:5000**

## 📊 使用说明

### Web 界面功能

1. **首页推荐**
   - 输入图书名称，获取个性化推荐
   - 支持 4 种不同推荐算法选择

2. **图书搜索**
   - 按书名、作者、ISBN 搜索图书
   - 查看图书详细信息

3. **模型对比**
   - 可视化展示 4 种算法的性能对比
   - 包含 Precision、Recall、NDCG 等指标

4. **图书详情**
   - 查看图书封面、作者、出版信息
   - 获取基于该书的相似推荐

### API 接口

系统提供 RESTful API：

```bash
# 获取推荐（基于图书名称）
GET /api/recommend?book_title=Harry Potter&n=5&model=ItemBasedCF

# 搜索图书
GET /api/search?query=Tolkien&limit=10

# 获取图书详情
GET /api/book/<isbn>
```

**参数说明**：
- `book_title`: 图书名称
- `n`: 推荐数量（默认 5）
- `model`: 推荐算法（ItemBasedCF / MatrixFactorization / LightFM / TwoStage）

## 🤖 算法介绍

### 1. Item-Based Collaborative Filtering (物品协同过滤)

**原理**: 基于"喜欢相似物品"的假设，计算物品之间的相似度，推荐与用户历史喜好相似的物品。

**实现**:
- 使用余弦相似度计算图书之间的相似性
- 构建物品-物品相似度矩阵
- 根据用户评分历史加权推荐

**优点**: 
- 解释性强，推荐结果直观
- 对于物品数量相对稳定的场景效果好

### 2. Matrix Factorization (矩阵分解)

**原理**: 将用户-物品评分矩阵分解为用户潜在因子矩阵和物品潜在因子矩阵。

**实现**:
- 使用 SVD (Singular Value Decomposition)
- 学习用户和图书的低维表示（Embedding）
- 通过内积预测评分

**优点**:
- 能够捕捉潜在特征
- 处理稀疏矩阵效果好
- 可扩展性强

### 3. LightFM (混合推荐模型)

**原理**: 结合协同过滤和内容特征的混合模型，支持冷启动问题。

**实现**:
- 使用 WARP (Weighted Approximate-Rank Pairwise) 损失函数
- 同时利用用户-物品交互和元数据特征
- 生成用户和物品的 Embedding

**优点**:
- 解决冷启动问题
- 结合协同和内容信息
- 训练效率高

### 4. LightGBM Ranker (梯度提升排序)

**原理**: 使用梯度提升决策树进行学习排序（Learning to Rank）。

**实现**:
- 提取用户-物品特征（评分统计、流行度等）
- 使用 LambdaRank 目标函数
- 对候选物品进行精排

**优点**:
- 特征工程灵活
- 排序效果优秀
- 可解释性较好

### 5. Two-Stage Recommender (两阶段推荐系统)

**架构**: 召回 (Recall) + 排序 (Ranking)

**召回阶段**:
- 使用多个模型（CF、MF、LightFM）生成候选集
- 加权融合多个召回源
- 快速筛选出 Top-N 候选

**排序阶段**:
- 使用 LightGBM 对候选集精排
- 基于更多特征进行打分
- 输出最终推荐列表

**优点**:
- 结合多模型优势
- 召回率和精准度兼顾
- 工业界主流架构

## 📈 评估指标

本项目使用以下指标评估推荐系统性能：

| 指标 | 说明 | 计算公式 |
|------|------|----------|
| **Precision@K** | 推荐列表中相关物品的比例 | $\frac{\text{相关推荐数}}{\text{推荐总数}}$ |
| **Recall@K** | 相关物品中被推荐的比例 | $\frac{\text{相关推荐数}}{\text{相关物品总数}}$ |
| **F1-Score@K** | Precision 和 Recall 的调和平均 | $\frac{2 \times P \times R}{P + R}$ |
| **NDCG@K** | 归一化折扣累积增益 | 考虑排序位置的质量指标 |
| **Coverage** | 推荐结果覆盖的物品比例 | $\frac{\text{被推荐物品数}}{\text{总物品数}}$ |
| **Diversity** | 推荐列表的多样性 | 基于物品相似度的平均差异 |

### 模型性能对比

运行模型对比脚本：

```bash
python src/evaluation/compare_four_models.py
```

结果示例：

| 模型 | Precision@5 | Recall@5 | NDCG@5 | Coverage |
|------|-------------|----------|---------|----------|
| ItemBasedCF | 0.245 | 0.182 | 0.267 | 0.432 |
| MatrixFactorization | 0.228 | 0.175 | 0.251 | 0.385 |
| LightFM | 0.236 | 0.179 | 0.259 | 0.411 |
| TwoStage | **0.268** | **0.198** | **0.289** | 0.456 |

*注：实际结果可能因数据集和参数而异*

## 🔧 配置说明

系统配置文件：`config.yaml`

```yaml
# 数据配置
data:
  min_user_ratings: 5      # 最小用户评分数量（过滤低活跃用户）
  min_book_ratings: 10     # 最小图书评分数量（过滤冷门图书）
  test_size: 0.2           # 测试集比例
  random_state: 42         # 随机种子

# 模型配置
models:
  # 协同过滤参数
  collaborative_filtering:
    similarity_metric: 'cosine'  # 相似度度量：cosine / pearson
    
  # 矩阵分解参数
  matrix_factorization:
    n_factors: 50           # 潜在因子数量
    learning_rate: 0.01     # 学习率
    regularization: 0.02    # 正则化系数
    n_epochs: 20            # 训练轮数
    
  # LightGBM 参数
  lightgbm:
    num_leaves: 31          # 叶子节点数
    learning_rate: 0.05     # 学习率
    n_estimators: 100       # 树的数量
    
  # LightFM 参数
  lightfm:
    no_components: 30       # Embedding 维度
    loss: 'warp'            # 损失函数：warp / bpr / logistic
    learning_rate: 0.05     # 学习率
    epochs: 10              # 训练轮数
    
  # 两阶段系统权重
  ensemble:
    equal_weight: true      # 是否使用等权重
    staged_pipeline:
      recall_weight: 0.3    # 召回阶段权重
      rerank_weight: 0.7    # 排序阶段权重

# 评估配置
evaluation:
  k: 5                      # Top-K 推荐
  n_samples: 100            # 评估样本数
  metrics:                  # 评估指标
    - precision
    - recall
    - f1
    - ndcg
    - coverage
```

### 修改配置

编辑 `config.yaml` 后重新训练模型即可生效：

```bash
python main.py
```

## 🛠️ 开发指南

### 添加新的推荐算法

1. **创建模型类**

在 `src/models/` 目录创建新文件，例如 `my_model.py`：

```python
class MyModel:
    def __init__(self):
        self.model = None
    
    def train(self, ratings_df, books_df, users_df=None):
        """训练模型"""
        # 实现训练逻辑
        pass
    
    def recommend(self, item_id, n=5):
        """生成推荐"""
        # 实现推荐逻辑
        return recommended_items
    
    def save_model(self, filepath):
        """保存模型"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    def load_model(self, filepath):
        """加载模型"""
        import pickle
        with open(filepath, 'rb') as f:
            loaded = pickle.load(f)
            self.__dict__.update(loaded.__dict__)
```

2. **在主程序中注册**

在 `main.py` 中添加：

```python
from models.my_model import MyModel

# 初始化
my_model = MyModel()

# 训练
my_model.train(ratings_df, books_df)

# 保存
my_model.save_model('data/models/my_model.pkl')
```

3. **在 Web 应用中集成**

在 `src/web/app.py` 中加载和使用新模型。

### 扩展评估指标

在 `src/evaluation/metrics.py` 添加新指标：

```python
def my_metric(y_true, y_pred):
    """自定义评估指标"""
    # 实现计算逻辑
    return score
```

### 数据处理流程

```python
# 1. 加载原始数据
from data_processing.data_loader import load_raw_data
books, ratings, users = load_raw_data()

# 2. 数据清洗
from data_processing.data_cleaner import clean_data
books_clean, ratings_clean, users_clean = clean_data(books, ratings, users)

# 3. 特征工程
from data_processing.feature_engineer import create_features
features = create_features(ratings_clean, books_clean)
```

## 📊 数据集说明

本项目使用 **Book-Crossing Dataset**，包含：

### Books.csv
- **ISBN**: 图书唯一标识符
- **Book-Title**: 图书标题
- **Book-Author**: 作者
- **Year-Of-Publication**: 出版年份
- **Publisher**: 出版社
- **Image-URL-S/M/L**: 封面图片链接

### Ratings.csv
- **User-ID**: 用户 ID
- **ISBN**: 图书 ISBN
- **Book-Rating**: 评分（0-10）

### Users.csv
- **User-ID**: 用户 ID
- **Location**: 地理位置
- **Age**: 年龄

**数据统计**（清洗后）：
- 图书数量：~50,000
- 用户数量：~10,000
- 评分记录：~200,000
- 评分范围：0-10（隐式评分 0，显式评分 1-10）

## 📝 Jupyter Notebooks

项目包含两个数据分析笔记本：

1. **01_data_exploration.ipynb**
   - 数据基本统计
   - 缺失值分析
   - 评分分布可视化
   - 用户/图书活跃度分析

2. **02_data_analysis.ipynb**
   - 图书流行度分析
   - 用户行为模式
   - 评分时间序列分析
   - 特征相关性分析

运行笔记本：

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## 🧪 测试

运行测试用例：

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python tests/test_basic.py
```

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码规范

- 遵循 PEP 8 编码规范
- 添加适当的注释和文档字符串
- 编写单元测试
- 更新 README 文档

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 👥 作者

**CS 5481 - Data Engineering**  
City University of Hong Kong（DG）  
2025 Semester A

## 🙏 致谢

- 数据集来源：[Book-Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)
- 参考框架：LightFM, LightGBM, Scikit-learn
- UI 框架：Bootstrap 5

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue：[GitHub Issues](https://github.com/YourUsername/book-recommendation-system/issues)
- Email: 72515790@cityu-dg.edu.cn

## 🔗 相关资源

- [推荐系统实践（项亮）](https://book.douban.com/subject/10769749/)
- [LightFM 文档](https://making.lyst.com/lightfm/docs/home.html)
- [LightGBM 文档](https://lightgbm.readthedocs.io/)
- [Flask 文档](https://flask.palletsprojects.com/)

---

⭐ 如果这个项目对您有帮助，请给个 Star！# CS5481-book_recommendation_system
