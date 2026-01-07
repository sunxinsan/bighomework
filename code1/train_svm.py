import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import re
import jieba
import joblib

class DataProcessor:
    def __init__(self, train_path, stopwords_path):
        self.train_path = train_path
        self.stopwords_path = stopwords_path
        self.stopwords = self.load_stopwords()

    def load_stopwords(self):
        """加载停用词表"""
        try:
            with open(self.stopwords_path, 'r', encoding='utf-8') as f:
                stopwords = set(line.strip() for line in f if line.strip())
        except Exception as e:
            print(f"Error loading stopwords from {self.stopwords_path}: {e}")
            stopwords = set()
        return stopwords

    def load_and_clean(self, file_path):
        """
        读取CSV并清洗文本。
        解析 'specific_dialogue_content'，只提取 left 方的发言。
        将 'is_fraud' 转换为 0/1 标签。
        去除 is_fraud 为空的行。
        """
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return pd.DataFrame()

        # 去除 specific_dialogue_content 和 is_fraud 列中的空值
        df = df.dropna(subset=['specific_dialogue_content', 'is_fraud'])

        # 标签转换: True -> 1, False -> 0
        def convert_label(x):
            if pd.isna(x):
                return x
            if isinstance(x, bool):
                return 1 if x else 0
            if isinstance(x, str):
                return 1 if x.lower() in ['true', '1', 'yes', '是', '诈骗'] else 0
            return 0 # 默认为非诈骗

        df['label'] = df['is_fraud'].apply(convert_label)

        # 去除 label 列中转换后仍为空的行
        df = df.dropna(subset=['label'])
        # 确保 label 是整数类型
        df['label'] = df['label'].astype(int)

        # 文本清洗函数 - 修改为只提取 left 方内容
        def clean_dialogue(text):
            # 去除 "音频内容：" 头部
            text = re.sub(r'^音频内容：\s*', '', str(text))
            # 去除首尾引号和多余的空白字符
            text = text.replace('"', '').strip()
            
            # 提取所有 left: 后的内容
            # 正则表达式解释：
            # left:\s* - 匹配 "left:" 后面可能跟着的空格
            # (.*?) - 非贪婪匹配，捕获冒号后的内容
            # (?=right:|$) - 前瞻断言，确保匹配到的内容后面是 "right:" 或者字符串结尾
            left_parts = re.findall(r'left:\s*(.*?)(?=right:|$)', str(text), re.DOTALL)
            
            # 将所有匹配到的 left 部分连接起来，用空格分隔
            left_text_combined = ' '.join([part.strip() for part in left_parts if part.strip()])
            
            # 规范化空白字符
            left_text_combined = re.sub(r'\s+', ' ', left_text_combined).strip()
            
            return left_text_combined

        df['text'] = df['specific_dialogue_content'].apply(clean_dialogue)

        # 注意：如果某行的 left 部分在清洗后为空字符串，可能需要处理
        # 这里可以选择保留（如果模型能处理空字符串）或丢弃
        # df = df[df['text'].str.len() > 0] # 如果需要丢弃空字符串行，取消此行注释

        return df[['text', 'label']]

    def preprocess_chinese_text(self, text):
        """
        对中文文本进行分词和去停用词
        """
        # 分词
        words = jieba.lcut(text)
        # 去除停用词和单字符词
        filtered_words = [word for word in words if word not in self.stopwords and len(word) > 1]
        return ' '.join(filtered_words)

    def get_processed_data(self):
        df = self.load_and_clean(self.train_path)

        if df.empty:
            print("数据加载失败或清洗后为空，请检查数据文件和路径。")
            return None, None, None

        # 简单统计
        print(f"原始数据集大小: {len(df)}")
        print("清洗后标签分布:\n", df['label'].value_counts())

        # 应用中文预处理
        df['processed_text'] = df['text'].apply(self.preprocess_chinese_text)

        # 返回处理后的文本和标签
        return df['processed_text'], df['label']

# ========================
# 主训练流程 (与之前相同)
# ========================

# 1. 初始化处理器
processor = DataProcessor(
    train_path='data/训练集结果.csv', # 修改为你的训练集路径
    stopwords_path='data/cn_stopwords.txt'     # 修改为你的停用词文件路径
)

# 2. 获取处理后的数据
X_raw, y = processor.get_processed_data()

if X_raw is None or y is None:
    print("无法获取有效数据，脚本退出。")
    exit()

# 3. 特征提取（TF-IDF）
vectorizer = TfidfVectorizer(
    max_features=5000,           # 控制特征数量，防止过拟合
    ngram_range=(1, 2),         # 使用 unigram 和 bigram
    min_df=2,                   # 忽略出现少于2次的词
    max_df=0.95                 # 忽略在超过95%文档中出现的词
)

X = vectorizer.fit_transform(X_raw)
y = y

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. 训练 SVM 模型
svm_model = SVC(
    kernel='linear',           # 线性核适合文本分类
    C=1.0,                     # 正则化参数，防止过拟合
    class_weight='balanced',   # 处理类别不平衡
    random_state=42
)

# 交叉验证评估
cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5, scoring='f1')
print(f"\nCross-validation F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 训练模型
svm_model.fit(X_train, y_train)

# 预测
y_pred = svm_model.predict(X_test)

# 输出评估结果
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 6. 保存模型和向量化器
joblib.dump(svm_model, 'svm_fraud_detector_left_only.pkl') # 修改模型保存名以区分
joblib.dump(vectorizer, 'tfidf_vectorizer_left_only.pkl') # 修改向量化器保存名以区分
joblib.dump(processor, 'data_processor_left_only.pkl') # 修改处理器保存名以区分

print("\n✅ 模型、向量化器和数据处理器（仅left）已保存！")

# ========================
# 如何使用训练好的模型预测新对话？（只看left）
# ========================

def predict_new_dialogue_with_threshold(dialogue_text, model, vectorizer, processor, threshold=0.0):
    """
    使用自定义阈值预测新对话是否为诈骗。只分析left方内容。

    Args:
        dialogue_text (str): 待预测的对话文本。
        model: 训练好的SVM模型。
        vectorizer: 训练好的TF-IDF向量化器。
        processor: 数据处理器实例。
        threshold (float): 判断为诈骗的置信度阈值。默认为0.0。

    Returns:
        tuple: (预测类别 (bool), 置信度分数 (float), 是否为诈骗 (bool))
    """
    # 1. 清洗原始文本，只提取left内容
    cleaned_raw_text = processor.load_and_clean.__code__.co_consts[1] # 获取内部函数，不优雅但可行
    # 更好的方式是将 clean_dialogue 和 preprocess_chinese_text 作为公共方法
    # 这里我们直接调用 processor 的方法
    left_only_text = processor.load_and_clean.__code__.co_consts[1](dialogue_text) # 获取内部函数
    # 实际上，我们应该将 clean_dialogue 作为公共方法
    # 假设我们将 clean_dialogue 方法改为 self._extract_left_text
    # left_only_text = processor._extract_left_text(dialogue_text)

    # 为了代码的可读性和复用性，我们在这里重新实现提取逻辑
    def extract_left_text(text):
        text = re.sub(r'^音频内容：\s*', '', str(text))
        text = text.replace('"', '').strip()
        left_parts = re.findall(r'left:\s*(.*?)(?=right:|$)', str(text), re.DOTALL)
        left_text_combined = ' '.join([part.strip() for part in left_parts if part.strip()])
        left_text_combined = re.sub(r'\s+', ' ', left_text_combined).strip()
        return left_text_combined

    left_only_text = extract_left_text(dialogue_text)

    # 2. 分词和去停用词
    processed_text = processor.preprocess_chinese_text(left_only_text)
    
    # 3. 向量化
    vec = vectorizer.transform([processed_text])
    
    # 4. 获取决策函数分数 (置信度)
    confidence_score = model.decision_function(vec)[0]
    
    # 5. 根据自定义阈值判断是否为诈骗
    is_fraud_prediction = confidence_score >= threshold
    
    return is_fraud_prediction, confidence_score, is_fraud_prediction

# 示例预测
new_dialogue = """
音频内容：

left: 喂，你好，我是小李淘宝客服专员。
right: 你好，有什么事情吗？
left: 我们注意到你前几天在我们店铺购买了一件外套，但是由于物流失误，然后导致你的包裹丢失了。
right: 是吗？那怎么办？
left: 为了补偿你，我们可以提供你双倍的退款，但是需要你点击我们提供的链接，填写一些必要的信息。
right: 好的，那我怎么操作呢？
left: 你只要点击这个链接，提供这个链接，嗯，按照提示操作，然后就可以快速完成退款。
right: 那我需要提供什么信息？
left: 主要是你的银行卡号和验证码，这样我们才能把钱退到你的账户。
right: 好的，我现在就去操作。
"""

# 设置一个示例阈值，例如 0.0 (SVM默认阈值)
# THRESHOLD = 0.0

# is_fraud_pred, conf_score, is_actually_fraud = predict_new_dialogue_with_threshold(
#     new_dialogue, loaded_model, loaded_vectorizer, loaded_processor, threshold=THRESHOLD
# )

# print(f"\n使用阈值 {THRESHOLD} (只分析left方):")
# print(f"提取的left方内容: '{left_only_text}'") # 验证提取是否正确
# print(f"决策函数分数 (置信度): {conf_score:.4f}")
# print(f"预测结果: {'是诈骗' if is_actually_fraud else '非诈骗'}")

# 注意：在实际使用时，你需要加载刚才保存的模型和处理器
# loaded_model = joblib.load('svm_fraud_detector_left_only.pkl')
# loaded_vectorizer = joblib.load('tfidf_vectorizer_left_only.pkl')
# loaded_processor = joblib.load('data_processor_left_only.pkl')