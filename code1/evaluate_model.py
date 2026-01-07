# evaluate_model.py
import pandas as pd
import jieba
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import joblib
import os
import sys
from datetime import datetime

# ======================
# 配置路径
# ======================
MODEL_PATH = "svm_fraud_detector_left_only.pkl"
VECTORIZER_PATH = "tfidf_vectorizer_left_only.pkl"
LOG_DIR = "logs"
RESULT_DIR = "results"

# ======================
# 必须与训练时完全一致！否则无法加载向量化器
# ======================
def chinese_tokenizer(text):
    return jieba.lcut(str(text))

# ======================
# 日志工具：同时输出到控制台和文件
# ======================
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ======================
# 加载模型和向量化器
# ======================
def load_model_and_vectorizer():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ 模型文件不存在: {os.path.abspath(MODEL_PATH)}")
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"❌ 向量化器文件不存在: {os.path.abspath(VECTORIZER_PATH)}")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✅ 模型与向量化器加载成功！")
    return model, vectorizer

# ======================
# 提取纯对话内容（去除 left/right 标记）
# ======================
def extract_dialogue(text):
    if pd.isna(text):
        return ""
    clean = str(text).replace("音频内容：", "").rstrip(" **")
    # 提取 left/right 后的内容
    turns = re.findall(r'(?:left|right):\s*(.*?)(?=\s*(?:left|right):|\s*$)', clean)
    return " ".join(turn.strip() for turn in turns if turn.strip())

# ======================
# 加载测试集（关键修复：处理非字符串标签）
# ======================
def load_test_data(csv_path, has_header=True):
    df = pd.read_csv(csv_path, header=0 if has_header else None, on_bad_lines='skip')
    
    # 确保列名正确（你的真实列名）
    expected_columns = ["specific_dialogue_content", "interaction_strategy", "call_type", "is_fraud", "fraud_type"]
    if has_header:
        # 检查是否包含关键列
        if "specific_dialogue_content" not in df.columns:
            raise ValueError(f"❌ 找不到列 'specific_dialogue_content'。当前列: {list(df.columns)}")
        if "is_fraud" not in df.columns:
            raise ValueError(f"❌ 找不到标签列 'is_fraud'。")
    else:
        # 如果无 header，强制命名（备用）
        if len(df.columns) >= 5:
            df.columns = expected_columns
        else:
            raise ValueError("❌ 数据列数不足，无法匹配训练格式。")

    # 映射为内部使用的 raw_text
    df["raw_text"] = df["specific_dialogue_content"]
    
    # 提取干净文本
    df["text"] = df["raw_text"].apply(extract_dialogue)
    df = df[df["text"].str.len() >= 5].reset_index(drop=True)
    
    # =============== 关键修复：处理非字符串标签 ===============
    if "is_fraud" in df.columns:
        # 1. 删除缺失标签的行
        initial_count = len(df)
        df = df.dropna(subset=["is_fraud"])
        dropped_count = initial_count - len(df)
        if dropped_count > 0:
            print(f"⚠️ 已删除 {dropped_count} 条缺失标签的样本")
        
        # 2. 确保 is_fraud 是字符串类型（关键修复！）
        df["is_fraud"] = df["is_fraud"].astype(str).str.lower()
        
        # 3. 统一处理标签值（处理各种表示形式）
        df["is_fraud"] = df["is_fraud"].replace(["true", "1", "yes", "t", "y"], "true")
        df["is_fraud"] = df["is_fraud"].replace(["false", "0", "no", "f", "n"], "false")
        
        # 4. 映射为 0/1
        df["label"] = df["is_fraud"].map({"true": 1, "false": 0})
        
        # 5. 处理无效值（确保所有值都是 'true' 或 'false'）
        invalid_mask = ~df["is_fraud"].isin(["true", "false"])
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            print(f"⚠️ 发现 {invalid_count} 条无效标签，已统一替换为 'false'")
            df.loc[invalid_mask, "label"] = 0
    else:
        raise ValueError("❌ 标签列 'is_fraud' 不存在，无法评估。")
    
    print(f"📊 测试集加载成功！共 {len(df)} 条有效样本")
    print(f"   - 欺诈样本: {df['label'].sum()}")
    print(f"   - 正常样本: {len(df) - df['label'].sum()}")
    
    return df

# ======================
# 评估并保存结果
# ======================
def evaluate_and_save_results(test_df, model, vectorizer, result_csv_path):
    X_test = vectorizer.transform(test_df["text"])
    y_pred = model.predict(X_test)
    
    result_df = test_df.copy()
    result_df["predicted_label"] = y_pred
    result_df["predicted_class"] = result_df["predicted_label"].map({1: "欺诈", 0: "正常"})
    result_df["true_class"] = result_df["label"].map({1: "欺诈", 0: "正常"})
    result_df["correct"] = (result_df["label"] == result_df["predicted_label"])
    
    # 计算指标（现在 y_true 保证无 NaN）
    y_true = test_df["label"].values
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    print("\n=== 模型评估结果 ===")
    print(f"准确率 (Accuracy): {acc:.3f}")
    print(f"欺诈类精确率 (Precision): {prec:.3f}")
    print(f"欺诈类召回率 (Recall): {rec:.3f}")
    print(f"欺诈类 F1 分数: {f1:.3f}")
   
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=["正常", "欺诈"], digits=3))
    
    # 保存结果（支持中文 Excel）
    os.makedirs(os.path.dirname(result_csv_path), exist_ok=True)
    result_df.to_csv(result_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 预测结果已保存至: {result_csv_path}")
    
    return {"acc": acc, "f1": f1}

# ======================
# 主流程
# ======================
if __name__ == "__main__":
    # 创建输出目录
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"eval_{timestamp}.log")
    result_csv = os.path.join(RESULT_DIR, f"predictions_{timestamp}.csv")
    
    # 重定向输出到日志 + 控制台
    sys.stdout = Logger(log_file)
    
    print("=" * 60)
    print(f"🚀 开始评估模型 | 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型路径: {os.path.abspath(MODEL_PATH)}")
    
    # 👇 修改这里指定你的测试集路径（确保路径正确）
    test_csv = "data/测试集结果.csv"
    print(f"测试集路径: {os.path.abspath(test_csv)}")
    print(f"结果将保存至: {result_csv}")
    print("=" * 60)
    
    # 执行评估
    model, vectorizer = load_model_and_vectorizer()
    test_df = load_test_data(test_csv, has_header=True)  # ✅ 你的数据有表头
    evaluate_and_save_results(test_df, model, vectorizer, result_csv)
    
    print(f"\n📄 完整日志已保存至: {log_file}")