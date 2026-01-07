# camouflage_attack_svm_guided_final.py
# 最终版：SVM 指导改写 —— 提取高危关键词，让 GLM 重点替换

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import time
import pandas as pd
import joblib
from zhipuai import ZhipuAI
import jieba
import torch
from bert_score import score as bertscore
import numpy as np
import re
from scipy.sparse import issparse

# 必须与训练时完全一致！
def chinese_tokenizer(text):
    return jieba.lcut(text)

# ==============================
# 🔧 配置区（请根据实际情况修改）
# ==============================
CONFIG = {
    "input_csv": "data/测试集结果.csv",
    "output_csv": "results/camouflage_svm_guided_final.csv",
    "svm_model": "svm_fraud_detector_left_only.pkl",
    "vectorizer": "tfidf_vectorizer_left_only.pkl",
    "api_key": "df9feb23f1a649d585b804dce3eeb7d6.ExubWLd71EDIW2tk",  # 替换为你的 GLM API Key
    "max_samples": 10,
    "max_iterations": 4,
    "min_similarity": 0.70,
    "use_cuda": False,
    "top_k_keywords": 8  # SVM 指导时提取的高危词数量
}

def compute_bertscore(original: str, rewritten: str, use_cuda: bool = False) -> float:
    """使用中文专用 BERT 计算更准确的语义相似度"""
    try:
        P, R, F1 = bertscore(
            [rewritten],
            [original],
            lang="zh",
            model_type="bert-base-chinese",
            rescale_with_baseline=False,
            device="cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )
        return float(F1.mean().item())
    except Exception as e:
        print(f"⚠️ BERTScore 计算失败，返回 0.0: {e}")
        return 0.0

def extract_high_risk_keywords(text, vectorizer, svm_model, top_k=10):
    """
    从文本中提取高风险关键词
    """
    # 1. 文本预处理 (确保与训练时一致)
    clean_text = re.sub(r'^音频内容：\s*', '', str(text))
    clean_text = re.sub(r'(客服|用户|left|right):', '', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    # 分词
    processed_text = ' '.join([w for w in jieba.lcut(clean_text) if len(w) > 1])
    
    # 2. 向量化
    try:
        tfidf_vec = vectorizer.transform([processed_text])
    except Exception as e:
        print(f"向量化失败: {e}")
        return ["转账", "验证码", "公安局", "涉嫌", "冻结", "安全账户"]  # 默认敏感词
    
    # 3. 获取非零特征索引
    rows, cols = tfidf_vec.nonzero()
    
    # 4. 安全检查：确保索引在有效范围内
    valid_indices = []
    for idx in cols:
        # 检查索引是否在 [0, vocab_size-1] 范围内
        if idx < len(vectorizer.get_feature_names_out()):
            valid_indices.append(idx)
        else:
            print(f"⚠️  警告: 索引 {idx} 超出了词汇表范围 (max_index={len(vectorizer.get_feature_names_out())-1})")
    
    # 如果没有有效的索引，返回默认敏感词
    if not valid_indices:
        return ["转账", "验证码", "公安局", "涉嫌", "冻结", "安全账户"]
    
    # 5. 计算每个特征的贡献度 (TF-IDF * SVM 权重)
    # 注意：这里假设 svm_model.coef_ 是密集数组或可以被索引
    # 如果是稀疏矩阵，需要转换或使用 .toarray()
    if issparse(svm_model.coef_):
        coef_array = svm_model.coef_.toarray()[0]  # 转换为密集数组
    else:
        coef_array = svm_model.coef_[0]
    
    contributions = []
    feature_names = vectorizer.get_feature_names_out()
    
    for idx in valid_indices:
        word = feature_names[idx]
        weight = coef_array[idx]
        contribution = tfidf_vec[0, idx] * weight
        contributions.append((word, contribution))
    
    # 6. 排序并返回前 top_k 个
    sorted_contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
    high_risk_words = [word for word, _ in sorted_contributions[:top_k]]
    
    return high_risk_words

def build_attacker_prompt(original: str, history: list, high_risk_words: list) -> str:
    """构建包含 SVM 高危词指导的 Prompt"""
    
    SPEAKER_PREFIXES = ["left:", "right:", "客服:", "用户:", "A:", "B:", "[User]:", "Agent:"]
    has_speaker = any(original.strip().startswith(p) for p in SPEAKER_PREFIXES)
    
    if has_speaker:
        for prefix in SPEAKER_PREFIXES:
            if original.startswith(prefix):
                speaker = prefix
                content = original[len(prefix):].strip()
                break
        instruction_prefix = f"【原始句子】\n{original}\n\n【改写要求】\n仅改写冒号后的内容，必须保留 '{speaker}' 前缀不变。"
    else:
        speaker = None
        instruction_prefix = f"【原始句子】\n{original}"

    # 默认同义替换库（用于 fallback 和示例）
    default_synonyms = {
        "转账": ["打款", "汇款", "把钱转过去", "资金划转"],
        "验证码": ["短信密码", "动态码", "手机收到的6位数", "一次性口令"],
        "公安局": ["公安部门", "警方", "当地派出所", "公安机关"],
        "涉嫌": ["可能涉及", "有嫌疑", "被指称"],
        "立即": ["尽快", "马上", "抓紧时间"],
        "安全账户": ["保护账户", "专用账号", "核查用的账号", "临时保管账户"],
        "冻结": ["暂时锁住", "暂停使用", "限制操作"],
        "点击链接": ["点开那个网址", "访问一下页面", "打开提供的地址"],
        "配合调查": ["协助核实", "帮忙确认情况", "按流程走一下"]
    }

    # 动态生成高危词替换建议
    dynamic_examples = "【本次需重点替换的高危词（来自SVM分析）】\n"
    for word in high_risk_words:
        candidates = default_synonyms.get(word, [f"同义表达_{word}"])
        alt = "、".join(candidates[:3])
        dynamic_examples += f"- “{word}” → {alt}\n"

    base_instruction = f"""你是一位精通中文对抗性文本生成的专家。任务是对以下句子进行**同义改写**，目标是绕过诈骗检测系统。

【核心原则】
1. ✅ **必须优先对上述高危词进行替换**！这是绕过检测的关键。
2. 🔒 **金额、银行卡号、手机号、身份证号、具体时间等关键信息必须原样保留**。
3. 🗣️ **如果原文有 'left:'、'right:' 等前缀，必须原样保留，仅改写其后内容**。
4. 只改写以left为前缀的句子内容（如果有前缀的话）。
5. 在保持句子大概原意的情况下，改写整个句子，避免仅替换单词。

{dynamic_examples}

⚠️ 禁止行为：
- 删除、修改或省略说话人标识（如把 'left:' 删掉 ❌）；
- 添加解释性前缀（如“改写结果：”）；
- 改变事实（如“转账”→“收款”）。

【输出要求】
- 仅输出一行改写后的完整句子；
- 格式必须与原文完全一致（如有前缀，则必须保留）。

{instruction_prefix}

【改写结果】
"""

    if not history:
        return base_instruction

    # === 动态反馈机制 ===
    successful = [h for h in history if h["svm_pred"] == 0]
    last = history[-1]

    feedback = "\n【历史反馈与策略调整】\n"

    if successful:
        ex = successful[-1]["text"]
        feedback += f"- ✅ 成功案例：\"{ex[:70]}...\"\n"
        feedback += "- 请保持相同格式（包括对话方前缀）。\n"
    else:
        if last["similarity"] < CONFIG["min_similarity"]:
            feedback += "- ❌ 语义偏离过大 → 请确保金额、账号、说话人前缀均保留。\n"
        else:
            feedback += "- ⚠️ 语义足够但未骗过模型 → 请更积极替换高危词！\n"
            feedback += "  同时注意：**不要删掉 left/right 等对话方标识！**\n"
    
    feedback += "- 🔑 记住：说话人标识是格式的一部分，必须原样保留！"

    return base_instruction + feedback

def rewrite_with_glm(client, prompt: str) -> str:
    """调用 GLM，避免误删 left:/right: 等合法前缀"""
    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=256
            )
            result = resp.choices[0].message.content.strip()
            
            # 清理 LLM 可能添加的说明性前缀，但保留 left:/right:
            unwanted_prefixes = [
                "改写结果：", "输出：", "：", "句子：", "“", "”", 
                "改写：", "【改写结果】", "结果：", "答：", "改写后的句子："
            ]
            for p in unwanted_prefixes:
                if result.startswith(p):
                    result = result[len(p):].strip()
            
            return result if result else ""
        except Exception as e:
            print(f"  ⚠️ GLM 调用出错: {e}，重试...")
            time.sleep(1)
    return ""

def predict_with_svm(text: str, svm_model, vectorizer) -> int:
    try:
        # 与提取关键词时的清洗逻辑保持一致
        clean_text = re.sub(r'^音频内容：\s*', '', str(text))
        clean_text = re.sub(r'(客服|用户|left|right):', '', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        processed_text = ' '.join([w for w in jieba.lcut(clean_text) if len(w) > 1])
        
        vec = vectorizer.transform([processed_text])
        pred = svm_model.predict(vec)[0]
        return int(pred)
    except Exception as e:
        print(f"  ⚠️ SVM 预测异常: {e}")
        return 1

def attack_single_sample(
    client,
    original: str,
    svm_model,
    vectorizer,
    max_iters: int,
    min_sim: float,
    use_cuda: bool,
    top_k: int
) -> dict:
    # === 提取高危关键词（只做一次）===
    high_risk_words = extract_high_risk_keywords(original, vectorizer, svm_model, top_k=top_k)
    print(f"  🔍 SVM 高危词: {high_risk_words}")

    history = []
    best_result = original
    best_sim = 0.0
    final_pred = 1
    success = False
    used_iters = 0

    print(f"  📝 原始文本: {original}")

    for it in range(1, max_iters + 1):
        used_iters = it
        
        attacker_prompt = build_attacker_prompt(original, history, high_risk_words)
        rewritten = rewrite_with_glm(client, attacker_prompt)
        if not rewritten.strip():
            rewritten = original

        sim_score = compute_bertscore(original, rewritten, use_cuda)
        svm_pred = predict_with_svm(rewritten, svm_model, vectorizer)
        
        history.append({
            "text": rewritten,
            "similarity": sim_score,
            "svm_pred": svm_pred
        })

        if sim_score > best_sim:
            best_sim = sim_score
            best_result = rewritten
            final_pred = svm_pred

        status = "✅ 成功" if (svm_pred == 0 and sim_score >= min_sim) else "❌ 失败"
        print(f"    → 第 {it} 轮: [SVM={svm_pred}] [BERTScore={sim_score:.3f}] {status}")
        print(f"      改写: {rewritten[:150]}{'...' if len(rewritten) > 150 else ''}")

        if svm_pred == 0 and sim_score >= min_sim:
            success = True
            break

        time.sleep(0.3)

    return {
        "adversarial": best_result,
        "attack_success": success,
        "bertscore_similarity": best_sim,
        "svm_prediction_after": final_pred,
        "final_iteration": used_iters,
        "total_attempts": len(history),
        "high_risk_words": ",".join(high_risk_words)  # 便于保存分析
    }

def main():
    args = CONFIG
    
    os.makedirs(os.path.dirname(args["output_csv"]), exist_ok=True)
    client = ZhipuAI(api_key=args["api_key"])
    
    # 加载数据
    df = pd.read_csv(args["input_csv"])
    if "specific_dialogue_content" not in df.columns:
        raise ValueError("输入 CSV 必须包含 'specific_dialogue_content' 列")

    # 加载模型
    svm_model = joblib.load(args["svm_model"])
    vectorizer = joblib.load(args["vectorizer"])

    # 检查是否为线性 SVM
    if not hasattr(svm_model, 'coef_') or svm_model.coef_.ndim != 2:
        raise ValueError("❌ 仅支持线性 SVM（如 LinearSVC 或 SVC(kernel='linear')）")

    # === 用 SVM 预测所有样本，只攻击预测为 1 的 ===
    print("🔍 正在用 SVM 预测所有样本，筛选可攻击对象（SVM_pred == 1）...")
    svm_preds_all = []
    for idx, row in df.iterrows():
        text = str(row["specific_dialogue_content"]).strip()
        pred = predict_with_svm(text, svm_model, vectorizer) if text else 0
        svm_preds_all.append(pred)
    
    df["svm_prediction_original"] = svm_preds_all
    attackable_mask = df["svm_prediction_original"] == 1
    attackable_indices = df[attackable_mask].index.tolist()[:args["max_samples"]]
    print(f"🎯 共找到 {len(df[attackable_mask])} 个 SVM 成功检出的诈骗样本，将攻击前 {len(attackable_indices)} 个\n")

    # 初始化输出 DataFrame
    output_df = df.copy()
    extra_cols = [
        "adversarial", "attack_success", "bertscore_similarity",
        "svm_prediction_after", "final_iteration", "high_risk_words"
    ]
    for col in extra_cols:
        output_df[col] = None

    success_count = 0
    for i, idx in enumerate(attackable_indices, 1):
        original = str(df.at[idx, "specific_dialogue_content"]).strip()
        if not original:
            continue

        print(f"[{i}/{len(attackable_indices)}] 原文: {original[:60]}{'...' if len(original) > 60 else ''}")
        result = attack_single_sample(
            client=client,
            original=original,
            svm_model=svm_model,
            vectorizer=vectorizer,
            max_iters=args["max_iterations"],
            min_sim=args["min_similarity"],
            use_cuda=args["use_cuda"],
            top_k=args["top_k_keywords"]
        )

        for k, v in result.items():
            output_df.at[idx, k] = v

        if result["attack_success"]:
            success_count += 1

    # 所有未被攻击的样本直接复制原文
    not_attacked_mask = ~df.index.isin(attackable_indices)
    output_df.loc[not_attacked_mask, "adversarial"] = df.loc[not_attacked_mask, "specific_dialogue_content"]
    output_df.loc[not_attacked_mask, "attack_success"] = False
    output_df.loc[not_attacked_mask, "bertscore_similarity"] = 1.0
    output_df.loc[not_attacked_mask, "svm_prediction_after"] = df.loc[not_attacked_mask, "svm_prediction_original"]
    output_df.loc[not_attacked_mask, "final_iteration"] = 0
    output_df.loc[not_attacked_mask, "high_risk_words"] = ""

    # 保存结果
    output_df.to_csv(args["output_csv"], index=False, encoding="utf-8-sig")
    
    print("\n" + "="*70)
    print("✅ CAMOUFLAGE 对抗攻击完成！")
    print(f"攻击成功率: {success_count}/{len(attackable_indices)} ({100 * success_count / max(1, len(attackable_indices)):.1f}%)")
    print(f"结果已保存至: {os.path.abspath(args['output_csv'])}")
    print("💡 提示：'high_risk_words' 列记录了每次攻击使用的 SVM 高危词，可用于分析。")

if __name__ == "__main__":
    main()