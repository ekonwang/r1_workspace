import json
import re
import argparse
from pathlib import Path

def count_keywords_in_replies(jsonl_file, keywords):
    """
    统计 JSONL 文件中包含特定关键词的回复数量
    
    Args:
        jsonl_file (str): JSONL 文件路径
        keywords (list): 要搜索的关键词列表
    
    Returns:
        dict: 每个关键词出现的回复数量
    """
    # 初始化计数器
    keyword_counts = {keyword: 0 for keyword in keywords}
    total_replies = 0
    
    # 读取 JSONL 文件
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'reply' in data:
                    reply_text = data['reply'].lower()
                    reply_texts = [reply_text] 

                    # bon_reply_texts = [r.lower() for r in data['bon_replies']]
                    # reply_texts += bon_reply_texts
                    total_replies += len(reply_texts)
                    
                    
                    for reply_text in reply_texts:
                        # 检查每个关键词
                        for keyword in keywords:
                            # 使用正则表达式匹配整个单词
                            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                            if re.search(pattern, reply_text):
                                keyword_counts[keyword] += 1
            except json.JSONDecodeError:
                print(f"警告: 跳过无效的 JSON 行")
                continue
    
    return keyword_counts, total_replies

def main():
    parser = argparse.ArgumentParser(description='统计 JSONL 文件中包含特定关键词的回复数量')
    parser.add_argument('--jsonl_file', type=str, help='JSONL 文件路径')
    parser.add_argument('--keywords', type=str, nargs='+', 
                        default=['however', 'despite', 'on the contrary', 'whereas'],
                        help='要搜索的关键词列表')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    file_path = Path(args.jsonl_file)
    if not file_path.exists():
        print(f"错误: 文件 '{args.jsonl_file}' 不存在")
        return
    
    # 统计关键词
    keyword_counts, total_replies = count_keywords_in_replies(args.jsonl_file, args.keywords)
    
    # 打印结果
    print(f"\n分析文件: {args.jsonl_file}")
    print(f"总回复数: {total_replies}\n")
    print("关键词统计:")
    print("-" * 40)
    
    for keyword, count in keyword_counts.items():
        percentage = (count / total_replies * 100) if total_replies > 0 else 0
        print(f"'{keyword}': {count} 次 ({percentage:.2f}%)")
    
    # 计算至少包含一个关键词的回复总数
    # 注意：这需要再次遍历文件，因为一个回复可能包含多个关键词
    with open(args.jsonl_file, 'r', encoding='utf-8') as f:
        replies_with_keywords = 0
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'reply' in data:
                    reply_text = data['reply'].lower()
                    reply_texts = [reply_text] 

                    # bon_reply_texts = [r.lower() for r in data['bon_replies']]
                    # reply_texts += bon_reply_texts

                    for reply_text in reply_texts:
                        if any(re.search(r'\b' + re.escape(keyword.lower()) + r'\b', reply_text) 
                            for keyword in args.keywords):
                            replies_with_keywords += 1
            except json.JSONDecodeError:
                continue
    
    percentage = (replies_with_keywords / total_replies * 100) if total_replies > 0 else 0
    print("-" * 40)
    print(f"包含至少一个关键词的回复: {replies_with_keywords} ({percentage:.2f}%)")

if __name__ == "__main__":
    main()