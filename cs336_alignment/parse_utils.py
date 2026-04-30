import re


def parse_mmlu_response(model_response: str) -> str | None:
    """
    根据作业要求解析 MMLU 响应。
    匹配格式："The correct answer is [A-D]"
    """
    if not model_response:
        return None
    
    # 使用正则表达式匹配提示词要求的特定句子格式
    # [A-D] 捕获选项，忽略大小写和末尾可能的标点
    pattern = r"[Tt]he correct answer is\s*([A-D])"
    match = re.search(pattern, model_response)
    
    if match:
        return match.group(1).upper()
    
    return None