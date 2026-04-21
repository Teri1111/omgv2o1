import os
import json
import threading
def colorful(text, color="yellow"):
    if color == "yellow":
        text = "\033[1;33m" + str(text) + "\033[0m"
    elif color == "grey":
        text = "\033[1;30m" + str(text) + "\033[0m"
    elif color == "green":
        text = "\033[1;32m" + str(text) + "\033[0m"
    elif color == "red":
        text = "\033[1;31m" + str(text) + "\033[0m"
    elif color == "blue":
        text = "\033[1;94m" + str(text) + "\033[0m"
    else:
        pass
    return text

def save_to_json(data, filename='data.json'):
    # 如果文件不存在，初始化为空列表
    if not os.path.exists(filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump([], f)
    
    # 读取现有数据
    with open(filename, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)
    
    # 添加新数据
    existing_data.append(data)
    
    # 写回文件
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

# Create a lock object to synchronize file access


def save_to_json_safe(data, save_lock, filename='data.json'):
    with save_lock:
        # 如果文件不存在，初始化为空列表
        if not os.path.exists(filename):
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump([], f)
        
        # 读取现有数据
        with open(filename, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        # 添加新数据
        existing_data.append(data)
        
        # 写回文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)