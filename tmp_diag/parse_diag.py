#!/usr/bin/env python3
"""读取检测节点诊断"""
import json

with open("/tmp/diag_raw.txt") as f:
    content = f.read()
# 提取 data: 后的 JSON
idx = content.find("data: '")
if idx >= 0:
    start = idx + len("data: '")
    end = content.rfind("'")
    raw = content[start:end]
    d = json.loads(raw)
    for k, v in d.items():
        print(f"{k}: {v}")
else:
    print("not found, raw:", content[:300])
