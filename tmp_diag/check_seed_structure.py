#!/usr/bin/env python3
"""检查种子数据结构"""
import json
import numpy as np

seeds = json.load(open("/workspace/data/calibration_runs/20260817_062447/seeds.json"))
rec = seeds["seeds"][0]
print("seed keys:", list(rec.keys()))
for k in rec.keys():
    v = rec[k]
    if isinstance(v, list) and len(v) > 0:
        print(f"  {k}: list len={len(v)} first_type={type(v[0]).__name__}")
    elif isinstance(v, dict):
        print(f"  {k}: dict keys={list(v.keys())[:6]}")
    elif isinstance(v, str):
        print(f"  {k}: str (len {len(v)})")
    else:
        print(f"  {k}: {type(v).__name__} = {v}")
