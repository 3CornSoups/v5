# V5.1 · 本地 Omni LoRA 后端说明

## 改动摘要

- 新增 `backend=omni_lora`：调用本机 `http://127.0.0.1:8910`（Qwen2.5-Omni-7B + H3 Rewriter LoRA）
- **单次 HTTP** 产出 H3 结构化 prompt；跳过 Gemini 多轮（perceive / contract / expand / elaborate / format）
- `skill_router=hybrid` 时：快路径内 **降为 keyword**（不打路由 LLM），仍记录命中 skill/mechanism
- CLI：`--backend omni_lora`；环境变量：`PE_BACKEND=omni_lora`
- 对照脚本：`scripts/bench_ref2av_backends.py`

## 用法

```bash
# 确认 Omni 服务
curl -s http://127.0.0.1:8910/health

cd /kwkj-k8s/zwb/0904ir/V5.1
python scripts/run.py -m r2va --backend omni_lora --no-video \
  --intent-file ... --ref-image a.jpg --ref-image b.jpg --duration 10 \
  --skill-router hybrid --mechanism-router hybrid

# 速度对照
python scripts/bench_ref2av_backends.py --warmup
```

## Ref2AV 速度结论（本机实测 2026-09-07）

| 路径 | 耗时 | HTTP 次数 |
|------|------|-----------|
| Omni LoRA（hybrid→keyword 快路径） | **≈22.8 s** | **1** |
| 历史 V5 Gemini r2va 批跑墙钟（`0828测试素材`，n≈200，含并行） | median **≈36.5 s** | 多轮（约 5–8+） |

相对该历史中位墙钟约 **1.6×**，**尚未稳定达到 ≥2×** 门禁。  
当日 Gemini Cloudsway **401**，无法同机即时复测；质量回退 ≤2% 需补：修好 Key 后同用例 Gemini vs Omni + 裁判/人工。

## 距 2× 可再压的方向

1. Omni 侧：FlashAttention、更短 `max_new_tokens`、4bit 量化（0907 任务二）
2. V5 侧：Ref2AV 仅 Omni、彻底 `skill_router=off` 省关键词扫描（秒级）
3. 基线：用修好的 Gemini Key 跑同图同意图，以纯 V5 串行耗时作分母
