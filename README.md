# Agnes-Video-2.5-PE (ContextIR V5) 提示词增强服务

> **版本定位**：`V5` 版本是专为高质量视频生成设计的提示词增强（Prompt Enhancement, PE）中枢。集成多模态素材解析、主体外观一致性绑定、8 大官方风格 Skill、109 个 T8 Creative DNA 镜头语言机制以及自动化格式规范校验。

---

## 目录
- [一、 快速开始](#一-快速开始)
- [二、 支持的生成模式](#二-支持的生成模式)
- [三、 Python API 核心接口](#三-python-api-核心接口)
- [四、 命令行 CLI 调用方式](#四-命令行-cli-调用方式)
- [五、 HTTP 服务端部署接口](#五-http-服务端部署接口)
- [六、 核心特性与架构](#六-核心特性与架构)

---

## 一、 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置 API 凭证
在 `configs/gemini.yaml` 中配置 Gemini 大模型服务地址与 API Key：
```yaml
api_key: "YOUR_API_KEY"
endpoint: "YOUR_ENDPOINT"
model: "MaaS_Ge_3.1_flash_lite_preview_20260303"
protocol: "native"      # 支持 native 原生接口与 openai 兼容接口
timeout_sec: 300
max_retries: 3
```

---

## 二、 支持的生成模式

`V5` 支持 5 种标准的视频生成增强模式（`mode`）：

| 模式标识 (`mode`) | 适用场景 | 必需媒体输入参数 |
| :--- | :--- | :--- |
| **`t2va`** | 纯文生视频（Text-to-Video） | 无（仅需提示词 `intent`） |
| **`i2va`** | 首帧图生视频（Image-to-Video） | `first_frame`（首帧图片绝对路径） |
| **`fl2va`** | 首尾帧过渡生视频（First-Last-to-Video） | `first_frame` + `last_frame`（首尾帧路径） |
| **`l2va`** | 尾帧前推生视频（Last-Frame-to-Video） | `last_frame`（尾帧图片绝对路径） |
| **`r2va`** | 多素材角色/主体参考生成（Reference-to-Video） | `reference_images`（参考图片列表，≤ 9 张） |

---

## 三、 Python API 核心接口

### 1. 核心增强函数：`src.pipeline.enhance`

```python
from src.pipeline import enhance

result: dict = enhance(
    mode: str,                                    # 必填，任务模式: 't2va' | 'i2va' | 'fl2va' | 'l2va' | 'r2va'
    intent: str,                                  # 必填，用户输入的原始提示词 / 意图描述
    *,
    first_frame: str | None = None,               # i2va / fl2va 首帧图片绝对路径
    last_frame: str | None = None,                # fl2va / l2va 尾帧图片绝对路径
    reference_images: list[str] | None = None,    # r2va 参考图片绝对路径列表 (≤ 9 张)
    reference_videos: list[str] | None = None,    # r2va 参考视频绝对路径列表 (≤ 3 段)
    reference_audios: list[str] | None = None,    # r2va 参考音频绝对路径列表 (≤ 3 段)
    duration: int | None = None,                  # 目标视频时长 (秒，4~15，默认从意图推断或设为 5)
    skills: list[str] | None = None,              # 强制加载的风格 Skill ID 列表
    skill_router: str = "hybrid",                 # 风格路由策略: 'hybrid' | 'keyword' | 'llm' | 'off'
    mechanisms: list[str] | None = None,          # 强制加载的 T8 机制 ID 列表
    mechanism_router: str = "hybrid",             # 机制路由策略: 'hybrid' | 'keyword' | 'llm' | 'off'
    enable_verify: bool = True,                   # 是否开启格式规范校验与自动修复 (默认 True)
    verify_intent_llm: bool | None = None,        # 是否开启大模型语义一致性回验 (默认 False)
    out_dir: str | Path | None = None,            # 可选，落盘保存每步中间产物的输出目录
)
```

#### 入参详细说明
* **`mode`** (`str`): 生成模式，必须是 `t2va`、`i2va`、`fl2va`、`l2va`、`r2va` 之一；
* **`intent`** (`str`): 用户的原始中文/英文自然语言输入；
* **`first_frame`** / **`last_frame`** (`str`): 本地图像文件的绝对路径；
* **`reference_images`** (`list[str]`): 多参考素材路径列表，模型将对图片中的主体进行提取与 `<Subject 1>`、`<Subject 2>` 标签化绑定；
* **`duration`** (`int`): 生成视频时长（秒），支持 4 至 15 秒；
* **`skill_router`** (`str`): 风格 Skill 路由模式，默认为 `hybrid`（模型综合打分 ≥ 0.8 才加载）；
* **`mechanism_router`** (`str`): T8 镜头机制路由模式，默认为 `hybrid`。

---

### 2. 返回值结构说明

函数返回一个包含全量生成细节的 `dict`：

```json
{
  "mode": "r2va",
  "prompt": "integrated_multimodal_description: [Shot 1] ...\n\noverall_soundscape: ...\n\nnon_diegetic_music: ...",
  "intent": "小男孩带着柴犬在草地上奔跑",
  "duration": 5,
  "first_frame": null,
  "last_frame": null,
  "reference_images": [
    "/data/images/child.png",
    "/data/images/dog.png"
  ],
  "inventory": "<Subject 1> is the toddler in <Picture 1>... <Subject 2> is the Shiba Inu in <Picture 2>...",
  "contract": {
    "onscreen_text": [],
    "dialogue": [],
    "camera_rules": [],
    "style_rules": []
  },
  "expanded": "[Shot 1]\nStyle: ...\nAction: ...",
  "elaborated": "...",
  "style_skills": ["live-action-cinematic"],
  "mechanisms": ["T8-101"],
  "verify": {
    "status": "passed",
    "fixed": false,
    "issues": []
  },
  "steps": [
    {"stage": "perceive_refs", "text": "..."},
    {"stage": "expand", "text": "..."},
    {"stage": "elaborate", "text": "..."},
    {"stage": "format", "text": "..."}
  ],
  "created_at": "2026-08-28T02:20:00.000000+00:00"
}
```

* **`result["prompt"]`**：**最终交付给视频生成底模（如 MiniMax-H3 / SGLang）的完整结构化提示词**；
* **`result["inventory"]`**：多参考素材的主体识别与一致性绑定描述；
* **`result["verify"]`**：格式合规性检查与自愈修复报告；
* **`result["steps"]`**：各阶段（感知 $\to$ 扩写 $\to$ 细化 $\to$ 格式化）的完整中间文本。

---

### 3. 常见调用代码示例

#### 示例 1：纯文生视频（T2V）
```python
import sys
sys.path.insert(0, "/kwkj-k8s/zq/workspace-ContextIR/V5")
from src.pipeline import enhance

res = enhance(
    mode="t2va",
    intent="阳光穿过秋日金黄的银杏树林，微风吹落树叶，一位女孩坐在木椅上看书。",
    duration=5
)
print("=== 最终增强提示词 ===")
print(res["prompt"])
```

#### 示例 2：首帧图生视频（I2V）
```python
res = enhance(
    mode="i2va",
    intent="镜头由特写慢慢拉开，展示整个赛博朋克城市的雨夜霓虹全貌。",
    first_frame="/absolute/path/to/start_frame.png",
    duration=5
)
print(res["prompt"])
```

#### 示例 3：多素材参考生视频（Ref2V）
```python
res = enhance(
    mode="r2va",
    intent="主角带着宠物小狗在夕阳下的沙滩上奔跑漫步，动作自然连贯。",
    reference_images=[
        "/absolute/path/to/character.png",
        "/absolute/path/to/dog.png"
    ],
    duration=5
)
print(res["prompt"])
```

---

## 四、 命令行 CLI 调用方式

可以直接通过 `scripts/run.py` 进行命令行调用：

### 1. 常用命令模板

```bash
# 1. 纯文生视频 (T2V)
python scripts/run.py -m t2va --intent "一只金毛犬在草地上欢快奔跑" --no-video

# 2. 首帧图生视频 (I2V)
python scripts/run.py -m i2va --intent "汽车启动驶入隧道" --first-frame "assets/frame.png" --no-video

# 3. 首尾帧生视频 (FL2V)
python scripts/run.py -m fl2va --intent "日出渐变到正午" --first-frame "assets/f1.png" --last-frame "assets/f2.png" --no-video

# 4. 多素材参考生视频 (Ref2V)
python scripts/run.py -m r2va --intent "主角和小狗散步" \
  --ref-image "assets/person.png" \
  --ref-image "assets/dog.png" \
  --duration 5 \
  --no-video
```

### 2. 常用参数选项
* `-m, --mode`：必填，生成模式（`t2va` / `i2va` / `fl2va` / `l2va` / `r2va`）；
* `--intent`：短意图提示词文本；
* `--intent-file`：从本地文本文件读取短意图；
* `--first-frame`：首帧图像绝对路径；
* `--last-frame`：尾帧图像绝对路径；
* `--ref-image`：参考图片路径（可重复传入多个 `--ref-image A --ref-image B`）；
* `--duration`：出片时长（秒，4~15）；
* `--out-dir`：指定中间过程文件与最终提示词的保存目录；
* `--no-video`：仅生成与输出提示词，不调用后端渲染引擎。

---

## 五、 HTTP 服务端部署接口

V5 提供了基于轻量 HTTP 的独立增强微服务：

### 1. 启动服务
```bash
python scripts/pe_server.py --port 8000 --host 0.0.0.0
```

### 2. REST API 接口定义

* **请求端点**：`POST /v1/enhance`
* **Content-Type**：`application/json`

#### 请求体示例
```json
{
  "mode": "t2va",
  "intent": "暴雨中一辆黑色轿车在公路上飞驰",
  "duration": 5,
  "reference_images": []
}
```

#### 响应体示例
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "prompt": "integrated_multimodal_description: [Shot 1] ...",
    "mode": "t2va",
    "duration": 5
  }
}
```

---

## 六、 核心特性与架构

1. **五阶段流式编排**：
   * **感知阶段 (`perceive`)**：多图网格扫描与 `<Subject N>` 角色特征解耦；
   * **路由阶段 (`route`)**：双轨命中 8 大官方题材风格与 109 种 T8 镜头机制；
   * **扩写阶段 (`expand`)**：电影分镜级光影、镜头构图、时序动作展开；
   * **细化阶段 (`elaborate`)**：多模态音效 (`soundscape`) 与非器乐配乐 (`music`) 织入；
   * **格式化阶段 (`format`)**：强制对齐 MiniMax-H3 官方分段规范。
2. **多发言人对白与屏上文字硬隔离**：
   * 台词自动提取并绑定发言人；
   * 自动过滤非台词文本（如屏幕 UI、旁白、背景音乐歌词），防止被误念。
3. **Gemini 双通道安全容灾**：
   * 原生接口遇到违规词误判或安全拦截时，毫秒级自动切换至备用通道重试，确保大批量离线生产 0 中断。
