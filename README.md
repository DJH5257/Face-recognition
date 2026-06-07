# Face Verify Demo

基于 FastAPI 的人脸验证和交互式活体检测 Demo。项目不包含登录、数据库、账号系统或生产级风控，适合用于学习、原型验证和离线评估流程。

## 功能

- 上传一张本人基准人脸照片，使用 InsightFace / ArcFace 提取人脸特征。
- 浏览器通过 `getUserMedia` 打开摄像头。
- 后端随机生成 2-4 个动作：眨眼、张嘴、摇头、点头、微笑。
- 前端按动作连续采集摄像头帧，后端用 MediaPipe FaceMesh 判断动作是否完成。
- 活体检测通过后，才允许做 1:1 人脸特征比对。
- 验证同时检查动作完成度、帧连续性、重复帧、人脸稳定性、防翻拍和人脸一致性。

## 重要声明

本项目是 Demo，不是生产身份认证系统。

- 不要把默认阈值直接用于真实业务。
- 不要向公开仓库提交真实人脸照片、摄像头帧、人脸特征、身份资料或业务数据。
- RGB 摄像头活体和 PAD 模型是概率检测，无法保证拦截所有照片、屏幕、视频回放、面具、深度伪造或对抗样本。
- 生产环境还需要身份系统、鉴权、限流、审计、设备风险、摄像头流完整性校验、合规评估和专门的 PAD 数据集测试。

## 模型和许可证

源码使用 MIT License。第三方模型权重不自动继承本仓库源码许可证。

- InsightFace 源码为 MIT License，但其 Model Zoo 和 Python 包自动下载的模型包，包括 `buffalo_l`，上游说明为仅限非商业研究用途。
- MiniFASNet ONNX 权重来源可能不同。常见上游包括 Minivision Silent-Face-Anti-Spoofing 和 yakhyo/face-anti-spoofing；发布或商用前，请确认你使用的具体权重文件来源和许可证。
- `models/*.onnx` 已在 `.gitignore` 中排除。模型准备说明见 [models/README.md](models/README.md)。

## 目录结构

```text
face-verify-demo/
  backend/
    app/
      main.py              # FastAPI routes and static page service
      config.py            # Thresholds and model settings
      face_engine.py       # InsightFace detection, embeddings, similarity
      liveness.py          # Random liveness actions and action checks
      anti_spoofing.py     # MiniFASNet ONNX inference
      schemas.py           # API request / response models
      stores.py            # In-memory enrollment and challenge store
    static/
      index.html
      styles.css
      app.js
  models/
    README.md
  scripts/
    test_frontend_static.js
  tests/
    test_verification_guards.py
  requirements.txt
```

## 环境要求

- Python 3.10 或 3.11
- Node.js 18+，用于静态前端检查
- 支持摄像头 `getUserMedia` 的现代浏览器

## 安装运行

```bash
git clone <your-repo-url>
cd face-verify-demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

打开：

```text
http://127.0.0.1:8000
```

第一次运行 InsightFace 会下载 `buffalo_l` 模型。如果下载失败，可以提前把 InsightFace 模型放到本机的 `~/.insightface/models/buffalo_l` 目录。

如果需要使用 PyPI 镜像：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn -r requirements.txt
```

## 配置

复制 `.env.example` 后按需调整：

```bash
cp .env.example .env
```

常用配置：

```bash
FACE_DEMO_FACE_MATCH_THRESHOLD=0.42
FACE_DEMO_FACE_MATCH_MIN_THRESHOLD=0.30
FACE_DEMO_FACE_MATCH_MIN_PASS_RATIO=0.65
FACE_DEMO_LIVENESS_ACTION_MIN_COUNT=2
FACE_DEMO_LIVENESS_ACTION_MAX_COUNT=4
FACE_DEMO_MIN_FRAMES_PER_ACTION=6
FACE_DEMO_MAX_TOTAL_FRAMES=72
FACE_DEMO_ANTI_SPOOFING_THRESHOLD=0.35
FACE_DEMO_ANTI_SPOOFING_MODEL_PATH=models/MiniFASNetV1SE.onnx,models/MiniFASNetV2.yakhyo.onnx
```

阈值需要用你的摄像头、光照、真人样本、照片、屏幕、视频回放和低性能设备重新校准。

## API

### 上传基准人脸

```http
POST /api/enroll
Content-Type: multipart/form-data

file=<image>
```

如果没有检测到人脸，返回 `400`。如果检测到多张人脸，也会返回 `400`。

### 生成随机活体动作

```http
POST /api/liveness/challenge
Content-Type: application/json
```

```json
{
  "enrollment_id": "..."
}
```

返回：

```json
{
  "challenge_id": "...",
  "actions": ["blink", "mouth_open", "shake_head"],
  "labels": {
    "blink": "请眨眼",
    "mouth_open": "请张嘴"
  }
}
```

### 提交摄像头帧做活体检测

```http
POST /api/liveness/verify
Content-Type: application/json
```

```json
{
  "challenge_id": "...",
  "enrollment_id": "...",
  "frames": [
    {
      "action": "blink",
      "index": 0,
      "timestamp": 123456.7,
      "image": "data:image/jpeg;base64,..."
    }
  ]
}
```

### 活体通过后做人脸比对

```http
POST /api/compare
Content-Type: application/json
```

```json
{
  "enrollment_id": "...",
  "challenge_id": "..."
}
```

未通过活体时调用该接口会返回 `403`。

## 测试

```bash
PYTHONPYCACHEPREFIX=/private/tmp/face-verify-demo-pycache python -m py_compile backend/app/*.py tests/test_verification_guards.py
PYTHONPYCACHEPREFIX=/private/tmp/face-verify-demo-pycache python -m unittest discover -s tests -v
node --check backend/static/app.js
node scripts/test_frontend_static.js
```

真实验收请按 [人脸验证验收记录.md](人脸验证验收记录.md) 记录本人、非本人、照片、屏幕照片、视频回放、重复帧和低配电脑样本。自动化测试不能替代真实摄像头样本。

## 开源发布前检查

- 确认 `.venv/`、`.env`、真实样本、模型权重和本地缓存没有进入 Git。
- 确认模型权重的来源、许可证和再分发权限。
- 确认 README、SECURITY、CONTRIBUTING 和 LICENSE 适合公开仓库。
- 初始化 Git 仓库后运行 `git status --ignored --short` 检查忽略规则。
- 发布前至少运行一次自动测试和一次真实摄像头验收。

## 许可证

本仓库源码使用 [MIT License](LICENSE)。第三方依赖和模型资产遵循其各自许可证。
