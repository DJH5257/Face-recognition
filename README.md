# Face Verify Demo

基于 FastAPI 的人脸验证和交互式活体检测服务。项目仍不包含登录、账号和业务权限系统，但已提供可供业务系统调用的模板、核验会话和一次性 proof 接口；业务系统负责决定谁扫脸、何时扫脸以及最终业务放行。

## 功能

- 上传一张本人基准人脸照片，使用 InsightFace / ArcFace 提取人脸特征。
- 浏览器通过 `getUserMedia` 打开摄像头。
- 后端随机生成 1-3 个动作：眨眼、张嘴、摇头、点头、微笑。
- 前端按动作连续采集摄像头帧，后端用 MediaPipe FaceMesh 判断动作是否完成。
- 活体检测通过后，才允许做 1:1 人脸特征比对。
- 验证同时检查动作完成度、帧连续性、重复帧、人脸稳定性、防翻拍和人脸一致性。
- 生产接口使用服务端模板、短期上传 token 和一次性 proof，业务系统不再依赖单图 `/api/compare` 放行。

## 重要声明

本项目只负责人脸核验判定，不替代业务系统的登录、权限、审计和风控。

- 不要把默认阈值直接用于真实业务。
- 不要向公开仓库提交真实人脸照片、摄像头帧、人脸特征、身份资料或业务数据。
- RGB 摄像头活体和 PAD 模型是概率检测，无法保证拦截所有照片、屏幕、视频回放、面具、深度伪造或对抗样本。
- 生产环境还需要业务系统在最终写入口强制消费 proof，并配合限流、审计、设备风险、摄像头流完整性校验、合规评估和专门的 PAD 数据集测试。

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
      stores.py            # Demo 内存 enrollment/challenge + SQLite template/session/proof store
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
FACE_DEMO_DATABASE_PATH=data/face_verify.sqlite3
FACE_DEMO_INTERNAL_API_KEY=change-me-face-internal-key
FACE_DEMO_CORS_ORIGINS=*
FACE_DEMO_VERIFICATION_SESSION_TTL_SECONDS=300
FACE_DEMO_PROOF_TTL_SECONDS=300
FACE_DEMO_TEMPLATE_ENCRYPTION_KEY=
FACE_DEMO_TEMPLATE_ENCRYPTION_REQUIRED=0
FACE_DEMO_INSIGHTFACE_MODEL=buffalo_l
FACE_DEMO_INSIGHTFACE_ROOT=/opt/face-verify-demo/shared/insightface-home/.insightface
FACE_DEMO_INSIGHTFACE_DET_SIZE=480
FACE_DEMO_INSIGHTFACE_LIVE_DET_SIZE=320
FACE_DEMO_FACE_MATCH_EMBEDDING_SAMPLE=5
FACE_DEMO_VERIFICATION_MAX_CONCURRENT_REQUESTS=4
FACE_DEMO_VERIFICATION_FAILURE_LIMIT_ENABLED=1
FACE_DEMO_VERIFICATION_FAILURE_MAX_ATTEMPTS=5
FACE_DEMO_VERIFICATION_FAILURE_WINDOW_SECONDS=300
FACE_DEMO_VERIFICATION_FAILURE_COOLDOWN_SECONDS=180
FACE_DEMO_LIVENESS_ACTION_MIN_COUNT=1
FACE_DEMO_LIVENESS_ACTION_MAX_COUNT=3
FACE_DEMO_LIVENESS_ACTION_WEIGHTS=mouth_open=24,shake_head=24,nod_head=24,blink=18,smile=18
FACE_DEMO_MIN_FRAMES_PER_ACTION=6
FACE_DEMO_MAX_FRAMES_PER_ACTION=32
FACE_DEMO_MAX_TOTAL_FRAMES=90
FACE_DEMO_MIN_UNIQUE_FRAME_RATIO=0.55
FACE_DEMO_DUPLICATE_FRAME_HASH_SIZE=12
FACE_DEMO_IMAGE_QUALITY_SAMPLE_FRAMES=6
FACE_DEMO_IMAGE_QUALITY_MIN_LAPLACIAN=12
FACE_DEMO_IMAGE_QUALITY_MIN_BRIGHTNESS=35
FACE_DEMO_IMAGE_QUALITY_MAX_BRIGHTNESS=225
FACE_DEMO_IMAGE_QUALITY_MIN_CONTRAST=10
FACE_DEMO_IMAGE_QUALITY_TEMPLATE_MIN_FACE_RATIO=0.025
FACE_DEMO_IMAGE_QUALITY_LIVE_MIN_FACE_RATIO=0.02
FACE_DEMO_IMAGE_QUALITY_MAX_FACE_RATIO=0.92
FACE_DEMO_IMAGE_QUALITY_TEMPLATE_MAX_ABS_YAW=35
FACE_DEMO_IMAGE_QUALITY_TEMPLATE_MAX_ABS_PITCH=35
FACE_DEMO_IMAGE_QUALITY_LIVE_MAX_ABS_YAW=55
FACE_DEMO_IMAGE_QUALITY_LIVE_MAX_ABS_PITCH=55
FACE_DEMO_ANTI_SPOOFING_THRESHOLD=0.35
FACE_DEMO_ANTI_SPOOFING_MODEL_PATH=models/MiniFASNetV1SE.onnx,models/MiniFASNetV2.yakhyo.onnx
```

阈值需要用你的摄像头、光照、真人样本、照片、屏幕、视频回放和低性能设备重新校准。图像质量门禁会先拦截明显模糊、过暗、过曝、低对比度、脸过小、脸框明显越界或姿态明显偏转的登记照和活体抽样帧，避免烂图污染模板或进入重模型推理。

生产环境建议设置 `FACE_DEMO_TEMPLATE_ENCRYPTION_KEY` 并开启 `FACE_DEMO_TEMPLATE_ENCRYPTION_REQUIRED=1`，这样新登记的人脸 embedding 会加密落库。旧明文模板仍可兼容读取，确认密钥配置无误后再逐步迁移。失败次数限制和验证并发保护为进程内实现，单机可以防止异常重试拖垮服务，多实例场景需要在网关层配合限流。

## 生产部署建议

为了保持本地和线上效果一致，生产环境不要关闭任何一层检测：保留 InsightFace 检测与 ArcFace 特征比对、MediaPipe 随机动作活体、MiniFASNet 防翻拍、多帧连续性和人脸一致性校验。

推荐服务器至少 2 核 4GB 内存；更稳妥为 4 核 8GB。1 核 2GB 或 2 核 2GB 机器可以启动页面和接口，但首次加载 `buffalo_l`/ArcFace 模型可能非常慢，甚至拖慢 SSH 和其它服务。不要通过降低阈值、减少防翻拍或删除活体动作来规避这个问题，应该换更合适的服务器规格。

上线前把所有运行资源上传到服务器，避免首次请求时联网下载：

```text
/opt/face-verify-demo/current/                  # 当前代码
/opt/face-verify-demo/shared/.venv/             # Python 虚拟环境
/opt/face-verify-demo/shared/insightface-home/.insightface/models/buffalo_l/
  det_10g.onnx
  w600k_r50.onnx
  2d106det.onnx
  1k3d68.onnx
  genderage.onnx
/opt/face-verify-demo/current/models/
  MiniFASNetV1SE.onnx
  MiniFASNetV2.yakhyo.onnx
```

`FACE_DEMO_INSIGHTFACE_ROOT` 必须指向包含 `models/` 子目录的 `.insightface` 根目录，例如：

```bash
FACE_DEMO_INSIGHTFACE_MODEL=buffalo_l
FACE_DEMO_INSIGHTFACE_ROOT=/opt/face-verify-demo/shared/insightface-home/.insightface
```

高配服务器不建议设置 ONNXRuntime 线程数，让运行时按硬件自动调度以获得最佳识别性能。4 核 CPU 服务器可先用 2/1 控制单请求 CPU 占用；2 核机器建议用 1/1 或不设置：

```bash
FACE_DEMO_ONNX_INTRA_OP_THREADS=2
FACE_DEMO_ONNX_INTER_OP_THREADS=1
```

公网摄像头采集必须使用 HTTPS。没有域名时可以先用带 IP SAN 的自签证书测试；正式环境建议使用域名和可信证书。部署后至少验证：

```bash
curl -k https://<host>:<port>/api/health
curl -k https://<host>:<port>/api/ready
curl -k -I https://<host>:<port>/static/app.js?v=20260606-face-verify
PYTHONPYCACHEPREFIX=/tmp/face-verify-demo-pycache .venv/bin/python -m unittest discover -s tests -v
```

## API

### 健康与就绪

```http
GET /api/health
GET /api/ready
```

`/api/health` 只表示进程可响应。`/api/ready` 会检查 SQLite 数据目录和 PAD 模型文件是否存在，不会预加载 InsightFace 或 PAD 大模型，避免探活拖慢低配机器。

### 生产接口鉴权

所有 `/v1/internal/*` 接口仅供 `chbzg` 后端调用，必须携带：

```http
X-Face-Api-Key: <FACE_DEMO_INTERNAL_API_KEY>
```

浏览器只允许调用 `/v1/verification-sessions/{session_id}/verify`，并使用 `chbzg` 从人脸服务拿到的短期上传令牌：

```http
Authorization: Bearer <upload_token>
```

浏览器不能指定人员 ID、模板 ID、业务场景、业务事件、阈值或动作列表。

### 生产：登记模板

```http
POST /v1/internal/templates
Content-Type: application/json
X-Face-Api-Key: ...
```

```json
{
  "subject_type": "doctor",
  "subject_id": "83",
  "image": "data:image/jpeg;base64,...",
  "source_type": "avatar",
  "request_id": "optional-id"
}
```

成功后返回 `template_id`、`template_version` 和 bbox，不返回 embedding。同一人员新增模板会原子吊销旧 active 模板。登记照会先经过轻量图像质量和人脸框质量门禁。

### 生产：查询和撤销模板

```http
GET /v1/internal/templates/subjects/{subject_type}/{subject_id}
POST /v1/internal/templates/{template_id}/revoke
X-Face-Api-Key: ...
```

查询接口返回该人员的模板版本列表、当前 active 模板 ID、状态、来源、图片 hash、bbox 和创建/激活/撤销时间。撤销接口会把 active 模板置为 `revoked`；撤销后该人员不能创建新的核验会话，直到重新上传模板。

### 生产：创建核验会话

```http
POST /v1/internal/verification-sessions
Content-Type: application/json
X-Face-Api-Key: ...
```

```json
{
  "request_id": "login-2112-20260612",
  "subject_type": "doctor",
  "subject_id": "83",
  "admin_id": "2112",
  "scene": "doctor_audit",
  "business_event_id": "biz-event-id",
  "record_id": "record-id",
  "action": "audit"
}
```

服务端按权重生成 1-3 个随机动作、绑定 active 模板并返回 `session_id`、`upload_token`、`actions` 和 `expires_at`。相同 `request_id` 会返回同一会话。

### 生产：提交连续帧核验

```http
POST /v1/verification-sessions/{session_id}/verify
Authorization: Bearer <upload_token>
Content-Type: application/json
```

```json
{
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

只有图像质量、动作、重复帧、单人脸稳定性、PAD 和本人比对全部通过才签发 `proof_id`。失败不会签发 proof。服务端会把 `action_results` 持久化到核验会话，便于后续排查高频失败动作、图像质量、PAD 或人脸比对问题。

### 生产：查询和完成 proof

```http
POST /v1/internal/proofs/introspect
POST /v1/internal/proofs/{proof_id}/finalize
```

`introspect` 返回 proof 是否仍有效以及绑定的人员、场景、业务事件、记录、动作和模板版本。`finalize` 必须传入相同 `business_event_id`；业务系统应在自身事务成功后调用它。

### Demo：上传基准人脸

```http
POST /api/enroll
Content-Type: multipart/form-data

file=<image>
```

如果没有检测到人脸，返回 `400`。如果检测到多张人脸，也会返回 `400`。

### Demo：生成随机活体动作

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

### Demo：提交摄像头帧做活体检测

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

### Demo：活体通过后的人脸比对（兼容旧流程）

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

当前前端默认在 `/api/liveness/verify` 里直接拿到最终结果，`/api/compare` 仅保留兼容调用。

生产业务放行不要依赖 `/api/compare`，应使用 `/v1` 签发的一次性 proof。

## 测试

```bash
PYTHONPYCACHEPREFIX=/private/tmp/face-verify-demo-pycache python -m py_compile backend/app/*.py tests/*.py
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
