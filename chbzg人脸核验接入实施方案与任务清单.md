# chbzg 人脸核验接入实施方案与任务清单

更新日期：2026-06-13

## 1. 实施结论

现有独立人脸服务 `https://faces.chbzg.com.cn` 继续负责人脸模板、随机动作、
连续帧校验、PAD 防翻拍、多帧本人比对和一次性 proof。`chbzg` 只负责业务规则：
谁需要核验、何时核验、核验对象是谁、proof 可以放行哪一次登录或审核。

正式接入不嵌入当前 Demo 页面，也不继续使用旧单图比对结果。采用以下链路：

```text
chbzg 后端判断是否需要核验
→ 创建不可由浏览器篡改的业务事件
→ chbzg 后端调用 faces 创建 verification session
→ 浏览器使用短期 upload_token 直传连续帧到 faces
→ faces 综合通过后返回一次性 proof_id
→ 浏览器把 proof_id 提交给 chbzg 最终业务接口
→ chbzg 后端 introspect 并逐字段校验 proof
→ chbzg 本地事务写入 proof 唯一消费记录并完成业务操作
→ 事务成功后通知 faces finalize proof
```

核心原则：

- 浏览器不决定是否需要扫脸，也不能指定核验人员、模板、场景、动作或阈值。
- `chbzg` 最终登录和审核接口必须强制检查 proof，不能只改前端弹窗。
- proof 必须绑定人员、场景、业务事件；审核场景还必须绑定记录和动作。
- 人脸服务异常时，对需要核验的操作失败关闭，不允许静默跳过。
- 旧 `feature` 不能直接转为新版向量，必须从有效头像原图或重新采集图片生成。
- 药师开关当前保持关闭，但代码结构一次性兼容，避免后续重复改造。

## 2. 当前基础与接入前置条件

### 2.1 已可直接复用的能力

- `POST /v1/internal/templates`：登记新版人脸模板。
- `POST /v1/internal/verification-sessions`：创建绑定业务字段的核验会话。
- `POST /v1/verification-sessions/{session_id}/verify`：浏览器提交连续帧。
- `POST /v1/internal/proofs/introspect`：业务后端查询 proof 和全部绑定字段。
- `POST /v1/internal/proofs/{proof_id}/finalize`：业务成功后完成 proof。
- 1-3 个随机动作，每动作 2 秒。
- 双 MiniFASNet PAD、InsightFace 多帧比对、重复帧和多脸拒绝。
- SQLite 持久化模板、session 和 proof，适合当前单实例部署。

### 2.2 接入前必须补齐的人脸服务任务

这些任务改动小，但直接影响正式业务接入的正确性：

| 编号 | 任务 | 原因 | 完成标准 |
| --- | --- | --- | --- |
| F1 | 配置实际 `chbzg` 页面 HTTPS Origin | 浏览器将跨域直传帧，当前 CORS 需加入业务域名 | 仅允许 faces 自身和明确的 chbzg 生产/测试 Origin |
| F2 | 强化 session 幂等冲突校验 | 当前相同 `request_id` 会直接返回旧 session | 重复请求字段完全一致才复用，否则返回 409 |
| F3 | 实现模板登记幂等 | 模板请求虽有 `request_id`，当前未参与存储 | 网络重试不会生成新模板版本 |
| F4 | 将 upload token 改为服务端签名短期 token | 当前 session 同时保存 token 明文和哈希；只存哈希又无法支持幂等重试 | token 可根据 session 重建和验签，数据库不保存可直接使用的明文 |
| F5 | 增加模板状态查询接口 | chbzg 需要校验 active 模板和同步 profile | 可按 subject 查询 active template，不返回 embedding |
| F6 | 增加模板人工吊销接口 | 人员离职、头像错误、模板泄露时需要立即失效 | 内部鉴权、幂等吊销、留存操作记录 |
| F7 | 补齐策略/模型/阈值版本字段 | 审计时需要知道使用了哪套规则 | session 和 proof 返回并持久化版本 |
| F8 | 增加基础限流与并发保护 | 防止重复点击或请求洪峰拖垮单实例模型 | 超限返回 `ERROR_SERVICE_BUSY`，不破坏已有 session |

F1-F4 属于接入阻塞项。F5-F8 应在正式灰度前完成。

## 3. 业务对象与字段约定

### 3.1 人员类型

固定使用：

```text
doctor
pharmacist
```

`subject_id` 使用医生或药师业务表主键字符串。`admin_id` 使用后台管理员主键字符串。
浏览器不能提交或覆盖这两个字段。

### 3.2 场景和动作

固定使用以下枚举，禁止前后端自由拼写：

| scene | action | 用途 |
| --- | --- | --- |
| `login` | `login` | 医生或药师后台登录 |
| `doctor_audit` | `pass` | 医生审核通过 |
| `doctor_audit` | `reject` | 医生审核驳回，如当前业务支持 |
| `pharmacist_audit` | `pass` | 药师审核通过 |
| `pharmacist_audit` | `reject` | 药师审核驳回 |

医生审核若当前只有统一 `audit` 动作，可暂时使用 `action=audit`，但必须在接入前
确认最终写入口，不能同一环境混用 `audit` 和 `pass/reject`。

### 3.3 业务事件 ID

`business_event_id` 由 `chbzg` 后端生成，建议使用 UUID，不包含可猜测业务含义。
一次业务操作对应一个事件：

```text
登录：一次正确密码校验后的待登录事件
审核：人员 + 记录 + 最终动作 + 时间窗的一次审核事件
```

相同事件重试必须复用同一个 `business_event_id` 和 session；重新开始核验时创建新事件。
若动作、PAD 或人脸比对已经明确失败，该 session 已终止，应创建新的业务事件和 session；
只有网络错误、模型暂时不可用或最终业务提交重试才复用原事件。

### 3.4 proof 通过条件

`FaceVerifyGuard` 必须同时检查：

```text
valid == true
status == ready
result_code == PASS
subject_type == 当前业务人员类型
subject_id == 当前真实人员 ID
admin_id == 当前管理员 ID（适用时）
scene == 当前业务场景
business_event_id == 当前事件 ID
record_id == 当前记录 ID（审核场景）
action == 当前最终动作
expires_at > 当前时间
```

任一字段不一致都拒绝，不允许“只要 proof 存在就通过”。

### 3.5 当前 faces 接口契约

服务间调用统一携带：

```http
X-Face-Api-Key: <server-secret>
Content-Type: application/json
```

登记模板：

```http
POST https://faces.chbzg.com.cn/v1/internal/templates
```

```json
{
  "subject_type": "doctor",
  "subject_id": "83",
  "image": "data:image/jpeg;base64,...",
  "source_type": "avatar",
  "request_id": "template-doctor-83-avatar-hash"
}
```

创建核验 session：

```http
POST https://faces.chbzg.com.cn/v1/internal/verification-sessions
```

```json
{
  "request_id": "event-request-id",
  "subject_type": "doctor",
  "subject_id": "83",
  "admin_id": "2112",
  "scene": "doctor_audit",
  "business_event_id": "event-uuid",
  "record_id": "record-9",
  "action": "pass"
}
```

返回：

```json
{
  "session_id": "session-uuid",
  "upload_token": "short-lived-token",
  "actions": ["blink", "shake_head"],
  "labels": {
    "blink": "请眨眼",
    "shake_head": "请摇头"
  },
  "expires_at": "2026-06-13T10:00:00+00:00"
}
```

浏览器提交连续帧：

```http
POST https://faces.chbzg.com.cn/v1/verification-sessions/{session_id}/verify
Authorization: Bearer <upload_token>
```

浏览器只提交 `frames`，每帧包含 `action/index/timestamp/image`。成功响应必须同时满足
`passed=true`、`result_code=PASS` 且存在 `proof_id`。

最终业务接口查询 proof：

```http
POST https://faces.chbzg.com.cn/v1/internal/proofs/introspect
```

```json
{"proof_id": "proof-uuid"}
```

业务提交成功后完成 proof：

```http
POST https://faces.chbzg.com.cn/v1/internal/proofs/{proof_id}/finalize
```

```json
{"business_event_id": "event-uuid"}
```

客户端不得使用 `/api/enroll`、`/api/liveness/*` 或 `/api/compare` 参与生产放行。

## 4. chbzg 数据库设计

### 4.1 `fa_face_verify_event`

记录业务侧一次核验事件，不保存原始人脸图片或向量。

| 字段 | 建议类型 | 说明 |
| --- | --- | --- |
| `id` | bigint | 主键 |
| `business_event_id` | varchar(64) | 唯一 |
| `request_id` | varchar(128) | 创建 faces session 的幂等键，唯一 |
| `subject_type` | varchar(32) | doctor/pharmacist |
| `subject_id` | bigint | 真实业务人员 ID |
| `admin_id` | bigint nullable | 管理员 ID |
| `scene` | varchar(32) | login/doctor_audit/pharmacist_audit |
| `record_id` | varchar(64) nullable | 审核记录 ID |
| `action` | varchar(32) | login/pass/reject/audit |
| `window_key` | varchar(32) nullable | 医生审核时间窗 |
| `policy_snapshot` | text/json | 当次服务端规则快照 |
| `required` | tinyint | 是否要求核验 |
| `session_id` | varchar(64) nullable | faces session |
| `proof_id` | varchar(128) nullable | 核验成功 proof |
| `status` | varchar(32) | created/session_issued/verified/consumed/failed/expired |
| `result_code` | varchar(64) nullable | 机器结果码 |
| `client_nonce_hash` | varchar(64) nullable | 登录前事件绑定浏览器 |
| `finalize_status` | varchar(32) nullable | pending/succeeded/failed |
| `finalize_attempts` | int | finalize 重试次数 |
| `last_error` | varchar(255) nullable | 脱敏后的最后错误 |
| `expires_at` | datetime nullable | 事件有效期 |
| `created_at` | datetime | 创建时间 |
| `verified_at` | datetime nullable | 核验成功时间 |
| `consumed_at` | datetime nullable | 业务消费时间 |

索引：

- `UNIQUE(business_event_id)`
- `UNIQUE(request_id)`
- `UNIQUE(proof_id)`
- `INDEX(subject_type, subject_id, scene, created_at)`
- `INDEX(record_id, action)`

### 4.2 `fa_face_verify_consumption`

使用独立消费表形成数据库唯一约束，避免 proof 重放。

| 字段 | 建议类型 | 说明 |
| --- | --- | --- |
| `id` | bigint | 主键 |
| `proof_id` | varchar(128) | 唯一 |
| `business_event_id` | varchar(64) | 唯一 |
| `scene` | varchar(32) | 场景 |
| `subject_type` | varchar(32) | 人员类型 |
| `subject_id` | bigint | 人员 ID |
| `record_id` | varchar(64) nullable | 审核记录 |
| `action` | varchar(32) | 最终动作 |
| `consumed_at` | datetime | 消费时间 |

索引：

- `UNIQUE(proof_id)`
- `UNIQUE(business_event_id)`

### 4.3 `fa_face_profile`

只保存业务人员到人脸服务模板的映射和迁移状态。

| 字段 | 建议类型 | 说明 |
| --- | --- | --- |
| `subject_type` | varchar(32) | 人员类型 |
| `subject_id` | bigint | 人员 ID |
| `active_template_id` | varchar(64) nullable | 当前模板 |
| `template_version` | int | 模板版本 |
| `avatar_hash` | varchar(64) nullable | 来源图片哈希 |
| `status` | varchar(32) | pending/active/pending_reenroll/revoked |
| `failure_code` | varchar(64) nullable | 迁移失败原因 |
| `updated_at` | datetime | 更新时间 |

索引：

- `UNIQUE(subject_type, subject_id)`

旧 `feature` 字段保持只读兼容一个回滚周期，不写入新版 InsightFace 向量。

## 5. chbzg 后端组件

### 5.1 `FaceVerifyPolicy`

职责：

- 从服务端当前用户、角色和配置解析真实 `subject_type/subject_id/admin_id`。
- 保留 `face_verify_doctor`、`face_verify_doctor_ids`、
  `face_verify_pharmacist`、`face_verify_pharmacist_ids` 原语义。
- 保留医生审核时间窗和“一个时间窗完成一次后不再触发”的规则。
- 在最终写入口重新计算是否要求核验，前端查询结果仅用于展示。
- 输出结构化 `PolicyDecision`，至少包含 `required`、`scene`、`window_key`、
  `policy_version` 和原因。

### 5.2 `FaceVerifyClient`

职责：

- 只在后端保存 faces 基础地址和 API Key。
- 封装模板登记、创建 session、introspect 和 finalize。
- 连接超时建议 2 秒，普通接口总超时建议 5 秒；模型 verify 由浏览器直传。
- 创建 session 可对网络错误重试一次，但必须复用同一 `request_id`。
- `introspect` 不自动重试业务拒绝，只对连接错误做有限重试。
- 统一映射 HTTP 状态和 faces 结果码，不根据响应中是否存在某字段判断成功。
- 日志隐藏 API Key、upload token、完整 Base64 和原始人脸图片。

### 5.3 `FaceVerifyEventService`

职责：

- 创建业务事件并保存 policy snapshot。
- 调用 faces 创建 session。
- 将 session、动作、标签、短期 token 和过期时间返回给前端。
- 接收前端返回的 proof ID，并把它绑定到当前事件。
- 管理 failed/expired 状态和重新发起逻辑。
- 登录事件只能在账号密码已校验正确后创建。

### 5.4 `FaceVerifyGuard`

职责：

- 在最终业务接口调用 `introspect`。
- 按第 3.4 节逐字段校验。
- 在本地数据库事务内插入 `fa_face_verify_consumption`。
- 与审核状态更新或登录事件消费放在同一事务边界内。
- 唯一索引冲突按 proof 重放拒绝。
- 调用远程 introspect 前先查询本地消费记录；相同 event/proof 的重复请求返回原业务结果，
  不再次执行业务更新，也不因为远程 proof 已 finalized 而误报失败。
- 事务提交后调用 faces `finalize`。
- finalize 网络失败不回滚已完成业务，由补偿任务重试；本地唯一消费仍可阻止重放。

### 5.5 `FaceVerifyFinalizeJob`

职责：

- 查询本地已 consumed 但 faces 尚未 finalize 的事件。
- 使用相同 `proof_id + business_event_id` 幂等重试。
- 指数退避并记录最后错误。
- 超过重试上限产生告警，不自动取消已经成功的业务操作。

## 6. 医生登录接入流程

### 6.1 推荐接口

```text
POST /admin/index/login
POST /admin/face_verify/session
POST /admin/index/face_login_complete
```

也可以保留单一登录路由，但后端状态机必须等价。

### 6.2 完整流程

1. 浏览器提交用户名、密码和验证码。
2. `Index::login()` 完成现有账号密码校验，但暂不建立正式登录态。
3. 服务端解析真实 admin 和 doctor/pharmacist ID。
4. `FaceVerifyPolicy` 判断是否需要核验。
5. 不需要核验时，执行现有登录成功流程。
6. 需要核验时，创建短期 pre-login 事件，不保存明文密码。
7. 服务端调用 faces 创建 `scene=login, action=login` session。
8. 返回 `face_required=true`、事件 ID、session ID、upload token、动作和过期时间。
9. 浏览器按动作采集连续帧并直传 faces。
10. faces 成功返回 proof ID。
11. 浏览器调用 `face_login_complete`，只提交事件 ID 和 proof ID。
12. 后端确认 pre-login 事件仍有效，并使用 Guard 校验 proof。
13. proof 绑定正确后消费事件，建立正式登录态。
14. 事务/登录完成后 finalize proof。

### 6.3 登录安全要求

- 未通过密码校验不得创建指定人员的人脸 session。
- pre-login 事件绑定浏览器会话 nonce、IP 摘要或现有 CSRF/session 标识。
- 不在前端保存密码，不为待核验用户提前创建完整后台登录态。
- proof 不能用于另一个账号、另一次登录事件或审核场景。
- 页面刷新可恢复未过期事件，但不能创建无限并行 session。
- 登录 completion 失败后，同一 proof 是否允许重试只限相同事件；一旦成功不得复用。
- completion 已消费但浏览器未收到响应时，相同事件、proof 和 client nonce 应幂等恢复登录，
  不重新执行 proof 消费。

## 7. 医生审核接入流程

### 7.1 触发规则

保留现有四个时间窗：

```text
09:00-11:00
11:00-13:00
13:00-17:00
17:00-20:00
```

后端以服务器时间计算 `window_key`。当前时间窗已有成功核验记录时，可按原规则不再要求；
失败、取消、前端关闭弹窗均不能写入成功记录。

### 7.2 完整流程

1. 用户点击审核通过或驳回。
2. 前端先向 chbzg 请求本次审核事件，不直接向 faces 指定人员。
3. 后端从当前登录态解析医生 ID，并根据记录、动作、时间窗计算 Policy。
4. 不需要核验时，返回 `required=false`，前端继续原审核接口。
5. 需要核验时，后端创建绑定医生、记录、动作、时间窗的事件和 faces session。
6. 浏览器完成连续帧核验并获得 proof ID。
7. 前端把 `business_event_id + proof_id` 随最终审核请求提交。
8. 最终审核写入口重新计算 Policy，并调用 Guard 校验 proof 全部字段。
9. 同一数据库事务中插入 proof 消费记录、更新审核业务状态、记录时间窗核验成功。
10. 提交成功后 finalize proof。

### 7.3 必须防止的绕过

- 直接调用审核写接口但不传 proof。
- 使用登录 proof 做审核。
- 使用医生 A 的 proof 操作医生 B。
- 使用记录 A 的 proof 操作记录 B。
- 使用 `pass` proof 执行 `reject`。
- 使用上一时间窗或上一业务事件的 proof。
- 重复提交同一 proof 处理第二笔记录。
- 前端伪造“本时间窗已核验”日志。

## 8. 药师审核接入

当前 `face_verify_pharmacist=0` 时必须保持现有业务行为，不强制核验。

实现时仍完成以下结构：

- Policy 支持药师关闭、全员和指定 ID 三种模式。
- `Record::pass()` 与 `Record::reject()` 均接入最终 Guard。
- proof 分别绑定 `action=pass/reject`。
- 不再将临时抓拍保存到公开上传目录。
- 开关关闭时不创建 session，不调用 faces，不增加审核延迟。

药师正式开启必须单独灰度，不与医生首批切换同时进行。

## 9. 前端接入

### 9.1 可复用内容

从当前 `backend/static/app.js` 提取独立采集模块，保留：

- 360px 帧宽。
- JPEG 质量 0.7。
- 200ms 采样间隔。
- 每动作完整采集 2 秒。
- `action/index/timestamp/image` 协议。
- 摄像头就绪、关闭摄像头、超时、错误分类和重复点击保护。

建议封装为：

```text
public/assets/js/backend/face_verify_client.js
```

对外只暴露：

```text
start({sessionId, uploadToken, actions, labels, expiresAt})
cancel()
destroy()
```

成功返回 `proofId`，不向业务页面暴露阈值控制。

### 9.2 网络方式

浏览器直接请求：

```text
https://faces.chbzg.com.cn/v1/verification-sessions/{session_id}/verify
```

优点是连续帧不经过 PHP，降低业务服务器内存、请求体解析和超时压力。

上线前必须确认：

- 实际 chbzg HTTPS 页面域名已加入 faces CORS。
- 页面和 faces 均为 HTTPS，避免摄像头权限和 mixed content 问题。
- Nginx 请求体上限、超时和 CSP `connect-src` 允许 faces 域名。
- upload token 只存在当前页面内存，不写 localStorage、日志或 URL。

## 10. 模板迁移

### 10.1 迁移步骤

1. 按当前配置导出需要核验的医生，首批只处理 `83,96,84`。
2. 从服务器本地头像文件读取原图，不让 faces 服务抓取外部 URL。
3. 计算头像 SHA-256，与 `fa_face_profile.avatar_hash` 比较。
4. chbzg 后端调用 `/v1/internal/templates` 上传图片。
5. 保存返回的 template ID、version 和 active 状态。
6. 无脸、多脸、损坏、过小或模型失败人员标记 `pending_reenroll`。
7. 失败人员通过管理员入口重新上传清晰正脸。
8. 新模板成功后再启用该人员新人脸链路。

### 10.2 迁移约束

- 不读取和转换旧 `feature` 作为新版向量。
- 不覆盖药师指纹 `template`。
- 相同头像哈希重复执行不生成新版本。
- 模板图片默认不在 faces 服务长期保存。
- 批处理输出成功、跳过、失败和需重采清单。

## 11. 异常、超时与一致性

| 情况 | chbzg 行为 |
| --- | --- |
| 没有 active 模板 | 阻断需要核验的操作，提示联系管理员登记 |
| session 过期或已使用 | 创建新业务事件重新核验 |
| 动作/PAD/人脸失败 | 显示可操作原因，重新发起，不放行业务 |
| faces 连接超时 | 不放行；允许用户稍后重试 |
| faces 模型 503 | 不把它记为用户核验失败，保留受控重试 |
| introspect 返回无效 | 拒绝并重新核验 |
| 本地业务事务失败 | 本地消费回滚；同一事件有效期内允许重试 |
| 业务成功、finalize 失败 | 业务保持成功，补偿任务重试 finalize |
| 重复业务提交 | 本地 proof/business event 唯一索引拒绝第二次执行 |

不得采用“远程服务异常就跳过人脸”的降级方式。

## 12. 配置与密钥

`chbzg` 服务端新增配置：

```text
face_verify_base_url=https://faces.chbzg.com.cn
face_verify_api_key=<secret>
face_verify_connect_timeout=2
face_verify_timeout=5
face_verify_enabled=0/1
face_verify_shadow_mode=0/1
face_verify_policy_version=<version>
```

要求：

- API Key 只保存在服务器环境变量或受控配置，不进入前端和 Git。
- 测试和生产使用不同 Key。
- 日志统一脱敏 proof、token、API Key 和 Base64。
- Key 轮换时允许短期双 Key 过渡，随后吊销旧 Key。

## 13. 灰度与切换

### 阶段 0：接入前收口

- 完成 F1-F8。
- 建表并实现四个后端组件。
- 确认实际 chbzg 域名、CORS、CSP 和 HTTPS。
- 为灰度医生准备 active 模板。

### 阶段 1：影子模式

- 只记录新人脸结果，不影响现有业务放行。
- 仅覆盖医生 `83,96,84`。
- 对比成功率、失败码、总耗时和服务错误。
- 不允许影子模式写入“已核验时间窗”正式记录。

### 阶段 2：医生登录强制

- `Index::login()` 改为密码通过后创建 pre-login 事件。
- 新 proof 成为指定医生登录唯一人脸凭证。
- 旧单图比对不再参与登录判定。

### 阶段 3：医生审核强制

- 最终审核写入口启用 Guard。
- 时间窗规则保持不变。
- 完成禁 JS、直连接口、重放和跨记录测试。

### 阶段 4：药师兼容灰度

- 总开关继续关闭。
- 选少量 ID 单独开启验证。
- 验证 pass/reject 两条最终写路径。

### 阶段 5：旧链路下线

- 停止调用旧人脸远程接口。
- 停止写入旧 `feature`。
- 收紧旧公开接口权限。
- 清理公开目录临时抓拍。
- 旧表只读保留一个回滚周期后再决定删除。

## 14. 回滚方案

回滚必须是发布操作，不是运行时自动放行：

1. 保留数据库迁移，关闭新人脸强制开关。
2. 回滚 chbzg 前后端代码到上一版本。
3. faces 服务保持运行，已有模板和 proof 数据不删除。
4. 旧链路仅在明确的紧急回滚窗口临时启用。
5. 记录回滚原因、操作人、开始时间、结束时间和受影响人员。

任何回滚方案都不能把“需要核验但服务异常”自动变成“无需核验”。

## 15. 实施任务清单

### A. 人脸服务接入前收口

- [ ] A01 确认 chbzg 生产和测试页面的准确 HTTPS Origin。
- [ ] A02 更新 faces CORS 白名单并验证 OPTIONS 预检。
- [ ] A03 session 幂等复用时校验所有绑定字段，冲突返回 409。
- [ ] A04 模板登记实现 `request_id` 幂等。
- [ ] A05 将 upload token 改为可重建、可验签、带有效期的服务端签名 token。
- [ ] A06 增加 active template 查询接口。
- [ ] A07 增加模板吊销接口和审计字段。
- [ ] A08 session/proof 增加 policy/model/threshold version。
- [ ] A09 增加单实例并发上限和 `ERROR_SERVICE_BUSY`。
- [ ] A10 补充上述接口、冲突、令牌和并发测试。

### B. chbzg 数据库

- [ ] B01 新增 `fa_face_verify_event`。
- [ ] B02 新增 `fa_face_verify_consumption`。
- [ ] B03 新增 `fa_face_profile`。
- [ ] B04 建立 proof、business event 和 request ID 唯一索引。
- [ ] B05 编写可回滚迁移并在测试库验证。

### C. chbzg 后端公共组件

- [ ] C01 实现 `FaceVerifyPolicy`。
- [ ] C02 为四个现有配置项补齐关闭、全员、名单模式测试。
- [ ] C03 实现医生审核时间窗和成功记录判断。
- [ ] C04 实现 `FaceVerifyClient`。
- [ ] C05 实现 `FaceVerifyEventService`。
- [ ] C06 实现 `FaceVerifyGuard` 全字段校验。
- [ ] C07 实现本地 proof 原子消费和重放拒绝。
- [ ] C08 实现 finalize 补偿任务。
- [ ] C09 增加统一结果码和用户错误文案映射。
- [ ] C10 增加脱敏日志和接口耗时记录。

### D. 模板迁移

- [ ] D01 导出首批医生 `83,96,84` 的 admin、doctor、avatar 映射。
- [ ] D02 实现头像文件读取、哈希、格式和大小校验。
- [ ] D03 批量调用模板登记接口。
- [ ] D04 保存 profile、模板版本和迁移状态。
- [ ] D05 输出无脸、多脸、损坏和失败清单。
- [ ] D06 实现管理员重新登记入口。
- [ ] D07 验证重复迁移不会产生多余模板版本。

### E. 登录接入

- [ ] E01 重构 `Index::login()`，密码正确后再判断人脸策略。
- [ ] E02 实现短期 pre-login 事件，不保存明文密码。
- [ ] E03 实现登录 session 创建接口。
- [ ] E04 实现 `face_login_complete`。
- [ ] E05 正式登录态只在 proof Guard 通过后建立。
- [ ] E06 验证错误密码不能创建人脸 session。
- [ ] E07 验证无 proof、过期、跨账号、跨场景和重放均失败。

### F. 医生审核接入

- [ ] F01 明确医生审核最终写入口和 action 枚举。
- [ ] F02 后端创建绑定记录、动作和时间窗的业务事件。
- [ ] F03 最终写入口重新计算 Policy。
- [ ] F04 Guard 与审核业务更新放入同一事务。
- [ ] F05 核验成功后才写时间窗完成记录。
- [ ] F06 验证跨记录、跨动作、跨医生和重复 proof 均失败。
- [ ] F07 验证业务事务失败后相同事件可安全重试。

### G. 药师审核兼容

- [ ] G01 Policy 支持药师关闭、全员和名单模式。
- [ ] G02 `Record::pass()` 接入 Guard。
- [ ] G03 `Record::reject()` 接入 Guard。
- [ ] G04 删除公开临时抓拍依赖。
- [ ] G05 保持总开关关闭时零额外调用。

### H. 前端采集

- [ ] H01 从现有 app.js 提取通用连续帧采集模块。
- [ ] H02 登录页面接入核验弹层。
- [ ] H03 医生审核页面接入核验弹层。
- [ ] H04 药师审核页面预留接入。
- [ ] H05 upload token 只保存在页面内存。
- [ ] H06 配置 CSP `connect-src` 和 faces CORS。
- [ ] H07 验证取消、过期、重复点击、摄像头拒绝和网络超时。
- [ ] H08 验证低配电脑采集参数未被放大。

### I. 自动化与防绕过测试

- [ ] I01 API 契约与结果码测试。
- [ ] I02 session/template 幂等冲突测试。
- [ ] I03 proof 全字段绑定测试。
- [ ] I04 proof 本地唯一消费与 finalize 补偿测试。
- [ ] I05 登录禁 JS 和直连接口测试。
- [ ] I06 审核禁 JS 和直连接口测试。
- [ ] I07 跨人员、跨场景、跨记录、跨动作测试。
- [ ] I08 proof 过期、重放和并发双提交测试。
- [ ] I09 faces 超时、503、连接中断测试。
- [ ] I10 数据库事务失败和补偿任务测试。

### J. 灰度、上线与清理

- [ ] J01 灰度开关和影子模式配置完成。
- [ ] J02 首批医生模板全部 active。
- [ ] J03 影子模式运行并检查错误率和耗时。
- [ ] J04 医生登录强制切换。
- [ ] J05 医生审核强制切换。
- [ ] J06 完成回滚演练。
- [ ] J07 药师小范围独立灰度。
- [ ] J08 停止旧远程人脸接口调用。
- [ ] J09 停止旧 `feature` 写入。
- [ ] J10 收紧旧接口并清理公开临时图片。

## 16. 开发顺序与完成判定

推荐严格按以下顺序开发：

```text
A 人脸服务收口
→ B 数据库
→ C 公共组件
→ D 首批模板
→ H 通用前端采集
→ E 医生登录
→ F 医生审核
→ I 防绕过测试
→ J 医生灰度
→ G 药师兼容与后续灰度
```

医生首批接入完成必须同时满足：

- 三名灰度医生都有 active 新模板。
- 登录和审核最终后端入口均不能绕过。
- proof 跨人员、场景、记录、动作、事件和重复使用全部失败。
- faces 网络或模型异常不会静默放行。
- 业务事务失败、finalize 失败均有明确且可重试的处理路径。
- 旧单图结果不再参与最终通过判定。
- 自动化测试、灰度测试和回滚演练均有记录。

真实人脸效果验收本轮按用户要求暂缓，但在扩大灰度名单或全量开启前仍必须完成。
