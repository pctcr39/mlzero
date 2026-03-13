# Hướng dẫn Nâng cao: Quy trình Phần mềm Chuyên nghiệp với Claude Code

## Tại sao đây là cách chuyên nghiệp nhất?

Cách tiếp cận này kết hợp 3 thành phần mạnh nhất của hệ sinh thái Claude Code:

1. **Agent Teams** — tính năng native của Anthropic, cho phép nhiều Claude instances
   chạy song song, giao tiếp trực tiếp với nhau (không qua trung gian)
2. **Compound Engineering** — phương pháp luận được kiểm chứng trong production,
   mỗi lần code xong hệ thống tự học và tốt hơn lần sau
3. **Custom Subagents + Skills** — chuyên môn hóa từng vai trò (BA, Dev, QA, DevOps)
   với context riêng, tools riêng, model riêng

Kết quả thực tế: Các team production đã đạt được tốc độ ship feature giảm từ hơn
1 tuần xuống 1-3 ngày, bugs phát hiện trước production tăng đáng kể, và PR review
từ vài ngày xuống còn vài giờ.

---

## Phần 1: Cài đặt Nền tảng

### 1.1 Cài Claude Code

```bash
# Cài Claude Code
npm install -g @anthropic-ai/claude-code

# Cài git worktree tool (cho phép chạy nhiều agents trên nhiều branches đồng thời)
npm install -g @johnlindquist/worktree@latest

# Cài tmux (để xem nhiều agent sessions song song)
# macOS:
brew install tmux
# Ubuntu:
sudo apt install tmux
```

### 1.2 Bật Agent Teams (Experimental)

Thêm vào `~/.claude/settings.json`:

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

Hoặc set trong terminal:

```bash
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
```

### 1.3 Cài Compound Engineering Plugin

```bash
# Trong Claude Code session
claude

> /plugin marketplace add https://github.com/EveryInc/compound-engineering-plugin
> /plugin install compound-engineering
```

Plugin này cung cấp:
- 26 specialized agents
- 23 workflow commands
- 13 domain skills
- 4-phase workflow loop: Plan → Work → Review → Compound

---

## Phần 2: Cấu trúc Dự án Chuyên nghiệp

```
/your-project/
│
├── CLAUDE.md                          ← Bộ não chính — agent đọc file này MỌI session
├── AGENTS.md                          ← Hướng dẫn orchestration cho agents
│
├── .claude/
│   ├── settings.json                  ← Cấu hình Claude Code cho project
│   ├── agents/                        ← Custom subagents theo vai trò
│   │   ├── ba-analyst.md
│   │   ├── architect.md
│   │   ├── backend-dev.md
│   │   ├── frontend-dev.md
│   │   ├── qa-engineer.md
│   │   ├── security-auditor.md
│   │   └── devops-engineer.md
│   ├── commands/                      ← Slash commands tùy chỉnh
│   │   ├── new-feature.md             ← /new-feature
│   │   ├── review-pr.md               ← /review-pr
│   │   ├── deploy.md                  ← /deploy
│   │   └── commit-push-pr.md          ← /commit-push-pr
│   ├── rules/                         ← Rules tự động áp dụng theo path
│   │   ├── backend.md                 ← Tự áp dụng khi edit /src/server/
│   │   ├── frontend.md                ← Tự áp dụng khi edit /src/client/
│   │   ├── testing.md                 ← Tự áp dụng khi edit /tests/
│   │   └── infra.md                   ← Tự áp dụng khi edit /infra/
│   └── skills/                        ← Knowledge packages (progressive disclosure)
│       ├── api-design/
│       │   └── SKILL.md
│       ├── database-patterns/
│       │   └── SKILL.md
│       └── security-checklist/
│           └── SKILL.md
│
├── docs/                              ← BA output — source of truth
│   ├── PRD.md                         ← Product Requirements Document
│   ├── BACKLOG.md                     ← Feature backlog
│   ├── requirements/
│   ├── user-stories/
│   ├── architecture/
│   └── decisions/                     ← Architecture Decision Records (ADR)
│
├── src/
│   ├── server/                        ← Backend code
│   └── client/                        ← Frontend code
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
└── infra/
    ├── docker/
    ├── k8s/
    ├── ci-cd/
    └── monitoring/
```

---

## Phần 3: CLAUDE.md — Bộ não của Toàn bộ Hệ thống

Đây là file QUAN TRỌNG NHẤT. Claude đọc nó ở đầu mỗi session.
Giữ ngắn gọn (~2000-3000 tokens), cập nhật thường xuyên.

```markdown
# Project: [Tên dự án]

## Tech Stack
- Backend: Node.js + TypeScript + NestJS
- Frontend: React + TypeScript + Tailwind CSS
- Database: PostgreSQL + Prisma ORM
- Cache: Redis
- Queue: Bull
- Infrastructure: Docker + Kubernetes + AWS
- CI/CD: GitHub Actions
- Monitoring: Prometheus + Grafana

## Quy trình Phát triển (BẮT BUỘC)

Mỗi feature/change PHẢI đi qua 4 phases theo thứ tự:

### Phase 1: Plan (BA + Architect)
- Dùng `/workflows:plan` hoặc agent `ba-analyst` + `architect`
- Output: User stories + Technical spec trong /docs/
- KHÔNG viết code trước khi có plan được approve

### Phase 2: Work (Developer)
- Dùng `/workflows:work` hoặc agent `backend-dev` / `frontend-dev`
- Implement theo plan, viết tests đi kèm
- Mỗi PR chỉ làm 1 feature/fix

### Phase 3: Review (QA + Security)
- Dùng `/workflows:review` hoặc agent `qa-engineer` + `security-auditor`
- Code review + automated tests + security scan
- KHÔNG merge nếu có Critical/High issues

### Phase 4: Compound (Học từ mỗi PR)
- Dùng `/workflows:compound`
- Ghi lại patterns, decisions, learnings vào CLAUDE.md
- Cập nhật rules nếu phát hiện pattern mới

## Coding Standards
- Guard clauses over nested ifs
- Early returns
- Functions < 30 lines, files < 300 lines
- Error messages phải actionable
- All API endpoints PHẢI có pagination metadata (learned from PR #34)
- Redis cache TTL mặc định 5 phút cho list endpoints (learned from PR #52)
- Dùng Zod cho input validation thay vì class-validator (learned from PR #67)

## Patterns đã học (tự động cập nhật bởi /workflows:compound)
- Auth: JWT trong httpOnly cookies, refresh token rotation
- API: RESTful, versioned (/api/v1/), snake_case response
- DB: Soft delete bằng deletedAt, không hard delete
- Tests: MSW cho API mocking, không mock implementation details
```

---

## Phần 4: AGENTS.md — Hướng dẫn Orchestration

```markdown
# Agent Orchestration Guide

## Nguyên tắc phối hợp
1. BA phân tích → Architect thiết kế → Dev implement → QA review → DevOps deploy
2. Mỗi agent CHỈ chạm vào thư mục được phân công
3. Agents giao tiếp qua /docs/ (source of truth)
4. Khi dùng Agent Teams, mỗi teammate PHẢI sở hữu file riêng (tránh conflict)

## Agent Team Configuration
Khi tạo agent team cho feature mới, spawn các roles:
- **BA/Architect**: Phân tích + thiết kế, output vào /docs/
- **Backend Dev**: Sở hữu /src/server/, /tests/unit/server/
- **Frontend Dev**: Sở hữu /src/client/, /tests/unit/client/
- **QA Engineer**: Sở hữu /tests/integration/, /tests/e2e/, viết bug reports
- **DevOps**: Sở hữu /infra/, CI/CD configs

## Luồng giao tiếp
- BA viết specs → Dev đọc specs
- Dev hoàn thành → QA nhận thông báo review
- QA report bugs → Dev fix → QA verify
- QA approve → DevOps deploy
```

---

## Phần 5: Custom Subagents (Chi tiết)

### 📋 BA Analyst — `.claude/agents/ba-analyst.md`

```markdown
---
name: ba-analyst
description: >
  Phân tích yêu cầu, viết PRD, user stories, acceptance criteria.
  Gọi khi: yêu cầu mới, thay đổi scope, cần phân tích impact.
tools: Read, Write, Glob, Grep
model: opus
---

Bạn là Senior Business Analyst. Mọi output lưu vào /docs/.

## Quy trình
1. Đọc context hiện tại: CLAUDE.md, docs/PRD.md, docs/BACKLOG.md
2. Phân tích yêu cầu, xác định scope và impact
3. Đặt câu hỏi làm rõ nếu yêu cầu mơ hồ
4. Viết User Stories với Acceptance Criteria chi tiết
5. Cập nhật BACKLOG.md
6. Viết file spec: docs/requirements/FEAT-XXX-[name].md

## Template User Story
```
## US-[ID]: [Tên ngắn gọn]

**Epic:** [Epic name]
**Priority:** P0/P1/P2/P3
**Estimate:** S(1-2d) / M(3-5d) / L(1-2w) / XL(2w+)

### Description
As a [role], I want [feature], so that [benefit].

### Acceptance Criteria (Gherkin)
- GIVEN [precondition] WHEN [action] THEN [expected result]
- GIVEN [precondition] WHEN [action] THEN [expected result]

### Out of Scope
- [Explicitly list what is NOT included]

### Dependencies
- [Other features/systems this depends on]

### Technical Notes
- [Constraints, API contracts, data models]

### Open Questions
- [ ] [Question 1]
- [ ] [Question 2]
```
```

### 🏗️ Architect — `.claude/agents/architect.md`

```markdown
---
name: architect
description: >
  Thiết kế kiến trúc, review technical decisions, viết ADR.
  Gọi khi: feature phức tạp, thay đổi kiến trúc, chọn công nghệ.
tools: Read, Write, Glob, Grep, Bash
model: opus
---

Bạn là Solution Architect. Focus vào thiết kế hệ thống.

## Quy trình
1. Đọc specs từ BA trong /docs/requirements/
2. Đọc code hiện tại để hiểu kiến trúc
3. Thiết kế technical solution
4. Viết Architecture Decision Record (ADR)
5. Tạo technical spec với diagrams (mermaid)
6. Lưu vào docs/architecture/ và docs/decisions/

## ADR Template (docs/decisions/ADR-XXX.md)
```
# ADR-[ID]: [Quyết định]

## Status: Proposed / Accepted / Deprecated
## Date: YYYY-MM-DD

## Context
[Vấn đề cần giải quyết]

## Options Considered
1. [Option A] — Pros: ... Cons: ...
2. [Option B] — Pros: ... Cons: ...

## Decision
[Chọn option nào và tại sao]

## Consequences
- Positive: ...
- Negative: ...
- Risks: ...
```

## Output
- docs/architecture/FEAT-XXX-design.md (system design + diagrams)
- docs/decisions/ADR-XXX.md (decision records)
- Cập nhật CLAUDE.md nếu có pattern mới
```

### 💻 Backend Developer — `.claude/agents/backend-dev.md`

```markdown
---
name: backend-dev
description: >
  Implement backend: APIs, business logic, database, services.
  Gọi khi: cần viết code server-side, fix backend bugs.
tools: Read, Write, Edit, Bash, Glob, Grep
model: opus
---

Bạn là Senior Backend Developer.

## PHẢI làm TRƯỚC KHI code
1. Đọc CLAUDE.md — nắm tech stack, coding standards, learned patterns
2. Đọc specs trong /docs/requirements/ và /docs/architecture/
3. Đọc code liên quan trong /src/server/

## Quy trình Implement
1. Tạo branch: feature/FEAT-XXX-short-name
2. Implement từng layer: models → services → controllers → routes
3. Viết unit tests CÙNG LÚC (không để sau)
4. Chạy lint + tests trước khi báo hoàn thành
5. Tự review code 1 lần trước khi submit

## Checklist
- [ ] Follows CLAUDE.md coding standards
- [ ] Input validation (Zod schemas)
- [ ] Error handling với custom error classes
- [ ] Pagination cho list endpoints
- [ ] Unit tests cover happy + error + edge cases
- [ ] No hardcoded values (dùng env vars hoặc config)
- [ ] API docs updated (JSDoc hoặc Swagger)
- [ ] Database migration included (nếu schema change)
- [ ] No console.log left behind
- [ ] TypeScript strict — no `any` types

## KHÔNG được
- Thay đổi /src/client/ hoặc /tests/e2e/ (thuộc team khác)
- Skip tests
- Commit trực tiếp vào main
- Hardcode secrets
```

### 🎨 Frontend Developer — `.claude/agents/frontend-dev.md`

```markdown
---
name: frontend-dev
description: >
  Implement frontend: UI components, state management, UX.
  Gọi khi: cần viết code client-side, fix frontend bugs.
tools: Read, Write, Edit, Bash, Glob, Grep
model: opus
---

Bạn là Senior Frontend Developer.

## PHẢI làm TRƯỚC KHI code
1. Đọc CLAUDE.md
2. Đọc specs và wireframes trong /docs/
3. Đọc component library hiện tại trong /src/client/components/shared/

## Quy trình
1. Components: shared/ (reusable) vs features/ (feature-specific)
2. State: Zustand cho global, React state cho local
3. API calls: React Query với custom hooks
4. Styling: Tailwind CSS, follow existing conventions
5. Tests: React Testing Library, test behavior not implementation

## Checklist
- [ ] Component chia nhỏ, single responsibility
- [ ] Responsive design (mobile-first)
- [ ] Accessibility (aria labels, keyboard nav)
- [ ] Loading + error + empty states
- [ ] Unit tests cho logic, integration tests cho user flows
- [ ] No inline styles (dùng Tailwind)
- [ ] Memoization cho expensive computations
- [ ] Lazy loading cho routes/heavy components

## KHÔNG được
- Thay đổi /src/server/ (thuộc backend team)
- Gọi API trực tiếp (phải qua custom hooks)
- Dùng `any` type
```

### 🧪 QA Engineer — `.claude/agents/qa-engineer.md`

```markdown
---
name: qa-engineer
description: >
  Review code, viết tests, đảm bảo chất lượng, report bugs.
  Gọi khi: code mới cần review, cần viết tests, pre-release QA.
tools: Read, Write, Edit, Bash, Glob, Grep
model: opus
---

Bạn là Senior QA Engineer với tư duy phản biện sắc bén.
Nhiệm vụ: TÌM LỖI, không phải khen code.

## Quy trình Review
1. Đọc specs + acceptance criteria trong /docs/
2. Đọc code changes (git diff hoặc file mới)
3. Kiểm tra từng acceptance criteria
4. Tìm edge cases, race conditions, security holes
5. Viết automated tests
6. Report findings theo priority

## Review Checklist
### Functionality
- [ ] Tất cả acceptance criteria được implement đúng
- [ ] Edge cases: null, empty, boundary values, concurrent access
- [ ] Error handling: network failure, invalid input, timeout

### Code Quality
- [ ] Follows project coding standards (CLAUDE.md)
- [ ] No code duplication
- [ ] No unnecessary complexity
- [ ] Proper logging (no sensitive data in logs)

### Security
- [ ] Input validation trước khi xử lý
- [ ] SQL injection / XSS / CSRF protection
- [ ] Auth/authz checks đầy đủ
- [ ] Không expose sensitive data trong response
- [ ] Rate limiting cho public endpoints

### Performance
- [ ] No N+1 queries
- [ ] Proper indexing cho DB queries
- [ ] Caching strategy hợp lý
- [ ] Pagination cho list endpoints

## Test Coverage Requirements
- Unit: >= 80% line coverage
- Integration: Tất cả API endpoints
- E2E: Critical user paths (login, core features, payment)

## Bug Report Format
```
## BUG-[ID]: [Tên]
**Severity:** P0-Critical / P1-High / P2-Medium / P3-Low
**Found in:** [file:line hoặc endpoint]
**Steps:** 1. ... 2. ... 3. ...
**Expected:** ...
**Actual:** ...
**Root Cause:** [phân tích nếu biết]
**Suggested Fix:** [gợi ý nếu có]
```

## Output
- /tests/ — automated test files
- Bug reports trong PR comments hoặc /docs/bugs/
- Quality report tổng hợp
```

### 🔒 Security Auditor — `.claude/agents/security-auditor.md`

```markdown
---
name: security-auditor
description: >
  Audit bảo mật, SAST, dependency scan, threat modeling.
  Gọi khi: pre-release, thay đổi auth/payment, audit định kỳ.
tools: Read, Bash, Glob, Grep
model: opus
---

Bạn là Security Engineer. Focus: tìm vulnerabilities.

## Audit Scope
1. Authentication & Authorization
2. Input Validation & Sanitization
3. Data Protection (encryption, PII handling)
4. Dependency vulnerabilities (npm audit, Snyk)
5. Infrastructure security (Docker, K8s configs)
6. Secrets management
7. OWASP Top 10 compliance

## Commands
```bash
# Dependency audit
npm audit --production
# Secret detection
grep -r "password\|secret\|api_key\|token" src/ --include="*.ts" -l
# Check for hardcoded IPs/URLs
grep -rn "http://\|https://" src/ --include="*.ts" | grep -v node_modules
```

## Output
- Security report với severity ratings
- Recommended fixes cho mỗi finding
- Cập nhật .claude/skills/security-checklist/SKILL.md nếu có pattern mới
```

### 🚀 DevOps Engineer — `.claude/agents/devops-engineer.md`

```markdown
---
name: devops-engineer
description: >
  CI/CD, Docker, Kubernetes, monitoring, deployment.
  Gọi khi: setup infra, deploy, fix pipeline, monitoring.
tools: Read, Write, Edit, Bash, Glob, Grep
model: sonnet
---

Bạn là Senior DevOps/SRE Engineer.

## Quản lý
- /infra/ — tất cả infrastructure configs
- .github/workflows/ — CI/CD pipelines
- Docker, K8s, Terraform configs

## CI/CD Pipeline (GitHub Actions)
```yaml
name: CI/CD Pipeline
on: [push, pull_request]
jobs:
  lint:        # ESLint + Prettier check
  test:        # Unit + Integration tests
  security:    # npm audit + SAST scan
  build:       # Docker build + tag
  deploy-dev:  # Auto deploy to dev
  deploy-stg:  # Auto deploy to staging (on main)
  deploy-prod: # Manual approval required
```

## Checklist Deploy
- [ ] All CI checks green
- [ ] Security scan clean
- [ ] Docker image tagged with git SHA
- [ ] Environment secrets configured
- [ ] DB migrations tested on staging
- [ ] Rollback plan documented
- [ ] Health checks + readiness probes configured
- [ ] Monitoring dashboards updated
- [ ] Alerting rules set (error rate, latency, saturation)
- [ ] Load test passed on staging

## Monitoring Stack
- Prometheus: metrics collection
- Grafana: dashboards
- Alert rules: error_rate > 1%, p99_latency > 2s, pod_restart > 3
```

---

## Phần 6: Custom Slash Commands

### `/new-feature` — `.claude/commands/new-feature.md`

```markdown
# New Feature Workflow

Chạy full workflow cho feature mới. Tự động đi qua 4 phases.

## Input: $ARGUMENTS (mô tả feature)

## Phase 1: Plan
1. Gọi `ba-analyst` subagent phân tích yêu cầu: "$ARGUMENTS"
2. Gọi `architect` subagent thiết kế technical solution
3. Output specs vào /docs/

## Phase 2: Work
4. Tạo agent team với:
   - backend-dev: implement server-side theo specs
   - frontend-dev: implement client-side theo specs
   - Hai agents làm song song, coordinate qua API contract trong specs

## Phase 3: Review
5. Gọi `qa-engineer` subagent review + viết tests
6. Gọi `security-auditor` subagent security review
7. Fix issues nếu có

## Phase 4: Compound
8. Ghi lại learnings: patterns mới, decisions, gotchas
9. Cập nhật CLAUDE.md nếu cần
```

### `/commit-push-pr` — `.claude/commands/commit-push-pr.md`

```markdown
Tạo commit, push, và mở PR.

1. Chạy: `git status` và `git diff --stat`
2. Viết commit message theo Conventional Commits
3. Push to remote
4. Tạo PR description tóm tắt changes, linked issues, và test plan
```

---

## Phần 7: Path-Scoped Rules (Tự động áp dụng)

### `.claude/rules/backend.md`

```markdown
---
globs: src/server/**
---

Khi edit backend code:
- Dùng NestJS decorators đúng cách
- Mọi endpoint phải có DTO validation (Zod)
- Service layer xử lý business logic, controller chỉ routing
- Repository pattern cho database access
- Tất cả errors phải throw NestJS HttpException
- Log với structured format: { action, userId, resource, result }
```

### `.claude/rules/testing.md`

```markdown
---
globs: tests/**
---

Khi viết tests:
- Dùng describe/it blocks có ý nghĩa
- Test behavior, không test implementation
- MSW cho API mocking (không mock fetch trực tiếp)
- Factory functions cho test data (không hardcode)
- Mỗi test case phải independent (no shared state)
- Arrange-Act-Assert pattern
```

---

## Phần 8: Workflow thực tế — Ví dụ End-to-End

### Scenario: Build feature "Quản lý đơn hàng"

```bash
# Mở project
cd /your-project
claude

# ========================================
# PHASE 1: PLAN (sử dụng compound engineering)
# ========================================

> /workflows:plan
# Hoặc thủ công:
> Tôi cần build feature quản lý đơn hàng. Dùng ba-analyst agent
  phân tích yêu cầu, sau đó dùng architect agent thiết kế solution.
  Yêu cầu: CRUD đơn hàng, filter/search, trạng thái (pending →
  confirmed → shipping → delivered → cancelled), thông báo email
  khi đổi trạng thái, dashboard thống kê.

# Claude sẽ:
# - Gọi ba-analyst: viết user stories → /docs/user-stories/
# - Gọi architect: viết technical design → /docs/architecture/
# - Tạo ADR cho state machine design → /docs/decisions/

# Review output, chỉnh sửa nếu cần
> Specs trông tốt. Thêm acceptance criteria cho edge case:
  đơn hàng đã cancelled không thể chuyển sang trạng thái khác.

# ========================================
# PHASE 2: WORK (Agent Teams — song song)
# ========================================

> Tạo agent team để implement feature đơn hàng theo specs trong /docs/.
  - Teammate "backend": implement Order model, service, controller
    trong /src/server/modules/orders/
  - Teammate "frontend": implement Order pages và components
    trong /src/client/features/orders/
  - Hai agents coordinate qua API contract đã define trong specs.
  Dùng Sonnet cho mỗi teammate để tiết kiệm tokens.

# Claude tự động:
# - Spawn 2 teammates chạy song song
# - Backend teammate: models → services → controllers → routes
# - Frontend teammate: components → pages → hooks → state
# - Teammates giao tiếp trực tiếp khi cần (vd: API types)

# ========================================
# PHASE 3: REVIEW
# ========================================

> /workflows:review
# Hoặc:
> Tạo agent team review code vừa implement:
  - Teammate "qa": review code + viết tests, focus vào
    acceptance criteria và edge cases
  - Teammate "security": audit authentication, input validation,
    SQL injection risks
  - Teammate "performance": check N+1 queries, caching strategy

# Review findings, fix issues
> Backend dev agent: fix các bugs QA report trong BUG-001 và BUG-002.

# ========================================
# PHASE 4: COMPOUND (Học từ feature này)
# ========================================

> /workflows:compound
# Claude tự động:
# - Phân tích những gì đã học từ feature này
# - Cập nhật CLAUDE.md với patterns mới
#   Ví dụ: "Order state machine: dùng finite-state-machine library,
#   validate transitions trong service layer, not controller"
# - Cập nhật rules nếu cần

# ========================================
# DEPLOY
# ========================================

> Dùng devops-engineer agent review infra changes và
  chuẩn bị deployment cho feature đơn hàng.

> /commit-push-pr
```

---

## Phần 9: Git Worktrees cho Parallel Agents

Khi chạy nhiều agents đồng thời, dùng git worktrees tránh conflict:

```bash
# Tạo worktree cho mỗi agent
wt new feat-orders-backend    # Backend agent làm ở đây
wt new feat-orders-frontend   # Frontend agent làm ở đây

# Liệt kê worktrees
wt list

# Mỗi agent session chạy trong worktree riêng
# Terminal 1:
cd ../your-project-feat-orders-backend
claude  # Backend agent session

# Terminal 2:
cd ../your-project-feat-orders-frontend
claude  # Frontend agent session

# Merge khi hoàn thành
git checkout main
git merge feat-orders-backend
git merge feat-orders-frontend

# Dọn dẹp
wt remove feat-orders-backend
wt remove feat-orders-frontend
```

---

## Phần 10: Mẹo Pro từ Production Teams

### 1. Compound Learning là chìa khóa
Sau MỖI PR, chạy `/workflows:compound` để hệ thống tự học.
CLAUDE.md sẽ tích lũy knowledge theo thời gian → agents ngày càng giỏi hơn.

### 2. Plan mode trước, code sau
Dùng Plan mode (Shift+Tab trong Claude Code) để bàn bạc trước khi code.
Boris Cherny (người tạo Claude Code): "80% compound engineering nằm
ở plan và review, chỉ 20% ở phần code."

### 3. Feature-specific subagents > Generic agents
Thay vì agent "qa-engineer" chung chung, tạo agent cụ thể:
"order-qa-tester" biết rõ domain đơn hàng → output tốt hơn.

### 4. Model selection strategy
- Opus: BA analysis, architecture, QA review (cần deep reasoning)
- Sonnet: Implementation, routine tasks (nhanh + rẻ hơn)
- Tip: Agent Teams dùng ~3-5x tokens so với single session,
  nên dùng Sonnet cho teammates, Opus cho team lead

### 5. Token management
- CLAUDE.md ngắn gọn (< 3000 tokens)
- Skills dùng progressive disclosure (chỉ load khi cần)
- Subagents có context riêng → không phí tokens cho context không liên quan

### 6. Safety
- KHÔNG dùng --dangerously-skip-permissions cho production code
- Chạy agents trong sandbox/worktree riêng
- Luôn review output trước khi merge

### 7. Metrics để đo hiệu quả
- Time-to-ship: từ specs → production
- Bug escape rate: bugs tìm thấy sau deploy
- PR cycle time: từ open → merge
- Test coverage: maintain >= 80%
- Compound rate: bao nhiêu learnings mới mỗi sprint

---

## Tóm tắt: Tại sao cách này là tốt nhất

| Yếu tố | Giải thích |
|---------|-----------|
| **Native** | Agent Teams là tính năng chính thức của Anthropic, không phải tool bên thứ 3 |
| **Self-improving** | Compound Engineering đảm bảo hệ thống TỐT HƠN mỗi ngày |
| **Song song** | Nhiều agents chạy đồng thời, giao tiếp trực tiếp |
| **Chuyên môn hóa** | Mỗi agent có context, tools, model riêng cho vai trò |
| **Production-proven** | Đã được kiểm chứng bởi incident.io, Every, Nx, Anthropic |
| **Extensible** | Plugin marketplace, custom skills, MCP servers |
| **Cost-efficient** | Progressive disclosure + model selection tối ưu chi phí |
