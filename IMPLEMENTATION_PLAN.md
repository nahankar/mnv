# AI Story-to-Video Pipeline – Final Tech Stack and Architecture

## Final Recommendation

- **Core language**: Python (AI/ML, media processing) + TypeScript (Next.js dashboard)
- **Frontend**: Next.js (admin dashboard: model settings, runs, analytics)
- **Backend (AI services)**: Python FastAPI microservices
  - `story-service`: GPT/Claude/Mistral
  - `tts-service`: ElevenLabs/OpenAI/Azure
  - `music-service`: Suno/Mubert
  - `image-service`: DALL·E/SDXL
  - `video-service`: FFmpeg/MoviePy (optionally Runway/Pika/Veo for premium)
- **Orchestration**: Prefect (primary, workflows-as-code)
  - Optional: n8n as a satellite for social uploads/notifications if desired
- **Queue/async**: Prefect task runners; optional Redis/RQ for bursty tasks
- **Storage**: S3/GCS for media; Postgres for metadata; Redis for cache
- **Auth/Secrets**: Managed secrets (AWS Secrets Manager/GCP Secret Manager)
- **Observability**: Prometheus + Grafana, Sentry, structured logs
- **CI/CD & Packaging**: Docker + GitHub Actions
- **Cost/safety**: Budget circuit breakers, moderation, licensing evidence store

## Single Source of Truth – Model Settings
- Versioned config stored in Postgres (cached in Redis): defaults, allowed models, A/B splits, constraints
- Admin can override per run; pipeline enforces capability/cost/latency constraints

## Architecture (Mermaid)

```mermaid
flowchart LR
  subgraph Admin[Admin Dashboard (Next.js)]
    UI[Model Settings • Runs • Analytics]
  end

  subgraph Orchestrator[Prefect Flows]
    F1[Schedule/Trigger]
    F2[Resolve Model Config]
    F3[Fan-out Tasks]
    F4[Enforce Budget/SLAs]
    F5[Aggregate & Publish]
  end

  subgraph Services[Python FastAPI Microservices]
    S1[story-service\n(GPT/Claude/Mistral)]
    S2[tts-service\n(ElevenLabs/OpenAI/Azure)]
    S3[music-service\n(Suno/Mubert)]
    S4[image-service\n(DALL·E/SDXL)]
    S5[video-service\n(FFmpeg/MoviePy\n+ optional Runway/Pika/Veo)]
  end

  subgraph Storage[Data & Media]
    DB[(Postgres\nmetadata/config)]
    Cache[(Redis\ncache/rate limits)]
    Blob[(S3/GCS\nmedia artifacts)]
  end

  subgraph Distribution[Publishers]
    YT[YouTube API]
    IG[Instagram Graph]
    TT[TikTok Business]
    FB[Facebook Graph]
  end

  UI -->|CRUD model versions, run overrides| DB
  UI -->|start run| F1
  F1 --> F2 --> F3
  F2 --> DB
  F3 -->|calls| S1 & S2 & S3 & S4
  S1 --> Blob
  S4 --> Blob
  S2 --> Blob
  S3 --> Blob
  S5 --> Blob
  F3 -->|assemble request| S5
  F4 --> Cache
  F5 -->|upload| YT & IG & TT & FB
  F5 --> DB

  classDef box fill:#f7faff,stroke:#6aa9ff;
  classDef svc fill:#f2f7ff,stroke:#7a8;
  classDef store fill:#fff7e6,stroke:#e6a23c;
  class Admin box; Orchestrator box; Services svc; Storage store; Distribution box;
```

## Minimal Implementation Notes

- Ship MVP with: GPT-4o (story), ElevenLabs (TTS), Mubert (music), DALL·E (images), FFmpeg/MoviePy (video), YouTube uploads.
- Implement versioned Model Settings (defaults, allowed, constraints, A/B); validate capability matrix.
- Add budget guardrails and retries with idempotency; store license proofs for music/images.
- Prefer Prefect for orchestration; optionally use n8n only for social notifications or simple cross-app automations.

## Platform/Format Guardrails
- Long-form: 1920x1080, H.264/AAC; Shorts/Reels/TikTok: 1080x1920, ≤60s; subtitles optional.

## Security/Compliance
- Secrets in cloud secret manager; RBAC on admin; audit trails for model changes; content moderation before publish; COPPA/GDPR retention windows.
