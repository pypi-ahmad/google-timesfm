# TimesFM — Agent Entry Point

This repository ships a first-party **Agent Skill** for TimesFM at:

```
timesfm-forecasting/
└── SKILL.md    ← read this for the full skill
```

## Install the skill

Copy the skill directory into your agent's skills folder:

```bash
# Cursor / Claude Code / OpenCode / Codex (global install)
cp -r timesfm-forecasting/ ~/.cursor/skills/
cp -r timesfm-forecasting/ ~/.claude/skills/

# Or project-level
cp -r timesfm-forecasting/ .cursor/skills/
```

Any agent that supports the open [Agent Skills standard](https://agentskills.io) will discover it automatically.

## Working in this repo

If you are developing TimesFM itself (not using it), the source lives in `src/timesfm/`.
Historical v1/v2 code is available from earlier Git revisions and releases.

See `README.md` for full developer setup.

<!-- okf:start -->
## Open Knowledge Format v0.2

Canonical governed project knowledge lives in `knowledge/index.md`.

- Read the index before architecture, policy, runbook, or domain work.
- Load only concepts relevant to the current task.
- Warn before relying on draft, deprecated, stale, or unverified concepts.
- Native instructions govern behavior; current source and tests govern factual conflicts.
<!-- okf:end -->
