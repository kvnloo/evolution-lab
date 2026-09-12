# Verify

Tests are necessary, not sufficient. Generation and verification are separate.

## Pyramid

1. **Unit** — `python -m unittest discover -s tests -p 'test_*.py'` on the touched surface.
2. **TDD** — red command, then green command (`skills/tdd/SKILL.md`).
3. **Mutation** — `n/a` in this tree. Do not invent a score.
4. **Runtime** — `python -m evolution_lab gym-smoke --n 8`. Hosted joules stay `unknown`.

## Receipt

Bind every result to `head_revision`. Tests from another SHA are not evidence. Fill `.github/PULL_REQUEST_TEMPLATE.md`.

The implementer does not self-approve. Independent review is a different person or a frozen evaluator. Workers never merge.

## Fail closed

- Unknown mutation tool → `n/a`, not `80`.
- Runtime you did not exercise → list it under `limitations`.
- Secrets, tokens, `.env` → stop.
