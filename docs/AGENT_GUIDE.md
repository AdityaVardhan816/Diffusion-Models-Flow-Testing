# Agent Guide

## Mission

Help build and document a small, reproducible comparison of diffusion and flow matching for image generation. Keep the implementation aligned with the research question in [the project brief](PROJECT_BRIEF.md).

## Before coding

1. Inspect the existing workspace and follow any repository-specific instructions.
2. Identify available hardware and dependencies before choosing model size or dataset.
3. Check the project brief and implementation plan for unsettled decisions.
4. If a decision materially affects feasibility or fairness, explain the options and choose a conservative default when possible.

## Working principles

- Keep shared data and model components shared when practical.
- Make experiment settings explicit and save them with every run.
- Use separate, clearly named configs for method-specific settings.
- Keep training and sampling budgets configurable.
- Preserve logs and generated samples needed to reproduce reported results.
- Explain which settings are matched and which differ.
- Do not claim flow matching inherently produces sharper images or always samples faster.
- Distinguish the training objective from the sampler or ODE solver.
- Mark unverified results as unverified; never invent metrics or claim runs occurred unless logs support that claim.
- Keep large production models as context, not as the implementation target.

## Suggested structure if code is added

Adapt this to the existing repository rather than imposing it if a structure already exists:

```text
configs/
src/
  data/
  models/
  methods/
  sampling/
  evaluation/
scripts/
results/
docs/
```

Keep generated checkpoints and large datasets out of source control unless the repository explicitly intends to store them.

## Reporting changes

When finishing a task, summarize what changed, how it relates to the project question, and any key assumptions or unverified items. Do not report an experiment as complete until its run artifacts and evaluation outputs exist.
