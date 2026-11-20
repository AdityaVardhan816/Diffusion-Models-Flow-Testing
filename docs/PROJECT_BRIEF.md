# Project Brief

## Working title

**Diffusion vs. Flow Matching: A Controlled Study of Image Generation**

## Summary

Build small image-generation models using a diffusion objective and a flow-matching objective, then compare their training behavior, sample quality, and generation cost under a clearly documented compute budget.

This is an educational, controlled experiment. It is not an attempt to reproduce large production systems such as Stable Diffusion 3 or FLUX from scratch.

## Research question

Under a comparable model architecture, dataset, and training budget, how do diffusion and flow matching compare in:

- optimization behavior and training stability;
- image quality at different sampling budgets;
- sampling latency and computational cost?

## Goals

1. Implement or adapt one compact diffusion model and one compact flow-matching model.
2. Keep shared experimental choices aligned where possible: data preprocessing, model capacity, training examples, hardware, and evaluation procedure.
3. Measure quality against sampling compute, rather than reporting a single quality or speed number.
4. Save enough configuration, logs, and sample outputs for another person to reproduce the comparison.
5. Explain limitations and differences in the objectives, rather than presenting the models as perfectly interchangeable.

## Out of scope

- Training a foundation-scale text-to-image model from scratch.
- Claiming flow matching is always faster, produces sharper images, or needs fewer steps.
- Treating one sampler or solver's performance as an inherent property of the training objective.
- Using visual examples alone as proof that one method is better.

## Proposed starting scope

Start with an image-only dataset small enough for the available hardware. CIFAR-10 is a practical initial candidate; a smaller curated subset may be used if compute is limited. Confirm the final dataset and resolution before implementation. Prefer a shared small U-Net or another compact architecture to reduce confounding from model differences.

## Expected outputs

- Reproducible training and sampling code.
- Configurations for both methods.
- Training and evaluation logs.
- Sample grids at selected checkpoints and sampling budgets.
- A short report that states the setup, results, limitations, and conclusions supported by the measurements.

## Decisions to make before the experiment

- Dataset and image resolution.
- Framework and hardware.
- Compute or time budget.
- Model architecture and parameter-count target.
- Diffusion noise schedule and sampler.
- Flow-matching probability path, target velocity, and ODE solver.
- Evaluation implementation and number of generated samples.

Record each choice in the experiment configuration and report. Do not silently change settings between methods.
