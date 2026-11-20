# Evaluation Guide

The purpose of evaluation is to compare methods under stated, reproducible conditions. No single metric captures image quality or computational cost completely.

## Experimental controls

Record and align these where possible:

- dataset, split, preprocessing, and resolution;
- backbone and parameter count;
- training examples, epochs or updates, batch size, optimizer, and learning-rate schedule;
- hardware, software versions, precision, and wall-clock training time;
- random seed and checkpoint selection rule;
- sampler or ODE solver, tolerances, and model function evaluations;
- number of generated samples and metric implementation.

If a setting cannot be matched, state the difference and its likely effect.

## Training behavior

Track the training objective over updates, throughput, wall-clock time, and numerical failures such as non-finite losses or gradients. Loss values across different objectives are not directly comparable as measures of sample quality; use them to describe each method's optimization behavior.

## Sample quality

Use an established dataset-level metric such as FID when the dataset and evaluation setup support it. State the feature extractor, sample count, preprocessing, and implementation. For small datasets or small sample counts, metric estimates can be noisy. Report that limitation and avoid overinterpreting tiny differences.

Include sample grids for qualitative inspection, but do not use a few hand-picked samples as the sole basis for a conclusion.

## Sampling speed and cost

For each method, measure generation at several function-evaluation budgets. Report:

- model function evaluations per sample;
- batch size and hardware;
- latency after warm-up, with the timing method stated;
- throughput for a stated batch size;
- quality metric at each budget.

An ODE solver step and a diffusion sampler step do not always represent identical work. Function evaluations and wall-clock time help make the comparison clearer.

## Recommended result views

1. Training objective versus update count (one panel per method or clearly labeled axes).
2. FID versus model function evaluations.
3. FID versus measured sampling latency.
4. Sample grids at matching evaluation budgets.

## Conclusions

Conclude only what the tested setup supports. A result on one small dataset, architecture, and compute budget does not establish that one framework is universally faster or higher quality.
