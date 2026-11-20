# Implementation Plan

This is a staged plan. Confirm project constraints before committing to a dataset, model size, or library. Keep the code small enough to inspect and rerun.

## 1. Inspect the environment

- Identify the available framework, accelerator, memory, and runtime constraints.
- Check whether a project repository or existing code exists before creating a new structure.
- Choose a dataset and resolution that fit the measured constraints.
- Document the environment and the initial experiment choices.

## 2. Establish shared components

- Build one data pipeline used by both methods.
- Use the same image normalization and train/validation split.
- Prefer one shared backbone or matched model capacities.
- Add configuration-driven random seeds, batch size, optimizer, and training budget.
- Save model checkpoints, configs, logs, and generated samples in separate run directories.

## 3. Implement diffusion baseline

- Define the forward noising process and noise schedule.
- Train the model with a clearly stated prediction target (for example, noise prediction).
- Implement a documented reverse sampler.
- Make the number of model function evaluations configurable.

## 4. Implement flow-matching baseline

- Specify the base distribution and probability path.
- Derive and document the velocity target used by the training objective.
- Train a time-conditioned vector-field model.
- Generate samples by integrating the learned ODE with a documented solver.
- Make the number of vector-field evaluations configurable.

## 5. Validate each method independently

- Check tensor shapes, time conditioning, and data ranges.
- Confirm the loss is finite and gradients update model parameters.
- Overfit a tiny subset as a basic implementation sanity check.
- Generate and inspect a small batch of samples before launching full runs.

## 6. Run the comparison

- Match the number of training examples or training compute as closely as practical.
- Save the exact configuration and software environment for each run.
- Generate samples at multiple evaluation budgets.
- Repeat runs with multiple random seeds if resources permit; otherwise state this limitation.
- Avoid tuning one method more extensively than the other without reporting it.

## 7. Summarize findings

- Plot training curves and quality-versus-compute results.
- Include representative samples, labeled with method, seed, and sampling budget.
- Separate measured results from interpretation.
- Discuss confounds, uncertainty, and what the experiment cannot establish.
