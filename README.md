# Diffusion vs. Flow Matching: Controlled Image-Generation Study

This project implements and compares small diffusion and flow-matching image-generation models. It includes matched training code, 1/2-Rectified Flow reflow training, sampling benchmarks, sanity tests, an interactive presentation, and a live synthesis server.

The project is an educational, controlled experiment. It is not an attempt to reproduce foundation-scale systems such as Stable Diffusion 3 or FLUX.

## Project overview

### Working title

**Diffusion vs. Flow Matching: A Controlled Study of Image Generation**

### Summary

Build compact image-generation models using a diffusion objective and a flow-matching objective, then compare:

- optimization behavior and training stability;
- image quality at different sampling budgets;
- sampling latency and computational cost.

The implementation uses MNIST resized to 32 x 32, a shared CompactUNet, matched training settings, and separately named training and evaluation artifacts.

### Research question

Under a comparable model architecture, dataset, and training budget, how do diffusion and flow matching compare in optimization behavior, image quality, and generation cost?

### Goals

1. Implement one compact diffusion model and one compact flow-matching model.
2. Keep data preprocessing, model capacity, training examples, hardware, and evaluation procedures aligned where possible.
3. Measure quality against sampling compute instead of reporting a single isolated quality or speed number.
4. Save configurations, logs, checkpoints, and sample outputs needed for reproduction.
5. Explain limitations and distinctions between objectives, samplers, and solvers.

### Scope and limitations

The project does not train a foundation-scale text-to-image model. It does not prove that either framework is universally better. A result on one small dataset and architecture does not establish general image-quality or speed conclusions.

Do not claim that flow matching always produces sharper images or always requires fewer steps. The measured cost is affected by the objective, sampler or ODE solver, implementation, batch size, hardware, and number of function evaluations.

## Concepts

### Diffusion models

A common diffusion formulation progressively corrupts clean data with Gaussian noise according to a schedule. A neural network learns a quantity such as the added noise, the clean sample, or a score.

Generation begins from Gaussian noise and uses a reverse-time process or numerical sampler. The cost depends on the model, sampler, and chosen number of function evaluations.

This implementation:

- uses a 1000-step noise schedule;
- trains noise prediction on a CompactUNet;
- uses DDIM sampling for configurable evaluation budgets.

### Flow matching

Flow matching trains a time-dependent vector field that transports samples from a simple base distribution, usually Gaussian noise, toward the data distribution. The learned field is represented as an ODE and integrated with a numerical solver.

The training objective depends on the selected probability path and velocity target. Straight interpolation between noise and data is used in this project, but straight paths are not a requirement of flow matching generally.

This implementation:

- uses straight interpolation with `sigma_min = 0.0001`;
- trains a time-conditioned CompactUNet;
- uses Euler integration for the baseline;
- reports vector-field evaluations as model function evaluations.

### Rectified flow and reflow

Rectified flow aims to straighten probability-transport paths so numerical integration can become effective with fewer steps. A straighter path can improve coarse sampling, but it does not guarantee a particular quality or latency result.

This project includes 2-Rectified Flow training:

1. generate coupled noise/data pairs from the baseline 1-Rectified Flow model;
2. retrain the velocity field on those paired endpoints;
3. support 1-step Euler generation.

### Important experimental distinction

Training objective and sampling algorithm are separate choices. A measured difference may originate from the objective, sampler, ODE solver, step count, implementation, or hardware. Quality-versus-compute comparisons are more informative than a result from one arbitrary step count.

## Repository structure

```text
app.py                  Live presentation and synthesis server
index.html              Interactive presentation and laboratory
train.py                Diffusion and baseline flow-matching training
train_reflow.py         2-Rectified Flow training
evaluate.py             Cross-method NFE and latency evaluation
run_experiments.py      Convenience experiment runner
test_sanity.py          Shape, loss, gradient, and sampling sanity checks
requirements.txt        Python dependencies

configs/
  diffusion.json        Diffusion experiment settings
  flow_matching.json    Flow-matching experiment settings

src/
  data.py               Shared MNIST/CIFAR-10/Fashion-MNIST loader
  model.py              CompactUNet backbone
  diffusion.py          Gaussian diffusion and DDIM sampler
  flow_matching.py      Flow-matching objective and ODE samplers
  reflow.py             Rectified-flow/reflow implementation
  evaluate.py           Grid generation and evaluation utilities

docs/
  PROJECT_BRIEF.md      Original project scope
  CONCEPTS.md           Method concepts and terminology
  IMPLEMENTATION_PLAN.md Staged implementation plan
  EVALUATION.md         Fair-evaluation procedure
  AGENT_GUIDE.md        Repository change and reporting conventions

results/
  diffusion_baseline/   Diffusion checkpoints, history, and samples
  flow_matching_baseline/ Flow-matching checkpoints, history, and samples
  reflow_baseline/      Reflow checkpoint and coupled pairs
  eval/                 NFE grids, comparisons, and latency summaries
```

Supporting documents are also preserved in [docs/](docs/).

## Implemented architecture

The shared backbone is `CompactUNet` with one input and output channel for MNIST:

- base channels: 32;
- channel multipliers: (1, 2, 4);
- time embedding dimension: 128;
- image resolution: 32 x 32.

The shared data pipeline normalizes data to [-1, 1]. MNIST is resized to 32 x 32. The configured runs use the same seed, batch size, optimizer learning rate, weight decay, and epoch count where practical.

## Experiment configuration

Both baseline configurations use:

| Setting | Diffusion | Flow matching |
|---|---:|---:|
| Dataset | MNIST | MNIST |
| Seed | 42 | 42 |
| Batch size | 128 | 128 |
| Epochs | 15 | 15 |
| Learning rate | 0.0003 | 0.0003 |
| Weight decay | 0.0001 | 0.0001 |
| Base channels | 32 | 32 |
| Channel multipliers | (1, 2, 4) | (1, 2, 4) |
| Time dimension | 128 | 128 |

Diffusion additionally uses 1000 timesteps with beta values from 0.0001 to 0.02. Flow matching uses a minimum path noise of 0.0001.

## Implementation plan

### 1. Environment and constraints

- identify the framework, accelerator, memory, and runtime constraints;
- confirm that checkpoints and dependencies are compatible with the active Python environment;
- choose a dataset and resolution that fit the available hardware;
- document environment and initial experiment choices.

### 2. Shared components

- use one data pipeline for all methods;
- use shared normalization and validation behavior;
- match backbone capacity where practical;
- configure seeds, batch size, optimizer, and training budget;
- store checkpoints, configuration, logs, and samples in separate run directories.

### 3. Diffusion baseline

- define the forward noising process and noise schedule;
- train noise prediction with an explicit objective;
- implement reverse/DDIM sampling;
- make model function evaluations configurable.

### 4. Flow-matching baseline

- specify the base distribution and probability path;
- derive and document the target velocity;
- train a time-conditioned vector field;
- integrate the learned ODE with Euler or Heun sampling;
- make the number of vector-field evaluations configurable.

### 5. Independent validation

- verify tensor shapes and [-1, 1] data ranges;
- confirm losses stay finite and gradients update parameters;
- overfit a tiny subset before launching full runs;
- generate and visually inspect small sample batches.

### 6. Controlled comparison

- match training examples or training compute as closely as practical;
- save exact configuration and software environment;
- generate samples at several budgets;
- repeat seeds only when resources permit and state the limitation otherwise;
- avoid tuning only one method more extensively without reporting it.

### 7. Results

- plot training objective versus update count;
- plot quality versus model function evaluations;
- plot quality or sample inspection against measured latency;
- include labeled sample grids with method, seed, and sampling budget.

## Installation

Use the Python interpreter that will execute `app.py`, and install packages through that interpreter:

```powershell
python3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project dependencies are:

- PyTorch
- torchvision
- NumPy
- Matplotlib
- SciPy

## Running sanity checks

```powershell
python test_sanity.py
```

The sanity test verifies the CompactUNet shape, finite diffusion and flow-matching losses, DDIM and Euler/Heun sampling, and 1-step reflow sampling.

## Training

Train the baseline models from their JSON configurations:

```powershell
python train.py --config configs/diffusion.json
python train.py --config configs/flow_matching.json
```

Optional sanity-check mode:

```powershell
python train.py --config configs/diffusion.json --sanity-check
```

Device selection is supported through `--device`:

```powershell
python train.py --config configs/diffusion.json --device cuda
python train.py --config configs/flow_matching.json --device cpu
```

Train 2-Rectified Flow after the baseline flow-matching checkpoint exists:

```powershell
python train_reflow.py
```

Useful options include `--num-pairs`, `--epochs`, `--batch-size`, `--lr`, `--force-pairs`, and `--device`.

## Evaluation

Evaluate a diffusion or flow-matching checkpoint across configurable sampling budgets:

```powershell
python evaluate.py ^
  --checkpoint results/diffusion_baseline/checkpoint_latest.pt ^
  --output-dir results/eval ^
  --steps 5 10 20 50 ^
  --batch-size 16

python evaluate.py ^
  --checkpoint results/flow_matching_baseline/checkpoint_latest.pt ^
  --output-dir results/eval ^
  --steps 5 10 20 50 ^
  --batch-size 16
```

Evaluation records solver type, requested NFE, generated sample grids, mean latency, latency standard deviation, and throughput.

## Interactive presentation server

Start the server with the same Python environment used for installation:

```powershell
python3 app.py --port 8000
```

Open:

```text
http://localhost:8000/
```

The live synthesis API is:

```text
/api/status
/api/generate?model=2rf&steps=1&seed=42&batch_size=16
```

Supported live models are:

- `1rf` - baseline flow matching;
- `2rf` - reflow model;
- `diffusion` - diffusion/DDIM baseline.

## Evaluation procedure

### Experimental controls

Record and align:

- dataset, split, preprocessing, and resolution;
- backbone and parameter count;
- training examples, epochs or updates, batch size, optimizer, learning-rate schedule;
- hardware, software versions, precision, and wall-clock training time;
- random seed and checkpoint selection rule;
- sampler or ODE solver, tolerance, and model function evaluations;
- number of generated samples and metric implementation.

If a setting cannot be matched, state the difference and its likely effect.

### Training behavior

Track:

- training objective versus updates;
- throughput;
- wall-clock time;
- non-finite losses or gradients.

Loss values from diffusion and flow matching are different objectives. They should not be compared as if they were the same physical measure of image quality.

### Sample quality

Use FID or another established dataset-level metric when the setup supports it. Report the feature extractor, sample count, preprocessing, and implementation. For small datasets or sample counts, metric estimates can be noisy.

Include sample grids for qualitative inspection, but do not use selected images alone as the basis for a conclusion.

### Sampling speed and cost

Measure each method at several function-evaluation budgets and report:

- model function evaluations per sample;
- batch size and hardware;
- latency after warm-up;
- throughput for a stated batch size;
- quality at each budget.

An ODE solver step and a diffusion sampler step do not always represent identical work. Function evaluations and wall-clock timing should both be reported.

## Current result artifacts

The repository contains generated checkpoints and evaluation outputs for:

- diffusion baseline;
- flow-matching baseline;
- 2-Rectified Flow;
- 1-step, 2-step, 4-step, and 8-step rectified-flow grids;
- diffusion and flow-matching grids at 5, 10, 20, and 50 NFE;
- latency summaries and loss curves.

The measured 16-sample CPU benchmark summaries show:

| Method | 5 NFE | 10 NFE | 20 NFE | 50 NFE |
|---|---:|---:|---:|---:|
| DDIM diffusion | 0.203 s | 0.402 s | 0.717 s | 2.116 s |
| Euler flow matching | 0.112 s | 0.212 s | 0.440 s | 1.058 s |

These measurements describe the saved run and local machine. They are not universal speed claims.

Key result files include:

- [results/eval/diffusion_eval_summary.json](results/eval/diffusion_eval_summary.json)
- [results/eval/flow_matching_eval_summary.json](results/eval/flow_matching_eval_summary.json)
- [results/eval/comparative_grid.png](results/eval/comparative_grid.png)
- [results/eval/loss_curves.png](results/eval/loss_curves.png)

## Repository and agent conventions

Before making project changes:

1. inspect existing workspace structure and repository instructions;
2. identify hardware and dependencies;
3. read the project brief and implementation plan;
4. document assumptions and material decisions;
5. choose a conservative default when a decision affects feasibility or fairness.

Working principles:

- keep shared data and model components shared;
- make experiment settings explicit;
- use separate, clearly named configurations;
- keep training and sampling budgets configurable;
- preserve logs and generated samples needed for reproduction;
- distinguish objective, sampler, and solver;
- mark unverified results clearly;
- never invent metrics or claim that a run occurred without artifacts;
- keep production-scale systems as context rather than implementation targets.

## Documentation index

- [Project brief](docs/PROJECT_BRIEF.md)
- [Concepts](docs/CONCEPTS.md)
- [Implementation plan](docs/IMPLEMENTATION_PLAN.md)
- [Evaluation guide](docs/EVALUATION.md)
- [Agent guide](docs/AGENT_GUIDE.md)
