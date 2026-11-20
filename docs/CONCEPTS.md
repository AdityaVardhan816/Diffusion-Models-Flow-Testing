# Concepts: Diffusion and Flow Matching

This guide provides the minimum shared vocabulary for implementing and discussing the comparison.

## Diffusion models

In a common diffusion formulation, a clean data sample is progressively corrupted with Gaussian noise according to a schedule. A neural network is trained to estimate a quantity such as the added noise, the clean sample, or a score. Generation starts from noise and uses a reverse-time process or numerical sampler to produce a sample.

The sequence of denoising evaluations can make sampling costly. The actual cost depends on the model, sampler, hardware, and chosen number of function evaluations; it is not fixed at dozens or hundreds for every setup.

## Flow matching

Flow matching trains a time-dependent vector field that transports samples along a chosen probability path from a simple base distribution (often Gaussian noise) to the data distribution. The learned field is used in an ordinary differential equation (ODE) to move a sample from the base distribution toward generated data.

The training objective depends on the selected path and velocity target. A straight interpolation between noise and data is one useful choice, but flow matching is a broader framework and does not require every trajectory to be straight.

## Rectified flow

Rectified flow is a related approach that aims to make transport paths straighter, which can make coarse numerical integration more effective. A straighter path may help reduce the number of solver evaluations, but it does not guarantee a particular speed or image quality. Those are empirical outcomes to measure.

## Important comparison distinction

The training method and the sampling algorithm are separate experimental choices. A measured speed difference may come from the objective, the chosen sampler or ODE solver, the number of function evaluations, implementation details, or hardware. Report these choices explicitly.

For a fair study, compare quality as a function of sampling cost (for example, quality at several function-evaluation budgets), not only at one arbitrarily chosen step count.

## Large model examples

Stable Diffusion 3 uses a rectified-flow formulation, and FLUX models are associated with flow-matching-style training. These are useful examples of modern generative modeling, but their full training and infrastructure are outside this project's proposed scope. The project should build a small controlled experiment and should not imply that its implementation reproduces those systems.
