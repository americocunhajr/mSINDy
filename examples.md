---
layout: default
title: "mSINDy MATLAB Examples and Methods"
description: "Open-source MATLAB software for discovering nonlinear differential equations from data using SINDy, sparse regression, and interpretable candidate libraries."
permalink: /examples/
---

# mSINDy: examples and computational methods

mSINDy is a modular MATLAB research and teaching framework for identifying interpretable governing differential equations from trajectory data using sparse regression and nonlinear candidate functions.

[Project homepage](https://sindycode.org/) · [MATLAB source repository](https://github.com/americocunhajr/mSINDy/tree/main/mSINDy-1.0)

## Available workflows

### Core SINDy routine

[**mSINDy.m**](https://github.com/americocunhajr/mSINDy/tree/main/mSINDy-1.0/mSINDy.m) — Sparse model identification, candidate libraries, and regression algorithms.

### Harmonic oscillator

[**Example01_HarmonicOscillator.m**](https://github.com/americocunhajr/mSINDy/tree/main/mSINDy-1.0/Example01_HarmonicOscillator.m) — Baseline linear oscillator for recovering governing equations from time-series data.

### Duffing oscillator

[**Example02_DuffingOscillator.m**](https://github.com/americocunhajr/mSINDy/tree/main/mSINDy-1.0/Example02_DuffingOscillator.m) — Nonlinear oscillator with polynomial restoring-force terms.

### van der Pol oscillator

[**Example03_VanDerPolOscillator.m**](https://github.com/americocunhajr/mSINDy/tree/main/mSINDy-1.0/Example03_VanDerPolOscillator.m) — Self-excited nonlinear oscillator with nonlinear damping.

### Rössler chaotic system

[**Example04_RosslerSystem.m**](https://github.com/americocunhajr/mSINDy/tree/main/mSINDy-1.0/Example04_RosslerSystem.m) — Data-driven identification for a nonlinear chaotic dynamical system.

## Getting started

Clone the [mSINDy repository](https://github.com/americocunhajr/mSINDy/), open `mSINDy-1.0/` in MATLAB, and select the relevant example or workflow. Consult the source files for model assumptions, required input data, and numerical parameters. This page describes existing source code; it does not make claims about benchmark accuracy or results not established by the repository.

## Frequently asked questions

### What is SINDy?

Sparse Identification of Nonlinear Dynamics: an approach that identifies a parsimonious dynamical model from a library of candidate terms using sparse regression.

### Which language does mSINDy use?

The available code and tutorials are implemented in MATLAB.

### Does the repository include nonlinear and chaotic benchmarks?

Yes. It includes Duffing, van der Pol, and Rössler examples in addition to a harmonic oscillator.

## Research citation

The repository [README](https://github.com/americocunhajr/mSINDy/blob/main/README.md) includes the associated 2026 book-chapter citation and BibTeX record: *Data-driven Evolution Equations via Sparse Identification of Nonlinear Dynamics*.
