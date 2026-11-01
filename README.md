# Fruit Fly Playing League of Legends

**FlyQ  Connectome-Based Bio-Inspired Sensorimotor Control**

*An independent experimental project, not affiliated with or endorsed by Riot Games.*

FlyQ is an experimental bio-inspired AI project that uses the
**Google/Janelia MaleCNS v1.0 adult male fruit fly connectome** to
investigate visual sensorimotor control in controlled simulated
environments.

The project processes visual input, drives a connectome-based neural
simulation, and decodes selected neural activity into movement
decisions.

> **Research prototype:** FlyQ is an engineered connectome simulation,
> not a living fruit fly brain or a complete biological model. Its
> visual and motor mappings are engineering approximations.

## Overview

FlyQ connects three main components:

1.  **Visual input:** RGB frames are converted into luminance samples
    mapped to retained R1--R6 photoreceptor inputs. This is not object
    recognition.
2.  **Neural processing:** The inputs drive the MaleCNS connectome
    through the [DoomFly](https://github.com/nftechie/doomfly)
    simulation infrastructure.
3.  **Motor decoding:** Selected neural activity is processed by
    engineered decoders to produce movement decisions.

### Neural readouts

  Function              Neural readout
  --------------------- ---------------------------------------
  Turning               DNp20 left--right activity difference
  Movement              Bilateral DNpe017 activity
  Experimental attack   MN9 onset through a latch decoder

These are engineered mappings, not proof that the biological neurons
naturally perform the corresponding game actions. In particular, MN9 is
mouthpart-related; it is not a natural attack neuron.

## Current scientific scope

-   Uses a fixed connectome.
-   Synaptic weights are not modified during experiments.
-   No learning or dopamine/reward-gated plasticity is currently
    implemented.
-   Thresholds, decoders, and state machines are engineering components.
-   Simulated task performance does not establish biological equivalence
    or natural fly behavior.

Reward-gated plasticity is a possible future research direction, not a
current feature.

## Project structure

  -----------------------------------------------------------------------
  Directory                           Purpose
  ----------------------------------- -----------------------------------
  `neural/`                           Connectome runtime, neural
                                      readouts, and client utilities

  `control/`                          Movement and attack decoders,
                                      calibration, and live-control
                                      utilities

  `experiments/`                      Visual, temporal, navigation, and
                                      sensorimotor experiments

  `telemetry/`                        CSV experiment outputs and recorded
                                      measurements

  `tests/`                            Automated tests
  -----------------------------------------------------------------------

## Current experimental results

The following results summarize the current experimental milestone,
**35A-V2c**.

  Experiment                              Result
  ------------------------------------- --------
  Sequential target navigation             10/10
  Perspective-scaled target sequences      10/10
  Moving targets                             6/6
  Occluded moving targets                    5/6
  Active visual search                       6/6
  Multi-object selection                     6/6

**Important qualifications:**

-   Perspective-scaled target performance does not establish depth
    perception.
-   Active visual search uses an engineered search strategy.
-   Multi-object selection uses explicit luminance-salience
    preprocessing; it is not evidence of biological attention.
-   These results come from controlled experimental environments and
    should not be interpreted as general real-world performance.

## Requirements

-   Linux or WSL Ubuntu
-   Python 3.11
-   A separate DoomFly checkout
-   The MaleCNS v1.0 source data and generated model files required by
    DoomFly

FlyQ expects the following directory layout:

``` text
~/FlyQ/
├── flyq/
└── upstream-doomfly/
```

The path matters: `neural/runtime.py` expects DoomFly at
`~/FlyQ/upstream-doomfly`.

## Installation

### 1. Clone the repositories

Replace `YOUR_GITHUB_USERNAME` with your GitHub username.

``` bash
mkdir -p ~/FlyQ
cd ~/FlyQ
git clone https://github.com/YOUR_GITHUB_USERNAME/FlyQ.git flyq
git clone https://github.com/nftechie/doomfly.git upstream-doomfly
```

### 2. Install the DoomFly neural environment

``` bash
cd ~/FlyQ/upstream-doomfly
python3.11 -m venv .venv-neural
source .venv-neural/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-neural.txt -r doom/requirements.txt --build-constraint neural-build-constraints.txt
```

### 3. Prepare the MaleCNS model

Follow the [DoomFly documentation](https://github.com/nftechie/doomfly)
to download the required MaleCNS v1.0 source files into
`connectome_data/malecns_v1/` and verify them against the supplied
provenance checksums.

Then, from the DoomFly repository root, build and audit the model:

``` bash
python -m doom.connectome malecns_v1
python -m doom.prepare
python -m doom.audit_data
python -m doom.build_kernel
```

The generated model is expected at:

``` text
outputs/doom/malecns_v1/
├── graph.npz
└── manifest.json
```

The generated graph is a large external model artifact and is not
included in the FlyQ repository.

## Running FlyQ

Activate or use the DoomFly Python environment. From the FlyQ repository
root:

``` bash
cd ~/FlyQ/flyq
```

Run an experiment using Python's module syntax:

``` bash
~/FlyQ/upstream-doomfly/.venv-neural/bin/python -m experiments.sequential_targets_2d
```

Other available experiments include:

``` bash
~/FlyQ/upstream-doomfly/.venv-neural/bin/python -m experiments.sequential_depth_targets_2d
~/FlyQ/upstream-doomfly/.venv-neural/bin/python -m experiments.moving_targets_2d
~/FlyQ/upstream-doomfly/.venv-neural/bin/python -m experiments.occluded_moving_targets_2d
~/FlyQ/upstream-doomfly/.venv-neural/bin/python -m experiments.active_visual_search_2d
~/FlyQ/upstream-doomfly/.venv-neural/bin/python -m experiments.multi_object_selection_2d_v2
```

Experiment outputs are written to CSV files under `telemetry/`. Running
an experiment again may overwrite its existing output file.

## Tests

Run the automated test suite from the FlyQ repository root:

``` bash
~/FlyQ/upstream-doomfly/.venv-neural/bin/python -m pytest -q
```

## Live neural utilities

The neural client and live direction decoder communicate with a
compatible neural service at:

``` text
127.0.0.1:18765
```

These client utilities do not start the neural service themselves. A
compatible DoomFly service must already be running.

## Data and references

-   [MaleCNS Connectome Portal ---
    Janelia](https://male-cns.janelia.org/)
-   [Google Research --- A connectomics milestone: mapping the complete
    male fruit fly
    brain](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)
-   [DoomFly --- Neural simulation
    infrastructure](https://github.com/nftechie/doomfly)

## Project status

FlyQ is an ongoing experimental research project. Current work focuses
on connectome-based visual sensorimotor behavior, neural readout
analysis, and controlled navigation experiments.

Future directions may include more robust motor decoding, improved
experimental evaluation, and investigation of reward-gated plasticity.
