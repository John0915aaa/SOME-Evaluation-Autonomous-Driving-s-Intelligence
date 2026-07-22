# SOME: Evaluation Autonomous Driving's Intelligence (Lightweight Reproduction)

![Autonomous Driving Intelligence Evaluation Process](overview.png)

## Overview

This repository provides a lightweight reproduction of an autonomous-driving intelligence evaluation workflow. It evaluates vehicle interactions from five dimensions, combines ten normalized objective metrics with LLM-based subjective scores, and trains an attention-based neural network to approximate the final intelligence score.

The repository includes 705 indexed Waymo interaction samples and a lightweight Waymo/trajdata cache. It can therefore reproduce the main workflow without downloading the complete Waymo or InterHub dataset.

## Repository Structure

```text
project-root/
├── data/
│   ├── waymo_data index.csv
│   ├── dlmm_interaction_index.csv
│   └── waymo_lightweight_cache/
│       ├── waymo_0-299/
│       └── ...
│
├── 1-objective_metrics/
│   ├── metric.py
│   ├── safety_metrics.py
│   ├── comfort_metrics.py
│   ├── ...
│   ├── src/
│   └── output/
│
├── 2-subjective_evaluation/
│   ├── LLM-based_evaluate.py
│   ├── src/
│   │   ├── evaluation_guidelines.json
│   │   ├── get_prompt_cp1.py
│   │   └── ...
│   └── output/
│
├── 3-evaluation_model/
│   ├── train_evaluation_model.py
│   ├── src/
│   │   ├── config.py
│   │   └── get_training_data.py
│   └── output/
│
├── requirements.txt
└── README.md
```

## Main Components

### 1. Objective Metric Extraction

The `1-objective_metrics` module extracts ten normalized metrics from five evaluation dimensions.

| Dimension | Metric | Output column | Description |
| --- | --- | --- | --- |
| Safety | Time to Collision | `TTC` | Measures potential collision risk during an interaction. |
| Safety | Post-Encroachment Time | `PET` | Measures the temporal separation at a potential conflict area. |
| Comfort | Longitudinal acceleration | `a_p` | Evaluates longitudinal acceleration comfort. |
| Comfort | Lateral acceleration | `a_l` | Evaluates lateral acceleration comfort. |
| Comfort | Jerk | `jerk` | Uses the time derivative of the acceleration vector. |
| Comfort | Yaw rate | `yaw_rate` | Evaluates steering-related motion comfort. |
| Efficiency | Task time | `task_time` | Evaluates the efficiency of completing the interaction path. |
| Efficiency | Average delay | `avg_delay` | Compares the ego mean speed with surrounding-vehicle mean speeds. |
| Social interaction | Interaction Orientation | `IO` | Evaluates whether the AV behavior agrees with the annotated right of way. |
| Traffic impact | Background-vehicle impact | `impact` | Evaluates surrounding-vehicle speed reduction after AV entry. |

The recommended entry point is `metric.py`. It extracts the metrics listed above from the Waymo interaction dataset and standardizes them using the formulations defined in the accompanying paper, making the results suitable for subsequent evaluation-model training. For each indexed interaction, the script loads the trajectory context once and shares that context across all ten metric calculations. This avoids loading the same scene once per metric or evaluation dimension.

The main output is:

```text
1-objective_metrics/output/metrics.csv
```

The module also saves dimension-specific CSV files under the `safety`, `comfort`, `efficiency`, `interaction`, and `impact` output folders.

In addition to running the integrated `metric.py` workflow, each dimension-specific script can be run independently to obtain only the corresponding results: `safety_metrics.py`, `comfort_metrics.py`, `efficiency_metrics.py`, `interaction_metrics.py`, or `impact_metrics.py`.

### 2. LLM-Based Subjective Evaluation

The `2-subjective_evaluation` module integrates three operations into one script:

1. Generate a DLMM prompt from one interaction context.
2. Ask an LLM to evaluate the AV's intelligence.
3. Extract the final numerical score.

The LLM evaluates each interaction using structured guidelines. Safety is considered first, followed by comfort, efficiency, social interaction, and traffic-system impact. The response is requested in the form:

```text
Final overall score: x/10
```

Each interaction row is loaded once and processed in the following order:

```text
context --> prompt --> LLM evaluation --> score extraction --> CSV
```

The output is:

```text
2-subjective_evaluation/output/evaluation_score.csv
```

The script supports checkpoint saving and resuming. Existing indexes are skipped by default, so an interrupted evaluation can continue without evaluating completed interactions again.

The evaluation scores currently provided in the `output` folder were produced with a fine-tuned GPT-4o model. Each interaction was evaluated independently three times, and the mean of the three scores was used as the final result. Users without an OpenAI API key can directly use the provided results to train the evaluation model. Users with an OpenAI API key are still encouraged to perform three independent evaluations and use their mean to reduce the effect of response variability.

For a new application, it is practical to begin with a readily available general-purpose model, such as GPT-4o, and inspect several sample evaluations before investing in a higher-capability or fine-tuned model. OpenAI fine-tuning jobs can be managed through the [OpenAI Fine-tuning Dashboard](https://platform.openai.com/finetune), and the corresponding workflow is described in the [official model-optimization and fine-tuning documentation](https://developers.openai.com/api/docs/guides/model-optimization).

The LLM backend is replaceable. Developers may integrate another hosted API, such as Claude or DeepSeek, or a locally deployed language model. The recommended approach is to preserve the existing application-level contract—structured messages as input and evaluation text as output—while replacing the provider-specific client initialization and completion request in `src/llm_api_utils.py`. This keeps prompt generation, score extraction, checkpointing, and downstream model training unchanged.

### 3. Evaluation Model Training

The `3-evaluation_model` module trains an attention-based multilayer perceptron using the outputs of the first two stages.

It reads:

```text
1-objective_metrics/output/metrics.csv
2-subjective_evaluation/output/evaluation_score.csv
```

The ten normalized objective metrics are used as input features, and the normalized LLM score is used as the training label.

Before training, `src/get_training_data.py` checks that:

- both CSV files exist and are not empty;
- both files contain an `index` column;
- indexes are valid integers and contain no duplicates;
- both files contain exactly the same index set;
- all required features and labels are numeric.

The two files are then merged one-to-one by `index`. The model applies feature standardization, an attention layer, and a multilayer perceptron to learn the mapping from the ten objective metrics to the subjective intelligence score.

The main outputs are:

```text
3-evaluation_model/output/
├── evaModel_lightweight.pth
├── loss.png
└── attention_matrix.png
```

## Environment Setup

### Supported Operating Systems

The recommended Linux environments are:

- Ubuntu 20.04
- Ubuntu 22.04
- Ubuntu 24.04

Windows users are advised to use Ubuntu through WSL2.

### Create a Conda Environment

Open a terminal and enter the repository root:

```bash
cd /path/to/project-root
```

Create and activate a Python 3.9 environment:

```bash
conda create -n some_eval python=3.9 -y
conda activate some_eval
python -m pip install --upgrade pip
```

### Optional: Use the Alibaba Cloud PyPI Mirror

Users in regions with slow access to the default PyPI server may optionally configure the Alibaba Cloud mirror:

```bash
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set install.trusted-host mirrors.aliyun.com
```

To restore the official PyPI source later:

```bash
pip config unset global.index-url
pip config unset install.trusted-host
```

### Install Dependencies

Run the following command from the repository root:

```bash
python -m pip install -r requirements.txt
```

Verify the main dependencies:

```bash
python -c "import torch, trajdata, pandas, openai; print('Environment setup completed.')"
```

## Usage

The three stages should normally be run in order. The repository already includes example outputs, so users who only want to test model training may run Stage 3 directly.

### Stage 1: Extract Objective Metrics

Enter the module directory:

```bash
cd 1-objective_metrics
```

Calculate all ten metrics for all indexed interactions:

```bash
python metric.py
```

Calculate only one interaction for testing:

```bash
python metric.py --target-id 1
```

Results are saved under:

```text
1-objective_metrics/output/
```

> **Note:** A run with `--target-id` writes only the selected interaction to the output files. Run `python metric.py` again before full-data training.

### Stage 2: Generate LLM-Based Scores

Enter the module directory:

```bash
cd 2-subjective_evaluation
```

Set an OpenAI API key before running the script:

```bash
export OPENAI_API_KEY="your_api_key"
```

The API key must be available in the same terminal session used to run the evaluation. Never hard-code an API key or commit it to GitHub.

Multiple API keys can be separated by commas:

```bash
export OPENAI_API_KEYS="key_1,key_2,key_3"
```

Test one interaction first:

```bash
python LLM-based_evaluate.py --target-id 1
```

Evaluate all interactions:

```bash
python LLM-based_evaluate.py
```

The default model is `gpt-4o`. A model can also be selected explicitly:

```bash
python LLM-based_evaluate.py --model gpt-4o
```

Only evaluate selected prompt cases:

```bash
python LLM-based_evaluate.py --prompt-cases cp1 cp2
```

Restart the evaluation without reusing existing scores:

```bash
python LLM-based_evaluate.py --overwrite
```

> **Warning:** `--overwrite` rebuilds the score file from the current run. Combining `--overwrite` with `--target-id` may leave the output with only one interaction.

The subjective output is saved to:

```text
2-subjective_evaluation/output/evaluation_score.csv
```

Calling an online LLM may take time and incur API charges. If the provided `evaluation_score.csv` is sufficient, this stage can be skipped.

### Stage 3: Train the Evaluation Model

Enter the module directory:

```bash
cd 3-evaluation_model
```

Start training with the default configuration:

```bash
python train_evaluation_model.py
```

The default configuration uses 122 epochs, a batch size of 350, and a learning rate of 0.005.

Training parameters can be changed from the terminal:

```bash
python train_evaluation_model.py \
  --epochs 200 \
  --batch-size 128 \
  --learning-rate 0.001
```

Custom input and output paths are also supported:

```bash
python train_evaluation_model.py \
  --metrics-path /path/to/metrics.csv \
  --scores-path /path/to/evaluation_score.csv \
  --output-dir /path/to/output
```

Training results are saved under:

```text
3-evaluation_model/output/
```

## Training a Model for a Different Evaluation Target

The same workflow can be adapted to another vehicle, driving system, dataset, or evaluation target. A simple replacement process is recommended:

1. **Prepare interaction samples.**
   Create an interaction index in which every sample has a unique `index`. Provide the trajectory and scene information required by your data loader.

2. **Define objective metrics.**
   Keep the existing ten metrics or replace them with metrics suitable for the new target. Each sample must finally produce one row in `metrics.csv`.

3. **Update the feature configuration.**
   If the metric columns change, update `FEATURE_COLUMNS` in `3-evaluation_model/src/config.py` so that its order exactly matches the model input.

4. **Customize the subjective evaluation.**
   Modify the prompt generators and `2-subjective_evaluation/src/evaluation_guidelines.json` to describe the new target, evaluation criteria, and expected behavior.

5. **Generate normalized labels.**
   Produce `evaluation_score.csv` with `index` and `score`. Keep `score` in `[0, 1]`, and make sure its indexes exactly match `metrics.csv`.

6. **Train and inspect the model.**
   Run `train_evaluation_model.py`, inspect the loss curve and attention matrix, and adjust the model configuration if necessary.

The minimum required interface between modules is:

```text
metrics.csv:
index,<feature_1>,<feature_2>,...

evaluation_score.csv:
index,score
```

## Data and Acknowledgements

This lightweight package uses Waymo interaction samples organized through a trajdata-compatible cache. For the complete interaction dataset and related extraction tools, refer to the [InterHub repository](https://github.com/zxc-tju/InterHub).

Users are responsible for following the licenses and usage requirements of Waymo Open Dataset, InterHub, trajdata, OpenAI services, and any replacement datasets used in their own experiments.
