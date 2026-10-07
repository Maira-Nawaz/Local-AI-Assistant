# Offline Local AI Assistant Using Small Language Models

A local AI assistant built with small language models (SLMs) running entirely on local hardware using Ollama.

This project explores the practical trade-offs of running small language models on constrained hardware, including **inference performance, task quality, structured-output reliability, and quantization**.

The goal is not simply to build a chatbot, but to experimentally evaluate and engineer a local AI system that can produce useful and reliable outputs without depending on cloud-based LLM APIs.

---

## Why This Project?

Cloud-based LLM applications can introduce:

- API costs
- Network dependency
- Data privacy concerns
- Variable latency
- Dependency on external infrastructure

Small language models provide an alternative for use cases where **local inference, privacy, lower cost, or offline capability** are important.

However, running an SLM locally introduces its own challenges:

- Limited CPU and RAM
- Slower inference
- Smaller model capacity
- Output reliability
- Model-selection trade-offs
- Quantization trade-offs

This project investigates these challenges experimentally.

---

# Project Goals

The project focuses on four main questions:

### 1. Model Selection

Which small language model performs best on constrained hardware?

### 2. Performance

How fast can different SLMs generate responses locally?

### 3. Reliability

Can a small local model consistently produce structured data that can safely be consumed by another application?

### 4. Efficiency

Can quantization reduce model footprint while maintaining acceptable performance and quality?

---

# System Architecture

```text
                    User
                     │
                     ▼
             Local AI Assistant
                     │
                     ▼
                  Ollama
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
       Qwen       Llama       Phi
       1.5B        1B         3.5
          │          │          │
          └──────────┼──────────┘
                     │
                     ▼
              Evaluation Layer
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
    Performance     Quality    Reliability
    Benchmark      Testing      Testing
        │            │            │
        ▼            ▼            ▼
    TTFT/Latency   40 prompts   JSON
    Tokens/sec                  Pydantic
                                Retry
```

---

# Hardware & Environment

The experiments were performed on a constrained local machine:

| Component | Specification |
|---|---|
| OS | Windows 11 |
| CPU | AMD Athlon Silver 3050U |
| CPU Cores | 2 |
| RAM | 8 GB |
| GPU | Integrated AMD Radeon |
| Python | 3.12.10 |
| Ollama | 0.35.1 |

The limited hardware makes the experiments representative of environments where GPU resources are unavailable or restricted.

---

# Models Evaluated

Three small/local language models were evaluated:

- **Qwen 2.5 1.5B**
- **Llama 3.2 1B**
- **Phi-3.5**

All models were evaluated using the same hardware and standardized evaluation methodology.

---

# 1. Performance Benchmark

The performance benchmark measures:

- **Time to First Token (TTFT)**
- **Total response latency**
- **Tokens per second**
- System memory snapshots

Each model was tested using the same controlled benchmark prompt and five runs.

## Baseline Results

| Model | Avg TTFT | Avg Latency | Avg Tokens/sec |
|---|---:|---:|---:|
| Qwen 2.5 1.5B | **0.72s** | **9.95s** | **7.85** |
| Llama 3.2 1B | 1.73s | 15.10s | 7.64 |
| Phi-3.5 | 2.81s | 27.47s | 3.66 |

### Observation

Qwen 2.5 1.5B provided the strongest overall inference performance in the tested environment.

It achieved:

- Lowest average TTFT
- Lowest average latency
- Highest average throughput

> **Note:** Benchmark averages include cold-start effects. In particular, some runs showed substantially higher TTFT while the model was being loaded.

---

# 2. Standardized Quality Evaluation

A standardized evaluation dataset containing **40 prompts** was created to compare model quality.

The prompts are divided into eight categories:

1. Knowledge
2. Explanation
3. Summarization
4. Classification
5. Information Extraction
6. Reasoning
7. Instruction Following
8. Structured Output

Each model was evaluated using the same prompts and scoring methodology.

## Quality Results

| Model | Correct | Incorrect | Accuracy |
|---|---:|---:|---:|
| Qwen 2.5 1.5B | 29/40 | 11 | **72.5%** |
| Llama 3.2 1B | 29/40 | 11 | **72.5%** |
| Phi-3.5 | 28/40 | 12 | **70.0%** |

### Observation

Qwen 2.5 1.5B and Llama 3.2 1B achieved the same overall accuracy in the standardized evaluation.

However, Qwen provided significantly better inference performance on the tested hardware.

This made Qwen 2.5 1.5B the strongest candidate for the subsequent reliability and efficiency experiments.

> **Evaluation note:** These are custom task scores from the project's 40-prompt evaluation set, not results from a standardized public benchmark.

---

# 3. Structured Output Reliability

A local AI system becomes much more useful when its output can be consumed by another application.

For example, instead of returning:

```text
My name is Ali, I am 25 years old and I live in Pakistan.
```

the model can produce:

```json
{
  "name": "Ali",
  "age": 25,
  "country": "Pakistan"
}
```

This structured representation can then be passed to:

- APIs
- Databases
- Automation workflows
- Data pipelines
- Other application components

However, an LLM's output cannot simply be trusted.

The project therefore adds a validation layer using **Pydantic**.

---

## Structured Output Pipeline

```text
User Input
    │
    ▼
Local SLM
    │
    ▼
JSON Output
    │
    ▼
JSON Parsing
    │
    ▼
Pydantic Validation
    │
    ├─────────────── Valid ───────────────► Validated Object
    │
    └──────────── Invalid
                       │
                       ▼
                    Retry
                       │
                       ▼
                Pydantic Validation
                       │
                 ┌─────┴─────┐
                 │           │
               Valid       Invalid
                 │           │
                 ▼           ▼
              Return      Graceful
                           Failure
```

---

# Reliability Experiment

A separate dataset containing **20 structured extraction prompts** was used to test the selected Qwen model.

The system measured:

- First-attempt validity
- Retry recovery
- Final failure rate
- Overall structured-output reliability

## Qwen 2.5 1.5B Results

| Metric | Result |
|---|---:|
| Total prompts | 20 |
| Valid on first attempt | **20/20** |
| Recovered after retry | 0/20 |
| Failed after retry | 0/20 |
| Final reliability | **100%** |

### Observation

Qwen 2.5 1.5B produced Pydantic-valid structured output for all 20 test prompts on the first attempt under the tested configuration.

The retry mechanism remains available as a reliability layer for cases where model output fails validation.

---

# 4. Quantization Experiment

After selecting Qwen 2.5 1.5B as the strongest candidate for the tested hardware, a more aggressively quantized Q3 variant was evaluated.

The experiment compares:

- **Q4 baseline:** `qwen2.5:1.5b`
- **Q3 quantized:** `qwen2.5:1.5b-instruct-q3_K_M`

The goal was to measure the trade-off between model footprint, inference performance, task quality, and structured-output reliability.

## Quantization Results

| Metric | Q4 Baseline | Q3 Quantized | Change |
|---|---:|---:|---:|
| Model size | **986 MB** | **824 MB** | **-16.4%** |
| Avg TTFT | **0.72s** | 4.71s | Higher |
| Avg latency | **9.95s** | 13.34s | Higher |
| Avg tokens/sec | **7.85** | 6.72 | **-14.4%** |
| Accuracy | **72.5%** | 67.5% | **-5.0 pp** |
| Structured reliability | **100%** | **100%** | No change |

### Quantization Observation

The Q3 variant reduced the local model footprint by approximately **16.4%**.

However, on the tested CPU hardware, the more aggressive quantization also resulted in:

- Lower average throughput
- Higher average latency
- A 5 percentage-point decrease in task accuracy

Structured-output reliability remained at **100%** in the 20-prompt test.

This demonstrates that **more aggressive quantization is not automatically better for every hardware configuration**. The optimal quantization level depends on the hardware, workload, and acceptable quality trade-offs.

> **Benchmark note:** The Q3 five-run average was affected by cold-start/outlier runs. Warm runs were substantially faster than the first and fifth runs, so the five-run average should be interpreted together with the individual-run variability.

---

# Overall Model Selection

Based on the experiments, **Qwen 2.5 1.5B in the baseline Q4 configuration** was selected as the preferred configuration for this project.

The decision was based on:

- Highest baseline throughput
- Lowest baseline latency
- Lowest baseline TTFT
- Joint-highest task accuracy
- 100% structured-output validity in the initial reliability test

The Q3 configuration provides a smaller model footprint, but its measured performance and task-quality trade-offs were less favorable on this specific hardware.

---

# Engineering Approach

The project follows an experimental engineering workflow:

```text
        Define Problem
              │
              ▼
       Select Candidate Models
              │
              ▼
        Benchmark Models
              │
              ▼
       Evaluate Task Quality
              │
              ▼
        Select Best Model
              │
              ▼
     Engineer Reliable Output
              │
              ▼
       Test Structured Data
              │
              ▼
        Test Quantization
              │
              ▼
        Analyze Trade-offs
```

This approach avoids selecting a model based solely on popularity or model size.

Instead, the model is selected based on **measured performance and task quality on the target hardware**.

---

# Project Structure

```text
local-ai-assistant/
│
├── assistant.py
├── benchmark.py
├── structured_output.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── benchmark_results/
│   ├── qwen25_1.5b_baseline.csv
│   ├── phi3.5_baseline.csv
│   ├── llama3.2_1b_baseline.csv
│   └── qwen2.5_1.5b-instruct-q3_K_M_baseline.csv
│
└── evaluation/
    ├── prompts.csv
    ├── structured_prompts.csv
    ├── evaluate.py
    ├── score_results.py
    ├── test_structured_output.py
    │
    └── results/
        ├── qwen25_1.5b_results.csv
        ├── qwen25_1.5b_scored.csv
        ├── phi3.5_results.csv
        ├── phi3.5_scored.csv
        ├── llama3.2_1b_results.csv
        ├── llama3.2_1b_scored.csv
        ├── qwen2.5_1.5b-instruct-q3_K_M_results.csv
        ├── qwen2.5_1.5b-instruct-q3_K_M_scored.csv
        └── structured_results.csv
```

---

# Installation

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd local-ai-assistant
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

## 3. Activate the environment

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Install Ollama

Install Ollama separately, then pull the desired model:

```bash
ollama pull qwen2.5:1.5b
```

For the quantization experiment:

```bash
ollama pull qwen2.5:1.5b-instruct-q3_K_M
```

---

# Running the Assistant

Run the local assistant:

```bash
python assistant.py
```

Example:

```text
Local AI Assistant
Type 'exit' to quit.

You: What is a data pipeline?

Assistant: ...
```

---

# Running the Benchmarks

## Performance Benchmark

```bash
python benchmark.py qwen2.5:1.5b
```

Other models:

```bash
python benchmark.py phi3.5
python benchmark.py llama3.2:1b
```

Quantized Qwen:

```bash
python benchmark.py qwen2.5:1.5b-instruct-q3_K_M
```

---

# Running the Quality Evaluation

Run the standardized 40-prompt evaluation:

```bash
python evaluation/evaluate.py qwen2.5:1.5b
```

Then score the results:

```bash
python evaluation/score_results.py qwen2.5:1.5b
```

The same process can be used for the other models.

---

# Running Structured Output Testing

Run the single structured-output example:

```bash
python structured_output.py
```

For the 20-prompt reliability evaluation:

```bash
python evaluation/test_structured_output.py
```

---

# Key Findings

### Model Performance

Qwen 2.5 1.5B achieved the strongest baseline inference performance among the tested models.

### Model Quality

Qwen 2.5 1.5B and Llama 3.2 1B both achieved **72.5%** accuracy across the 40-prompt evaluation.

### Local Inference

Small language models can provide usable inference performance even on modest CPU-based hardware.

### Structured Outputs

Pydantic validation provides a deterministic validation layer between an LLM and downstream application logic.

### Reliability

Qwen 2.5 1.5B achieved **100% first-attempt structured-output validity** across the initial 20-prompt reliability experiment.

The quantized Q3 configuration also achieved **100%** in the same structured-output test.

### Quantization

The Q3 variant reduced model size by **16.4%**, but on the tested hardware it also reduced throughput by **14.4%** and task accuracy by **5 percentage points**.

This highlights the importance of evaluating quantization empirically rather than assuming that a smaller model will always be faster or better.

---

# Limitations

This project has several limitations:

- The experiments were performed on a single hardware configuration.
- The quality evaluation contains 40 prompts.
- The structured-output reliability test contains 20 prompts.
- System memory measurements are snapshots rather than dedicated model-memory profiling.
- The benchmark focuses on CPU-based local inference.
- Results may differ significantly on other hardware.
- The quality scoring uses a custom evaluation methodology rather than a standardized public benchmark.
- Quantization results are based on one Q3 configuration rather than a complete quantization sweep.

Therefore, the results should be interpreted as **hardware- and methodology-specific experimental results**, not universal model rankings.

---

# Future Improvements

Possible extensions include:

- Larger evaluation datasets
- More structured-output test cases
- Additional quantization levels
- Dedicated model-memory profiling
- GPU inference comparison
- FastAPI service layer
- Docker deployment
- Monitoring and observability
- RAG integration
- Edge-device deployment
- Concurrent request testing

---

# Technologies Used

- Python
- Ollama
- Qwen
- Llama
- Phi
- Pydantic
- psutil
- CSV-based evaluation
- Git / GitHub

---

# Conclusion

This project demonstrates a practical workflow for evaluating and engineering local small language models under hardware constraints.

Rather than treating an LLM as a black-box chatbot, the project evaluates the system across multiple engineering dimensions:

```text
Performance
     +
Task Quality
     +
Structured Output
     +
Validation
     +
Reliability
     +
Quantization
     =
Local AI System Engineering
```

The experiments show that small language models can be viable for local AI applications when model selection, output validation, and hardware constraints are considered together.

The final Qwen comparison also demonstrates an important engineering lesson: **reducing model size does not necessarily improve real-world inference performance on every hardware configuration.** Quantization should therefore be evaluated as a measurable trade-off between footprint, speed, and quality rather than treated as an automatic optimization.
