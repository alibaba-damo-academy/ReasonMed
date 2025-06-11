# ReasonMed: A 370K Multi-Agent Generated Dataset for Advancing Medical Reasoning

<p align="center">
  <a href="https://arxiv.org/">[📖Paper]</a> &nbsp;&nbsp;
  <a href="https://huggingface.co/datasets/YuSun-AI/ReasonMed">[🤗ReasonMed Dataset]</a>
</p>


<p align="center">
  <a href="https://huggingface.co/YuSun-AI/ReasonMed">[🤗ReasonMed-7B model]</a> &nbsp;&nbsp;
  <a href="https://huggingface.co/YuSun-AI/CoTMed">[🤗CoTMed-7B model]</a> &nbsp;&nbsp;
  <a href="https://huggingface.co/YuSun-AI/ResponseMed">[🤗ReasponseMed-7B model]</a>
</p>

**Table of Contents**  

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Modules](#modules)
   - 3.1 [Generate CoTs](#generate-cots)
   - 3.2 [Evaluate CoTs](#evaluate-cots)
   - 3.3 [Quality Ranker](#quality-ranker)
   - 3.4 [Error Refiner](#error-refiner)
   - 3.5 [Diff Optimizer](#diff-optimizer)
   - 3.6 [Response Summarizer](#response-summarizer)
   - 3.7 [Score Evaluator](#score-evaluator)
4. [Example Pipeline](#example-pipeline)
   - 4.1 [Easy Pipeline](#easy-pipeline)
   - 4.2 [Medium Pipeline](#medium-pipeline)
   - 4.3 [Difficult Pipeline](#difficult-pipeline)
5. [Conclusion](#conclusion)


## Introduction
ReasonMed is a comprehensive multi-agent generated dataset designed to advance medical reasoning capabilities. It is equipped with various tools and modules for generating, validating, optimizing, ranking, summarizing, and evaluating Chain-of-Thought (CoT) responses in the medical domain. ReasonMed's goal is to help researchers and practitioners improve and assess medical reasoning in clinical decision-making.

This README provides an overview of ReasonMed's core functionality, installation instructions, usage examples, and how to integrate each module into your medical reasoning workflow.

<!-- --- -->

## Installation

### Clone the Repository
To get started, clone the ReasonMed repository:

```bash
git clone https://github.com/YuSun-Work/ReasonMed.git
cd ReasonMed
```

### Requirements


```bash
conda create -n reasonmed python=3.11 -y
conda activate reasonmed
pip install -r requirements.txt
```


Note: Ensure that you have access to the models or endpoints mentioned for inference in each script.

<!-- --- -->

## Modules

### Generate CoTs
This module generates multiple Chain-of-Thought (CoT) responses from three different models, each generating three CoTs for a given question.

#### Command Example:
```bash
python generate_9cot.py --data_path /path/to/question.json --model_path1 /path/to/model1 --model_path2 /path/to/model2 --model_path3 /path/to/model3 --json_path /path/to/save_cot.json
```

#### Input Example (`question.json`):
```json
[
    {
        "question": "Chronic urethral obstruction due to benign prostatic hyperplasia can lead to the following change in kidney parenchyma",
        "options": [
            "Hyperplasia",
            "Hypertrophy",
            "Atrophy",
            "Dysplasia"
        ]
    }
]
```

- **`data_path`**: The path to the JSON file containing clinical questions and multiple-choice options.
- **`model_path1`, `model_path2`, `model_path3`**: Paths to the three models used for generating CoTs.
- **`json_path`**: Path to save the generated CoTs.

#### Output:
The generated CoTs will be saved in a specified JSON file.

#### Intermediate File:
`intermediate.json` can be used to store results after each stage, useful for debugging or troubleshooting.

<!-- --- -->

### Evaluate CoTs
This script validates the generated CoTs by verifying their correctness against clinical reasoning.

#### Command Example:
```bash
python verifier.py --input_json /path/to/save_cot.json --model_path /path/to/eval_model
```

- **`input_json`**: Path to the JSON file containing generated CoTs.
- **`model_path`**: Path to the model used for evaluating the CoTs.

#### Output:
Validates the CoTs and outputs a verdict (e.g., `Correct`, `Error`).

<!-- --- -->

### Quality Ranker
The quality ranker ranks the CoTs generated for each clinical question, keeping the top two most valid CoTs.

#### Command Example:
```bash
python quality_ranker.py --input_json /path/to/save_cot.json --model_path /path/to/eval_model --intermediate_file /path/to/intermediate.json --final_output /path/to/final_results.json
```

- **`input_json`**: The JSON file with the CoTs to be evaluated and ranked.
- **`model_path`**: Path to the model used for ranking.
- **`intermediate_file`**: Path to save intermediate ranking results.
- **`final_output`**: Path to save the final ranked CoTs.

#### Output:
Ranks the CoTs and saves the best two CoTs per clinical question.

<!-- --- -->

### Error Refiner
This module refines CoTs that have errors or incomplete reasoning by leveraging error feedback to improve the reasoning process.

#### Command Example:
```bash
python error_refiner_openai.py --input_json /path/to/save_cot.json --api_key /path/to/api_key --azure_endpoint /path/to/azure_endpoint --model /path/to/refine_model --output_json /path/to/refined_output.json
```

- **`input_json`**: Path to the JSON file with generated CoTs to be refined.
- **`api_key`**: Your Azure OpenAI API key.
- **`azure_endpoint`**: The endpoint for the Azure OpenAI API.
- **`model`**: Path to the refinement model.
- **`output_json`**: Path to save the refined CoTs.

#### Output:
Refined CoTs that incorporate error corrections from previous iterations.

<!-- --- -->

### Diff Optimizer
This module performs advanced optimizations on the CoTs using the Azure OpenAI API. It focuses on deep reasoning improvements based on detailed feedback.

#### Command Example:
```bash
python diff_opti.py --input_json /path/to/save_cot.json --output_json /path/to/optimized_cot.json --api_key /path/to/api_key --azure_endpoint /path/to/azure_endpoint --model /path/to/optimize_model
```

- **`input_json`**: Path to the JSON file with CoTs to be optimized.
- **`output_json`**: Path to save the optimized CoTs.
- **`api_key`**: Your Azure OpenAI API key.
- **`azure_endpoint`**: The Azure OpenAI endpoint.
- **`model`**: The model used for deep optimization.

#### Output:
Optimized CoTs that have undergone deeper analysis and improvement.

<!-- --- -->

### Response Summarizer
This module generates concise summaries for each CoT, transforming verbose reasoning into a one-sentence explanation.

#### Command Example:
```bash
python response_summarizer.py --input_json /path/to/save_cot.json --model /path/to/summary_model --azure_endpoint /path/to/azure_endpoint --api_key /path/to/api_key --results_file /path/to/summaries.json
```

- **`input_json`**: Path to the JSON file with CoTs to be summarized.
- **`model`**: The model used for summarization.
- **`azure_endpoint`**: The endpoint for the Azure OpenAI API.
- **`api_key`**: Your Azure OpenAI API key.
- **`results_file`**: Path to save the summarized CoTs.

#### Output:
A JSON file with concise summaries for each CoT.

<!-- --- -->

### Score Evaluator
This module evaluates the clinical accuracy of CoTs based on multiple criteria and generates scores for each CoT.

#### Command Example:
```bash
python score_evaluator.py --input_jsons /path/to/save_cot.json --model /path/to/score_model --azure_endpoint /path/to/azure_endpoint --api_key /path/to/api_key --final_output /path/to/scores.json
```

- **`input_jsons`**: The input JSON file with CoTs to be evaluated.
- **`model`**: Path to the model used for scoring.
- **`azure_endpoint`**: The endpoint for the Azure OpenAI API.
- **`api_key`**: Your Azure OpenAI API key.
- **`final_output`**: Path to save the final evaluation scores.

#### Output:
Scores for each CoT based on clinical accuracy, reasoning, and completeness.

<!-- --- -->

## Example Pipeline

### Easy Pipeline:
To generate and evaluate CoTs:
```bash
python generate_9cot.py --data_path /path/to/question.json --model_path1 /path/to/model1 --model_path2 /path/to/model2 --model_path3 /path/to/model3 --json_path /path/to/save_cot.json
python verifier.py --input_json /path/to/save_cot.json --model_path /path/to/eval_model
```

### Medium Pipeline:
To generate, evaluate, rank, and refine CoTs:
```bash
python generate_9cot.py --data_path /path/to/question.json --model_path1 /path/to/model1 --model_path2 /path/to/model2 --model_path3 /path/to/model3 --json_path /path/to/save_cot.json
python verifier.py --input_json /path/to/save_cot.json --model_path /path/to/eval_model
python quality_ranker.py --input_json /path/to/save_cot.json --model_path /path/to/eval_model --intermediate_file /path/to/intermediate.json --final_output /path/to/final_results.json
python error_refiner_openai.py --input_json /path/to/save_cot.json --api_key /path/to/api_key --azure_endpoint /path/to/azure_endpoint --model /path/to/refine_model --output_json /path/to/refined_output.json
```

### Difficult Pipeline:
For advanced optimizations:
```bash
python generate_9cot.py --data_path /path/to/question.json --model_path1 /path/to/model1 --model_path2 /path/to/model2 --model_path3 /path/to/model3 --json_path /path/to/save_cot.json
python verifier.py --input_json /path/to/save_cot.json --model_path /path/to/eval_model
python quality_ranker.py --input_json /path/to/save_cot.json --model_path /path/to/eval_model --intermediate_file /path/to/intermediate.json --final_output /path/to/final_results.json
python error_refiner_openai.py --input_json /path/to/save_cot.json --api_key /path/to/api_key --azure_endpoint /path/to/azure_endpoint --model /path/to/refine_model --output_json /path/to/refined_output.json
python diff_opti.py --input_json /path/to/save_cot.json --output_json /path/to/optimized_cot.json --api_key /path/to/api_key --azure_endpoint /path/to/azure_endpoint --model /path/to/optimize_model
```

<!-- --- -->

## Conclusion

ReasonMed provides an integrated framework for generating, optimizing, validating, and evaluating medical Chain-of-Thought responses. This comprehensive pipeline is crucial for advancing AI-powered clinical reasoning and decision-making models.
