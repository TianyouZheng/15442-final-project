import os
import json
import re
import time
import torch
import numpy as np
from tqdm import tqdm
from kvpress import *
from datasets import load_dataset
from transformers import pipeline, QuantizedCacheConfig, QuantoQuantizedCache

device = "cuda" if torch.cuda.is_available() else "cpu"

TEMPLATE = """The following is a math problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Math Problem]

{problem}

[Solution]

{tagged_response}

Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes "not found").

Please put your final answer (i.e., the index) in \\boxed{{}}."""


def extract_answer(solution_text: str):
    try:
        boxed_pattern = r"\\boxed\{([^}]*)\}"
        matches = re.findall(boxed_pattern, solution_text)
        if matches:
            return int(matches[-1].strip())
        return None
    except:
        return None


def prepare_input_boxed(template, input_d):
    problem = input_d["problem"]
    steps = input_d["steps"]
    tagged_response = ""
    for sdx, step in enumerate(steps):
        tagged_response += f"<paragraph_{sdx}>\n{step}\n</paragraph_{sdx}>\n\n"
    tagged_response = tagged_response.strip()
    return template.format(problem=problem, tagged_response=tagged_response)


def run(llm, output_dir, configs):
    os.makedirs(output_dir, exist_ok=True)
    for config in configs:
        if os.path.exists(os.path.join(output_dir, f"{config}_summary.json")):
            print(f"Skip {config}")
            continue

        start_time = time.time()
        dataset = load_dataset("Qwen/ProcessBench", split=config)
        prompts = [prepare_input_boxed(TEMPLATE, e) for e in dataset][1:10]
        # prompts = ["write a fib function in python"]
        generations = llm(prompts)

        res_data = []
        for input_data, output in zip(dataset, generations):
            pred = extract_answer(output)
            result = {
                "config": config,
                "generated_critique": output,
                "prediction": pred,
                "match": pred == input_data["label"],
                "label": input_data["label"],
            }
            res_data.append(result)

        error_data = [e for e in res_data if e["label"] != -1]
        correct_data = [e for e in res_data if e["label"] == -1]
        acc1 = np.mean([e["match"] for e in error_data]) * 100
        acc2 = np.mean([e["match"] for e in correct_data]) * 100
        f1 = 2 * acc1 * acc2 / (acc1 + acc2)
        summary = {
            "config": config,
            "error_acc": acc1,
            "correct_acc": acc2,
            "f1": f1,
            "time_used": time.time() - start_time,
        }
        print(summary)
        with open(os.path.join(output_dir, f"{config}_error.jsonl"), "w") as f:
            for e in error_data:
                f.write(json.dumps(e) + "\n")
        with open(os.path.join(output_dir, f"{config}_correct.jsonl"), "w") as f:
            for e in correct_data:
                f.write(json.dumps(e) + "\n")
        with open(os.path.join(output_dir, f"{config}_summary.json"), "w") as f:
            f.write(json.dumps(summary, indent=4) + "\n")


# ================== LLM ==================


def make_llm(
    model="Qwen/Qwen2.5-7B-Instruct",
    press=RandomPress(),
    temperature=0.0,
    quantization=4,
    max_new_tokens=1024,
    attn_implementation="flash_attention_2",
):
    model_kwargs = {"attn_implementation": attn_implementation}
    pipe = pipeline(
        "kv-press-text-generation",
        model=model,
        device_map="auto",
        torch_dtype=torch.float16 if quantization is not None else "auto",
        model_kwargs=model_kwargs,
    )
    kwargs = {
        "press": press,
        "question": "",
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "force_no_quantization_at_generation": True,
    }


    def generate(prompts):
        results = []
        tqdm_obj = tqdm(prompts)
        for prompt in tqdm_obj:
            if quantization is not None:
                cache_config = QuantizedCacheConfig(nbits=quantization, device=device)
                kwargs["cache"] = QuantoQuantizedCache(cache_config)
            result = pipe(prompt, **kwargs)["answer"]
            tqdm_obj.set_description_str(f"Length: {len(result)}")
            results.append(result)
            print(result)
            # with open("./quantization_outputs/quanto.txt", "a") as f:
            #     f.write(result + "\n")
        return results

    def free():
        nonlocal pipe
        del pipe
        torch.cuda.empty_cache()

    return generate, free


# ================== Main ==================


"""
conda create -n kvpress python=3.12
conda activate kvpress

cd kvpress && pip install -e .
pip install optimum-quanto
conda install -c conda-forge libstdcxx-ng
pip install flash-attn --no-build-isolation
"""

temperature = 0.0

configs = [
    "gsm8k",
    # "math",
    # "olympiadbench",
    # "omnimath",
]

quantizations = [
    # None,
    4,
    2,
]

models = [
    "Qwen/Qwen2.5-7B-Instruct",
    # "meta-llama/Llama-2-7b-chat-hf",
    # "Qwen/Qwen2.5-0.5B-Instruct",
    # "Qwen/Qwen2.5-Math-7B-Instruct",
    # "Llama-3.1-8B-Instruct",
]

presses = [
    # SnapKVPress,
    # StreamingLLMPress,
    # ExpectedAttentionPress,
    # TOVAPress,
    # ObservedAttentionPress,
    # QFilterPress,
    RandomPress,
    # KnormPress,
    # PyramidKVPress,
]

compression_ratios = [
    0.0,
    # 0.1,
    # 0.25,
    # 0.5,
    # 0.7,
    # 0.8,
    # 0.9,
    # 0.95,
]

attn = "sdpa"
# attn = "eager"
# attn = "flex_attention"
# attn = "flash_attention_2"


for model in models:
    for quantization in quantizations:
        for compression_ratio in compression_ratios:
            for press in presses:
                press_obj = press()
                press_name = press_obj.__class__.__name__
                model_name = model.replace("/", "--")
                run_id = f"{model_name}_{press_name}_r{compression_ratio}_q{quantization}"
                print(f"Running {run_id}")

                attn = "eager" if isinstance(press_obj, ObservedAttentionPress) else attn
                if hasattr(press_obj, "head_compression_ratio"):
                    press_obj.head_compression_ratio = compression_ratio
                elif hasattr(press_obj, "compression_ratio"):
                    press_obj.compression_ratio = compression_ratio

                max_new_tokens = 128 if compression_ratio > -1 else 1024

                llm, free = make_llm(
                    model=model,
                    press=press_obj,
                    temperature=temperature,
                    quantization=quantization,
                    max_new_tokens=max_new_tokens,
                    attn_implementation=attn,
                )
                run(llm, output_dir=f"./outputs/{run_id}/", configs=configs)
                free()
