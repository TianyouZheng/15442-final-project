import os
import math
import json
import re
import time
import torch
import numpy as np
from tqdm import tqdm
from kvpress import *
from datasets import load_dataset
from transformers import pipeline, QuantizedCacheConfig, QuantoQuantizedCache


TEMPLATE = """The following is a math problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Math Problem]

{problem}

[Solution]

{tagged_response}

Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes "not found").

Please put your final answer (i.e., the index) in \\boxed{{}}. Let's think step by step, justify each step in detail."""


def prepare_input_boxed(template, input_d):
    problem = input_d["problem"]
    steps = input_d["steps"]
    tagged_response = ""
    for sdx, step in enumerate(steps):
        tagged_response += f"<paragraph_{sdx}>\n{step}\n</paragraph_{sdx}>\n\n"
    tagged_response = tagged_response.strip()
    return template.format(problem=problem, tagged_response=tagged_response)


def run(llm, output_dir, configs):
    for config in configs:
        if os.path.exists(output_dir):
            print(f"File {output_dir} already exists. Skipping...")
            continue
        start_time = time.time()
        dataset = load_dataset("Qwen/ProcessBench", split=config)
        prompts = [prepare_input_boxed(TEMPLATE, e) for e in dataset][:5]
        start_time = time.time()
        raw_outputs = llm(prompts)
        duration = time.time() - start_time
        avg_time = duration / len(prompts)

        result = {
            "avg_time": avg_time,
            "raw_outputs": raw_outputs,
        }

        with open(f"{output_dir}", "a") as f:
            f.write(json.dumps(result, indent=4))


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
        torch_dtype="auto",
        model_kwargs=model_kwargs,
    )
    kwargs = {
        "press": press,
        "question": "",
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
    }

    if quantization is not None:
        kwargs["cache"] = QuantoQuantizedCache(QuantizedCacheConfig(nbits=quantization))

    def generate(prompts):
        results = []
        tqdm_obj = tqdm(prompts)
        for prompt in tqdm_obj:
            result = pipe(prompt, **kwargs)["answer"]
            tqdm_obj.set_description_str(f"Length: {len(result)}")
            results.append(result)
        return results

    def free():
        nonlocal pipe
        del pipe
        torch.cuda.empty_cache()

    return generate, free


# ================== Main ==================


temperature = 0.0

configs = [
    # "gsm8k",
    # "math",
    # "olympiadbench",
    "omnimath",
]

quantizations = [
    None,
    # 4,
    # 2,
]

models = [
    "Qwen/Qwen2.5-7B-Instruct",
    # "Qwen/Qwen2.5-3B-Instruct",
    # "Qwen/Qwen2.5-0.5B-Instruct",
    # "Qwen/Qwen2.5-Math-7B-Instruct",
    # "Llama-3.1-8B-Instruct",
]

presses = [
    RandomPress,
]

compression_ratios = [
    # 0.0,
    # 0.1,
    # 0.25,
    # 0.5,
    # 0.7,
    # 0.8,
    0.9,
    # 0.95,
]

attns = [
    "flex_attention",
    "sdpa",
    "flash_attention_2",
]
tokens = [
    64,
    128,
    256,
    512,
    # 1024,
]


for attn in attns:
    for token in tokens:
        for model in models:
            for quantization in quantizations:
                for compression_ratio in compression_ratios:
                    for press in presses:
                        press_obj = press(compression_ratio=compression_ratio)
                        run_id = f"{attn}_r{compression_ratio}_tok{token}"
                        print(f"Running {run_id}")

                        if hasattr(press_obj, "compression_ratio") and not isinstance(press_obj, ComposedPress):
                            press_obj.compression_ratio = compression_ratio

                        llm, free = make_llm(
                            model=model,
                            press=press_obj,
                            temperature=temperature,
                            quantization=quantization,
                            max_new_tokens=token,
                            attn_implementation=attn,
                        )
                        run(llm, output_dir=f"./test_speed/{run_id}.json", configs=configs)
                        free()
