import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json


@torch.no_grad()
def warmup_and_benchmark(
    model,
    tokenizer,
    max_seq_len,
    num_batches,
    max_new_tokens,
):
    inputs = tokenizer("Hi" * max_seq_len, return_tensors="pt").to("cuda")

    # warmup
    _ = model.generate(
        **inputs,
        max_new_tokens=20,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    with torch.no_grad():
        start_event.record()
        for _ in range(num_batches):
            _ = model.generate(
                **inputs,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                use_cache=False,
            )
        end_event.record()
        torch.cuda.synchronize()

    forward_timing = (start_event.elapsed_time(end_event) * 1.0e-3) / num_batches

    return forward_timing


if __name__ == "__main__":

    num_batches = 1
    max_new_tokens = 128
    model_id = "Qwen/Qwen2.5-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    ).to("cuda")

    model_fa = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0},
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )

    native_total_time_dict = {}
    fa2_total_time_dict = {}
    forward_speedups = {}
    for max_seq_len in [32, 64, 128, 256, 1024, 2048, 4096]:
        print(f"Running for sequence length {max_seq_len}")
        native_timing = warmup_and_benchmark(
            model,
            tokenizer,
            max_seq_len,
            num_batches,
            max_new_tokens,
        )
        native_total_time_dict[f"{max_seq_len}"] = native_timing

        fa2_timing = warmup_and_benchmark(
            model_fa,
            tokenizer,
            max_seq_len,
            num_batches,
            max_new_tokens,
        )
        fa2_total_time_dict[f"{max_seq_len}"] = fa2_timing

        forward_speedups[f"{max_seq_len}"] = native_timing / fa2_timing

        results = {
            "sdpa_time": native_total_time_dict,
            "fa2_time": fa2_total_time_dict,
        }
        with open("timing.json", "w") as f:
            json.dump(results, f, indent=4)
