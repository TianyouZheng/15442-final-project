# Full script for evaluating models with KV-Press and accuracy debugging
# Dependencies: transformers, datasets, torch, numpy, tqdm, kvpress
# Ensure 'kvpress' library is installed and compatible.

import argparse
import numpy as np
import os
import torch
from collections import Counter
from transformers import pipeline
from datasets import load_dataset
import re
from tqdm import tqdm
import math
import multiprocessing
import traceback # For detailed error printing

# --- Check for kvpress library ---
try:
    from kvpress import ExpectedAttentionPress, ObservedAttentionPress
    KVPRESS_AVAILABLE = True
except ImportError:
    print("WARNING: 'kvpress' library not found. KV-Press features will be disabled.")
    KVPRESS_AVAILABLE = False
    # Define dummy classes if kvpress not found to avoid NameErrors later
    class DummyPress:
        def __init__(self, *args, **kwargs):
            pass
    ExpectedAttentionPress = DummyPress
    ObservedAttentionPress = DummyPress
# ---------------------------------

# Global flag to control internal debugging prints
DEBUG_ENABLED = False

# Helper function (MODIFIED for debugging)
def extract_answer(solution_text: str):
    """Extract the answer from the boxed solution"""
    global DEBUG_ENABLED
    debug_prefix = "[extract_answer Debug]"

    # Initial logging if debug enabled
    if DEBUG_ENABLED:
        print(f"\n{debug_prefix} --- Input ---")
        print(f"{debug_prefix} Type: {type(solution_text)}")
        print(f"{debug_prefix} Text (first 500 chars): {str(solution_text)[:500]}")
        print(f"{debug_prefix} ---------------")

    # Ensure input is a string before regex processing
    if isinstance(solution_text, list) and len(solution_text) > 0:
        # Attempt to extract from expected structure: [{'generated_text': '...'}]
        item = solution_text[0]
        if isinstance(item, dict) and 'generated_text' in item:
             solution_text = item.get('generated_text', '')
             if DEBUG_ENABLED: print(f"{debug_prefix} Extracted text from list/dict structure.")
        else:
             # Fallback if structure is different (e.g., list of strings)
             solution_text = str(item)
             if DEBUG_ENABLED: print(f"{debug_prefix} Converted first list element to string.")
    elif not isinstance(solution_text, str):
        if DEBUG_ENABLED: print(f"{debug_prefix} Input is not a string or expected list structure, returning None.")
        return None

    # Regex pattern to find \boxed{...}
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    try:
        matches = re.findall(boxed_pattern, solution_text)
    except Exception as e:
        if DEBUG_ENABLED: print(f"{debug_prefix} Regex error: {e}, returning None.")
        return None # Handle potential regex errors on weird inputs


    if DEBUG_ENABLED:
        print(f"{debug_prefix} Regex matches found: {matches}")

    if matches:
        # Return the *last* match found
        extracted = matches[-1].strip()
        if DEBUG_ENABLED: print(f"{debug_prefix} Returning extracted value: '{extracted}'")
        return extracted

    if DEBUG_ENABLED: print(f"{debug_prefix} No matches found, returning None.")
    return None

# Helper function
def prepare_input_boxed(template, example):
    """Prepare input from example dictionary"""
    problem = example.get('problem', '[Problem missing]') # Use .get for safety
    steps = example.get('steps', [])

    tagged_response = ''
    if isinstance(steps, list):
        for sdx, step in enumerate(steps):
            tagged_response += f'<paragraph_{sdx}>\n{step}\n</paragraph_{sdx}>\n\n'
    tagged_response = tagged_response.strip()

    # Handle potential missing keys in format
    try:
        prompt = template.format(problem=problem, tagged_response=tagged_response)
    except KeyError as e:
        print(f"Warning: Missing key in template formatting: {e}")
        # Provide a fallback or adjust template as needed
        prompt = f"Problem:\n{problem}\n\nSolution:\n{tagged_response}\n\nAre there any errors?"

    return prompt


def main():
    global DEBUG_ENABLED, KVPRESS_AVAILABLE # Allow modifying the global flag

    parser = argparse.ArgumentParser(description="Evaluate LLMs with optional KV-Press compression.")
    # --- Basic Arguments ---
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                        choices=['gsm8k', 'math', 'olympiadbench', 'omnimath'],
                        help='Dataset configurations to evaluate (default: all).')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the Hugging Face model.')
    parser.add_argument("--output_dir", type=str, default='./outputs',
                        help='Directory to save metrics files.')
    parser.add_argument('--use_voting', action='store_true',
                        help='Use multiple generations and voting for prediction.')
    parser.add_argument('--voting_n', type=int, default=8,
                        help='Number of generations for voting.')
    parser.add_argument('--processing_batch_size', type=int, default=16,
                        help='Number of examples to process in one pipeline call.')

    # --- KV-Press Arguments ---
    kv_group = parser.add_argument_group('KV-Press Options (Requires kvpress library)')
    kv_group.add_argument('--use_kvpress', action='store_true',
                           help='Enable KV-Press compression (requires kvpress library).')
    kv_group.add_argument('--compression_ratio', type=float, default=0.5,
                           help='KV-Press compression ratio (if --use_kvpress).')
    kv_group.add_argument('--press_type', type=str, default='expected',
                           choices=['expected', 'observed'],
                           help='KV-Press type (if --use_kvpress).')

    # --- Technical Arguments ---
    tech_group = parser.add_argument_group('Technical Options')
    tech_group.add_argument('--attn_implementation', type=str, default='sdpa',
                           choices=['eager', 'flash_attention_2', 'sdpa'],
                           help='Attention implementation backend (sdpa recommended).')
    tech_group.add_argument('--trust_remote_code', action='store_true',
                            help='Set trust_remote_code=True when loading model/pipeline.')

    # --- Debugging Arguments ---
    debug_group = parser.add_argument_group('Debugging Options')
    debug_group.add_argument('--debug_accuracy', action='store_true',
                             help='Enable printing debug info for accuracy calculation.')
    debug_group.add_argument('--num_debug_samples', type=int, default=5,
                             help='Number of samples per config to print debug info for.')

    args = parser.parse_args()
    args.model_name = os.path.basename(args.model_path)

    # Handle KV-Press option based on availability and user flag
    if args.use_kvpress and not KVPRESS_AVAILABLE:
        print("ERROR: --use_kvpress specified, but 'kvpress' library is not installed. Exiting.")
        return
    if not args.use_kvpress:
        print("KV-Press is disabled (--use_kvpress not specified).")
    elif KVPRESS_AVAILABLE:
         print("KV-Press is ENABLED.")


    # Set the global debug flag based on the argument
    if args.debug_accuracy:
        DEBUG_ENABLED = True
        print(f"*** Accuracy debugging ENABLED for the first {args.num_debug_samples} samples per config ***")

    # --- Load Template ---
    try:
        # Consider making template path an argument?
        template_path = './src/template.txt'
        TEMPLATE = open(template_path).read().strip()
        print(f"Loaded template from: {template_path}")
    except FileNotFoundError:
        print(f"Warning: Template file not found at {template_path}. Using fallback template.")
        TEMPLATE = """Please identify if there are any mathematical errors in the following problem and solution. If there are errors, explain the error and provide the correct answer.

Problem:
{problem}

Solution:
{tagged_response}

Are there any errors in the solution? If yes, what is the correct answer?"""

    # --- Pipeline Creation ---
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Target device: {device}")
    # Use float16 only if CUDA is available and compute capability is >= 7.0
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else \
                  (torch.float16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7 else torch.float32)
    print(f"Using torch_dtype: {torch_dtype}")

    model_kwargs = {}
    if args.attn_implementation and args.attn_implementation != 'eager':
         # Only set if supported, otherwise Transformers handles default ('sdpa' is often default now)
         model_kwargs["attn_implementation"] = args.attn_implementation
         print(f"Requested attn_implementation: {args.attn_implementation}")
    # Add other model_kwargs if needed

    # Determine pipeline task based on whether KV-Press is used
    pipeline_task = "kv-press-text-generation" if args.use_kvpress and KVPRESS_AVAILABLE else "text-generation"
    print(f"Using pipeline task: '{pipeline_task}'")

    pipe = None
    try:
        print(f"Attempting to create pipeline...")
        pipe = pipeline(
            pipeline_task,
            model=args.model_path,
            device_map="auto", # Recommended for multi-GPU or large models
            torch_dtype=torch_dtype,
            model_kwargs=model_kwargs,
            trust_remote_code=args.trust_remote_code
        )
        print(f"Pipeline created successfully.")

    except Exception as e:
        print(f"\n--- ERROR: Failed to create pipeline '{pipeline_task}' ---")
        if pipeline_task == "kv-press-text-generation":
             print(f"Ensure 'kvpress' library is installed correctly and compatible.")
        if args.trust_remote_code == False:
             print(f"Consider adding '--trust_remote_code' flag if the model requires it.")
        print(f"Original error: {e}\n")
        traceback.print_exc()
        return # Exit if pipeline creation fails

    # --- Create press object (only if using KV-Press) ---
    press = None
    if args.use_kvpress and KVPRESS_AVAILABLE:
        if args.press_type == 'expected':
            press = ExpectedAttentionPress(compression_ratio=args.compression_ratio)
        else:
            press = ObservedAttentionPress(compression_ratio=args.compression_ratio)
        print(f"Using {args.press_type} KV-Press with compression ratio {args.compression_ratio}")

    # --- Dataset Configuration ---
    if args.configs is None:
        args.configs = ['gsm8k', 'math', 'olympiadbench', 'omnimath']
    print(f"Selected dataset configs: {args.configs}")


    # --- Main Evaluation Loop ---
    for config in args.configs:
        # Reset debug counter for each config
        debug_samples_printed_this_config = 0

        # Construct output directory path, handle potential naming conflicts if needed
        kvp_suffix = f'_kvpress_{args.press_type}_{args.compression_ratio}' if args.use_kvpress and KVPRESS_AVAILABLE else ''
        output_subdir_name = f'{args.model_name}{kvp_suffix}'
        output_dir = os.path.join(args.output_dir, output_subdir_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n===== Processing config: {config} =====")
        try:
            # Load dataset into memory for easier batch slicing
            dataset = load_dataset('Qwen/ProcessBench', split=config)
            dataset_list = list(dataset) # Convert to list
            total_examples = len(dataset_list)
            print(f"Loaded {total_examples} examples from {config}")
            if total_examples == 0:
                print("Dataset is empty, skipping config.")
                continue
        except Exception as e:
            print(f"ERROR: Failed to load dataset for config {config}: {e}")
            traceback.print_exc()
            continue # Skip to the next config

        # Initialize metrics lists for this config
        error_matches = []
        correct_matches = []

        # Batch Processing Loop
        num_batches = math.ceil(total_examples / args.processing_batch_size)

        for i in tqdm(range(num_batches), desc=f"Processing {config} (Batch Size: {args.processing_batch_size})"):
            start_idx = i * args.processing_batch_size
            end_idx = min((i + 1) * args.processing_batch_size, total_examples)
            current_batch_indices = range(start_idx, end_idx)
            batch_examples = dataset_list[start_idx:end_idx]
            if not batch_examples: continue

            # Prepare batch data
            try:
                batch_prompts = [prepare_input_boxed(TEMPLATE, example) for example in batch_examples]
                batch_labels = [example.get('label', None) for example in batch_examples] # Use .get for safety
                # Check if any label is None, indicating potential dataset issue
                if None in batch_labels:
                     print(f"Warning: Found None label in batch {i} for config {config}. Check dataset integrity.")

            except Exception as e:
                 print(f"\n--- ERROR preparing batch {i} for config '{config}' (Indices {start_idx}-{end_idx-1}) ---")
                 print(f"Error: {e}")
                 traceback.print_exc()
                 # Decide how to handle: skip batch? assign default values?
                 print(f"Skipping batch {i} due to preparation error.")
                 continue # Skip this batch

            # Store raw critiques and predictions for this batch
            batch_critiques = ["<Error during generation>"] * len(batch_examples)
            batch_preds = [None] * len(batch_examples)

            try:
                # --- Pipeline Call ---
                # Common arguments for generation
                generation_args = {
                    "max_new_tokens": 8192, # Consider making this an arg
                    "return_full_text": False,
                    # Add other relevant generation params here if needed globally
                    # "eos_token_id": pipe.tokenizer.eos_token_id, # Example
                }
                # Add KV-Press object if enabled
                if args.use_kvpress and KVPRESS_AVAILABLE and press:
                    generation_args["press"] = press

                # Add voting arguments if enabled
                if args.use_voting:
                    generation_args.update({
                        "temperature": 0.7 if 'Qwen2.5-Math' in args.model_path else 1.0, # Model-specific defaults
                        "top_p": 0.8 if 'Qwen2.5-Math' in args.model_path else 0.9,
                        "top_k": 20 if 'Qwen2.5-Math' in args.model_path else 50,
                        "do_sample": True,
                        "num_return_sequences": args.voting_n
                    })

                # Execute pipeline call
                responses = pipe(batch_prompts, **generation_args)

                # --- Process Responses ---
                if not args.use_voting:
                    # Expecting list of dicts: [{'generated_text': '...'}]
                    for idx, response in enumerate(responses):
                        # Safely extract critique text
                        critique = "<Extraction Error>"
                        if isinstance(response, list) and len(response)>0 and isinstance(response[0], dict):
                           critique = response[0].get('generated_text', '<Missing generated_text key>')
                        elif isinstance(response, dict):
                           critique = response.get('generated_text', '<Missing generated_text key>')
                        else:
                            print(f"Warning: Unexpected response format in non-voting case: {type(response)}")
                            critique = str(response) # Fallback

                        batch_critiques[idx] = critique
                        pred_str = extract_answer(critique)
                        numeric_pred = None # Default to None
                        if pred_str is not None:
                            try:
                                # Try converting to int first (handles '-1', '0', '1', etc.)
                                numeric_pred = int(pred_str)
                            except (ValueError, TypeError):
                                try:
                                    # If int fails, try float
                                    numeric_pred = float(pred_str)
                                except (ValueError, TypeError):
                                    # If both fail, numeric_pred remains None
                                    if args.debug_accuracy: # Print warning only if debugging
                                        tqdm.write(f"[Warning] Extracted '{pred_str}' could not be converted to number (Index: {start_idx + idx}).")
                        batch_preds[idx] = numeric_pred
                else: # Handle voting results
                    # Expecting list of lists: [[{'gen...'}*N], [{'gen...'}*N], ...]
                    if isinstance(responses, list) and len(responses) > 0 and isinstance(responses[0], list):
                        for prompt_idx, prompt_responses in enumerate(responses):
                            # Ensure we don't go out of bounds for batch_critiques/batch_preds
                            if prompt_idx >= len(batch_critiques):
                                 print(f"Warning: More responses than expected for batch {i}, prompt_idx {prompt_idx}")
                                 continue

                            # Extract critiques for this prompt
                            critiques_for_prompt = []
                            for resp in prompt_responses:
                                if isinstance(resp, dict):
                                    critiques_for_prompt.append(resp.get('generated_text', '<Missing generated_text key>'))
                                else:
                                     print(f"Warning: Unexpected item type in prompt_responses: {type(resp)}")
                                     critiques_for_prompt.append(str(resp)) # Fallback

                            batch_critiques[prompt_idx] = critiques_for_prompt # Store list of critiques

                            # Extract answers and vote
                            preds_for_prompt = []
                            for critique in critiques_for_prompt:
                                extracted = extract_answer(critique)
                                if extracted is not None:
                                    preds_for_prompt.append(extracted)

                            # Determine final prediction from votes
                            final_pred = None
                            if preds_for_prompt: # Only vote if we extracted any answers
                                try:
                                    # Use Counter for efficient voting
                                    vote_counts = Counter(preds_for_prompt)
                                    most_common_pred_str = vote_counts.most_common(1)[0][0]
                                    # Attempt conversion of the most common prediction
                                    final_pred = int(most_common_pred_str) if most_common_pred_str.isdigit() else \
                                                 (float(most_common_pred_str) if '.' in most_common_pred_str else None) # Basic float check
                                    if final_pred is None:
                                         print(f"Warning: Most common vote '{most_common_pred_str}' could not be converted to int/float.")
                                except Exception as vote_err:
                                     print(f"Error during voting logic for prompt {prompt_idx}: {vote_err}")

                            batch_preds[prompt_idx] = final_pred # Store final voted prediction

                    else: # Handle unexpected response structure for voting
                        print(f"Warning: Unexpected response structure for voting in batch {i}. Type: {type(responses)}. Accuracy debugging might be compromised for this batch.")
                        # Attempt a flattened interpretation if feasible, otherwise batch_preds remains None
                        # This part might need adjustment based on observed incorrect structures

                # --- Process Batch Results and Debug ---
                for idx_in_batch, (pred, label, critique_data) in enumerate(zip(batch_preds, batch_labels, batch_critiques)):
                    current_global_idx = start_idx + idx_in_batch # Calculate index within the dataset config
                    # Handle potential None labels safely
                    if label is None:
                        match = False # Cannot match if label is None
                        print(f"Warning: Skipping match calculation for global index {current_global_idx} due to None label.")
                    else:
                         # Try converting label to type of pred if needed, or vice-versa
                         # Basic type comparison and conversion attempt
                         try:
                             if isinstance(pred, (int, float)) and isinstance(label, (int, float)):
                                 match = (pred == label)
                             elif pred is not None: # Attempt conversion if pred exists but types differ
                                 match = (type(label)(pred) == label)
                             else: # pred is None
                                 match = False
                         except (ValueError, TypeError):
                              match = False # Conversion failed or types incompatible

                    # --- Accuracy Debugging Logic ---
                    # Print if debug enabled and haven't printed enough samples for *this config* yet
                    if args.debug_accuracy and debug_samples_printed_this_config < args.num_debug_samples:
                        print(f"\n--- Debugging Sample {debug_samples_printed_this_config + 1}/{args.num_debug_samples} (Config: {config}, Index: {current_global_idx}) ---")
                        print(f"Raw Generated Critique(s):")
                        critique_limit = 500 # Limit print length
                        if isinstance(critique_data, list): # Voting case
                            print(f"  (Voting used - {len(critique_data)} responses)")
                            for v_idx, c_text in enumerate(critique_data):
                                print(f"    Vote {v_idx+1}: {c_text[:critique_limit]}{'...' if len(c_text)>critique_limit else ''}")
                        else: # Single generation case
                            print(f"  {critique_data[:critique_limit*2]}{'...' if len(critique_data)>critique_limit*2 else ''}") # Print more for single
                        print(f"Extracted Prediction (pred): {pred} (type: {type(pred)})")
                        print(f"Ground Truth Label (label): {label} (type: {type(label)})")
                        print(f"Prediction == Label? : {match}")
                        print("-" * 40)
                        debug_samples_printed_this_config += 1

                    # --- Update metrics lists ---
                    # Determine if it's an 'error' or 'correct' example based on label
                    # Assuming label == -1 means the original solution was correct (no error)
                    # Assuming label != -1 means the original solution had an error, and label is the correct answer
                    if label is not None: # Only record metrics if label is valid
                         if label != -1:
                             error_matches.append(match) # Did we correctly identify/fix the error?
                         else: # label == -1
                             correct_matches.append(match) # Did we correctly identify no error (pred should be None or match -1?)
                             # NOTE: If label is -1 (original is correct), should pred be None or -1?
                             # This depends on the task definition. Assuming pred should also be None or -1 for a match.
                             # Let's refine the match logic for label == -1
                             if label == -1:
                                 match_neg_one = (pred is None or pred == -1) # Redefine match for this case
                                 correct_matches[-1] = match_neg_one # Update the last appended value


                    # --- Periodic Progress Update ---
                    # Print progress roughly every 10% or every 50 samples, whichever is more frequent
                    samples_processed_so_far = len(error_matches) + len(correct_matches)
                    print_interval = max(50, total_examples // 10) if total_examples > 0 else 50
                    if (samples_processed_so_far + 1) % print_interval == 0 or samples_processed_so_far == total_examples -1 :
                         error_acc = np.mean(error_matches) * 100 if error_matches else 0
                         correct_acc = np.mean(correct_matches) * 100 if correct_matches else 0
                         tqdm.write(f"Progress ({config}): {samples_processed_so_far+1}/{total_examples} samples processed - "
                                    f"Error Acc: {error_acc:.1f}% ({len(error_matches)} samples), "
                                    f"Correct Acc: {correct_acc:.1f}% ({len(correct_matches)} samples)")


            except Exception as e:
                print(f"\n--- ERROR processing batch {i} during generation/results handling (Config: '{config}') ---")
                print(f"Indices: {start_idx}-{end_idx-1}")
                print(f"Error: {e}")
                traceback.print_exc()
                num_failed = len(batch_prompts)
                # Append False matches for metrics for the failed batch
                for label in batch_labels:
                     if label is not None:
                         if label != -1: error_matches.append(False)
                         else: correct_matches.append(False)
                # Skip debug printing increment if batch failed before results processing
                print(f"Skipped results processing for {num_failed} examples in this batch due to error.")
                continue # Move to the next batch

        # --- End of Batch Processing Loop for Config ---

        # --- Final Metrics Calculation for Config ---
        print(f"\nCalculating final metrics for config: {config}")
        acc1 = np.mean(error_matches) * 100 if error_matches else 0
        acc2 = np.mean(correct_matches) * 100 if correct_matches else 0
        f1 = (2 * acc1 * acc2 / (acc1 + acc2)) if (acc1 + acc2) > 0 else 0

        metrics = {
            "config": config,
            "model_name": args.model_name,
            "use_kvpress": args.use_kvpress and KVPRESS_AVAILABLE,
            "press_type": args.press_type if args.use_kvpress and KVPRESS_AVAILABLE else "N/A",
            "compression_ratio": args.compression_ratio if args.use_kvpress and KVPRESS_AVAILABLE else "N/A",
            "attn_implementation": args.attn_implementation,
            "processing_batch_size": args.processing_batch_size,
            "use_voting": args.use_voting,
            "voting_n": args.voting_n if args.use_voting else "N/A",
            "error_accuracy": f"{acc1:.2f}",
            "correct_accuracy": f"{acc2:.2f}",
            "f1_score": f"{f1:.2f}",
            "error_examples_count": len(error_matches),
            "correct_examples_count": len(correct_matches),
            "total_examples_processed": len(error_matches) + len(correct_matches),
            "total_examples_in_dataset": total_examples
        }

        # Save metrics
        metrics_file_name = f'{config}_metrics.txt'
        metrics_file_path = os.path.join(output_dir, metrics_file_name)
        try:
            with open(metrics_file_path, 'w') as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")
            print(f"Metrics saved to: {metrics_file_path}")
        except IOError as e:
            print(f"ERROR: Could not write metrics file to {metrics_file_path}: {e}")

        # Print final results for the config
        print(f'\nFinal results for {config}:')
        print(f'  Error accuracy (Acc1): {acc1:.2f}% ({len(error_matches)} examples)')
        print(f'  Correct accuracy (Acc2): {acc2:.2f}% ({len(correct_matches)} examples)')
        print(f'  F1 score: {f1:.2f}')
        print(f'  Total examples processed: {len(error_matches) + len(correct_matches)} / {total_examples}')
        print(f"=========================================")

    print("\n--- Evaluation script finished ---")


if __name__ == '__main__':
    print("Setting up multiprocessing context...")
    # Set multiprocessing start method crucial for CUDA with multiprocessing
    try:
        current_context = multiprocessing.get_context()
        # Set only if not already 'spawn' to avoid issues in some environments
        if current_context.get_start_method(allow_none=True) != 'spawn':
             multiprocessing.set_start_method('spawn', force=True)
             print("Set multiprocessing start method to 'spawn'.")
        else:
             print("Multiprocessing start method already 'spawn' or default.")
    except RuntimeError as e:
        # Context might already be started by another library import
        print(f"Info: Could not set multiprocessing start method (may already be set): {e}. Proceeding...")
        pass
    except Exception as e:
         print(f"Warning: Unexpected error setting multiprocessing start method: {e}")

    print("Starting main evaluation function...")
    main()
    print("Main function finished.")