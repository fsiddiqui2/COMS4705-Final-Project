import json
import os
import re
import tinker 
from tinker import types as ttypes
from transformers import AutoTokenizer
from datasets import load_dataset

# Global Model ID
MODEL_ID = "Qwen/Qwen3-235B-A22B-Instruct-2507"
BATCH_SIZE = 8  # Number of concurrent requests

# ==========================================
# 1. Tinker Client & Tokenizer Setup
# ==========================================

def setup_tinker():
    """
    Initializes the connection to the Qwen3 model via Tinker and loads the tokenizer.
    """
    try:
        print(f"🔌 Connecting to Tinker Service for {MODEL_ID}...")
        service = tinker.ServiceClient()
        client = service.create_sampling_client(base_model=MODEL_ID)
        
        print(f"📖 Loading Tokenizer for {MODEL_ID}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        return client, tokenizer
    except Exception as e:
        print(f"⚠️ Failed to initialize Tinker client: {e}")
        return None, None

# ==========================================
# 2. Prompt Engineering & Formatting
# ==========================================

def construct_teacher_prompt(question, answer, context_paragraphs):
    """
    Updated for OPEN BOOK setting: Encourages explicit citations.
    """
    # Format the paragraphs clearly with identifiers
    context_str = ""
    for idx, doc in enumerate(context_paragraphs):
        context_str += f"Document [{idx+1}] (Title: {doc['title']}):\n{doc['content']}\n\n"

    system_instruction = "You are an expert Professor of Knowledge Graph Reasoning and Information Retrieval."
    
    user_instruction = f"""
I will give you a multi-hop question, the correct answer, and a set of documents (paragraphs).
Most of these documents are **irrelevant distractors**. Only two (or sometimes three) contain the necessary information.

Your task is to write a **Chain-of-Thought (CoT) reasoning trace** that explicitly selects the relevant documents and uses them to derive the answer.

### CRITICAL RULES FOR REASONING STYLE (OPEN BOOK):
1. **Cite Your Sources:** You MUST explicitly refer to the documents by their ID when extracting facts (e.g., "According to Document [2]...", "Document [5] states that...").
2. **Structure:**
   - **Step 1: Information Selection.** Explicitly identify which documents are relevant to the entities in the question.
   - **Step 2: The Bridge.** Combine information from the selected documents.
   - **Step 3: Conclusion.** Derive the final answer.
3. **Brevity:** Keep the reasoning concise (3-5 sentences max).

### INPUT DATA:
Question: {question}
Correct Answer: {answer}

### AVAILABLE DOCUMENTS:
{context_str}

### OUTPUT FORMAT:
Return only the reasoning trace, ending with "Therefore, the answer is {answer}."
"""
    return system_instruction, user_instruction

def format_for_qwen(messages):
    formatted_prompt = ""
    for msg in messages:
        formatted_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    formatted_prompt += "<|im_start|>assistant\n"
    return formatted_prompt

# ==========================================
# 3. Validation Logic
# ==========================================

def validate_trace(trace, answer):
    # === REMOVED FORBIDDEN PHRASES CHECK ===
    # Since this is open book, phrases like "According to the passage" are now allowed and encouraged.

    # === ROBUST ANSWER CHECK ===
    # Verify that the trace ENDS with the ground truth answer.
    clean_truth = answer.strip().lower()
    if clean_truth.endswith('.'):
        clean_truth = clean_truth[:-1]

    # Pattern: "the answer is <escaped_answer>" followed by optional punctuation/whitespace at end of string
    pattern = r"the answer is\s+" + re.escape(clean_truth) + r"[.!?\s]*$"
    
    # We check the lowercased trace for the pattern
    match = re.search(pattern, trace.lower())
    
    if not match:
        snippet = trace[-50:].replace("\n", " ") if len(trace) > 50 else trace
        return False, f"Rule Violation: Trace did not end with 'Therefore, the answer is {answer}.'. End of trace was: '...{snippet}'"
    
    return True, "Valid"

# ==========================================
# 4. Batch Generation Logic
# ==========================================

def process_batch(client, tokenizer, batch_data, max_retries=3):
    """
    Processes a list of examples concurrently.
    """
    active_items = {} 
    results = [None] * len(batch_data)
    
    for i, item in enumerate(batch_data):
        sys_msg, user_msg = construct_teacher_prompt(
            item['question'], 
            item['answer'], 
            item['all_paragraphs']
        )
        active_items[i] = {
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg}
            ],
            "attempts": 0,
            "done": False,
            "data": item
        }

    while True:
        pending_indices = [
            idx for idx, state in active_items.items() 
            if not state["done"] and state["attempts"] < max_retries
        ]
        
        if not pending_indices:
            break
            
        print(f"   >> Sending {len(pending_indices)} requests...")
        
        futures_map = {} 
        
        for idx in pending_indices:
            state = active_items[idx]
            state["attempts"] += 1
            
            raw_prompt_str = format_for_qwen(state["messages"])
            
            try:
                input_ids = tokenizer.encode(raw_prompt_str)
                # Ensure we don't exceed context limits with 10 paragraphs
                if len(input_ids) > 30000: 
                    print(f"⚠️ Warning: Input length {len(input_ids)} is high.")

                prompt_input = ttypes.ModelInput.from_ints(input_ids)
                
                params = ttypes.SamplingParams(
                    max_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    stop=["<|im_end|>"] 
                )

                future = client.sample(
                    prompt=prompt_input, 
                    sampling_params=params, 
                    num_samples=1
                )
                futures_map[future] = idx
                
            except Exception as e:
                print(f"❌ Tokenization/Request Error for item {idx}: {e}")
        
        # Blocking wait for results
        for future, idx in futures_map.items():
            state = active_items[idx]
            try:
                result = future.result()
                
                if result.sequences:
                    output_tokens = result.sequences[0].tokens
                    response_text = tokenizer.decode(output_tokens)
                    response_text = response_text.replace("<|im_end|>", "").strip()
                    
                    is_valid, critique = validate_trace(response_text, state['data']['answer'])
                    
                    if is_valid:
                        results[idx] = response_text
                        state["done"] = True
                    else:
                        # Retry with feedback
                        state["messages"].append({"role": "assistant", "content": response_text})
                        state["messages"].append({
                            "role": "user", 
                            "content": f"{critique}\nPlease rewrite the reasoning trace. Ensure you end exactly with: 'Therefore, the answer is {state['data']['answer']}.'"
                        })
                else:
                    print(f"❌ Item {idx}: No sequences returned.")
            except Exception as e:
                print(f"❌ Item {idx} API Error: {e}")

    return results

# ==========================================
# 5. Dataset Processing Helper
# ==========================================

def process_hotpot_sample(raw_sample):
    """
    Extracts ALL 10 paragraphs from the 'context' field for open-book reasoning.
    """
    question = raw_sample['question']
    answer = raw_sample['answer']
    
    all_paragraphs = []
    titles = raw_sample['context']['title']
    sentences_list = raw_sample['context']['sentences']
    
    # Zip them to keep titles and content together
    for title, sentences in zip(titles, sentences_list):
        full_text = " ".join(sentences)
        all_paragraphs.append({
            "title": title,
            "content": full_text
        })
    
    return {
        "question": question,
        "answer": answer,
        "all_paragraphs": all_paragraphs
    }

def batch_iterator(iterable, size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch

# ==========================================
# 6. Main Execution
# ==========================================

if __name__ == "__main__":
    TARGET_COUNT = 10000
    OUTPUT_FILE = f"hotpot_openbook_reasoning_{TARGET_COUNT}.jsonl"
    
    # --- RESUME LOGIC ---
    processed_count = 0
    if os.path.exists(OUTPUT_FILE):
        print(f"🔍 Found existing file: {OUTPUT_FILE}. Counting progress...")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for _ in f:
                processed_count += 1
        print(f"✅ Resuming from index {processed_count}")
    else:
        print("🆕 Starting fresh run.")

    if processed_count >= TARGET_COUNT:
        print("🎉 Target count already reached. Exiting.")
        exit()

    # 1. Setup
    client, tokenizer = setup_tinker()
    
    if client and tokenizer:
        print(f"💾 Loading HotpotQA (distractor, train)... Target: {TARGET_COUNT}")
        dataset = load_dataset("hotpot_qa", "distractor", split="train", streaming=True)
        
        remaining_count = TARGET_COUNT - processed_count
        iterable_data = dataset.skip(processed_count).take(remaining_count)
        
        print(f"🚀 Starting Batch Generation (Batch Size: {BATCH_SIZE})...")

        for batch_idx, raw_batch in enumerate(batch_iterator(iterable_data, BATCH_SIZE)):
            print(f"\nProcessing Batch {batch_idx + 1}...")
            
            # 1. Pre-process
            formatted_batch = []
            for raw_sample in raw_batch:
                formatted_batch.append(process_hotpot_sample(raw_sample))
            
            # 2. Run concurrent generation
            batch_traces = process_batch(client, tokenizer, formatted_batch)
            
            # 3. Save immediately
            current_batch_success = 0
            new_lines = []
            
            for raw_sample, trace in zip(raw_batch, batch_traces):
                if trace:
                    output_obj = raw_sample.copy()
                    output_obj['reasoning_trace'] = trace
                    new_lines.append(json.dumps(output_obj))
                    current_batch_success += 1
            
            if new_lines:
                with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                    for line in new_lines:
                        f.write(line + "\n")
            
            processed_count += len(raw_batch)
            print(f"   Batch finished. Success: {current_batch_success}/{len(raw_batch)}")
            print(f"   📈 Total Progress: {processed_count}/{TARGET_COUNT}")

        print("Done.")