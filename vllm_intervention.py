# vllm_intervention.py
import os
os.environ["VLLM_USE_V1"] = "0"
import random, sys, re
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from typing import List, Dict

import torch.multiprocessing as mp
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# ───────────────────────────────────────── tokenizer & phrase
MODEL_NAME = "Qwen/Qwen3-8B" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

INJECTION_TEMPLATE = "\n{options_text}\n</think>"

# ───────────────────────────────────────── logits-processor
class ForcePhrase:
    def __init__(self, tokenizer, injection_text: str, hesitation_threshold: int = 6):
        
        self.tokenizer = tokenizer
        self.ids = tokenizer.encode(injection_text, add_special_tokens=False)
        self.hesitation_threshold = hesitation_threshold  
        
        self.idx = None
        self.wait_count = 0
        self.last_sequence_length = 0
        

        self.wait_tokens = set()
        

        wait_core_variants = [
            "wait",    
            "Wait",    
            " wait",   
            " Wait"   
        ]
        
        for variant in wait_core_variants:
            tokens = tokenizer.encode(variant, add_special_tokens=False)
            if len(tokens) == 1: 
                self.wait_tokens.update(tokens)
        
        # Pre-encode </think> tokens for fast detection
        self.think_close_tokens = set(tokenizer.encode("</think>", add_special_tokens=False))
        think_tokens = tokenizer.encode("</think>", add_special_tokens=False)
        self.think_close_token_id = think_tokens[0] if think_tokens else None
    
    def activate(self):
        """Activate injection"""
        if self.idx is None:
            self.idx = 0

    def is_done(self):
        """Check if injection is completed"""
        return self.idx is not None and self.idx >= len(self.ids)

    def __call__(self, input_ids, logits):
        # If in the injection process, force generation of injection tokens
        if self.idx is not None and self.idx < len(self.ids):
            tgt = self.ids[self.idx]
            logits.fill_(float("-inf"))   # Set all to negative infinity
            
            # Handle different dimensions of logits
            if logits.dim() == 1:
                # 1D case: [vocab_size]
                logits[tgt] = 0.0
            else:
                # 2D case: [batch, vocab_size]
                logits[:, tgt] = 0.0
            
            self.idx += 1
            return logits
        
        # If injection is already completed, no more intervention
        if self.idx is not None and self.idx >= len(self.ids):
            return logits
        
        # Check if injection needs to be triggered (only when not activated)
        if self.idx is None:
            # Trigger condition 1: detect if next token is </think>
            if self.think_close_token_id is not None:
                # Get the token with highest probability
                if logits.dim() == 1:
                    predicted_token = torch.argmax(logits).item()
                else:
                    predicted_token = torch.argmax(logits[0]).item()
                
                # If the next token to be generated is </think>, immediately activate injection
                if predicted_token == self.think_close_token_id:
                    self.activate()
                    return self.__call__(input_ids, logits)
            
            # Trigger condition 2: wait word prediction detection
            try:
                # First check if current wait count has reached n-1
                if self.wait_count >= self.hesitation_threshold - 1:
                    # Get the token with highest probability
                    if logits.dim() == 1:
                        predicted_token = torch.argmax(logits).item()
                    else:
                        predicted_token = torch.argmax(logits[0]).item()
                    
                    # If the next token to be generated is wait or Wait, immediately activate injection
                    if predicted_token in self.wait_tokens:
                        self.activate()
                        return self.__call__(input_ids, logits)
                
                # Handle different types of input_ids to update wait count
                if isinstance(input_ids, list):

                    if len(input_ids) > 0 and isinstance(input_ids[0], list):
                        # Batch processing case
                        current_tokens = input_ids[0]
                    else:
                        current_tokens = input_ids
                else:
                    # Other versions: might be tensor
                    if hasattr(input_ids, 'dim') and input_ids.dim() > 1:
                        current_tokens = input_ids[0]
                        if hasattr(current_tokens, 'cpu'):
                            current_tokens = current_tokens.cpu().tolist()
                    else:
                        current_tokens = input_ids
                        if hasattr(current_tokens, 'tolist'):
                            current_tokens = current_tokens.tolist()
                
                # Ensure current_tokens is a list
                if not isinstance(current_tokens, list):
                    current_tokens = [current_tokens] if isinstance(current_tokens, int) else list(current_tokens)
                
                current_length = len(current_tokens)
                
                # Detect wait words in newly added tokens
                if current_length > self.last_sequence_length:
                    # Get newly added tokens
                    new_tokens = current_tokens[self.last_sequence_length:]
                    
                    # Count new wait/Wait occurrences
                    wait_found = 0
                    for token in new_tokens:
                        if token in self.wait_tokens:
                            wait_found += 1
                            # Output debug info, showing detected token and corresponding text

                    
                    # Update wait count
                    if wait_found > 0:
                        self.wait_count += wait_found
                    
                    # Update sequence length record
                    self.last_sequence_length = current_length
                    
            except Exception as e:
                print(f"[DEBUG] wait detection error: {e}")
                pass
        
        return logits


# ───────────────────────────────────────── Main inference class
class VLLMInterventionGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen3-8B"):
        mp.set_start_method("spawn", force=True)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Initialize vLLM engine
        self.llm = LLM(
            model=model_name, 
            trust_remote_code=True,
            max_num_seqs=1,  # Single sequence generation
            gpu_memory_utilization=0.9,
            max_model_len=7000,

            # enforce_eager=True,
        )

    def stream_response(
        self,
        messages: List[Dict[str, str]],
        options_text: str,
        hesitation_threshold: int = 5,
        max_new_tokens: int = 2048,
        temperature: float = 0,
        top_p: float = 1.0,
    ) -> str:
        
        # Build prompt
        chat_template_kwargs = {"enable_thinking": True}
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            chat_template_kwargs=chat_template_kwargs
        )
        
        # Create new ForcePhrase instance for each request to ensure thread safety
        injection_text = INJECTION_TEMPLATE.format(options_text=options_text)
        force_phrase = ForcePhrase(self.tokenizer, injection_text, hesitation_threshold)
        
        # Initialize sampling parameters
        sampler = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            logits_processors=[force_phrase]   # New instance each time
        )
        
        try:
            # vLLM's generate returns a list of RequestOutput
            outputs = self.llm.generate([prompt], sampler, use_tqdm=False)
            
            for request_output in outputs:
                for completion_output in request_output.outputs:
                    # Get complete generated text
                    generated_text = completion_output.text
                    # print(generated_text)
                    return generated_text
                        
        except Exception as e:
            print(f"\nError in generation: {e}")
            return ""
        
        return ""


# ───────────────────────────────────────── Run example
def main():
    # Instantiate generator
    generator = VLLMInterventionGenerator(model_name="Qwen/Qwen3-8B")
    
    # Prepare test input
    messages = [
        {"role": "system", "content": "Read the following story and answer the question by selecting the correct choice. Output your final answer strictly in the format: Answer: [Letter]"},
        {
            "role": "user",
            "content": "[Story]: The following story happens in chronological order. You will be given a multiple-choice question and a note at the end. Directly output the answer without explanation.\n1 Ethan, Aria, Emily, Owen and William entered the crawlspace.\n2 The pumpkin is in the green_pantry.\n3 Ethan moved the pumpkin to the green_box.\n4 Ethan exited the crawlspace.\n5 Aria moved the pumpkin to the red_bottle.\n6 Aria exited the crawlspace.\n7 Ethan saw a dog.\n8 Emily made no movements and stayed in the crawlspace for 1 minute.\n9 Emily exited the crawlspace.\n10 Owen moved the pumpkin to the green_suitcase.\n11 Owen exited the crawlspace.\n12 William moved the pumpkin to the green_pantry.\n13 William exited the crawlspace.\n14 Ethan, Aria, Emily, Owen and William entered the waiting_room.\n15 Emily, Owen and Ethan entered the crawlspace.\n16 The peach is in the green_envelope.\n17 Emily moved the peach to the green_pantry.\n18 Owen saw a mouse.\n19 Emily exited the crawlspace.\n20 Owen made no movements and stayed in the crawlspace for 1 minute.\n21 Owen exited the crawlspace.\n22 Ethan made no movements and stayed in the crawlspace for 1 minute.\n23 Ethan exited the crawlspace.\n24 Emily, Owen and Ethan entered the waiting_room.\n25 Owen saw a monkey.\n26 Emily publicly claimed that peach is in the green_pantry now.\n27 Ethan privately told Aria that the peach is in the green_pantry now.\n\n [Question]:Where does William think Emily thinks Owen thinks the pumpkin is?"
        }
    ]
    # messages = [{'role': 'user', "content": "who are you"}]
    
    options_text = (
        "Now, I need to choose an answer from fast intuition: A. green_envelope, B. green_pantry, C. green_suitcase, D. red_bottle, E. green_box, F. green_drawer, G. red_pantry, H. blue_box, I. green_cupboard, J. blue_basket, K. green_bottle, L. blue_cupboard, M. red_box, N. red_treasure_chest, O. red_basket"
    )
    
    print("--- Starting vLLM intervention generation ---\n")
    result = generator.stream_response(
        messages=messages,
        options_text=options_text,
        hesitation_threshold=5,
        temperature=0,
        max_new_tokens=2048,
        

    )
    print(f"\n\n--- Generation completed ---")
    print(f"Total length: {len(result)} characters")


if __name__ == "__main__":
    main()
