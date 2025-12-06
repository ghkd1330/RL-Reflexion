#!/usr/bin/env python3
"""
Real Qwen2-VL Client (NO MOCK)

Loads the REAL Qwen2-VL-7B-Instruct model in 4-bit quantization.
No fallback to mock - if loading fails, raises an error.
"""

import os
import torch
from typing import List
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class Qwen2VLClientReal:
    """
    REAL Qwen2-VL-7B client with 4-bit quantization.
    
    No mock fallback - requires actual model loading.
    """
    
    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct", load_in_4bit=True, device="cuda"):
        """
        Initialize REAL Qwen2-VL client.
        
        Args:
            model_name: Hugging Face model name
            load_in_4bit: Use 4-bit quantization (required for 12GB VRAM)
            device: Device to use
            
        Raises:
            RuntimeError: If model fails to load
        """
        print("="*70)
        print("Loading REAL Qwen2-VL-7B-Instruct (NO MOCK)")
        print("="*70)
        
        self.model_name = model_name
        self.device = device
        
        # Check for HF token (Qwen2-VL may require authentication)
        hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
        if not hf_token:
            print("⚠ Warning: No HF_TOKEN found in environment")
            print("   If model download fails, set: export HF_TOKEN=your_token")
        
        try:
            print(f"\n[1/3] Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=hf_token
            )
            print(f"✓ Processor loaded")
            
            print(f"\n[2/3] Loading model in 4-bit...")
            print(f"  Model: {model_name}")
            print(f"  Quantization: 4-bit (bitsandbytes)")
            print(f"  This may take a few minutes on first run...")
            
            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    token=hf_token
                )
            else:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    device_map="auto",
                    trust_remote_code=True,
                    token=hf_token
                )
            
            print(f"✓ Model loaded successfully")
            
            print(f"\n[3/3] Model ready for inference")
            print(f"✓ Qwen2-VL-7B-Instruct loaded in 4-bit (REAL VLM, no mock)")
            print("="*70)
            
        except Exception as e:
            print(f"\n✗ CRITICAL ERROR: Failed to load Qwen2-VL model")
            print(f"  Error: {e}")
            print(f"\n  Possible fixes:")
            print(f"  1. Set HF_TOKEN: export HF_TOKEN=your_huggingface_token")
            print(f"  2. Ensure GPU has enough memory (need ~6-8GB for 4-bit)")
            print(f"  3. Check internet connection for model download")
            print("="*70)
            raise RuntimeError(f"Failed to load REAL Qwen2-VL model: {e}") from e
    
    def generate(self, images: List[str], prompt: str, max_new_tokens=512) -> str:
        """
        Generate text from images and prompt using REAL VLM.
        
        Args:
            images: List of image file paths (PNG/JPEG)
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            generated_text: Model output (REAL, not mock)
        """
        try:
            # Prepare messages in Qwen2-VL format
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img_path} for img_path in images],
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Prepare inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = inputs.to(self.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False  # Greedy decoding for consistency
                )
            
            # Decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return output_text
            
        except Exception as e:
            print(f"✗ Error during generation: {e}")
            raise RuntimeError(f"VLM generation failed: {e}") from e


# Test
if __name__ == '__main__':
    print("Testing Qwen2VLClientReal...")
    
    try:
        # Load client
        client = Qwen2VLClientReal()
        print("\n✓ Client loaded successfully")
        print("  This is a REAL VLM, not a mock!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
