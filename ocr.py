import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def extract_text_with_qwen(image_path: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Could not find '{image_path}'. Check your 'data' folder.")

    print("Loading Qwen2-VL-2B model into memory (this may take a moment)...")
    
    # Load the model. Using bfloat16 reduces memory usage significantly.
    # device_map="auto" automatically places the model on your GPU if available.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto" 
    )
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    print(f"Processing '{image_path}'...")

    # Qwen2-VL requires a specific message format for multimodal inputs
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text", 
                    "text": (
                        "Transcribe all handwritten and printed text in this image verbatim into clean Markdown. "
                        "Use standard Markdown headers (##), bold text (**text**), and bullet points (-). "
                        "Do NOT use LaTeX commands unless there is an actual mathematical equation. "
                        "Preserve the original layout, structure, and dates accurately."
                    )
                },
            ],
        }
    ]

    # Prepare inputs for the model
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    print("Generating transcription...")
    
    # Generate the output text
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    
    # Trim the input tokens from the generated output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Free up memory (useful if calling this repeatedly later)
    del model
    torch.cuda.empty_cache()

    return output_text[0]

if __name__ == "__main__":
    image_file = os.path.join("data", "answer4.jpeg")
    output_file = "qwen_ocr_output.md"
    
    try:
        extracted_text = extract_text_with_qwen(image_file)
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(extracted_text)
            
        print(f"\n[+] Clean Markdown successfully saved to '{output_file}'")
        
    except Exception as e:
        print(f"\n[-] Error: {e}")