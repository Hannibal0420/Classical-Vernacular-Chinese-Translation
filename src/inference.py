import argparse
import json
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
import torch


parser = argparse.ArgumentParser(description='Run validation.')

# Add arguments
parser.add_argument('--base_model', type=str, default="yentinglin/Taiwan-LLM-7B-v2.0-chat", required=True, help='Path to the saved model')
parser.add_argument('--peft_config', type=str, required=True, help='Path to the PEFT config.')
parser.add_argument('--input_file', type=str, required=True, help='Path to the input file')
parser.add_argument('--sampling', action="store_true", help='Do sampling')
parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')

# Execute the parse_args() method
args = parser.parse_args()


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(arg.base_model)
model = AutoModelForCausalLM.from_pretrained(arg.base_model, quantization_config=bnb_config, device_map={"":0})


model = PeftModel.from_pretrained(model, args.peft_config)

def get_prompt(instruction: str) -> str:
    instruction.replace('\n', '')
    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"

def reformat(df):
    # df.rename(columns={'output': 'labels'}, inplace=True)
    df['instruction'] = df['instruction'].apply(lambda instruction: get_prompt(instruction))
    return df

if args.sampling:
    inference_df = pd.read_json(args.input_file).head(3)
else:
    inference_df = pd.read_json(args.input_file)
inference_df = reformat(inference_df)

id_list = []
output_list = []
device = "cuda:0"
pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff。，、]+')

for i in range(len(inference_df)):
    inputs = tokenizer(inference_df["instruction"][i], max_length=512, padding="max_length", truncation=True, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        generation_config=GenerationConfig(
            do_sample=True,
            max_new_tokens=256,
            num_beams=3
        )
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sliced_output_text = output_text[output_text.rfind("ASSISTANT:") + len("ASSISTANT:"):].strip()
    sliced_output_text = ''.join(pattern.findall(sliced_output_text))
    
    id_list.append(inference_df["id"][i])
    output_list.append(sliced_output_text)

with open(args.output_file, 'w', encoding='utf-8') as f:
    for data_id, output in zip(id_list, output_list):
        data = {"id": data_id, "output": output}
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")
