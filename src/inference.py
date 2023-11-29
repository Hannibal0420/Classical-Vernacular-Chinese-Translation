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

import pandas as pd
import datasets
from accelerate import Accelerator
import re

accelerator = Accelerator()

def get_prompt(instruction: str) -> str:
    instruction.replace('\n', '')
    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"

def reformat(df):
    # df.rename(columns={'output': 'labels'}, inplace=True)
    df['instruction'] = df['instruction'].apply(lambda instruction: get_prompt(instruction))
    return df


inference_df = pd.read_json(args.input_file)
inference_df = reformat(inference_df)

inference_dataset = datasets.Dataset.from_pandas(inference_df)
# print(train_dataset)

text_column = 'instruction'
id_column = "id"
column_names = inference_dataset.column_names
id_list = inference_dataset[id_column]


def preprocess_function(examples):
    inputs = examples[text_column]
    ids = examples[id_column]
    model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)
    # model_inputs["id"] = ids
    return model_inputs

def postprocess_text(predictions):
    pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff。，、]+')

    predictions = [pred[pred.rfind("ASSISTANT:") + len("ASSISTANT:"):].strip() for pred in predictions]
    predictions = [''.join(pattern.findall(pred)) for pred in predictions]
    return predictions

with accelerator.main_process_first():
    inference_dataset = inference_dataset.map(preprocess_function,batched=True,remove_columns=column_names,desc="Running tokenizer on dataset")
    
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, GenerationConfig

label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8 if accelerator.use_fp16 else None,
)
eval_dataloader = DataLoader(inference_dataset, collate_fn=data_collator, batch_size=2)

# Prepare everything with our `accelerator`.
model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
model.eval()

decoded_preds_list = []
preds_ids_list = []

for step, batch in enumerate(eval_dataloader):
    with torch.no_grad():
        generated_tokens = accelerator.unwrap_model(model).generate(
            **batch,
            # batch["input_ids"],
            # attention_mask=batch["attention_mask"],
            generation_config=GenerationConfig(
                do_sample=True,
                max_new_tokens=256,
                num_beams=3
            )
        )
        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )
        generated_tokens = accelerator.gather_for_metrics((generated_tokens))
        generated_tokens = generated_tokens.cpu().numpy()

        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_preds = postprocess_text(decoded_preds)
        
        decoded_preds_list.extend(decoded_preds)


with open(args.output_file, 'w', encoding='utf-8') as f:
    for data_id, output in zip(id_list, decoded_preds_list):
        data = {"id": data_id, "output": output}
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")

accelerator.end_training()
