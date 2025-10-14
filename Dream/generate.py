import torch
from transformers import AutoModel, AutoTokenizer
import types

model_path = "Zigeng/dParallel_Dream_7B_Instruct"
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = model.to("cuda").eval()

from model.generation_utils_semiar import DreamGenerationMixin
model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
model._sample = types.MethodType(DreamGenerationMixin._sample, model)


messages = [
    {"role": "user", "content": "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep? Let's think step by step."}
]

inputs =  tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
)
input_ids = inputs.input_ids.to(device="cuda")
attention_mask = inputs.attention_mask.to(device="cuda")

output, nfe = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256,
        output_history=False,
        return_dict_in_generate=True,
        steps=256,
        temperature=0.,
        top_p=None,
        alg="entropy_threshold",
        alg_temp=0.1,
        top_k=None,
        block_length=32,
        threshold=0.5,
    )

generations = [
    tokenizer.decode(g[0:].tolist())
    for p, g in zip(input_ids, output.sequences)
]

print(generations[0].split(tokenizer.eos_token)[0])
print("NFE:", nfe)
