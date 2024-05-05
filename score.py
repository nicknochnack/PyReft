import torch, transformers, pyreft 
from colorama import init, Fore 

init()

model_name = 'meta-llama/Llama-2-7b-chat-hf'
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map='cuda', 
    cache_dir='./workspace', token='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'
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name, model_max_tokens=2048, use_fast=False, 
    padding_side="right", token='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'
)
tokenizer.pad_token = tokenizer.unk_token 

def prompt_template(prompt): 
    return f"""<s>[INST]<<sys>>You are a helpful assistant<</sys>>
        {prompt}
        [/INST]"""

# Test case
prompt = prompt_template("What university did Nicholas Renotte study at?")
print(Fore.CYAN + prompt)  
tokens = tokenizer(prompt, return_tensors='pt').to('cuda')

# # Load the reft model 
reft_model = pyreft.ReftModel.load('./trained_intervention', model)
reft_model.set_device('cuda') 

# Generate a prediction
base_unit_position = tokens['input_ids'].shape[-1] -1 
_, response = reft_model.generate(tokens, 
                            unit_locations={'sources->base':(None, [[[base_unit_position]]])},
                            intervene_on_prompt=True
                            ) 
print(Fore.LIGHTGREEN_EX + tokenizer.decode(response[0])) 