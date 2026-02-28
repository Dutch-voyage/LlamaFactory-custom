from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

config = AutoConfig.from_pretrained("/home/yyx/models/Qwen3.5-35B-A3B", trust_remote_code=True)

with open("auto_config" , "w") as f:
    f.write(str(config))
print(config)

model_class = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get("qwen3_5_moe_text", None)

# print(config.sub_configs["text_config"])

model = AutoModelForCausalLM.from_config(config.text_config)

model = AutoModelForCausalLM.from_pretrained("/home/yyx/models/Qwen3.5-35B-A3B", trust_remote_code=True)

print(model.config)
with open("model_config" , "w") as f:
    f.write(str(model.config))

difference_in_config = set(model.config.__dict__.keys()) - set(config.__dict__.keys())
with open("difference_in_config" , "w") as f:
    f.write(str(difference_in_config))

# config = model_class.config_class.from_pretrained("/home/yyx/models/Qwen3.5-35B-A3B", trust_remote_code=True)

# print(config)
# print(model_class.__config)