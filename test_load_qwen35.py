from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer



TRAIN_ARGS = {
    "num_train_epochs": 1,
    "overwrite_cache": True,
    "report_to": "tensorboard",
    "logging_dir": "./logs",
    "deepspeed": "./examples/deepspeed/ds_z3_config.json",
    "gradient_checkpointing": "yes",
    "trust_remote_code": True,
    "disable_shuffling": True,
    "stage": "pt",
    "model_name_or_path": "/home/yyx/models/Qwen3.5-35B-A3B",
    "do_train": True,
    "flash_attn": "fa2",
    "dataset": "alpaca_en_demo",
    "preprocessing_num_workers": 16,
    "cutoff_len": 1024,
    "template": "qwen3",
    "finetuning_type": "full",
    "output_dir": "./output/qwen3-30b-alpaca-en-demo",
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "eval_accumulation_steps": 1,
    "lr_scheduler_type": "cosine",
    "logging_steps": 1,
    "save_strategy": "steps",
    "eval_steps": 20000000,
    "save_steps": 20000000,
    "warmup_ratio": 0.0,
    "learning_rate": 1.25e-5,
    "save_total_limit": 1,
    "plot_loss": True,
    "seed": 42,
    "bf16": True,
    "ddp_timeout": 180000000,
    "max_steps": 200,
}


def test_base():
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(TRAIN_ARGS)
    model_args, _, _, finetuning_args, _ = get_train_args(TRAIN_ARGS)
    tokenizer = load_tokenizer(model_args)["tokenizer"]
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    print(model)
    
if __name__ == "__main__":
    test_base()