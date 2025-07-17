import fire
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from torch.distributed._shard.checkpoint import FileSystemReader, load_state_dict
import torch
from fms_fsdp.utils.config_utils import get_model_config


def main(model_variant, load_path, save_path, tokenizer_name_or_path, upi_path=None):
    print("Initializing model...")
    config_data = get_model_config(model_variant)
    mamba_config = MambaConfig(**config_data)
    model = MambaLMHeadModel(mamba_config)

    print(f"Reading state dict from {load_path}")
    state_dict = {"model_state": model.state_dict()}
    load_state_dict(
        state_dict=state_dict, storage_reader=FileSystemReader(load_path), no_dist=True
    )

    # Overwrite UPI mask within the model 
    if upi_path: 
        if "upi" not in model_variant:
            model_variant_upi = model_variant+"_upi"
        else:
            model_variant_upi = model_variant
        config_data = get_model_config(model_variant_upi)
        mamba_config = MambaConfig(**config_data)
        model = MambaLMHeadModel(mamba_config)

        print("Overwritting UPI masks...")
        upi_mask_dict = torch.load(upi_path)
        for i in range(config_data['n_layer']): # Iterate through all layers
            if i not in config_data['attn_layer_idx']: # Exclude transformer layers
                if "upi" not in model_variant:
                    state_dict['model_state'][f'backbone.layers.{i}.mixer.upi_mask'] = upi_mask_dict[i].to(torch.bfloat16)
                else:
                    state_dict['model_state'][f'backbone.layers.{i}.mixer.upi_mask'] *= upi_mask_dict[i].to(torch.bfloat16)

    print("Loading state dict into the model...")
    model.load_state_dict(state_dict["model_state"])

    print("Saving model to HF-compatible format...")
    model.save_pretrained(save_path)

    print("Copying tokenizer...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    tokenizer.save_pretrained(save_path)

    print(f"Model saving at {save_path}")


if __name__ == "__main__":
    fire.Fire(main)
