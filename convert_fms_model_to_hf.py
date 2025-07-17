import subprocess
from fms_to_hf_mamba_transformers import convert_mamba_ssm_checkpoint_file_to_huggingface_model_file
from fms_to_hf_mamba import main as fms_to_hf

import glob
import shutil
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir', type=str, required=True, help="Model source directory, (example: /gpfs/davis/bamba_tune/zloss-500k-step-128k/checkpoints/step_6000_ckp)")
    parser.add_argument('--model-name', type=str, required=True, help="Model name, (example: zloss-500k-step-128k)")
    parser.add_argument('--model-dir', type=str, required=True, help="Saving directory")
    parser.add_argument('--model-variant', type=str, required=True, help="Mamba model type, (example: mamba_9.8b)")
    parser.add_argument('--upi-path', type=str, default=None, help="Path to an UPI scaling mask, if applicable")

    args = parser.parse_args()

    DEST_DIR = args.model_dir + '/' + args.model_name
    TOKENIZER_DIR = '/datasets/tokenizers/llama3'

    # --src-dir=/gpfs/hshen/bamba_upi_tune/bamba_upi_32k_layer/pth/step_6000/consolidated.00.pth
    fms_to_hf(args.model_variant, args.src_dir, DEST_DIR, TOKENIZER_DIR, upi_path=args.upi_path)

    convert_mamba_ssm_checkpoint_file_to_huggingface_model_file(
        DEST_DIR , 'fp32', DEST_DIR + '/hf', save_model='sharded'
    )

    for file in glob.glob(DEST_DIR + '/*token*'):
        shutil.copy2(file, DEST_DIR + '/hf/')
