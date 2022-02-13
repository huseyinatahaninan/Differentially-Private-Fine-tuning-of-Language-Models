## Instructions for DP-LoRA

First, run the following command.
```
pip install --editable . --user
```

**Important:** You need to run the above command **every time** when you switch between different methods, otherwise the code from other folders will be executed.


Here is an example command to fine-tune the model with DP-LoRA.
```
python run_exp.py --gpu_id 0 --task SST-2 --k 16 --eps 8 --delta 1e-5 --clip 10. --accountant moments --batch_size 2000 --lr 1e-3 --epoch 50  --sess lora_debug  --to_console
```

The `--k` flag specifies the bottleneck dimension. 

The `--eps` and `--delta` flags specify the privacy parameters. 

The `--clip` flag specifies the clipping threshold of pre-example gradients. 

See `run_exp.py` for the introduction of all flags.

The following command fine-tunes the RoBERTa.Large model with full-precision on the SST-2 dataset.
```
python run_exp.py --gpu_id 1 --task SST-2 --k 16 --eps 8 --delta 1e-5 --clip 2. --accountant moments --batch_size 2000 --lr 1e-3 --epoch 50  --sess lora_bertx_debug --arch roberta.large  --to_console --fp32
```
