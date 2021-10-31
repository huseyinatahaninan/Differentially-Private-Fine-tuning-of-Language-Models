## Instructions for fine-tuning the full model with DPSGD.

First, run the following command.
```
pip install --editable . --user
```

**Important:** You need to run the above command **every time** when you switch between different methods, otherwise the code from other folders will be executed.


Here is an example command to fine-tune the full model with DPSGD.
```
python run_exp.py --gpu_id 0 --task SST-2 --clip 0.1 --eps 8 --delta 1e-5 --accountant moments  --batch_size 2000 --lr 1e-3 --epoch 10 --sess dpsgd_debug --to_console
```

The `--eps` and `--delta` flags specify the privacy parameters. 

The `--clip` flag specifies the clipping threshold of pre-example gradients. 

See `run_exp.py` for the introduction of all flags.

