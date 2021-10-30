## Instructions for DP-Adapter

First, run the following command.
```
pip install --editable . --user
```

**Important:** You need to run the above command **every time** when you switch between different methods, otherwise the code from other folders will be executed.


Here is an example command to fine-tune the model with DP-Adapter.
```
python run_exp.py --gpu_id 0 --task SST-2 --k 16 --eps 8 --delta 1e-5 --clip 10. --accountant moments --batch_size 2000 --lr 1e-3 --epoch 50  --sess adapter_debug  --to_console
```

The `--k` flag specifies the bottleneck dimension. 

The `--eps` and `--delta` flags specify the privacy parameters. 

The `--clip` flag specifies the clipping threshold of pre-example gradients. 

See `run_exp.py` for the introduction of all flags.
