# GPT-2 DP Fine-Tuning

The code in this directory is based on the [original LoRA code for fine-tuning
GPT-2](https://github.com/microsoft/LoRA/tree/main/examples/NLG), however, we think that 
all fine-tuning methods discussed in our paper should achieve comparable accuracy. 

For differentially-private fine-tuning of GPT-2 model series, we use the [Opacus
library](https://github.com/pytorch/opacus). Notice that we heavily rely on distributed
training capabilities that, as of now, are only available in the latest release
of Opacus on Github -- in particular, we use the `DPDDP` wrapper which
increments the code with manual calls to `torch.distributed.all_reduce`.

As the LoRA layers are not natively supported by Opacus, we have specifically
coded the per-sample gradient computation for these layers, and registered them
in Opacus by means of the `register_grad_sampler` decorator. The setting 
is currently specific to lora_dropout=0.0 and enable_lora=[True, False, True].

Below we provide the setting we use in our experiments. For more information on 
how to run the code, please check the original LoRA repository linked above.

## Replicating Our Result on E2E

1. Fine-tune GPT-2 (Medium) with DP
```
python -m torch.distributed.launch --nproc_per_node=16 src/gpt2_ft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --train_batch_size 8 \
    --grad_acc 1 \
    --noise_multiplier 0.6 \
    --max_grad_norm 1.0 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card gpt2.md \
    --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
    --platform local \
    --clip 0.0 \
    --lr 0.0004 \
    --weight_decay 0.01 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler constant \
    --warmup_step 0 \
    --max_epoch 20 \
    --save_interval 1000 \
    --lora_dim 4 \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    --label_smooth 0.1 \
    --work_dir ./trained_models/GPT2_M/e2e \
    --random_seed 110
```

2. Generate outputs from the trained model using beam search:
```
python -m torch.distributed.launch --nproc_per_node=16 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M/e2e \
    --platform local \
    --lora_dim 4 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M/e2e \
    --output_file predict.jsonl \
    --hyperparam
```

3. Decode outputs from step (2)
```
python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/e2e/predict.jsonl \
    --input_file ./data/e2e/test_formatted.jsonl \
    --output_ref_file e2e_ref.txt \
    --output_pred_file e2e_pred.txt
```

4. Run evaluation on E2E test set

```
python eval/e2e/measure_scores.py e2e_ref.txt e2e_pred.txt -p
```
