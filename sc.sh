python pca.py \
        --model microsoft/phi-2 \
        --save-dir models/step1/exp \
        --sparsity 0 \
        --device cuda:0 \
        --eval-baseline \
        --no-wandb \
        --cal-batch-size 1 \
        --ppl-eval-batch-size 1 \

python pca.py \
        --model microsoft/phi-2 \
        --sparsity 0 \
        --device cuda:0 \
        --eval-baseline \
        --no-wandb \
        --cal-batch-size 1 \
        --ppl-eval-batch-size 1 \

torchrun --standalone --nproc_per_node=1 hn_in.py \
        --model microsoft/phi-2 \
        --sliced-model-path models/step1/exp \
        --device cuda:0 \
        --no-wandb \
        --train-hn \
        --hn-lam 6 \
        --hn-p 0.7 \
        --hn-steps 10 \
        --hn-use-bf16 \
        --hn-block-size 1 \
        --dtype fp32 \
        --hn-out-dir models/step2/exp \