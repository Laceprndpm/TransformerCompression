python pca.py \
        --model microsoft/phi-2 \
        --save-dir sm/step1/exp \
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
        --sliced-model-path sm/step1/exp \
        --sparsity 0 \
        --device cuda:0 \
        --eval-baseline \
        --no-wandb \
        --cal-batch-size 1 \
        --ppl-eval-batch-size 1 \
        --ppl-eval-seqlen 512 \
        --train-hn \
        --hn-out-dir sm/step2/exp 