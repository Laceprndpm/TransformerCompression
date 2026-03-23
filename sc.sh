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
        --hn-lam 1 \
        --hn-p 0.7 \
        --hn-steps 10 \
        --hn-use-bf16 \
        --hn-block-size 1 \
        --dtype fp32 \
        --hn-out-dir models/step2/exp \

python pca.py \
        --model meta-llama/Llama-2-7b-hf \
        --save-dir models/step1/llama \
        --sparsity 0 \
        --device cuda:0 \
        --eval-baseline \
        --no-wandb \
        # --cal-batch-size 1 \
        # --ppl-eval-batch-size 1 \

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 hn_in.py \
        --model meta-llama/Llama-2-7b-hf \
        --sliced-model-path models/step1/llama \
        --device cuda \
        --hn-use-fsdp \
        --no-wandb \
        --train-hn \
        --hn-model-kind llama \
        --hn-lam 4 \
        --hn-p 0.7 \
        --hn-steps 2000 \
        --hn-use-bf16 \
        --hn-block-size 512 \
        --dtype fp32 \
        --hn-out-dir models/step2/llama \
        2>&1 out.log
