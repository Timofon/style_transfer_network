# Style transfer network

### Done by Tim Senin in HSE, 2024 as course project

This is my modification of style transfer network, original model is taken from [DreamStyler paper](https://arxiv.org/abs/2309.06933)

## Training
```
accelerate launch dreamstyler/train.py \
  --num_stages 6 \
  --train_image_path "./images/03.png" \
  --context_prompt "A painting of pencil, pears and apples on a cloth, in the style of {}" \
  --placeholder_token "<sks03>" \
  --output_dir "./outputs/sks03" \
  --learnable_property style \
  --initializer_token painting \
  --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
  --resolution 512 \
  --train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 500 \
  --save_steps 100 \
  --learning_rate 0.002 \
  --lr_scheduler constant \
  --lr_warmup_steps 0
```

## Inference

```
python dreamstyler/inference_style_transfer.py \
  --sd_path "runwayml/stable-diffusion-v1-5" \
  --embedding_path ./outputs/sks03/embedding/final.bin \
  --content_image_path ./images/content.png \
  --saveroot ./outputs/sample03 \
  --placeholder_token "<sks03>" \
  --threshold 0.3
```

Threshold parameter is responsible for the intensity of the transferred style
