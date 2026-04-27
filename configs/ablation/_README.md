# Compact HF-U-Net ablation configs

Run the 9-variant subset with:

```bash
python scripts/train_all.py \
  --models "unet,unet_conv_bottleneck,unet_fft_bottleneck,proposal_hf_unet,hf_unet_wo_hartley,hf_unet_wo_fourier_kernel,hf_unet_wo_residual,hf_unet_encoder_stage4,hf_unet_decoder_stage" \
  --config-dir configs/ablation \
  --dataset cvc_clinicdb \
  --data-root data \
  --image-size 352 \
  --seed 42 \
  --device auto \
  --output-root outputs_ablation_cvc_clinicdb
```

For a quick smoke test, reduce image size and epochs:

```bash
python scripts/train_all.py \
  --models "unet,unet_conv_bottleneck,unet_fft_bottleneck,proposal_hf_unet,hf_unet_wo_hartley,hf_unet_wo_fourier_kernel,hf_unet_wo_residual,hf_unet_encoder_stage4,hf_unet_decoder_stage" \
  --config-dir configs/ablation \
  --dataset custom \
  --data-root data_tiny \
  --image-size 64 \
  --batch-size 2 \
  --epochs 1 \
  --num-workers 0 \
  --device cpu \
  --output-root outputs_smoke_ablation
```
