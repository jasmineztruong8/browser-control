**GPU:** NVIDIA L4 (22.0 GB)  
**Model:** Qwen/Qwen2.5-3B-Instruct, 4-bit=True, bfloat16, LoRA r=16

| Measurement | Value |
| --- | --- |
| Peak GPU memory (any stage) | 5.74 GB |
| Model load | 3.96 s |
| Prompt length (mean / max) | 328 / 378 tokens |
| SFT step (batch 4) | 1.19 s, peak 5.74 GB |
| Rollout step (generation + browser) | 0.73 + 0.16 s |
| RL update per sequence | 0.55 s, peak 3.51 GB |
| Adapter save / reload | 28.2 MB, weights match: True, same top token: True |
| **RL interactions per GPU hour** | **2500** |
| 1,000 interactions | 0.4 h per condition, 1.2 h for B+C+D |
| 2,500 interactions | 1.0 h per condition, 3.0 h for B+C+D |
| 5,000 interactions | 2.0 h per condition, 6.0 h for B+C+D |
| 10,000 interactions | 4.0 h per condition, 12.0 h for B+C+D |
