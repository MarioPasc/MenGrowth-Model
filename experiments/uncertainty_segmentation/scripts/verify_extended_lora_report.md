# Extended-LoRA verification report

Generated: 2026-04-13 10:39:47  
Device: `cuda:0`  
GPU phase: fp16 autocast, batch=1. Spatial extent is tried from the
largest candidate down; first one that fits wins.

| Label | r | stages | types | targets | wired | params | spatial | fwd shape | grads | peak MB | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stages34_qkv_r8 | 8 | 3,4 | qkv | 4/4 | 4 | 36,864 | 128^3 | (1, 3, 128, 128, 128) | ok | 3864 | **PASS** |
| stages123_qkv_r8 | 8 | 1,2,3 | qkv | 6/6 | 6 | 21,504 | 96^3 | (1, 3, 96, 96, 96) | ok | 3882 | **PASS** |
| stages123_extended_r2 | 2 | 1,2,3 | qkv,proj,fc1,fc2 | 24/24 | 24 | 21,504 | 96^3 | (1, 3, 96, 96, 96) | ok | 4136 | **PASS** |
| stages123_extended_r8 | 8 | 1,2,3 | qkv,proj,fc1,fc2 | 24/24 | 24 | 86,016 | 96^3 | (1, 3, 96, 96, 96) | ok | 4146 | **PASS** |
| stages123_extended_r32 | 32 | 1,2,3 | qkv,proj,fc1,fc2 | 24/24 | 24 | 344,064 | 96^3 | (1, 3, 96, 96, 96) | ok | 4198 | **PASS** |
| stages1234_extended_r2 | 2 | 1,2,3,4 | qkv,proj,fc1,fc2 | 32/32 | 32 | 46,080 | 96^3 | (1, 3, 96, 96, 96) | ok | 4139 | **PASS** |
| stages1234_extended_r8 | 8 | 1,2,3,4 | qkv,proj,fc1,fc2 | 32/32 | 32 | 184,320 | 96^3 | (1, 3, 96, 96, 96) | ok | 4150 | **PASS** |
| stages1234_extended_r32 | 32 | 1,2,3,4 | qkv,proj,fc1,fc2 | 32/32 | 32 | 737,280 | 96^3 | (1, 3, 96, 96, 96) | ok | 4204 | **PASS** |

## Notes
- **stages123_qkv_r8** (PASS): Functional check ran at spatial=96^3 (fallback from 128^3 due to local VRAM).
- **stages123_extended_r2** (PASS): Functional check ran at spatial=96^3 (fallback from 128^3 due to local VRAM).
- **stages123_extended_r8** (PASS): Functional check ran at spatial=96^3 (fallback from 128^3 due to local VRAM).
- **stages123_extended_r32** (PASS): Functional check ran at spatial=96^3 (fallback from 128^3 due to local VRAM).
- **stages1234_extended_r2** (PASS): Functional check ran at spatial=96^3 (fallback from 128^3 due to local VRAM).
- **stages1234_extended_r8** (PASS): Functional check ran at spatial=96^3 (fallback from 128^3 due to local VRAM).
- **stages1234_extended_r32** (PASS): Functional check ran at spatial=96^3 (fallback from 128^3 due to local VRAM).
