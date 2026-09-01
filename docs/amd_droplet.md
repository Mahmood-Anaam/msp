
# Training and Evaluation of MSP Models on AMD Droplet

## Check the GPU and ROCm version on the AMD Droplet
```bash

amd-smi

+------------------------------------------------------------------------------+
| AMD-SMI 26.2.2+97f5574fe2    amdgpu version: 6.16.13  ROCm version: 7.2.4    |
| VBIOS version: 00123529                                                      |
| Platform: Linux Guest                                                        |
|-------------------------------------+----------------------------------------|
| BDF                        GPU-Name | Mem-Uti   Temp   UEC       Power-Usage |
| GPU  HIP-ID  OAM-ID  Partition-Mode | GFX-Uti    Fan               Mem-Usage |
|=====================================+========================================|
| 0000:83:00.0 AMD Instinct MI300X VF | 0 %      37 °C   0           160/750 W |
|   0       0       3        SPX/NPS1 | 0 %        N/A           285/196288 MB |
+-------------------------------------+----------------------------------------+
+------------------------------------------------------------------------------+
| Processes:                                                                   |
|  GPU        PID  Process Name          GTT_MEM  VRAM_MEM  MEM_USAGE     CU % |
|==============================================================================|
|    0       1823  N/A                     0.0 B     0.0 B      0.0 B    0.0 % |
|    0       4212  N/A                     0.0 B     0.0 B      0.0 B    0.0 % |
+------------------------------------------------------------------------------+
Process Name may require elevated permissions.

```

```bash
rocminfo

[37mROCk module version 6.16.13 is loaded[0m
=====================    
HSA System Attributes    
=====================    
Runtime Version:         1.18
Runtime Ext Version:     1.15
System Timestamp Freq.:  1000.000000MHz
Sig. Max Wait Duration:  18446744073709551615 (0xFFFFFFFFFFFFFFFF) (timestamp count)
Machine Model:           LARGE                              
System Endianness:       LITTLE                             
Mwaitx:                  DISABLED
XNACK enabled:           NO
DMAbuf Support:          YES
VMM Support:             YES

==========               
HSA Agents               
==========               
*******                  
Agent 1                  
*******                  
  Name:                    INTEL(R) XEON(R) PLATINUM 8568Y+   
  Uuid:                    CPU-XX                             
  Marketing Name:          INTEL(R) XEON(R) PLATINUM 8568Y+   
  Vendor Name:             CPU                                
  Feature:                 None specified                     
  Profile:                 FULL_PROFILE                       
  Float Round Mode:        NEAR                               
  Max Queue Number:        0(0x0)                             
  Queue Min Size:          0(0x0)                             
  Queue Max Size:          0(0x0)                             
  Queue Type:              MULTI                              
  Node:                    0                                  
  Device Type:             CPU                                
  Cache Info:              
    L1:                      32768(0x8000) KB                   
  Chip ID:                 0(0x0)                             
  ASIC Revision:           0(0x0)                             
  Cacheline Size:          64(0x40)                           
  Max Clock Freq. (MHz):   0                                  
  BDFID:                   0                                  
  Internal Node ID:        0                                  
  Compute Unit:            20                                 
  SIMDs per CU:            0                                  
  Shader Engines:          0                                  
  Shader Arrs. per Eng.:   0                                  
  WatchPts on Addr. Ranges:1                                  
  Memory Properties:       
  Features:                None
  Pool Info:               
    Pool 1                   
      Segment:                 GLOBAL; FLAGS: FINE GRAINED        
      Size:                    247409212(0xebf2a3c) KB            
      Allocatable:             TRUE                               
      Alloc Granule:           4KB                                
      Alloc Recommended Granule:4KB                                
      Alloc Alignment:         4KB                                
      Accessible by all:       TRUE                               
    Pool 2                   
      Segment:                 GLOBAL; FLAGS: EXTENDED FINE GRAINED
      Size:                    247409212(0xebf2a3c) KB            
      Allocatable:             TRUE                               
      Alloc Granule:           4KB                                
      Alloc Recommended Granule:4KB                                
      Alloc Alignment:         4KB                                
      Accessible by all:       TRUE                               
    Pool 3                   
      Segment:                 GLOBAL; FLAGS: KERNARG, FINE GRAINED
      Size:                    247409212(0xebf2a3c) KB            
      Allocatable:             TRUE                               
      Alloc Granule:           4KB                                
      Alloc Recommended Granule:4KB                                
      Alloc Alignment:         4KB                                
      Accessible by all:       TRUE                               
    Pool 4                   
      Segment:                 GLOBAL; FLAGS: COARSE GRAINED      
      Size:                    247409212(0xebf2a3c) KB            
      Allocatable:             TRUE                               
      Alloc Granule:           4KB                                
      Alloc Recommended Granule:4KB                                
      Alloc Alignment:         4KB                                
      Accessible by all:       TRUE                               
  ISA Info:                
*******                  
Agent 2                  
*******                  
  Name:                    gfx942                             
  Uuid:                    GPU-43529fe3ce00c724               
  Marketing Name:          AMD Instinct MI300X VF             
  Vendor Name:             AMD                                
  Feature:                 KERNEL_DISPATCH                    
  Profile:                 BASE_PROFILE                       
  Float Round Mode:        NEAR                               
  Max Queue Number:        128(0x80)                          
  Queue Min Size:          64(0x40)                           
  Queue Max Size:          131072(0x20000)                    
  Queue Type:              MULTI                              
  Node:                    1                                  
  Device Type:             GPU                                
  Cache Info:              
    L1:                      32(0x20) KB                        
    L2:                      4096(0x1000) KB                    
    L3:                      262144(0x40000) KB                 
  Chip ID:                 29877(0x74b5)                      
  ASIC Revision:           1(0x1)                             
  Cacheline Size:          128(0x80)                          
  Max Clock Freq. (MHz):   2100                               
  BDFID:                   33536                              
  Internal Node ID:        1                                  
  Compute Unit:            304                                
  SIMDs per CU:            4                                  
  Shader Engines:          32                                 
  Shader Arrs. per Eng.:   1                                  
  WatchPts on Addr. Ranges:4                                  
  Coherent Host Access:    FALSE                              
  Memory Properties:       
  Features:                KERNEL_DISPATCH 
  Fast F16 Operation:      TRUE                               
  Wavefront Size:          64(0x40)                           
  Workgroup Max Size:      1024(0x400)                        
  Workgroup Max Size per Dimension:
    x                        1024(0x400)                        
    y                        1024(0x400)                        
    z                        1024(0x400)                        
  Max Waves Per CU:        32(0x20)                           
  Max Work-item Per CU:    2048(0x800)                        
  Grid Max Size:           4294967295(0xffffffff)             
  Grid Max Size per Dimension:
    x                        2147483647(0x7fffffff)             
    y                        65535(0xffff)                      
    z                        65535(0xffff)                      
  Max fbarriers/Workgrp:   32                                 
  Packet Processor uCode:: 189                                
  SDMA engine uCode::      24                                 
  IOMMU Support::          None                               
  Pool Info:               
    Pool 1                   
      Segment:                 GLOBAL; FLAGS: COARSE GRAINED      
      Size:                    200998912(0xbfb0000) KB            
      Allocatable:             TRUE                               
      Alloc Granule:           4KB                                
      Alloc Recommended Granule:2048KB                             
      Alloc Alignment:         4KB                                
      Accessible by all:       FALSE                              
    Pool 2                   
      Segment:                 GLOBAL; FLAGS: EXTENDED FINE GRAINED
      Size:                    200998912(0xbfb0000) KB            
      Allocatable:             TRUE                               
      Alloc Granule:           4KB                                
      Alloc Recommended Granule:2048KB                             
      Alloc Alignment:         4KB                                
      Accessible by all:       FALSE                              
    Pool 3                   
      Segment:                 GLOBAL; FLAGS: FINE GRAINED        
      Size:                    200998912(0xbfb0000) KB            
      Allocatable:             TRUE                               
      Alloc Granule:           4KB                                
      Alloc Recommended Granule:2048KB                             
      Alloc Alignment:         4KB                                
      Accessible by all:       FALSE                              
    Pool 4                   
      Segment:                 GROUP                              
      Size:                    64(0x40) KB                        
      Allocatable:             FALSE                              
      Alloc Granule:           0KB                                
      Alloc Recommended Granule:0KB                                
      Alloc Alignment:         0KB                                
      Accessible by all:       FALSE                              
  ISA Info:                
    ISA 1                    
      Name:                    amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-
      Machine Models:          HSA_MACHINE_MODEL_LARGE            
      Profiles:                HSA_PROFILE_BASE                   
      Default Rounding Mode:   NEAR                               
      Default Rounding Mode:   NEAR                               
      Fast f16:                TRUE                               
      Workgroup Max Size:      1024(0x400)                        
      Workgroup Max Size per Dimension:
        x                        1024(0x400)                        
        y                        1024(0x400)                        
        z                        1024(0x400)                        
      Grid Max Size:           4294967295(0xffffffff)             
      Grid Max Size per Dimension:
        x                        2147483647(0x7fffffff)             
        y                        65535(0xffff)                      
        z                        65535(0xffff)                      
      FBarrier Max Size:       32                                 
    ISA 2                    
      Name:                    amdgcn-amd-amdhsa--gfx9-4-generic:sramecc+:xnack-
      Machine Models:          HSA_MACHINE_MODEL_LARGE            
      Profiles:                HSA_PROFILE_BASE                   
      Default Rounding Mode:   NEAR                               
      Default Rounding Mode:   NEAR                               
      Fast f16:                TRUE                               
      Workgroup Max Size:      1024(0x400)                        
      Workgroup Max Size per Dimension:
        x                        1024(0x400)                        
        y                        1024(0x400)                        
        z                        1024(0x400)                        
      Grid Max Size:           4294967295(0xffffffff)             
      Grid Max Size per Dimension:
        x                        2147483647(0x7fffffff)             
        y                        65535(0xffff)                      
        z                        65535(0xffff)                      
      FBarrier Max Size:       32                                 
*** Done ***

```

## Train the MSP models on the AMD Droplet

```bash
cd shared-docker
export HF_USERNAME="<your_huggingface_username>"
export HF_TOKEN="<your_huggingface_token>"
export WANDB_API_KEY="<your_wandb_api_key>"
export WANDB_PROJECT="<your_wandb_project_name>"
export WANDB_ENTITY="<your_wandb_entity_name>"

git clone https://github.com/Mahmood-Anaam/msp.git
pip install -e ./msp/
clear

train ./msp/configs/train_msp_asr.json
train ./msp/configs/train_msp_vsr.json
train ./msp/configs/train_msp_avsr.json

```

## Evaluate the MSP models on the AMD Droplet

```bash

evaluate \
  --model_type msp_audio \
  --model_name_or_path {HF_USERNAME}/MSP-ASR \
  --processor_name_or_path {HF_USERNAME}/MSP-Processor-With-LM \
  --dataset_name lrs2 \
  --streaming_dataset true \
  --set_id "*" \
  --cache_dir /workspace/cache/huggingface \
  --dataset_cache_dir /workspace/data/cache \
  --device cuda


evaluate \
  --model_type msp_audio \
  --model_name_or_path {HF_USERNAME}/MSP-ASR \
  --processor_name_or_path {HF_USERNAME}/MSP-Processor-With-LM \
  --dataset_name AVCocktail \
  --streaming_dataset true \
  --set_id "*" \
  --cache_dir /workspace/cache/huggingface \
  --dataset_cache_dir /workspace/data/cache \
  --device cuda


evaluate \
  --model_type msp_visual \
  --model_name_or_path {HF_USERNAME}/MSP-VSR \
  --processor_name_or_path {HF_USERNAME}/MSP-Processor-With-LM \
  --dataset_name lrs2 \
  --streaming_dataset true \
  --set_id "test" \
  --cache_dir /workspace/cache/huggingface \
  --dataset_cache_dir /workspace/data/cache \
  --device cuda


evaluate \
  --model_type msp_visual \
  --model_name_or_path {HF_USERNAME}/MSP-VSR \
  --processor_name_or_path {HF_USERNAME}/MSP-Processor-With-LM \
  --dataset_name AVCocktail \
  --streaming_dataset true \
  --set_id "*" \
  --cache_dir /workspace/cache/huggingface \
  --dataset_cache_dir /workspace/data/cache \
  --device cuda



evaluate \
  --model_type msp \
  --model_name_or_path {HF_USERNAME}/MSP-AVSR \
  --processor_name_or_path {HF_USERNAME}/MSP-Processor-With-LM \
  --dataset_name lrs2 \
  --streaming_dataset true \
  --set_id "*" \
  --cache_dir /workspace/cache/huggingface \
  --dataset_cache_dir /workspace/data/cache \
  --device cuda



evaluate \
  --model_type msp \
  --modality msp_audio \
  --model_name_or_path {HF_USERNAME}/MSP-AVSR \
  --processor_name_or_path {HF_USERNAME}/MSP-Processor-With-LM \
  --dataset_name lrs2 \
  --streaming_dataset true \
  --set_id "*" \
  --cache_dir /workspace/cache/huggingface \
  --dataset_cache_dir /workspace/data/cache \
  --device cuda


evaluate \
  --model_type msp \
  --modality msp_visual \
  --model_name_or_path {HF_USERNAME}/MSP-AVSR \
  --processor_name_or_path {HF_USERNAME}/MSP-Processor-With-LM \
  --dataset_name lrs2 \
  --streaming_dataset true \
  --set_id "test" \
  --cache_dir /workspace/cache/huggingface \
  --dataset_cache_dir /workspace/data/cache \
  --device cuda



evaluate \
  --model_type msp \
  --model_name_or_path {HF_USERNAME}/MSP-AVSR \
  --processor_name_or_path {HF_USERNAME}/MSP-Processor-With-LM \
  --dataset_name AVCocktail \
  --streaming_dataset true \
  --set_id "*" \
  --cache_dir /workspace/cache/huggingface \
  --dataset_cache_dir /workspace/data/cache \
  --device cuda


evaluate \
  --model_type msp \
  --modality msp_audio \
  --model_name_or_path {HF_USERNAME}/MSP-AVSR \
  --processor_name_or_path {HF_USERNAME}/MSP-Processor-With-LM \
  --dataset_name AVCocktail \
  --streaming_dataset true \
  --set_id "*" \
  --cache_dir /workspace/cache/huggingface \
  --dataset_cache_dir /workspace/data/cache \
  --device cuda


  evaluate \
  --model_type msp \
  --modality msp_visual \
  --model_name_or_path {HF_USERNAME}/MSP-AVSR \
  --processor_name_or_path {HF_USERNAME}/MSP-Processor-With-LM \
  --dataset_name AVCocktail \
  --streaming_dataset true \
  --set_id "*" \
  --cache_dir /workspace/cache/huggingface \
  --dataset_cache_dir /workspace/data/cache \
  --device cuda

```
