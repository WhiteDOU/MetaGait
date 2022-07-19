# # **************** For CASIA-B ****************
# # Baseline
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 lib/main.py --cfgs ./config/baseline.yaml --phase test

# # GaitSet
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 lib/main.py --cfgs ./config/gaitset.yaml --phase test

# # GaitPart
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 lib/main.py --cfgs ./config/gaitpart.yaml --phase test

# GaitGL
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 12345 --nproc_per_node=4 lib/main.py --cfgs ./config/gaitgl.yaml --phase test

# # GLN 
# # Phase 1
# CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch --master_port 12345  --nproc_per_node=2 lib/main.py --cfgs ./config/gln/gln_phase1.yaml --phase test
# # Phase 2
# CUDA_VISIBLE_DEVICES=2,5 python -m torch.distributed.launch --nproc_per_node=2 lib/main.py --cfgs ./config/gln/gln_phase2.yaml --phase test


# # **************** For OUMVLP ****************
# # Baseline
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 lib/main.py --cfgs ./config/baseline_OUMVLP.yaml --phase test

# # GaitSet
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 lib/main.py --cfgs ./config/gaitset_OUMVLP.yaml --phase test

# # GaitPart
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 lib/main.py --cfgs ./config/gaitpart_OUMVLP.yaml --phase test

# GaitGL
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 lib/main.py --cfgs ./config/gaitgl_OUMVLP.yaml --phase test
