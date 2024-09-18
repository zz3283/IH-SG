#python scripts/evaluate_model.py DucoNet ./checkpoints/DucoNet256.pth \
#--resize-strategy Fixed256 \
#--gpu 0\
#--datasets HAdobe5k

#python scripts/evaluate_model.py DucoNet ./checkpoints/last_model/DucoNet1024.pth \
#--resize-strategy Fixed1024 \
#--gpu 1 \
#--datasets HAdobe5k1

python scripts/predict_for_dir0.py hrnet18_idih256  /mnt/data/ZJ/IS2AM/harmonization_exps/fixed256/hrnet18_idih/054_first-try/checkpoints/last_checkpoint.pth \
--resize  256 \
--images  /mnt/data/ZJ/data/Shadow-AR/noshadow/  \
--masks   /mnt/data/ZJ/data/Shadow-AR/mask/  \
--gt   /mnt/data/ZJ/data/Shadow-AR/shadow/ \
--gpu 0 \
--results-path  /mnt/data/ZJ/IS2AM/shadow_AR/

#python scripts/predict_for_dir.py hrnet18_idih256  /mnt/data/ZJ/IS2AM/harmonization_exps/fixed256/hrnet18_idih/054_first-try/checkpoints/last_checkpoint.pth \
#--resize  256 \
#--conf  /mnt/data/ZJ/dataset/IHD/Hday2night/Hday2night_test.txt \
#--images  /mnt/data/ZJ/dataset/IHD/Hday2night/composite_images/ \
#--masks   /mnt/data/ZJ/dataset/IHD/Hday2night/masks/  \
#--gt /mnt/data/ZJ/dataset/IHD/Hday2night/real_images/  \
#--gpu 1 \
#--results-path  /mnt/data/ZJ/IS2AM/Hday2night_out/

#python scripts/predict_for_dir.py hrnet18_idih256  /mnt/data/ZJ/IS2AM/harmonization_exps/fixed256/hrnet18_idih/054_first-try/checkpoints/last_checkpoint.pth \
#--resize  256 \
#--conf  /mnt/data/ZJ/dataset/IHD/HAdobe5k/HAdobe5k_test.txt \
#--images  /mnt/data/ZJ/dataset/IHD/HAdobe5k/composite_images/ \
#--masks   /mnt/data/ZJ/dataset/IHD/HAdobe5k/masks/  \
#--gt /mnt/data/ZJ/dataset/IHD/HAdobe5k/real_images/  \
#--gpu 1 \
#--results-path  /mnt/data/ZJ/IS2AM/HAdobe5k_out/

#python scripts/predict_for_dir1.py hrnet18_idih256  /mnt/data/ZJ/IS2AM/harmonization_exps/fixed256/hrnet18_idih/054_first-try/checkpoints/last_checkpoint.pth \
#--resize  256 \
#--conf  /mnt/data/ZJ/dataset/mydata/mydata_test.txt \
#--images  /mnt/data/ZJ/dataset/mydata/composite_images/  \
#--masks   /mnt/data/ZJ/dataset/mydata/masks/  \
#--gpu 0 \
#--results-path  /mnt/data/ZJ/mydata1/
