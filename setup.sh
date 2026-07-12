mkdir data
mkdir exp_data
mkdir pretrained_models
conda env create -f environment.yml
# python setup_bop.py install

unzip pretrained_models.zip -d pretrained_models/
wget https://huggingface.co/hamacojr/CAT-Seg/resolve/main/model_final_large.pth
mv model_final_large.pth  pretrained_models/catseg.pth

wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth
mkdir pretrained_models/groundingdino
mv groundingdino_swinb_cogcoor.pth pretrained_models/groundingdino/groundingdino_swinb_cogcoor.pth

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
mkdir pretrained_models/sam
mv sam_vit_h_4b8939.pth pretrained_models/sam/sam_vit_h_4b8939.pth