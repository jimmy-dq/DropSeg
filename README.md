# DropSeg
The official fine-tuning implementation of our VOS approach (DropSeg) for the **CVPR 2023** paper [_DropMAE: Masked Autoencoders with Spatial-Attention Dropout for Tracking Tasks_](https://arxiv.org/pdf/2304.00571.pdf).


## :sunny: Highlights

#### * Thanks for STCN (https://github.com/hkchengrex/STCN) library, which helps us to quickly implement the [_DropMAE_](https://github.com/jimmy-dq/DropMAE) VOS fine-tuning. The repository mainly follows the STCN repository.

#### * The proposed DropSeg uses pairs of frames for offline VOS training, and achieves SOTA results on existing VOS benchmarks w/ one-shot evaluation.

## Install the environment
**Option1**: The Anaconda is used to create the Python environment, which mainly follows the installation in [_DropMAE_](https://github.com/jimmy-dq/DropMAE) and partially in STCN (https://github.com/hkchengrex/STCN). The detailed installation packages are listed in :
```
conda create -n droptrack python=3.8
conda activate droptrack
pip install -r requirements.txt
```

## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```




## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```


## Training
* Download pre-trained DropMAE models in [_DropMAE_](https://github.com/jimmy-dq/DropMAE) and put it under `$PROJECT_ROOT$/pretrained_models`. 
* Modify the ```PRETRAIN_FILE``` tag in ```vitb_384_mae_ce_32x4_ep300.yaml``` or ```vitb_384_mae_ce_32x4_got10k_ep100.yaml``` to the name of your downloaded DropMAE pre-trained models. 
* Training Command on GOT-10K:
```
cd path_to_your_project
python tracking/train.py --script ostrack --config vitb_384_mae_ce_32x4_got10k_ep100 --save_dir sabe_path --mode multiple --nproc_per_node 4 --use_lmdb 0 --use_wandb 0
```
* Training Command on the other tracking datasets:
```
cd path_to_your_project
python tracking/train.py --script ostrack --config vitb_384_mae_ce_32x4_ep300 --save_dir save_path --mode multiple --nproc_per_node 4 --use_lmdb 0 --use_wandb 0
```



## Evaluation
Download the tracking model weights
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">K400-1600E-GOT10k</th>
<th valign="bottom">K700-800E-GOT10k</th>
<th valign="bottom">K700-800E-AllData</th>
<!-- TABLE BODY -->
<tr><td align="left">Tracking Models</td>
<td align="center"><a href="https://drive.google.com/file/d/1AHNr7dJ1B53CR8WigV26amEoFJLTtu7v/view?usp=sharing">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1OMYfyvkpxf7DVS7wYLUGmXPydS9TkskT/view?usp=sharing">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1l0YSK0QLPGVIGiNXwlaWp5uhIkJawJqh/view?usp=sharing">download</a></td>
</tbody></table>

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths. Note that the ```save_dir``` tag should be set to the downloaded tracking model path and you can also modify the tracking model name in ```lib/test/parameter/ostrack.py```.

Some testing examples:
- LaSOT or other off-line evaluated benchmarks (modify `--dataset` correspondingly)
```
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset lasot --threads 16 --num_gpus 4
python tracking/analysis_results.py # need to modify tracker configs and names
```
- GOT10K-test
```
python tracking/test.py ostrack vitb_384_mae_ce_32x4_got10k_ep100 --dataset got10k_test --threads 16 --num_gpus 4
python lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name vitb_384_mae_ce_32x4_got10k_ep100
```


## Acknowledgments
* Thanks for the [OSTrack](https://github.com/botaoye/OSTrack) library for convenient implementation.


## Citation
If our work is useful for your research, please consider cite:

```
@inproceedings{dropmae2023,
  title={DropMAE: Masked Autoencoders with Spatial-Attention Dropout for Tracking Tasks},
  author={Qiangqiang Wu and Tianyu Yang and Ziquan Liu and Baoyuan Wu and Ying Shan and Antoni B. Chan},
  booktitle={CVPR},
  year={2023}
}
```
