# DropSeg
The official fine-tuning implementation of our VOS approach (DropSeg) for the **CVPR 2023** paper [_DropMAE: Masked Autoencoders with Spatial-Attention Dropout for Tracking Tasks_](https://arxiv.org/pdf/2304.00571.pdf).
<p align="left">
  <img src="https://github.com/jimmy-dq/DropSeg/blob/main/figs/qualitative.png" width="960">
</p>


## :sunny: Highlights

#### * Thanks for the great [_STCN_](https://github.com/hkchengrex/STCN) library, which helps us to quickly implement the [_DropMAE_](https://github.com/jimmy-dq/DropMAE) VOS fine-tuning. The repository mainly follows the STCN repository.

#### * The proposed DropSeg uses pairs of frames for offline VOS training, and achieves SOTA results on existing VOS benchmarks w/ one-shot evaluation.

## Install the environment
The Anaconda is used to create the Python environment, which mainly follows the installation in [_DropMAE_](https://github.com/jimmy-dq/DropMAE) and partially in [_STCN_](https://github.com/hkchengrex/STCN). The detailed installation packages can be found in `environment.yaml`.

## Training

### Data preparation
We follow the same data preparation steps used in [_STCN_](https://github.com/hkchengrex/STCN). Download both DAVIS and YouTube-19 datasets:
```bash
├── DAVIS
│   ├── 2016
│   │   ├── Annotations
│   │   └── ...
│   └── 2017
│       ├── test-dev
│       │   ├── Annotations
│       │   └── ...
│       └── trainval
│           ├── Annotations
│           └── ...
├── YouTube
│   ├── all_frames
│   │   └── valid_all_frames
│   ├── train
│   ├── train_480p
│   └── valid
```
### Pre-trained model download
Download pre-trained DropMAE models in [_DropMAE_](https://github.com/jimmy-dq/DropMAE) (e.g., K700-800E).

### Training command
```
python -m torch.distributed.launch --master_port 9842 --nproc_per_node=8 train_dropseg.py --pretrained_net_path pretrained_model_path --id retrain_s03 --stage 3
```
`--pretrained_net_path` indicates your downloaded pre-trained model path. 

### Inference command
Download the [DropSeg](https://drive.google.com/file/d/167aMTSQrgX3NFimRAnY5LjB7apRuSP8P/view?usp=sharing) model here, and run the evaluation w/ the following commands. All evaluations are done in the 480p resolution.
```
Python submit_eval_davis17.py --davis_path path_to_davis17_dataset
```
```
Python submit_eval_davis16.py --davis_path path_to_davis16_dataset
```
After running the above evaluation, you could get the qualitative results saved in the root project directory. You could use the offline evaluation toolikit (https://github.com/davisvideochallenge/davis2017-evaluation) to get the validation performance on DAVIS-16/17. For `test-dev` on DAVIS-17, using the online evaluation server instead.



## Acknowledgments
* Thanks for the [_STCN_](https://github.com/hkchengrex/STCN) library for convenient implementation.


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
