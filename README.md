# An Idea to Improve Position Encoding in Transformers

The proposed Transformer variant was implemented by modifying position encoding in the [original Transformer model](https://arxiv.org/abs/1706.03762) implementation source code of [Fairseq](https://github.com/pytorch/fairseq).

The original Fairseq implementation is on the branch **original-implementation**.\
The proposed model variant implementation is on the branch **feature/rethinking-position-encoding**.\
Compare the two branches to see the introduced changes.

## Rationale

Why? Because I believed that summing position signals to feature vectors representing token embeddings does not attribute consistent meaning to such features. Instead, without introducing additional parameters and breaking the advantages of Transformers, I thought that position features could be concatenated to each layer block's input feature vectors, so as to make position information available to the attention mechanism and separate from other features.

## Architecture Differences

### Encoder:
![...loading...](https://github.com/MattiaSarti/rethinking-position-encoding-in-transformers/blob/feature/rethinking-position-encoding/readme_pictures/encoders_comparison.png?raw=true)

### Decoder:
![...loading...](https://github.com/MattiaSarti/rethinking-position-encoding-in-transformers/blob/feature/rethinking-position-encoding/readme_pictures/decoders_comparison.png?raw=true)

### Position Encoding:
![...loading...](https://github.com/MattiaSarti/rethinking-position-encoding-in-transformers/blob/feature/rethinking-position-encoding/readme_pictures/position_encoding_comparison.png?raw=true)

## Results on WMT'16 En-De [BLEU*]

| Original Transformer**   | Proposed Transformer**   |
|:------------------------:|:------------------------:|
| 26.61                    | YY.YY                    |

\* *BLEU score computed as in the [original paper](https://arxiv.org/abs/1706.03762) but without compound splitting*\
\*\* *"base" models compared, according to the [original terminology](https://arxiv.org/abs/1706.03762)*

## How To Reproduce Such Result Comparison

The dataset was downloaded and preprocessed as suggested in Fairseq [here](https://github.com/pytorch/fairseq/blob/master/examples/scaling_nmt/README.md), coherently with the [original work](https://arxiv.org/abs/1706.03762).

Training was always executed by running this command until 100000 training steps were reached, coherently with the [original base model training](https://arxiv.org/abs/1706.03762):
```
fairseq-train <your_dataset_directory> --arch transformer_wmt_en_de --share-all-embeddings --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0007 --min-lr 1e-09 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 --max-tokens 2048 --update-freq 16 --no-progress-bar --log-format json --log-interval 50 --save-interval-updates 1000 --keep-interval-updates 10 --tensorboard-logdir <your_tensorboard_directory>
```

Evaluation was always carried out, coherently with the [original work](https://arxiv.org/abs/1706.03762), by first averaging the last 5 checkpoints thanks to [this script](https://github.com/pytorch/fairseq/blob/master/scripts/average_checkpoints.py):
```
python average_checkpoints.py --inputs <your_checkpoints_directory> --num-update-checkpoints 5 --output <your_averaged_checkpoint_path>
```
and by evantually running this command:
```
fairseq-generate <your_dataset_directory> --path <your_averaged_checkpoint_path> --gen-subset test --beam 4 --batch-size 128 --remove-bpe --lenpen 0.6 > <your_output_text_file_path>
```

Assuming your local has a single GeForce GTX 1060 NVIDIA GPU whose software dependencies for interfacing with by PyTorch are already installed:
1. first, install Fairseq 0.10.0 via pip;
2. next, download and preprocess the dataset as above;
3. then, train and evaluate the original Transformer as implemented by fairseq by running the above-mentioned commands;
4. afterwards, modify the source code of the Fairseq library installed on your local by directly replacing the whole directory **fairseq**, wherever installed by pip, with the directory **fairseq** on the branch **feature/rethinking-position-encoding**;
5. finally, train and evaluate the proposed Transformer variant by running the above-mentioned commands.

## The Code Is Tested

The proposed architecture variant includes an additional module whose implementation has been extensively tested.\
The script **fairseq/modules/position_encoder_unit_tests.py** contains such unit tests.\
To run them, after installing Faiseq and modifying its source code as explained above, execute in the root directory of the package this command:
```
python modules/position_encoder_unit_tests.py
```
