# An Idea to Improve Position Encoding in Transformers

The proposed Transformer variant was implemented by modifying position encoding in the [original Transformer model](https://arxiv.org/abs/1706.03762) implementation source code of [Fairseq](https://github.com/pytorch/fairseq).

The original Fairseq implementation is on the branch [**original-implementation**](https://github.com/MattiaSarti/rethinking-position-encoding-in-transformers/tree/original-implementation).\
The proposed model variant implementation is on the branch [**rethinking-position-encoding/16-position-features**](https://github.com/MattiaSarti/rethinking-position-encoding-in-transformers/tree/rethinking-position-encoding/16-position-features).\
[Compare the two branches](https://github.com/MattiaSarti/rethinking-position-encoding-in-transformers/compare/original-implementation...rethinking-position-encoding/16-position-features) to see the introduced changes.

## Rationale

Why? Because I believed that summing position signals to feature vectors representing token embeddings does not attribute consistent meaning to such features. Instead, without introducing additional parameters and breaking the advantages of Transformers, I thought that position features could be concatenated to each layer block's input feature vectors, so as to make position information available to the attention mechanism and separate from other features.

## Related Work

[Liu et al.](https://arxiv.org/abs/2003.09229) and, previously, [Shaw et al.](https://arxiv.org/abs/1803.02155) were moved by similar concerns about position encoding in the original Transformer.\
Differently from these approaches, the proposed one 1) does not re-design the attention mechanism, 2) does not require additional learnable parameters and 3) introduces minimal computational overhead, yet preserving the benefits of the original architecture.

## Proposed Architectural Modifications

### Encoder:
![...loading...](https://github.com/MattiaSarti/rethinking-position-encoding-in-transformers/blob/documentation-and-results/readme_pictures/encoders_comparison.png?raw=true)

### Decoder:
![...loading...](https://github.com/MattiaSarti/rethinking-position-encoding-in-transformers/blob/documentation-and-results/readme_pictures/decoders_comparison.png?raw=true)

### Position Encoding:
![...loading...](https://github.com/MattiaSarti/rethinking-position-encoding-in-transformers/blob/documentation-and-results/readme_pictures/position_encoding_comparison.png?raw=true)

## Results on WMT'16 En-De [BLEU*]

| Original Transformer**   | Proposed Transformer**   |
|:------------------------:|:------------------------:|
| 26.61                    | 24.85                    |

\* *BLEU score computed as in the [original paper](https://arxiv.org/abs/1706.03762) but without compound splitting*\
\*\* *"base" models compared, according to the [original terminology](https://arxiv.org/abs/1706.03762)*

## Discussion

The proposed position encoding mechanism does not lead to a BLEU improvement, probably because of 1) the relatively relevant number of token features sacrified to inject position information, which are no longer available as additional representation dimensions, and 2) the scarse influence of position information on Transformers' performances, as proven by former literature.\
Further experients would be necessary to infer robust conclusions, both on additional datasets (as WMT'16 En-Fr) and with different lower percentages of token features allocated to position representation, but it was not possible to further investigate due to hardware costs.

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

Assuming your local has a single GeForce GTX 1060 NVIDIA GPU whose software dependencies for interfacing with PyTorch are already installed:
1. first, install Fairseq 0.10.0 via pip;
2. next, download and preprocess the dataset as above;
3. then, train and evaluate the original Transformer as implemented in Fairseq by running the above-mentioned commands;
4. afterwards, modify the source code of the Fairseq library installed on your local by directly replacing the whole directory **fairseq**, wherever installed by pip, with the directory [**fairseq**](https://github.com/MattiaSarti/rethinking-position-encoding-in-transformers/tree/rethinking-position-encoding/16-position-features/fairseq) on the branch [**rethinking-position-encoding/16-position-features**](https://github.com/MattiaSarti/rethinking-position-encoding-in-transformers/tree/rethinking-position-encoding/16-position-features);
5. finally, train and evaluate the proposed Transformer variant by running the above-mentioned commands.

## The Code Is Tested

The proposed architecture variant includes an additional module whose implementation has been extensively tested.\
The script [**fairseq/modules/position_encoder_unit_tests.py**](https://github.com/MattiaSarti/rethinking-position-encoding-in-transformers/blob/rethinking-position-encoding/16-position-features/fairseq/modules/position_encoder_unit_tests.py) contains such unit tests.\
To run them, after installing Faiseq and modifying its source code as explained above, execute the command ```python modules/position_encoder_unit_tests.py``` in the root directory of the package.\
Also, the code style of the added parts is both Pylint- and Flake8-compliant, as of respective 2.5.3 and 3.8.4 versions.

## My Attitude

I first learned about Transformers by implementing and training one myself from scratch: check [this](https://github.com/MattiaSarti/transformer-from-scratch) out!
