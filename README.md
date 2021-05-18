# An Idea to Improve Position Encoding in Transformers

The proposed Transformer variant was implemented by modifying position encoding in the [original Transformer model](https://arxiv.org/abs/1706.03762) implementation source code of [Fairseq](https://github.com/pytorch/fairseq).

The original Fairseq implementation is on the branch **original-implementation**.\
The proposed model variant implementation is on the branch **feature/rethinking-position-encoding**.\
Compare the two branches to see the introduced changes.

## Rationale

Why? Because I believed that summing position signals to feature vectors representing token embeddings does not attribute consistent meaning to such features. Instead, without introducing additional parameters and breaking the advantages of Transformers, I thought that position features could be concatenated to each layer block's input feature vectors, so as to make position information available to the attention mechanism and separate from other features.

## Architecture Differences

![alt text](https://github.com/MattiaSarti/temp-README/blob/main/readme_pictures/encoders.png?raw=true)
![alt text](https://github.com/MattiaSarti/temp-README/blob/main/readme_pictures/decoders.png?raw=true)
![alt text](https://github.com/MattiaSarti/temp-README/blob/main/readme_pictures/flowers.jpg?raw=true)

## Results on WMT'16 En-De [BLEU*]

| Original Transformer**   | Proposed Transformer**   |
|:------------------------:|:------------------------:|
| 26.61                    | YY.YY                    |

\* *BLEU score computed as in the [original paper](https://arxiv.org/abs/1706.03762) but without compound splitting*\
\*\* *"base" models compared, according to the [original terminology](https://arxiv.org/abs/1706.03762)*

## How To Reproduce Such Result Comparison

The dataset was downloaded and preprocessed as suggested in Fairseq [here](https://github.com/pytorch/fairseq/blob/master/examples/scaling_nmt/README.md).

Training was always executed by running this command:
```
Lorem Ipsum
```

Evaluation was always carried out by running this command:
```
Lorem Ipsum
```

Assuming your local has a single GeForce GTX 1060 NVIDIA GPU whose software dependencies for interfacing with by PyTorch are already installed:
1. first, install via pip both Fairseq 0.10.0 and the requirements in **requirements.txt**;
2. next, download and preprocess the dataset as above;
3. then, train and evaluate the original Transformer as implemented by fairseq by running the above-mentioned commands;
4. afterwards, modify the source code of the Fairseq library installed on your local by directly replacing the whole directory **fairseq**, wherever installed by pip, with the directory **fairseq** on the branch **feature/rethinking-position-encoding**;
5. finally, train and evaluate the proposed Transformer variant by running the above-mentioned commands.

## The Code Is Tested

The proposed architecture variant includes an additional module whose implementation has been extensively tested.\
The script **fairseq/modules/position_encoder_unit_tests.py** contains such unit tests.\
To run them, after installing Faiseq and modifying its source code as explained above, execute in the root directory of the package this command:
```
Lorem Ipsum
```
