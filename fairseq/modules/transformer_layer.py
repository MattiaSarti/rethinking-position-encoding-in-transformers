# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)
############################ BEGINNING OF CHANGES ############################
        self.position_encoder = PositionEncoder()
############################### END OF CHANGES ###############################

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

############################ BEGINNING OF CHANGES ############################
        x = self.position_encoder(x)
############################### END OF CHANGES ###############################
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False
############################ BEGINNING OF CHANGES ############################
        self.position_encoder = PositionEncoder()
############################### END OF CHANGES ###############################

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
############################ BEGINNING OF CHANGES ############################
        token_position: Optional[int] = None,
############################### END OF CHANGES ###############################
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

############################ BEGINNING OF CHANGES ############################
        x = self.position_encoder(x, incremental_position=token_position)
############################### END OF CHANGES ###############################
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
############################ BEGINNING OF CHANGES ############################


class PositionEncoder(nn.Module):
    """
    Layer encoding position with the proposed alternative methodology.
    """

    def __init__(
        self,
        position_dim: int = 16,
        upper_bound_max_seq_len: int = 1024
    ):
        assert isinstance(position_dim, int)
        super(PositionEncoder, self).__init__()

        self.pool_input_dim = position_dim * 2

        # position signals injected as positional features:
        position_signals = PositionEncoder.generate_position_signals(
            pos_dim=position_dim,
            max_seq_len=upper_bound_max_seq_len
        )
        self.register_buffer('position_signals', position_signals)

    @staticmethod
    def generate_position_signals(pos_dim: int, max_seq_len: int) -> Tensor:
        """
        Generate position signals covering any possible sequence length,
        to be furtherly selected and broadcasted, but so as to pre-compute
        all possible values, avoiding such step during forward propagation.

        Tensor Shapes:

            Returned:
                (seq_len, pos_dim)
        """
        from math import pi

        data_type = torch.float
        freq_interleaving_factor = 16

        # generating the abscissas of the base cosinusoidal position signal
        # (i.e. the one with lowest frequency), in radiants:
        base_signal_abscissas = torch.arange(
            start=0,
            end=pi,
            step=pi/max_seq_len,
            dtype=data_type
        )

        # imposing the relative frequencies, compared to the base signal's
        # one, of the resulting cosinusoidal positon signals:
        relative_frequencies = torch.arange(
            start=1,
            end=(pos_dim*freq_interleaving_factor)+1,
            step=freq_interleaving_factor,
            dtype=data_type
        )

        # computing the cosinusoidal position signals after modifying their
        # abscissas to match the desired frequency distribution, eventually
        # shifting and scaling the cosinusoidal values so that they are
        # linearly mapped from [-1; +1] to [0; +1] preserving the cosinusoidal
        # trend:
        signals_abscissas = (
            base_signal_abscissas.unsqueeze(dim=1).repeat(1, pos_dim)
            * relative_frequencies
        )
        return (torch.cos(signals_abscissas) + 1) / 2

    def forward(
        self,
        x: Tensor,
        incremental_position: Optional[int] = None
    ) -> Tensor:
        """
        Inject position by applying token-wise max-pooling to a subset of
        features first and then concatenating the respective position signals
        to each token feature vector, restoring the original feature
        dimension.

        Tensor Shapes:

            Args:
                x: (seq_len, batch, embed_dim)

            Returned:
                (seq_len, batch, embed_dim)
        """
        seq_len, batch_size, _ = x.shape

        # when predicting in mini-batches, with teacher forcing for decoding:
        if incremental_position is None:
            # selecting the position signals for the tokens existing for such
            # sequence length and broadcasting them for all the mini-batches:
            position_features = (
                self.position_signals[:seq_len, ].detach().unsqueeze(dim=1)
                .repeat(1, batch_size, 1)
            )
        # when predicting incrementally, at inference time, for decoding:
        else:
            # selecting the position signals for the tokens in that position
            # and broadcasting them for all the mini-batches:
            position_features = (
                self.position_signals[incremental_position, ].detach()
                .unsqueeze(dim=0).unsqueeze(dim=0).repeat(1, batch_size, 1)
            )

        # pooling the subset of features to be halved in size so as to let the
        # position features be concatenated without changing the overall token
        # feature vector dimensionality:
        pooled_features = nn.functional.max_pool1d(
            x[..., -self.pool_input_dim:],
            kernel_size=2
        )

        # concatenating to the remaining features of the original feature
        # vectors the pooled features and the position features, restoring the
        # same original feature dimensionality:
        return torch.cat(
            (
                x[..., :-self.pool_input_dim],
                pooled_features,
                position_features
            ),
            dim=-1
        )


if __name__ == '__main__':

    from math import pi
    from unittest import main as unittest_main, TestCase

    POSITION_FEATURE_DIMENSION_FOR_TESTS = 16
    MAX_SEQUENCE_LENGTH_FOR_TESTS = 1024

    class TestPositionEncoder(TestCase):
        """
        Unit tests for the proposed "PositionEncoder" module.
        """

        @classmethod
        def setUpClass(cls):
            """
            Common module instantiation to all tests, since it is not affected
            by re-utilization.
            """
            cls.module = PositionEncoder(
                position_dim=POSITION_FEATURE_DIMENSION_FOR_TESTS,
                upper_bound_max_seq_len=MAX_SEQUENCE_LENGTH_FOR_TESTS
            )

        def test_backpropagation(self):
            """
            Test that the module does not corrupt gradient backpropagation
            from its outputs to its inputs when inserted in a (toy) network
            (in-between two linear layers).
            """
            input_feature_dimension = 1024

            class ToyNetwork(nn.Module):
                """
                Toy neural network, with the proposed layer in-between two
                linear layers.
                """

                def __init__(self):
                    super(ToyNetwork, self).__init__()
                    self.first_linear_layer = nn.Linear(
                        in_features=input_feature_dimension,
                        out_features=128
                    )
                    self.last_linear_layer = nn.Linear(
                        in_features=128,
                        out_features=16
                    )
                    self.position_encoding_layer = PositionEncoder()
                
                def forward(self, x):
                    x = self.first_linear_layer(x)
                    x = self.position_encoding_layer(x)
                    x = self.last_linear_layer(x)
                    return x

            toy_network = ToyNetwork()
            optimizer = torch.optim.SGD(
                params=toy_network.parameters(),
                lr=1  # exaggeratedly high to avoid negligible updates
            )

            # taking a snapshot of the initial parameters:
            initial_parameter_dict = dict(
                toy_network.named_parameters()
            )
            # switching to training mode:
            toy_network.train()
            # forward propagation:
            input_tensor = torch.empty(100, 40, input_feature_dimension)
            output_tensor = toy_network(input_tensor)
            toy_loss = output_tensor.sum()**2 - 1
            # backward propagation:
            toy_loss.backward()
            # parameter update:
            optimizer.step()
            # taking a snapshot of the parameters now:
            final_parameter_dict = dict(
                toy_network.named_parameters()
            )
            # asserting that all parameters have been updated: 
            for name, initial_parameter_tensor in (
                initial_parameter_dict.items()
            ):
                self.assertFalse(
                    torch.equal(
                        initial_parameter_tensor,
                        final_parameter_dict[name]
                    )
                )

        def test_no_possibly_learnable_parameters(self):
            """
            Test that the module does not have any parameter that can become
            learnable.
            """
            possibly_learnable_parameters = list(self.module.parameters())
            self.assertEqual(possibly_learnable_parameters, [])

        def test_output_dtype_and_shape(self):
            """
            Test the output tensor data type and shape.
            """
            expected_output_dtype = torch.float

            test_cases = [
                {
                    'incremental_position': None,
                    'input_tensor_kwargs': {
                        'dtype': torch.float,
                        'size': (19, 104, 512)
                    },
                    'expected_output_dtype': expected_output_dtype,
                    'expected_output_shape': (19, 104, 512)
                },
                {
                    'incremental_position': 4,
                    'input_tensor_kwargs': {
                        'dtype': torch.float,
                        'size': (1, 512, 512)
                    },
                    'expected_output_dtype': expected_output_dtype,
                    'expected_output_shape': (1, 512, 512)
                },
                {
                    'incremental_position': None,
                    'input_tensor_kwargs': {
                        'dtype': torch.float,
                        'size': (23, 40, 256)
                    },
                    'expected_output_dtype': expected_output_dtype,
                    'expected_output_shape': (23, 40, 256)
                },
                {
                    'incremental_position': 137,
                    'input_tensor_kwargs': {
                        'dtype': torch.float,
                        'size': (1, 8, 256)
                    },
                    'expected_output_dtype': expected_output_dtype,
                    'expected_output_shape': (1, 8, 256)
                },
            ]

            for test_case in test_cases:

                test_name = "pos " + str(test_case['incremental_position']) +\
                    " & input " + str(test_case['input_tensor_kwargs'])

                input_tensor = torch.empty(**test_case['input_tensor_kwargs'])
                output_tensor = self.module(
                    input_tensor,
                    incremental_position=test_case['incremental_position']
                )

                with self.subTest("dtype for " + test_name):
                    self.assertEqual(
                        output_tensor.dtype,
                        test_case['expected_output_dtype']
                    )

                with self.subTest("shape for " + test_name):
                    self.assertEqual(
                        output_tensor.shape,
                        test_case['expected_output_shape']
                    )

        def test_output_values(self):
            """
            Test the output tensor values.
            """
            raise NotImplementedError

        def test_position_signals_generation(self):
            """
            Test the position signals' generation output.
            """
            freq_interleaving_factor = 16
            expected_dtype = torch.float

            test_cases_kwargs = [
                {
                    'pos_dim': 16,
                    'max_seq_len': 1024
                },
                {
                    'pos_dim': 4,
                    'max_seq_len': 10
                }
            ]

            for test_kwargs in test_cases_kwargs:

                test_name = str(test_kwargs)

                position_signals = PositionEncoder.generate_position_signals(
                    **test_kwargs
                )

                with self.subTest("dtype for " + test_name):
                    self.assertEqual(
                        position_signals.dtype,
                        expected_dtype
                    )

                with self.subTest("shape for " + test_name):
                    self.assertEqual(
                        position_signals.shape,
                        (test_kwargs['max_seq_len'], test_kwargs['pos_dim'])
                    )

                with self.subTest("non-negative values for " + test_name):
                    self.assertTrue(
                        torch.all(position_signals >= 0)
                    )

                with self.subTest("no values above +1 for" + test_name):
                    self.assertTrue(
                        torch.all(position_signals <= 1)
                    )

                for i, actual_cosinusoidal_signal in (
                    enumerate(position_signals.transpose(dim0=0, dim1=1))
                ):
                    subtest_name = (
                        "cosinusoidal signal n. " + str(i + 1) + " for "
                        + test_name
                    )
                    with self.subTest(subtest_name):
                        expected_relative_frequency = (
                            i*freq_interleaving_factor + 1
                        )
                        # cosinusoidal abscissas in radiants:
                        abscissas = torch.arange(
                            start=0,
                            end=pi,
                            step=pi/test_kwargs['max_seq_len'],
                            dtype=expected_dtype
                        )
                        abscissas *= expected_relative_frequency
                        # applying cosine:
                        expected_cosinusoidal_signal = torch.cos(abscissas)
                        # rescaling to [0; +1]:
                        expected_cosinusoidal_signal = (
                            (expected_cosinusoidal_signal + 1) / 2
                        )
                        # cosinusoidal signal assertion:
                        self.assertTrue(
                            torch.all(
                                torch.eq(
                                    actual_cosinusoidal_signal,
                                    expected_cosinusoidal_signal
                                )
                            )
                        )
                # redundant assertion to ensure this test is properly written:
                assert i + 1 == test_kwargs['pos_dim']

    unittest_main(verbosity=2)
############################### END OF CHANGES ###############################
