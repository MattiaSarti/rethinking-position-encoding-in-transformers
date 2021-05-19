############################ BEGINNING OF CHANGES ############################
"""
Unit tests for the proposed "PositionEncoder" module.
"""


from copy import deepcopy
from math import pi
from random import seed as random_seed
from unittest import main as unittest_main, TestCase

from numpy.random import seed as numpy_seed
import torch
from torch import nn
from torch.nn import functional as F

from transformer_layer import PositionEncoder


DEFAULT_POSITION_FEATURE_DIMENSION_FOR_TESTS = 16
DEFAULT_MAX_SEQUENCE_LENGTH_FOR_TESTS = 1024


def make_results_reproducible() -> None:
    """
    Make the subsequent instructions produce purely deterministic outputs
    by fixing all the relevant seeds.
    """
    random_seed(0)
    _ = numpy_seed(0)
    _ = torch.manual_seed(0)


class ReproducibleTest:  # pylint: disable=R0903
    """
    Common setup for reproducible tests.
    """

    def __init__(self, *args, **kwargs):  # pylint: disable=W0235
        super().__init__(*args, **kwargs)

    def setUp(self):  # pylint: disable=R0201,C0103
        """
        Setup executed before every method (test) for reproducible results.
        """
        make_results_reproducible()


class TestPositionEncoder(ReproducibleTest, TestCase):
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
            position_dim=DEFAULT_POSITION_FEATURE_DIMENSION_FOR_TESTS,
            upper_bound_max_seq_len=DEFAULT_MAX_SEQUENCE_LENGTH_FOR_TESTS
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

            def forward(self, x):  # pylint: disable=C0103
                """
                Forward propagation.
                """
                x = self.first_linear_layer(x)
                x = self.position_encoding_layer(x)
                x = self.last_linear_layer(x)
                return x

        toy_network = ToyNetwork()
        optimizer = torch.optim.SGD(
            params=toy_network.parameters(),
            lr=1  # exaggeratedly high to avoid negligible updates
        )

        # cleaning any possibly previously cumulated gradient:
        optimizer.zero_grad()
        # taking a snapshot of the initial parameters:
        initial_parameter_dict = deepcopy(
            dict(toy_network.named_parameters())
        )
        # switching to training mode:
        toy_network.train()
        # forward propagation:
        input_tensor = torch.ones(100, 40, input_feature_dimension)
        output_tensor = toy_network(input_tensor)
        toy_loss = output_tensor.sum()
        # backward propagation:
        toy_loss.backward()
        # parameter update:
        optimizer.step()
        # considering parameters now:
        final_parameter_dict = dict(toy_network.named_parameters())
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
                x=input_tensor,
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
        dtype = torch.float

        test_cases = [
            {
                'input': torch.rand(
                    size=(128, 40, 512),
                    dtype=dtype
                ),
                'pos_dim': 16,
                'max_seq_len': 1024,
                'incremental_position': None
            },
            {
                'input': torch.rand(
                    size=(1, 40, 512),
                    dtype=dtype
                ),
                'pos_dim': 16,
                'max_seq_len': 1024,
                'incremental_position': 3
            },
            {
                'input': torch.rand(
                    size=(6, 40, 1024),
                    dtype=dtype
                ),
                'pos_dim': 128,
                'max_seq_len': 12,
                'incremental_position': None
            },
            {
                'input': torch.rand(
                    size=(1, 40, 1024),
                    dtype=dtype
                ),
                'pos_dim': 128,
                'max_seq_len': 12,
                'incremental_position': 4
            },
        ]

        for i, test in enumerate(test_cases, start=1):

            seq_len, batch_size, _ = test['input'].shape

            module = PositionEncoder(
                position_dim=test['pos_dim'],
                upper_bound_max_seq_len=test['max_seq_len']
            )
            output = module(
                x=test['input'],
                incremental_position=test['incremental_position']
            )

            with self.subTest("unpooled features for test n. " + str(i)):
                self.assertTrue(
                    torch.equal(
                        output[..., :-module.pool_input_dim],
                        test['input'][..., :-module.pool_input_dim]
                    )
                )

            with self.subTest("max-pooled features for test n. " + str(i)):
                self.assertTrue(
                    torch.equal(
                        output[..., -module.pool_input_dim:-test['pos_dim']],
                        F.max_pool1d(
                            test['input'][..., -module.pool_input_dim:],
                            kernel_size=2
                        )
                    )
                )

            if test['incremental_position'] is None:
                with self.subTest("position features for test n. " + str(i)):
                    self.assertTrue(
                        torch.equal(
                            output[..., -test['pos_dim']:],
                            (
                                module.position_signals[:seq_len, ]
                                .unsqueeze(dim=1).repeat(1, batch_size, 1)
                            )
                        )
                    )
            else:
                with self.subTest("position features for test n. " + str(i)):
                    self.assertTrue(
                        torch.equal(
                            output[..., -test['pos_dim']:],
                            (
                                module.position_signals[
                                    test['incremental_position'],
                                ]
                                .unsqueeze(dim=0).unsqueeze(dim=0)
                                .repeat(1, batch_size, 1)
                            )
                        )
                    )

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


if __name__ == '__main__':

    unittest_main(verbosity=2)
############################### END OF CHANGES ###############################
