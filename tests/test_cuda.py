import pytest
import torch


def test_pytorch_can_be_imported() -> None:
    assert torch.__version__


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is not available")
def test_cuda_matrix_multiplication() -> None:
    device = torch.device("cuda")

    x = torch.randn((512, 512), device=device)
    result = x @ x

    assert result.is_cuda
    assert result.shape == (512, 512)
    assert torch.isfinite(result).all().item()
    assert torch.cuda.get_device_name(0)
