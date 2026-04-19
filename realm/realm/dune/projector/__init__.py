from .tp import TransformerProjector


def get_projector(input_dim: int, output_dim: int):
    assert input_dim > 0, input_dim
    assert output_dim > 0, output_dim
    return TransformerProjector(input_dim=input_dim, output_dim=output_dim)
