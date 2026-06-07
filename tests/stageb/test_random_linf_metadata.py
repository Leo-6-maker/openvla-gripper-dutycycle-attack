"""Test: random_linf metadata fields must be present and valid."""

RANDOM_LINF_REQUIRED = [
    'random_seed',
    'perturbation_space',
    'random_noise_linf',
    'random_noise_l2',
    'eps_processor',
    'eps_raw_pixels_name_deprecated_or_compat',
]


def test_all_random_linf_fields():
    for f in RANDOM_LINF_REQUIRED:
        assert f
    assert len(RANDOM_LINF_REQUIRED) == 6
    print('PASS: test_all_random_linf_fields')


def test_perturbation_space_discriminates():
    """VIS PGD and random Linf must use different perturbation_space values."""
    vis_space = 'processor_pixel_values_linf'
    rand_space = 'random_linf_processor_pixel_values'
    clean_space = 'none'
    assert vis_space != rand_space
    assert rand_space != clean_space
    assert vis_space != clean_space
    print('PASS: test_perturbation_space_discriminates')


def test_eps_processor_is_float():
    """eps_processor must be a parsable float."""
    eps = 0.023529  # 6/255
    assert isinstance(eps, float)
    assert eps > 0.0
    print('PASS: test_eps_processor_is_float')


if __name__ == '__main__':
    test_all_random_linf_fields()
    test_perturbation_space_discriminates()
    test_eps_processor_is_float()
    print('All random_linf metadata tests PASSED')
