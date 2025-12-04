# tests/test_tiles.py
import numpy as np

from pipeline.stap.tiles import extract_tiles_2d, make_tile_specs, overlap_add


def test_tiling_grid_and_overlap_identity():
    H, W = 32, 40
    th, tw, st = 8, 8, 4
    specs = make_tile_specs(H, W, th, tw, st)
    # sanity: last tile hits the end
    assert any(sp.y1 == H for sp in specs)
    assert any(sp.x1 == W for sp in specs)

    # Reconstruct a random image from its own tiles (overlap-add with weights) -> exact
    img = np.random.RandomState(0).rand(H, W).astype(np.float32)
    tiles = extract_tiles_2d(img, specs)
    recon = overlap_add(tiles, specs, H, W, dtype=np.float32)
    np.testing.assert_allclose(recon, img, rtol=1e-6, atol=1e-6)


def test_overlap_add_constant_field():
    H, W = 24, 24
    th, tw, st = 8, 8, 4
    specs = make_tile_specs(H, W, th, tw, st)
    const_tile = np.ones((th, tw), dtype=np.float32)
    tiles = [const_tile for _ in specs]
    out = overlap_add(tiles, specs, H, W, dtype=np.float32)
    assert np.allclose(out, 1.0, atol=1e-6)
