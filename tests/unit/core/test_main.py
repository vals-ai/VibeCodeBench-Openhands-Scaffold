import json

import pytest

from openhands.core.main import _write_trajectory


def test_write_trajectory_replaces_only_complete_checkpoints(tmp_path):
    file_path = tmp_path / 'trajectory.json'
    file_path.write_text('[{"checkpoint": "previous"}]')

    with pytest.raises(TypeError):
        _write_trajectory(file_path, [{'not_json': {1}}])

    assert json.loads(file_path.read_text()) == [{'checkpoint': 'previous'}]

    _write_trajectory(file_path, [{'checkpoint': 'current'}])

    assert json.loads(file_path.read_text()) == [{'checkpoint': 'current'}]
