from sbwrapper import SBAccess, MicroscopeStates, byte_util
from sbwrapper import cli


def test_public_api_exposes_expected_symbols():
    assert SBAccess is not None
    assert MicroscopeStates.CurrentObjective.name == "CurrentObjective"
    assert hasattr(byte_util, "uint16_to_bytes")


def test_cli_list_metadata(capsys):
    exit_code = cli.main(["--list-metadata"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "metadata" in captured.out.lower()
