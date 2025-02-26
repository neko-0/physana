import pytest
from click.testing import CliRunner
from collinearw import cli


@pytest.mark.parametrize(
    'command',
    [
        [],
        ['histmaker'],
        ['histmaker', 'run'],
        ['histmanipulate'],
        ['histmanipulate', 'abcd-fake'],
        ['histmanipulate', 'abcd-tf'],
        ['plotmaker'],
        ['plotmaker', 'run'],
        ['utility'],
        ['utility', 'browse'],
        ['utility', 'set'],
    ],
)
def test_help(command):
    runner = CliRunner()
    result = runner.invoke(cli.collinearw, command + ['--help'])
    assert result.exit_code == 0
