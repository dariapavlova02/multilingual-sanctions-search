"""Log values must stay private even in DEBUG mode and exception handling."""

import asyncio
import io
import logging
import json
import os
import subprocess
import sys
import uuid

import pytest

from ai_service.utils.log_privacy import LogPrivacyFilter
from ai_service.utils.logging_config import get_logger


def test_eager_strings_arguments_and_exception_payloads_are_not_emitted():
    private = uuid.uuid4().hex
    logger = get_logger(__name__)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    old_level, old_propagate = logger.level, logger.propagate
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        logger.info(f"Private person: {private}")
        logger.debug("Private identifier: %s", private)
        try:
            raise RuntimeError(private)
        except RuntimeError:
            logger.exception("Private failure")
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logger.propagate = old_propagate
    output = stream.getvalue()
    assert private not in output
    assert "error_type=RuntimeError" in output
    assert len(output.splitlines()) == 3


def test_untrusted_record_paths_do_not_trigger_source_reads(tmp_path):
    private = uuid.uuid4().hex
    source = tmp_path / 'foreign.py'
    source.write_text(f'logger.info("{private}")\n')
    record = logging.LogRecord('foreign', logging.ERROR, str(source), 1, private, (), None)
    assert LogPrivacyFilter().filter(record)
    assert record.getMessage() == 'Log event'


@pytest.mark.asyncio
async def test_real_normalization_keeps_person_names_out_of_captured_log_records(caplog):
    from ai_service.layers.normalization.normalization_service import NormalizationService

    with caplog.at_level(logging.DEBUG):
        results = await asyncio.gather(*[
            NormalizationService().normalize_async(text, language=language)
            for text, language in [('Саши Пушкина', 'ru'), ('Bill Smith', 'en')]
        ])
    assert [result.normalized for result in results] == ['Александр Пушкин', 'William Smith']
    messages = '\n'.join(record.getMessage() for record in caplog.records)
    assert caplog.records
    for value in ['Саши', 'Пушкина', 'Александр', 'Пушкин', 'Bill', 'William', 'Smith']:
        assert value not in messages
    assert '[redacted]' in messages  # Static application event templates remain useful.
    assert any(record.funcName and record.lineno for record in caplog.records)


def test_third_party_handler_filters_do_not_emit_credentials():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(LogPrivacyFilter())
    private = uuid.uuid4().hex
    record = logging.LogRecord('external', logging.WARNING, '/vendor/client.py', 20,
                               'Connection failed: %s', ('https://user:' + private + '@example.invalid',), None)
    handler.handle(record)
    assert private not in stream.getvalue()
    assert 'Log event' in stream.getvalue()


def test_json_formatter_emits_valid_json_without_exception_payload():
    from ai_service.utils.log_privacy import JsonFormatter

    private = uuid.uuid4().hex
    error = ValueError(private)
    record = logging.LogRecord('external', logging.ERROR, '/vendor/client.py', 20,
                               '"Private"\n%s', (private,), (ValueError, error, None))
    payload = JsonFormatter().format(record)
    assert json.loads(payload)['level'] == 'ERROR'
    assert private not in payload


@pytest.mark.parametrize('broken_config', [False, True])
def test_log_level_and_privacy_survive_configuration_fallback(tmp_path, broken_config):
    private = uuid.uuid4().hex
    env = os.environ.copy()
    env['LOG_LEVEL'] = 'WARNING'
    env['TEST_PRIVATE_VALUE'] = private
    env['TEST_LOG_DIR'] = str(tmp_path / 'logs')
    if broken_config:
        bad = tmp_path / 'logging.yml'
        bad.write_text('broken: [' + private + '\n')
        env['LOGGING_CONFIG'] = str(bad)
    else:
        env.pop('LOGGING_CONFIG', None)
    code = '''import logging,os
from ai_service.utils.logging_config import get_logger,setup_logging
setup_logging(log_dir=os.environ['TEST_LOG_DIR'])
logger=get_logger('ai_service.privacy_probe')
assert logger.getEffectiveLevel()==logging.WARNING
logger.warning(os.environ['TEST_PRIVATE_VALUE'])
print('PRIVACY_OK')
'''
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert 'PRIVACY_OK' in result.stdout
    assert private not in result.stdout + result.stderr
    for path in (tmp_path / 'logs').glob('*.log'):
        assert private not in path.read_text()
