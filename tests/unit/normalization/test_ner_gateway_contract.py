"""NER adapters must share contracts and requested processing must use their hints."""
import asyncio
from types import SimpleNamespace

import pytest

from ai_service.layers.normalization import ner_gateways as public
from ai_service.layers.normalization.ner_gateways import spacy_en, spacy_ru, spacy_uk
from ai_service.layers.normalization.ner_gateways.unified_spacy_gateway import UnifiedSpacyGateway, SupportedLanguage
from ai_service.layers.normalization.processors.normalization_factory import NormalizationFactory, NormalizationConfig


@pytest.mark.parametrize('module', [spacy_en, spacy_ru, spacy_uk])
def test_language_modules_export_the_canonical_hint_contract(module):
    assert module.NERHints is public.NERHints
    hints = module.NERHints(person_spans=[(0, 8)], org_spans=[], entities=[])
    assert hints.person_spans == [(0, 8)]


@pytest.mark.parametrize('language', list(SupportedLanguage))
async def test_gateway_async_output_preserves_entity_positions(monkeypatch, language):
    gateway = UnifiedSpacyGateway()
    text = 'Nova Labs and Nova'
    def model(value):
        assert value == text
        return SimpleNamespace(ents=[SimpleNamespace(text='Nova Labs', label_='ORG', start_char=0,end_char=9), SimpleNamespace(text='Nova',label_='PERSON',start_char=14,end_char=18)])
    monkeypatch.setattr(gateway, '_load_spacy_model', lambda lang:(model, True))
    hints = await gateway.get_ner_hints_async(text, language)
    assert hints.person_spans == [(14,18)]
    assert hints.org_spans == [(0,9)]
    assert all(text[e.start:e.end] == e.text for e in hints.entities)
    assert hints.persons == {'nova'} and hints.organizations == {'nova labs'}
    gateway.close()


async def test_factory_consumes_requested_ner_and_can_disable_it(monkeypatch):
    factory = NormalizationFactory()
    gateway = UnifiedSpacyGateway()
    calls=[]
    def model(text):
        calls.append(text)
        start=text.index('Nebula')
        return SimpleNamespace(ents=[SimpleNamespace(text='Nebula',label_='ORG',start_char=start,end_char=start+6)])
    monkeypatch.setattr(gateway,'_load_spacy_model',lambda language:(model,True))
    factory.ner_gateway = gateway
    config=NormalizationConfig(language='en',enable_spacy_en_ner=True,enable_cache=False,enable_nameparser_en=True,enable_morphology=False)
    result=await factory.normalize_text('Nebula John Smith',config)
    assert result.success
    assert calls, 'Requested NER was not dispatched'
    assert 'Nebula' not in result.normalized.split()
    assert any('ner_' in str(t) for t in result.trace)
    calls.clear()
    config.enable_spacy_en_ner=False
    result=await factory.normalize_text('Nebula John Smith',config)
    assert result.success and not calls
    gateway.close()


@pytest.mark.parametrize('failure', ['missing', 'exception', 'bad-span'])
async def test_requested_ner_failure_is_not_an_empty_success(monkeypatch, failure):
    gateway=UnifiedSpacyGateway()
    def model(text):
        if failure=='exception': raise RuntimeError('controlled inference error')
        return SimpleNamespace(ents=[SimpleNamespace(text='Different',label_='PER',start_char=0,end_char=4)])
    monkeypatch.setattr(gateway,'_load_spacy_model',lambda lang:(None,False) if failure=='missing' else (model,True))
    with pytest.raises(public.NERUnavailableError):
        await gateway.get_ner_hints_async('Nova',SupportedLanguage.ENGLISH)
    gateway.close()


async def test_requested_ner_failure_fails_normalization(monkeypatch):
    factory=NormalizationFactory()
    gateway=UnifiedSpacyGateway()
    monkeypatch.setattr(gateway,'_load_spacy_model',lambda lang:(None,False))
    factory.ner_gateway=gateway
    result=await factory.normalize_text('John Smith',NormalizationConfig(language='en',enable_spacy_en_ner=True))
    assert result.success is False and result.errors
    assert not result.normalized and not result.tokens
    gateway.close()


@pytest.mark.parametrize('language', ['en', 'ru', 'uk'])
async def test_requested_ner_failure_preserves_requested_context(monkeypatch, language):
    factory = NormalizationFactory()
    gateway = UnifiedSpacyGateway()
    monkeypatch.setattr(gateway, '_load_spacy_model', lambda lang: (None, False))
    factory.ner_gateway = gateway
    try:
        for tracing in (False, True):
            result = await factory.normalize_text('Nova', NormalizationConfig(
                language=language, enable_spacy_ner=True, debug_tracing=tracing))
            assert result.success is False and result.errors
            assert result.language == language and result.ner_disabled is False
            assert not result.normalized and not result.tokens and not result.persons
            assert bool(result.trace) is tracing
            if tracing:
                assert result.trace[0].rule == 'normalization_failed'
                assert result.trace[0].output == ''
    finally:
        gateway.close()


async def test_async_ner_is_bounded_and_does_not_block_event_loop(monkeypatch):
    import threading
    from ai_service.utils.inference_queue import InferenceUnavailableError
    entered,release=threading.Event(),threading.Event()
    gateway=UnifiedSpacyGateway(max_pending=0,timeout=2)
    def model(text):
        entered.set()
        assert release.wait(2)
        return SimpleNamespace(ents=[])
    monkeypatch.setattr(gateway,'_load_spacy_model',lambda language:(model,True))
    first=asyncio.create_task(gateway.get_ner_hints_async('Nova','en'))
    try:
        assert await asyncio.wait_for(asyncio.to_thread(entered.wait,1),1.5)
        with pytest.raises(InferenceUnavailableError):
            await gateway.get_ner_hints_async('Another','en')
    finally:
        release.set()
        await first
        gateway.close()


def test_model_diagnostics_are_lazy_and_language_cache_can_reload(monkeypatch):
    gateway=UnifiedSpacyGateway();loaded=[]
    def load(name, *, exclude):
        loaded.append(name)
        return lambda text:SimpleNamespace(ents=[])
    monkeypatch.setattr('spacy.load',load)
    assert all(not row['loaded'] for row in gateway.get_model_info().values())
    assert not loaded
    for _ in range(2): gateway.get_ner_hints('Nova','en')
    assert loaded==['en_core_web_sm']
    gateway.clear_cache('en')
    gateway.get_ner_hints('Nova','en')
    assert loaded==['en_core_web_sm','en_core_web_sm']
    gateway.close()


@pytest.mark.parametrize('language,texts', [
    ('en', ['Apple Inc. CEO Tim Cook announced new products', 'John Smith Jr.',
            'Stephen E. King', "John O'Neil-Smith", 'Smith-Jones',
            'J. Smith INN 001234567890 DOB 1990-01-01']),
    ('ru', ['ООО Рога и копыта', 'Анна Иванова работает в ООО Альфа',
            'И. Петров ИНН 1234567890', 'Иван Петров; Сергей Сидоров']),
    ('uk', ['Анна Ковальська', 'ТОВ ПРИВАТБАНК',
            'Олександр Петренко працює в ТОВ Альфа', 'І. Коваленко ІПН 001234567890']),
])
def test_ner_only_pinned_model_retains_full_pipeline_entities(language, texts):
    import spacy
    gateway = UnifiedSpacyGateway()
    try:
        model, available = gateway._load_spacy_model(SupportedLanguage(language))
        assert available and model.pipe_names == ['ner']
        full = spacy.load(gateway.MODEL_CONFIG[SupportedLanguage(language)]['model_name'])
        for text in texts:
            def entities(doc):
                return [(e.text, e.label_, e.start_char, e.end_char) for e in doc.ents]
            assert entities(model(text)) == entities(full(text))
    finally:
        gateway.close()


@pytest.mark.parametrize('tokens,person,organization,expected',[
    (['Nova','Labs','Nova'],[(10,14)],[(0,9)],['org','org','person']),
    (['J.','Smith','INN','001234567890'],[],[(0,25)],['initial','person','document-marker','id']),
])
def test_ner_preserves_repeated_token_positions_and_identity_evidence(tokens,person,organization,expected):
    from ai_service.layers.normalization.role_tagger_service import RoleTaggerService
    from ai_service.layers.normalization.processors.role_classifier import RoleClassifier
    tagger=RoleTaggerService(role_classifier=RoleClassifier())
    hints=public.NERHints(person_spans=person,org_spans=organization)
    tags=tagger.tag(tokens,'en',{'enable_ner':True,'ner_hints':hints})
    for tag, expected_role in zip(tags, expected):
        if expected_role == 'person':
            assert tag.role.value in {'given','surname','patronymic'}
        elif expected_role == 'document-marker':
            assert not tag.reason.startswith('ner_')
        else:
            assert tag.role.value == expected_role


@pytest.mark.parametrize('text', ['иван ПЕТРОВ','John Smith Jr.','Stephen E. King', 'Smith-Jones', "John O'Neil-Smith"])
async def test_real_ner_does_not_erase_known_persons_or_reclassify_name_roles(text):
    from ai_service.layers.normalization.normalization_service import NormalizationService
    from ai_service.utils.feature_flags import FeatureFlags
    service=NormalizationService()
    language='ru' if 'иван' in text else 'en'
    enabled=await service.normalize_async(text,language=language,feature_flags=FeatureFlags(enable_spacy_ner=True))
    disabled=await service.normalize_async(text,language=language,feature_flags=FeatureFlags(enable_spacy_ner=False))
    assert enabled.success and disabled.success
    assert enabled.normalized==disabled.normalized


async def test_mixed_script_models_do_not_erase_each_others_person_spans(monkeypatch):
    gateway=UnifiedSpacyGateway();calls=[]
    def load(language):
        def model(text):
            calls.append(language.value)
            rows = [('John',0,4,'PER' if language.value=='en' else 'ORG'),
                    ('Анна',5,9,'ORG' if language.value=='en' else 'PER')]
            return SimpleNamespace(ents=[SimpleNamespace(text=value,start_char=start,end_char=end,label_=label) for value,start,end,label in rows])
        return model,True
    monkeypatch.setattr(gateway,'_load_spacy_model',load)
    hints=await gateway.get_ner_hints_async('John Анна','mixed')
    assert hints.person_spans==[(0,4),(5,9)]
    assert hints.org_spans==[]
    assert calls==['en','ru','uk']
    gateway.close()


async def test_identifier_only_normalization_does_not_request_a_language_model(monkeypatch):
    factory=NormalizationFactory();gateway=UnifiedSpacyGateway()
    def unexpected(language):
        pytest.fail('Identifier-only input must not load NER models')
    monkeypatch.setattr(gateway,'_load_spacy_model',unexpected)
    factory.ner_gateway=gateway
    result=await factory.normalize_text('INN 001234567890',NormalizationConfig(language='unknown',enable_spacy_ner=True))
    assert result.success
    assert result.normalized==''
    assert '001234567890' in result.tokens
    gateway.close()
