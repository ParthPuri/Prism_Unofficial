import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from textaudit.cli import main
from textaudit.core import Config,LocalDataset
from textaudit.planner import plan
from textaudit.providers import HF_BASE_URL, HF_MODEL, GROQ_BASE_URL, GROQ_MODEL, resolve_provider
from scripts.context_demo import example_policies


class Response:
    def __init__(self,body): self.body=json.dumps(body).encode()
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def read(self,n): return self.body[:n]


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        p=self.root/"dataset.jsonl";p.write_text('{"text":"PRIVATE_MESSAGE_TEXT"}\n')
        self.dataset=LocalDataset(p,Config());self.addCleanup(self.dataset.close)
        self.policy=dict(example_policies())["event_monitoring"]

    def test_default_is_huggingface_with_explicit_model(self):
        p=resolve_provider()
        self.assertEqual(p.base_url,HF_BASE_URL);self.assertEqual(p.model,HF_MODEL)
        self.assertTrue(p.structured_schema)

    def test_missing_token_fails_before_any_network_or_dataset_scan(self):
        with patch.dict(os.environ,{},clear=True),patch('textaudit.planner.urllib.request.build_opener') as network:
            error=io.StringIO()
            with contextlib.redirect_stderr(error):
                code=main([str(self.root/'does-not-exist.csv'),'--use-case','Monitor event messages'])
            self.assertEqual(code,1);self.assertIn('HF_TOKEN is not set',error.getvalue())
            network.assert_not_called()

    def test_token_is_never_forwarded_to_local_provider(self):
        with patch.dict(os.environ,{'HF_TOKEN':'hf_test_secret','TEXTAUDIT_API_KEY':'another_secret'}):
            p=resolve_provider('local',model='qwen3:8b')
            self.assertIsNone(p.credentials());self.assertFalse(p.allow_remote)

    def test_huggingface_endpoint_cannot_be_overridden(self):
        with self.assertRaises(ValueError):resolve_provider('huggingface',base_url='https://wrong.example/v1')

    def test_local_rejects_remote_endpoint(self):
        with self.assertRaises(ValueError):resolve_provider('local',model='qwen',base_url='https://remote.example/v1')

    def test_local_requires_explicit_model(self):
        with self.assertRaises(ValueError):resolve_provider('local')

    def test_hf_request_schema_auth_and_safe_provenance(self):
        captured=[]
        def request(req,timeout):
            captured.append(req)
            return Response({'choices':[{'finish_reason':'stop','message':{'content':json.dumps(self.policy)}}],
                             'usage':{'prompt_tokens':100,'completion_tokens':80}})
        with patch.dict(os.environ,{'HF_TOKEN':'hf_test_secret'}):
            p=resolve_provider()
            with patch('textaudit.planner.urllib.request.build_opener') as opener:
                opener.return_value.open.side_effect=request
                actual,meta=plan(self.dataset,'Monitor events',p.model,p.base_url,self.root/'cache',
                    p.allow_remote,api_key=p.credentials(),provider=p.name,structured_schema=p.structured_schema)
        self.assertEqual(actual,self.policy)
        req=captured[0]; payload=json.loads(req.data)
        self.assertEqual(req.full_url,HF_BASE_URL+'/chat/completions')
        self.assertEqual(req.get_header('Authorization'),'Bearer hf_test_secret')
        self.assertEqual(payload['response_format']['type'],'json_schema')
        self.assertTrue(payload['response_format']['json_schema']['strict'])
        self.assertNotIn('PRIVATE_MESSAGE_TEXT',req.data.decode())
        self.assertNotIn('hf_test_secret',json.dumps(meta))
        for f in (self.root/'cache').glob('*.json'):
            self.assertNotIn('hf_test_secret',f.read_text())
        self.assertEqual(meta['provider'],'huggingface')

    def test_provider_error_does_not_log_response_secrets(self):
        with patch('textaudit.planner.urllib.request.build_opener') as opener:
            opener.return_value.open.side_effect=HTTPError(HF_BASE_URL,401,'secret_response',{},io.BytesIO(b'hf_test_secret'))
            with self.assertRaises(ValueError) as error:
                plan(self.dataset,'Monitor events',HF_MODEL,HF_BASE_URL,self.root/'cache',True,api_key='hf_test_secret')
        self.assertIn('HTTP 401',str(error.exception));self.assertNotIn('secret',str(error.exception))

    def test_truncated_output_does_not_create_policy_cache(self):
        with patch('textaudit.planner.urllib.request.build_opener') as opener:
            opener.return_value.open.return_value=Response({'choices':[{'finish_reason':'length','message':{'content':'{}'}}]})
            with self.assertRaisesRegex(ValueError,'truncated'):
                plan(self.dataset,'Monitor events','test-model','http://127.0.0.1:11434/v1',self.root/'cache')
        self.assertEqual(list((self.root/'cache').glob('*.json')),[])

    def test_groq_provider_resolution_and_credentials(self):
        with patch.dict(os.environ,{'GROQ_API_KEY':'gsk_test_secret'}):
            p=resolve_provider('groq')
            self.assertEqual(p.base_url,GROQ_BASE_URL)
            self.assertEqual(p.model,GROQ_MODEL)
            self.assertEqual(p.credentials(),'gsk_test_secret')
            self.assertTrue(p.allow_remote)
            self.assertFalse(p.structured_schema)

    def test_gemini_provider_resolution_and_credentials(self):
        with patch.dict(os.environ,{'GEMINI_API_KEY':'gemini_test_secret'}):
            p=resolve_provider('gemini')
            self.assertEqual(p.name,'gemini')
            self.assertEqual(p.base_url,'https://generativelanguage.googleapis.com/v1beta/openai')
            self.assertEqual(p.model,'gemini-3.8-flash')
            self.assertEqual(p.credentials(),'gemini_test_secret')
            self.assertTrue(p.allow_remote)
            self.assertFalse(p.structured_schema)

    def test_gemini_google_api_key_fallback(self):
        with patch.dict(os.environ,{'GOOGLE_API_KEY':'google_fallback_secret'},clear=True):
            p=resolve_provider('google')
            self.assertEqual(p.credentials(),'google_fallback_secret')

    def test_gemini_missing_token_raises(self):
        with patch.dict(os.environ,{},clear=True):
            p=resolve_provider('gemini')
            with self.assertRaisesRegex(ValueError,'GEMINI_API_KEY'):
                p.credentials()


if __name__=='__main__': unittest.main()

