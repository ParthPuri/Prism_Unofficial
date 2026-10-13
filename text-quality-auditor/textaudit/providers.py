"""Provider configuration; credentials are read at execution, never serialized."""
import os
import urllib.parse
from dataclasses import dataclass

HF_BASE_URL = "https://router.huggingface.co/v1"
HF_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507:nscale"
LOCAL_BASE_URL = "http://127.0.0.1:11434/v1"


@dataclass(frozen=True)
class Provider:
    name: str
    model: str
    base_url: str
    token_env: str | None
    allow_remote: bool
    structured_schema: bool

    def credentials(self) -> str | None:
        if self.token_env is None:
            return None
        token=os.getenv(self.token_env)
        if self.name=="huggingface" and not token:
            raise ValueError("HF_TOKEN is not set. Set a Hugging Face token with Inference Providers permission in your local environment; do not paste it into chat.")
        return token


def resolve_provider(name="huggingface",model=None,base_url=None,allow_remote=False) -> Provider:
    if name=="huggingface":
        if base_url and base_url.rstrip("/")!=HF_BASE_URL:
            raise ValueError("Hugging Face uses its fixed router endpoint. Use --provider compatible for a custom endpoint.")
        return Provider(name,model or HF_MODEL,HF_BASE_URL,"HF_TOKEN",True,True)
    if name=="local":
        endpoint=base_url or LOCAL_BASE_URL
        parsed=urllib.parse.urlparse(endpoint)
        if parsed.hostname not in {"localhost","127.0.0.1","::1"} or parsed.scheme not in {"http","https"} or parsed.username or parsed.password:
            raise ValueError("Local provider requires a loopback HTTP(S) endpoint.")
        if not model:
            raise ValueError("Local provider needs --model matching an installed local model.")
        # Never forward HF_TOKEN or TEXTAUDIT_API_KEY to a local model server.
        return Provider(name,model,endpoint,None,False,False)
    if name=="compatible":
        if not model or not base_url:
            raise ValueError("Compatible provider requires both --model and --base-url.")
        return Provider(name,model,base_url,"TEXTAUDIT_API_KEY",allow_remote,False)
    raise ValueError("Unknown inference provider.")
