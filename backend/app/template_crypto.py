from __future__ import annotations

import base64
import hashlib
import os

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


_MAGIC = b"FVENC1\x00"
_NONCE_SIZE = 12
_AAD = b"face-template-embedding-v1"


class TemplateCryptoError(RuntimeError):
    pass


def embedding_to_blob(embedding: np.ndarray, key: str, required: bool) -> tuple[bytes, int]:
    normalized = np.asarray(embedding, dtype=np.float32)
    raw = normalized.tobytes()
    if not key:
        if required:
            raise TemplateCryptoError("人脸模板加密密钥未配置，不能写入明文模板")
        return raw, int(normalized.size)

    aes_key = _normalize_key(key)
    nonce = os.urandom(_NONCE_SIZE)
    encrypted = AESGCM(aes_key).encrypt(nonce, raw, _AAD)
    return _MAGIC + nonce + encrypted, int(normalized.size)


def embedding_from_blob(blob: bytes, dimension: int, key: str) -> np.ndarray:
    payload = bytes(blob)
    if payload.startswith(_MAGIC):
        if not key:
            raise TemplateCryptoError("人脸模板已加密，但当前服务未配置解密密钥")
        nonce_start = len(_MAGIC)
        nonce_end = nonce_start + _NONCE_SIZE
        nonce = payload[nonce_start:nonce_end]
        encrypted = payload[nonce_end:]
        try:
            payload = AESGCM(_normalize_key(key)).decrypt(nonce, encrypted, _AAD)
        except InvalidTag as exc:
            raise TemplateCryptoError("人脸模板解密失败，请检查加密密钥") from exc

    embedding = np.frombuffer(payload, dtype=np.float32, count=dimension).copy()
    norm = max(float(np.linalg.norm(embedding)), 1e-12)
    return (embedding / norm).astype(np.float32)


def is_encrypted_blob(blob: bytes) -> bool:
    return bytes(blob).startswith(_MAGIC)


def _normalize_key(value: str) -> bytes:
    value = value.strip()
    if value.startswith("base64:"):
        decoded = base64.urlsafe_b64decode(_pad_base64(value[len("base64:") :]))
        if len(decoded) != 32:
            raise TemplateCryptoError("人脸模板加密密钥必须为 32 字节")
        return decoded
    raw = value.encode("utf-8")
    if len(raw) == 32:
        return raw
    return hashlib.sha256(raw).digest()


def _pad_base64(value: str) -> bytes:
    padded = value + "=" * (-len(value) % 4)
    return padded.encode("ascii")
