from __future__ import annotations

import hmac

from fastapi import HTTPException, Request, status


async def require_api_key(request: Request) -> None:
    settings = request.app.state.settings
    if settings.allow_unauthenticated:
        return
    authorization = request.headers.get("authorization", "")
    scheme, _, credential = authorization.partition(" ")
    expected = settings.api_key_value
    if (
        scheme.lower() != "bearer"
        or not credential
        or expected is None
        or not hmac.compare_digest(credential, expected)
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthorized")
