from fastapi import (
    Depends,
    HTTPException,
    status,
)
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

from src.auth.security import (
    decode_access_token,
    hash_password,
    verify_password,
)
from src.storage.user_store import UserStore


bearer_scheme = HTTPBearer(
    auto_error=False
)
user_store = UserStore()

# Performing the same password verification work for unknown emails
# reduces login timing differences without revealing user existence.
DUMMY_PASSWORD_HASH = hash_password(
    "development-dummy-password"
)


def authentication_error():
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials.",
        headers={
            "WWW-Authenticate": "Bearer"
        },
    )


def authenticate_user(email, password):
    stored_user = (
        user_store
        .get_user_credentials_by_email(
            email
        )
    )

    password_hash = (
        stored_user["password_hash"]
        if stored_user is not None
        else DUMMY_PASSWORD_HASH
    )
    password_is_valid = verify_password(
        password,
        password_hash,
    )

    if (
        stored_user is None
        or not password_is_valid
        or not stored_user["is_active"]
    ):
        return None

    stored_user.pop(
        "password_hash",
        None,
    )

    return stored_user


def get_current_user(
    credentials: HTTPAuthorizationCredentials
    | None = Depends(bearer_scheme),
):
    if credentials is None:
        raise authentication_error()

    try:
        user_id = decode_access_token(
            credentials.credentials
        )
    except ValueError as error:
        raise authentication_error() from error

    user = user_store.get_user_by_id(
        user_id
    )

    if (
        user is None
        or not user["is_active"]
    ):
        raise authentication_error()

    return user
