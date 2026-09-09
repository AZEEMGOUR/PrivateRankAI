import logging
import os
from datetime import datetime, timedelta, timezone

import jwt
from pwdlib import PasswordHash
from pwdlib.exceptions import PwdlibError


logger = logging.getLogger(__name__)

JWT_ALGORITHM = "HS256"
SECRET_KEY_ENVIRONMENT_VARIABLE = (
    "PRIVATERANK_SECRET_KEY"
)
TOKEN_EXPIRY_ENVIRONMENT_VARIABLE = (
    "PRIVATERANK_ACCESS_TOKEN_EXPIRE_MINUTES"
)
DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES = 60
MINIMUM_PASSWORD_LENGTH = 12

# Development only. Production deployments must set
# PRIVATERANK_SECRET_KEY to a strong, random secret.
DEVELOPMENT_SECRET_KEY = (
    "private-rank-development-only-change-this-secret"
)

password_hasher = PasswordHash.recommended()


def get_access_token_expire_minutes():
    raw_value = os.getenv(
        TOKEN_EXPIRY_ENVIRONMENT_VARIABLE,
        str(DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    try:
        minutes = int(raw_value)
    except ValueError as error:
        raise RuntimeError(
            f"{TOKEN_EXPIRY_ENVIRONMENT_VARIABLE} must be an integer."
        ) from error

    if minutes <= 0:
        raise RuntimeError(
            f"{TOKEN_EXPIRY_ENVIRONMENT_VARIABLE} must be positive."
        )

    return minutes


def get_secret_key():
    return (
        os.getenv(
            SECRET_KEY_ENVIRONMENT_VARIABLE,
            "",
        ).strip()
        or DEVELOPMENT_SECRET_KEY
    )


def warn_if_using_development_secret():
    if not os.getenv(
        SECRET_KEY_ENVIRONMENT_VARIABLE,
        "",
    ).strip():
        logger.warning(
            "PRIVATERANK_SECRET_KEY is not set. "
            "Using the development-only JWT secret; "
            "do not use this fallback in production."
        )


def hash_password(password):
    password = str(password or "")

    if len(password) < MINIMUM_PASSWORD_LENGTH:
        raise ValueError(
            "Password must be at least "
            f"{MINIMUM_PASSWORD_LENGTH} characters."
        )

    return password_hasher.hash(password)


def verify_password(password, password_hash):
    try:
        return password_hasher.verify(
            str(password or ""),
            str(password_hash or ""),
        )
    except (
        PwdlibError,
        TypeError,
        ValueError,
    ):
        return False


def create_access_token(user_id):
    issued_at = datetime.now(timezone.utc)
    expire_minutes = (
        get_access_token_expire_minutes()
    )

    return jwt.encode(
        {
            "sub": str(user_id),
            "iat": issued_at,
            "exp": issued_at
            + timedelta(
                minutes=expire_minutes
            ),
        },
        get_secret_key(),
        algorithm=JWT_ALGORITHM,
    )


def decode_access_token(token):
    try:
        payload = jwt.decode(
            token,
            get_secret_key(),
            algorithms=[JWT_ALGORITHM],
            options={
                "require": [
                    "sub",
                    "iat",
                    "exp",
                ]
            },
        )
    except jwt.InvalidTokenError as error:
        raise ValueError(
            "Invalid or expired access token."
        ) from error

    user_id = str(
        payload.get("sub", "")
    ).strip()

    if not user_id:
        raise ValueError(
            "Invalid or expired access token."
        )

    return user_id
