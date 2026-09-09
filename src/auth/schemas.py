from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from src.auth.roles import normalize_role
from src.auth.security import validate_password


class PublicUser(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: str
    updated_at: str


class LoginRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    email: str = Field(
        min_length=3,
        max_length=320,
    )
    password: str = Field(
        min_length=1,
        max_length=1024,
    )


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user: PublicUser


class UserCreateRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    email: str = Field(
        min_length=3,
        max_length=320,
    )
    full_name: str = Field(
        min_length=1,
        max_length=200,
    )
    password: str = Field(
        min_length=1,
        max_length=1024,
    )
    role: str
    is_active: bool = True

    @field_validator("email")
    @classmethod
    def validate_email(cls, value):
        email = value.strip().casefold()

        if (
            "@" not in email
            or email.startswith("@")
            or email.endswith("@")
        ):
            raise ValueError(
                "A valid email address is required."
            )

        return email

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, value):
        full_name = value.strip()

        if not full_name:
            raise ValueError(
                "Full name is required."
            )

        return full_name

    @field_validator("password")
    @classmethod
    def validate_new_password(cls, value):
        return validate_password(value)

    @field_validator("role")
    @classmethod
    def validate_user_role(cls, value):
        return normalize_role(value)


class UserUpdateRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    full_name: str | None = Field(
        default=None,
        max_length=200,
    )
    role: str | None = None
    is_active: bool | None = None

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, value):
        if value is None:
            return None

        full_name = value.strip()

        if not full_name:
            raise ValueError(
                "Full name cannot be empty."
            )

        return full_name

    @field_validator("role")
    @classmethod
    def validate_user_role(cls, value):
        if value is None:
            return None

        return normalize_role(value)

    @model_validator(mode="after")
    def validate_update_fields(self):
        if all(
            value is None
            for value in (
                self.full_name,
                self.role,
                self.is_active,
            )
        ):
            raise ValueError(
                "At least one update field is required."
            )

        return self


class PasswordResetRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    new_password: str = Field(
        min_length=1,
        max_length=1024,
    )

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, value):
        return validate_password(value)
