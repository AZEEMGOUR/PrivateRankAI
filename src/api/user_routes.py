from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from src.auth import dependencies as auth_dependencies
from src.auth.dependencies import require_admin
from src.auth.roles import (
    ADMIN_ROLE,
    role_matches,
)
from src.auth.schemas import (
    PasswordResetRequest,
    PublicUser,
    UserCreateRequest,
    UserUpdateRequest,
)
from src.auth.security import hash_password
from src.storage.user_store import (
    DuplicateEmailError,
    LastActiveAdminError,
)


router = APIRouter(
    prefix="/users",
    tags=["users"],
)


def get_user_or_404(user_id):
    user = (
        auth_dependencies.user_store
        .get_user_by_id(user_id)
    )

    if user is None:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail="User not found.",
        )

    return user


@router.get(
    "",
    response_model=list[PublicUser],
)
def list_users(
    current_admin=Depends(
        require_admin
    ),
):
    return (
        auth_dependencies.user_store
        .list_users()
    )


@router.post(
    "",
    response_model=PublicUser,
    status_code=status.HTTP_201_CREATED,
)
def create_user(
    request: UserCreateRequest,
    current_admin=Depends(
        require_admin
    ),
):
    try:
        return (
            auth_dependencies.user_store
            .create_user(
                email=request.email,
                full_name=request.full_name,
                password_hash=hash_password(
                    request.password
                ),
                role=request.role,
                is_active=request.is_active,
            )
        )
    except DuplicateEmailError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error


@router.get(
    "/{user_id}",
    response_model=PublicUser,
)
def get_user(
    user_id: str,
    current_admin=Depends(
        require_admin
    ),
):
    return get_user_or_404(user_id)


@router.patch(
    "/{user_id}",
    response_model=PublicUser,
)
def update_user(
    user_id: str,
    request: UserUpdateRequest,
    current_admin=Depends(
        require_admin
    ),
):
    get_user_or_404(user_id)

    if user_id == current_admin["id"]:
        if request.is_active is False:
            raise HTTPException(
                status_code=(
                    status.HTTP_400_BAD_REQUEST
                ),
                detail=(
                    "You cannot deactivate your own account."
                ),
            )

        if (
            request.role is not None
            and not role_matches(
                request.role,
                ADMIN_ROLE,
            )
        ):
            raise HTTPException(
                status_code=(
                    status.HTTP_400_BAD_REQUEST
                ),
                detail=(
                    "You cannot remove your own "
                    "administrator role."
                ),
            )

    try:
        user = (
            auth_dependencies.user_store
            .update_user(
                user_id,
                full_name=request.full_name,
                role=request.role,
                is_active=request.is_active,
                protected_role=ADMIN_ROLE,
            )
        )
    except LastActiveAdminError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=str(error),
        ) from error

    if user is None:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail="User not found.",
        )

    return user


@router.post(
    "/{user_id}/reset-password",
    response_model=PublicUser,
)
def reset_user_password(
    user_id: str,
    request: PasswordResetRequest,
    current_admin=Depends(
        require_admin
    ),
):
    get_user_or_404(user_id)

    user = (
        auth_dependencies.user_store
        .update_password(
            user_id,
            hash_password(
                request.new_password
            ),
        )
    )

    if user is None:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail="User not found.",
        )

    return user
