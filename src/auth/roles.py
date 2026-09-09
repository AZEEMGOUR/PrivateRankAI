ADMIN_ROLE = "Admin"

RECOGNIZED_ROLES = (
    ADMIN_ROLE,
    "HR",
    "Finance",
    "Employee",
)

_ROLES_BY_KEY = {
    role.casefold(): role
    for role in RECOGNIZED_ROLES
}


def normalize_role(role):
    clean_role = str(
        role or ""
    ).strip()

    try:
        return _ROLES_BY_KEY[
            clean_role.casefold()
        ]
    except KeyError as error:
        allowed_roles = ", ".join(
            RECOGNIZED_ROLES
        )
        raise ValueError(
            "Role must be one of: "
            f"{allowed_roles}."
        ) from error


def role_matches(role, expected_role):
    return str(role or "").casefold() == (
        str(expected_role or "").casefold()
    )
