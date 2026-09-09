import argparse
import getpass
import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.auth.security import hash_password
from src.storage.user_store import UserStore


def create_admin(
    *,
    email,
    full_name,
    password,
    store=None,
):
    store = store or UserStore()
    store.initialize_database()

    if store.get_user_by_email(email):
        raise ValueError(
            "A user with this email already exists."
        )

    return store.create_user(
        email=email,
        full_name=full_name,
        password_hash=hash_password(
            password
        ),
        role="Admin",
        is_active=True,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Create the first PrivateRankAI administrator."
        )
    )
    parser.add_argument(
        "--email",
        help="Administrator email address.",
    )
    parser.add_argument(
        "--name",
        dest="full_name",
        help="Administrator full name.",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    email = (
        arguments.email
        or input("Admin email: ")
    ).strip()
    full_name = (
        arguments.full_name
        or input("Admin full name: ")
    ).strip()

    store = UserStore()
    store.initialize_database()

    if store.get_user_by_email(email):
        print(
            "A user with this email already exists.",
            file=sys.stderr,
        )
        return 1

    password = getpass.getpass(
        "Password: "
    )
    confirmation = getpass.getpass(
        "Confirm password: "
    )

    if password != confirmation:
        print(
            "Passwords do not match.",
            file=sys.stderr,
        )
        return 1

    try:
        user = create_admin(
            email=email,
            full_name=full_name,
            password=password,
            store=store,
        )
    except (ValueError, sqlite3.IntegrityError) as error:
        print(
            str(error),
            file=sys.stderr,
        )
        return 1

    print(
        "Admin created successfully:"
    )
    print(
        f"  ID: {user['id']}"
    )
    print(
        f"  Email: {user['email']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
