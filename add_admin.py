"""Create an admin user from the command line.

Usage:
    python add_admin.py --name "Admin" --email admin@example.com --password "choose-a-strong-password"

The password is hashed with scrypt before storage (never plaintext).
"""
import argparse

from app import app, db, Faculty
from security import hash_password


def main():
    ap = argparse.ArgumentParser(description="Create a ProFace admin user")
    ap.add_argument("--name", default="Admin")
    ap.add_argument("--email", required=True)
    ap.add_argument("--password", required=True)
    args = ap.parse_args()

    with app.app_context():
        existing = Faculty.query.filter_by(email=args.email).first()
        if existing:
            existing.password = hash_password(args.password)
            existing.is_admin = True
            existing.approved = True
            db.session.commit()
            print(f"Updated existing faculty '{args.email}' to admin.")
            return
        admin = Faculty(
            name=args.name,
            email=args.email,
            password=hash_password(args.password),
            is_admin=True,
            approved=True,
        )
        db.session.add(admin)
        db.session.commit()
        print(f"Admin '{args.email}' created!")


if __name__ == "__main__":
    main()
