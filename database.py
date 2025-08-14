import json
import os
import logging
from typing import List, Dict, Any, Optional
import uuid

# Initialize logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define database paths
DB_FOLDER = 'data'
USERS_FILE = os.path.join(DB_FOLDER, 'users.json')
MISSING_PERSONS_FILE = os.path.join(DB_FOLDER, 'missing_persons.json')

# Ensure database folder exists
os.makedirs(DB_FOLDER, exist_ok=True)


def initialize_database() -> None:
    """Initialize the database files if they don't exist"""
    # Initialize users file
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump([], f)
        logger.info(f"Created users database at {USERS_FILE}")

    # Initialize missing persons file
    if not os.path.exists(MISSING_PERSONS_FILE):
        with open(MISSING_PERSONS_FILE, 'w') as f:
            json.dump([], f)
        logger.info(f"Created missing persons database at {MISSING_PERSONS_FILE}")


# User functions
def get_all_users() -> List[Dict[str, Any]]:
    """Get all users from the database"""
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error reading users: {str(e)}")
        return []


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Get a user by username"""
    users = get_all_users()
    for user in users:
        if user['username'] == username:
            return user
    return None


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Get a user by ID"""
    users = get_all_users()
    for user in users:
        if user['id'] == user_id:
            return user
    return None


def add_user(user_data: Dict[str, Any]) -> bool:
    """Add a new user to the database"""
    try:
        users = get_all_users()
        users.append(user_data)
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error adding user: {str(e)}")
        return False


def update_user(user_id: str, updated_data: Dict[str, Any]) -> bool:
    """Update an existing user"""
    try:
        users = get_all_users()
        for i, user in enumerate(users):
            if user['id'] == user_id:
                # Update the user data
                users[i].update(updated_data)
                with open(USERS_FILE, 'w') as f:
                    json.dump(users, f, indent=2)
                return True
        return False
    except Exception as e:
        logger.error(f"Error updating user: {str(e)}")
        return False


# Missing persons functions
def get_all_missing_persons() -> List[Dict[str, Any]]:
    """Get all missing persons from the database"""
    try:
        with open(MISSING_PERSONS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error reading missing persons: {str(e)}")
        return []


def get_missing_person(person_id: str) -> Optional[Dict[str, Any]]:
    """Get a missing person by ID"""
    persons = get_all_missing_persons()
    for person in persons:
        if person['id'] == person_id:
            return person
    return None


def add_missing_person(person_data: Dict[str, Any]) -> bool:
    """Add a new missing person to the database"""
    try:
        persons = get_all_missing_persons()
        persons.append(person_data)
        with open(MISSING_PERSONS_FILE, 'w') as f:
            json.dump(persons, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error adding missing person: {str(e)}")
        return False


def update_missing_person(person_id: str, updated_data: Dict[str, Any]) -> bool:
    """Update an existing missing person"""
    try:
        persons = get_all_missing_persons()
        for i, person in enumerate(persons):
            if person['id'] == person_id:
                # Update the person data
                persons[i].update(updated_data)
                with open(MISSING_PERSONS_FILE, 'w') as f:
                    json.dump(persons, f, indent=2)
                return True
        return False
    except Exception as e:
        logger.error(f"Error updating missing person: {str(e)}")
        return False


def remove_missing_person(person_id: str) -> bool:
    """Remove a missing person from the database"""
    try:
        persons = get_all_missing_persons()
        for i, person in enumerate(persons):
            if person['id'] == person_id:
                # Remove the person
                del persons[i]
                with open(MISSING_PERSONS_FILE, 'w') as f:
                    json.dump(persons, f, indent=2)
                return True
        return False
    except Exception as e:
        logger.error(f"Error removing missing person: {str(e)}")
        return False
