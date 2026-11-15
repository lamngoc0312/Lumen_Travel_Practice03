"""Firebase Authentication by REST API. No database is used."""
import os
import requests

BASE_URL = 'https://identitytoolkit.googleapis.com/v1/accounts'


class AuthError(Exception):
    """A readable Firebase authentication error."""


ERROR_MESSAGES = {
    'EMAIL_EXISTS': 'This email is already registered.',
    'EMAIL_NOT_FOUND': 'Incorrect email or password.',
    'INVALID_PASSWORD': 'Incorrect email or password.',
    'INVALID_LOGIN_CREDENTIALS': 'Incorrect email or password.',
    'USER_DISABLED': 'This account has been disabled.',
    'WEAK_PASSWORD': 'The password is too weak (use at least 6 characters).',
    'OPERATION_NOT_ALLOWED': 'Enable Email/Password in Firebase Authentication.',
    'INVALID_EMAIL': 'Please enter a valid email address.',
    'TOO_MANY_ATTEMPTS_TRY_LATER': 'Too many attempts. Please try again later.',
    'API_KEY_INVALID': 'FIREBASE_API_KEY is invalid. Copy apiKey again from Firebase Project settings.',
    'API key not valid. Please pass a valid API key.':
        'FIREBASE_API_KEY is invalid. Copy apiKey again from Firebase Project settings.',
}


def _request(endpoint: str, email: str, password: str) -> dict:
    key = os.getenv('FIREBASE_API_KEY', '').strip()
    if not key:
        raise AuthError('FIREBASE_API_KEY is missing from .env.')
    if not email.strip() or not password:
        raise AuthError('Email and password are required.')

    try:
        result = requests.post(
            f'{BASE_URL}:{endpoint}',
            params={'key': key},
            json={
                'email': email.strip(),
                'password': password,
                'returnSecureToken': True,
            },
            timeout=20,
        )
    except requests.RequestException as exc:
        raise AuthError('Cannot contact Firebase. Check your internet connection.') from exc

    try:
        data = result.json()
    except ValueError as exc:
        raise AuthError('Firebase returned an invalid response.') from exc

    if not result.ok:
        error_code = str(data.get('error', {}).get('message', 'AUTH_ERROR'))
        # Some errors add a suffix with details; preserve a safe friendly message.
        readable = ERROR_MESSAGES.get(error_code, ERROR_MESSAGES.get(error_code.split(':')[0],
                      'Authentication failed. Check your Firebase configuration and credentials.'))
        raise AuthError(readable)

    if not data.get('idToken') or not data.get('email'):
        raise AuthError('Firebase did not return a valid sign-in response.')
    return data


def sign_up(email: str, password: str) -> dict:
    return _request('signUp', email, password)


def sign_in(email: str, password: str) -> dict:
    return _request('signInWithPassword', email, password)
