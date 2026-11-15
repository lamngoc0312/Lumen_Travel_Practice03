"""Offline checks for readable Firebase Authentication behavior."""
import unittest
from unittest.mock import Mock, patch

import firebase_client


class TestFirebaseClient(unittest.TestCase):
    def test_missing_api_key_is_reported_before_network_call(self):
        with patch.dict(firebase_client.os.environ, {}, clear=True), \
                patch.object(firebase_client.requests, 'post') as post:
            with self.assertRaisesRegex(firebase_client.AuthError, 'FIREBASE_API_KEY is missing'):
                firebase_client.sign_in('student@example.com', 'secret1')
        post.assert_not_called()

    def test_invalid_credentials_have_safe_message(self):
        response = Mock(ok=False)
        response.json.return_value = {'error': {'message': 'INVALID_LOGIN_CREDENTIALS'}}
        with patch.dict(firebase_client.os.environ, {'FIREBASE_API_KEY': 'test-key'}), \
                patch.object(firebase_client.requests, 'post', return_value=response):
            with self.assertRaisesRegex(firebase_client.AuthError, 'Incorrect email or password'):
                firebase_client.sign_in('student@example.com', 'wrong-password')

    def test_invalid_api_key_has_actionable_message(self):
        response = Mock(ok=False)
        response.json.return_value = {
            'error': {'message': 'API key not valid. Please pass a valid API key.'},
        }
        with patch.dict(firebase_client.os.environ, {'FIREBASE_API_KEY': 'invalid-key'}), \
                patch.object(firebase_client.requests, 'post', return_value=response):
            with self.assertRaisesRegex(firebase_client.AuthError, 'Copy apiKey again'):
                firebase_client.sign_in('student@example.com', 'secret1')

    def test_successful_sign_up_returns_firebase_session(self):
        response = Mock(ok=True)
        response.json.return_value = {
            'email': 'student@example.com',
            'idToken': 'test-token',
            'expiresIn': '3600',
        }
        with patch.dict(firebase_client.os.environ, {'FIREBASE_API_KEY': 'test-key'}), \
                patch.object(firebase_client.requests, 'post', return_value=response) as post:
            result = firebase_client.sign_up('student@example.com', 'secret1')
        self.assertEqual(result['idToken'], 'test-token')
        self.assertTrue(post.call_args.args[0].endswith('/accounts:signUp'))


if __name__ == '__main__':
    unittest.main()
