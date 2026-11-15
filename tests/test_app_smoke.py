"""Smoke-test the signed-out Streamlit page without external services."""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class TestAppSmoke(unittest.TestCase):
    def test_signed_out_user_only_sees_authentication_ui(self):
        app_path = Path(__file__).resolve().parents[1] / 'app.py'
        app = AppTest.from_file(str(app_path)).run(timeout=10)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual([tab.label for tab in app.tabs], ['Sign in', 'Create account'])
        self.assertEqual(len(app.chat_input), 0)
        self.assertIn('Sign in to access', app.info[0].value)


if __name__ == '__main__':
    unittest.main()
