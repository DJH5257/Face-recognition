import unittest

from backend.app.business_store import BusinessStore, BusinessStoreError
from backend.app.config import Settings


class BusinessStoreConfigTests(unittest.TestCase):
    def test_disabled_store_does_not_connect(self):
        store = BusinessStore(Settings())
        self.assertFalse(store.enabled)
        self.assertEqual(store.check_schema(), {"enabled": False})

    def test_enabled_store_requires_dsn(self):
        with self.assertRaises(BusinessStoreError):
            BusinessStore(Settings(business_db_enabled=True))

    def test_table_names_are_validated(self):
        with self.assertRaises(BusinessStoreError):
            BusinessStore(Settings(business_db_profile_table="fa_face_profile;DROP"))
