from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from student_detection.api.main import app


class TestApiUI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_serve_ui_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Student & People Counting", response.text)
        self.assertIn("Interactive Inspection Lab", response.text)

    def test_serve_ui_alias(self):
        response = self.client.get("/ui")
        self.assertEqual(response.status_code, 200)
        self.assertIn("image-canvas", response.text)

    def test_get_server_info(self):
        response = self.client.get("/api/info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("server", data)
        self.assertIn("model", data)
        self.assertIn("pipeline", data)
        self.assertIn("conf_threshold", data["pipeline"])

    def test_get_sample_image(self):
        response = self.client.get("/api/sample-image")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")


if __name__ == "__main__":
    unittest.main()
