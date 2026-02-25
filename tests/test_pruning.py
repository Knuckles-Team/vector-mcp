import sys
import unittest
from dataclasses import dataclass

sys.path.insert(0, "/home/genius/Workspace/vector-mcp")
from vector_mcp.vector_agent import prune_large_messages


@dataclass
class MockMessage:
    role: str
    content: str


class TestPruning(unittest.TestCase):
    def test_prune_large_string_content_dict(self):
        large_content = "A" * 6000
        msg = {"role": "tool", "content": large_content}
        messages = [msg]

        pruned = prune_large_messages(messages, max_length=5000)
        self.assertEqual(len(pruned), 1)
        self.assertLess(len(pruned[0]["content"]), 5000)
        self.assertTrue("truncated" in pruned[0]["content"])

    def test_prune_large_string_content_object(self):
        large_content = "B" * 6000
        msg = MockMessage(role="tool", content=large_content)
        messages = [msg]

        pruned = prune_large_messages(messages, max_length=5000)
        self.assertEqual(len(pruned), 1)

        self.assertLess(len(pruned[0].content), 5000)
        self.assertTrue("truncated" in pruned[0].content)

    def test_keep_small_content(self):
        small_content = "Small content"
        msg = {"role": "user", "content": small_content}
        messages = [msg]

        pruned = prune_large_messages(messages, max_length=5000)
        self.assertEqual(len(pruned), 1)
        self.assertEqual(pruned[0]["content"], small_content)


if __name__ == "__main__":
    unittest.main()
