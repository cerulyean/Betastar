import os
import unittest
from collections import Counter

from parser import Parser, UnitInitExclusionList
from models import Message, BuildingEvent, MacroStatistics, UnitLifetime

REPLAY_DIR = os.path.join(os.path.dirname(__file__), "replays")
REPLAY_FILENAMES = [
    "AbyssalReefLE.SC2Replay",
    "Ultralove.SC2Replay",
]


class TestParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parsers = {
            fn: Parser(os.path.join(REPLAY_DIR, fn)) for fn in REPLAY_FILENAMES
        }

    @classmethod
    def _foreach(cls):
        for name, parser in cls.parsers.items():
            yield name, parser

    def test_can_load_tracker_events(self):
        for name, parser in self._foreach():
            with self.subTest(replay=name):
                self.assertGreater(
                    len(parser.trackerEvents),
                    0,
                    f"No tracker events decoded for {name}.",
                )

    def test_messages(self):
        for name, parser in self._foreach():
            with self.subTest(replay=name):
                messages = parser.get_messages()
                self.assertTrue(all(isinstance(m, Message) for m in messages))
                self.assertEqual(messages, sorted(messages, key=lambda m: m.gameloop))

    def test_building_events(self):
        for name, parser in self._foreach():
            with self.subTest(replay=name):
                events = parser.get_building_events()
                self.assertTrue(all(isinstance(e, BuildingEvent) for e in events))
                self.assertFalse(
                    any(e.buildingName in UnitInitExclusionList for e in events),
                    f"Exclusion list unit leaked into building events in {name}",
                )
                tag_counts = Counter(e.buildingTag for e in events)
                self.assertTrue(any(count > 1 for count in tag_counts.values()))
                self.assertEqual(events, sorted(events, key=lambda e: e.gameloop))

    def test_macro_statistics(self):
        for name, parser in self._foreach():
            with self.subTest(replay=name):
                stats = parser.get_macro_statistics()
                self.assertTrue(all(isinstance(s, MacroStatistics) for s in stats))
                self.assertEqual(stats, sorted(stats, key=lambda s: s.gameloop))

    def test_unit_lifetimes(self):
        for name, parser in self._foreach():
            with self.subTest(replay=name):
                lifetimes = parser.get_unit_lifetimes()
                self.assertTrue(
                    all(isinstance(u, UnitLifetime) for u in lifetimes)
                )
                self.assertTrue(all(len(u.positions) > 0 for u in lifetimes))
                self.assertTrue(any(u.dead for u in lifetimes))
                self.assertTrue(any(not u.dead for u in lifetimes))


if __name__ == "__main__":
    unittest.main(verbosity=2)
