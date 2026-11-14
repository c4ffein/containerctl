"""The map of the classes in containerctl's module docstring: a table that lies is worse than none, so every
top-level class must be in it, once, nothing in it may be anything else, and the boxes must close."""

import inspect
import unittest

import containerctl


def tables() -> list[list[str]]:
    """The docstring's tables, each as its lines (borders included)"""
    found, current = [], []
    for line in containerctl.__doc__.splitlines():
        if line[:1] in ("┌", "│", "└"):
            current.append(line)
        elif current:
            found.append(current)
            current = []
    return found + ([current] if current else [])


def mapped() -> list[str]:
    """The names in the first column, in the order they are written"""
    return [line.split("│")[1].strip() for table in tables() for line in table if line.startswith("│")]


def defined() -> list[str]:
    """The classes containerctl.py itself defines, in the order of the file"""
    classes = [c for c in vars(containerctl).values() if inspect.isclass(c) and c.__module__ == containerctl.__name__]
    return [c.__name__ for c in sorted(classes, key=lambda c: inspect.getsourcelines(c)[1])]


class ClassMapTests(unittest.TestCase):
    def test_every_class_is_in_the_map_and_nothing_else(self):
        self.assertEqual(sorted(mapped()), sorted(defined()))

    def test_no_class_is_there_twice(self):
        names = mapped()
        self.assertEqual(len(names), len(set(names)))

    def test_every_row_says_something(self):
        for table in tables():
            for line in table:
                if line.startswith("│"):
                    self.assertTrue(line.split("│")[2].strip(), line)

    def test_boxes_are_closed_and_aligned(self):
        self.assertTrue(tables())
        for table in tables():
            self.assertEqual((table[0][0], table[0][-1]), ("┌", "┐"))
            self.assertEqual((table[-1][0], table[-1][-1]), ("└", "┘"))
            self.assertEqual(len({len(line) for line in table}), 1, table[0])
            # the middle bar sits under the ┬ on every row
            self.assertEqual({line.index("│", 1) for line in table[1:-1]}, {table[0].index("┬")})
