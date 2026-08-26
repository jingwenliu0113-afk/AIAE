"""Colour, kept apart from shape on purpose.

The generation track is colourless: a structure is decided first, and colours
are assigned to its brick slots afterwards from a ``(part, colour)`` stock.
Cross-multiplying eight shapes by dozens of colours into one label set would
multiply the classes for no gain, so shape recognition, colour recognition and
colour assignment are three separate things here.

Assignment is deterministic and never invents stock: if a shape's colours do
not add up to the number of that shape the structure needs, it is refused by
name.
"""
