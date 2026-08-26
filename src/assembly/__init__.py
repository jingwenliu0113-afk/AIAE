"""A build order for a finished structure, and the checks at every step.

Given a validated brick list, produce a sequence of steps that a person could
follow: start on the ground, never place a brick with nothing under it, allow
several grounded sub-assemblies to be joined later by a beam, and re-verify
bounds, collisions, inventory and the accumulated structure after every step.

``stud_only_connected`` is connectivity and nothing else -- adjacent-layer
footprint overlap. This package does not check centre of mass, moments or
whether a model stands up, and nothing here may be described as support or
stability.
"""
