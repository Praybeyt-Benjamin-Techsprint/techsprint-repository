"""Shared gesture label configuration for data collection and training."""

from __future__ import annotations


ACTIONS = [
    "hello",
    "thank_you",
    "see_you_later",
    "see",
    "you",
    "later",
    "yes",
    "no",
    "help",
    "me",
    "father",
    "mother",
    "abuse",
    "please",
    "want",
    "what",
    "eat_food",
    "more",
    "go_to",
    "fine",
    "like",
    "name",
    "meet",
    "nice",
    "Sorry",
    "where",
    "call",
]

STOP_ACTION = "stop"
DELETE_ACTION = "delete"

CONTROL_ACTIONS = [
    STOP_ACTION,
    DELETE_ACTION,
]

MODEL_ACTIONS = ACTIONS + CONTROL_ACTIONS

# Backward-compatible lowercase alias for callers that expect an `actions` array.
actions = ACTIONS
