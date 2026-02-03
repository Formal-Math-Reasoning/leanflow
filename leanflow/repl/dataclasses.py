from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass(order=True, unsafe_hash=True)
class Pos:
    """Represents a position (line, column) in a file.

    Attributes:
        line (int): The line number (1-indexed).
        column (int): The column number (0-indexed).
    """
    line: int
    column: int

    def __str__(self) -> str:
        return f"({self.line}, {self.column})"

@dataclass(frozen=True)
class Sorry:
    """Represents a 'sorry' in the Lean code, a placeholder for an incomplete proof.

    Attributes:
        pos (Pos): Start position of the sorry.
        endPos (Pos): End position of the sorry.
        goal (str): The goal state at the sorry.
        proofState (int): The ID of the proof state.
    """
    pos: Pos
    endPos: Pos
    goal: str
    proofState: int

    def serialize(self) -> dict[str, Any]:
        """Serializes the Sorry object to a dictionary."""
        return {
            "pos": str(self.pos),
            "endPos": str(self.endPos),
            "goal": self.goal,
            "proofState": self.proofState,
        }

@dataclass(frozen=True)
class LeanError:
    """A data container for an error message from the Lean REPL.

    Attributes:
        message (str): The error message text.
        source (str): The error origin (Lean REPL, timeout, server, etc.)
    """
    message: str
    source: str = "REPL"

    def __str__(self) -> str:
        return "(" + self.source + ") " + self.message

    def serialize(self) -> dict[str, Any]:
        """Serializes the LeanError to a dictionary."""
        return {"message": self.message, "source": self.source}

@dataclass(frozen=True)
class Message:
    """Represents a message (info, warning, or error) from the Lean compiler.

    Attributes:
        severity (str): The severity level (e.g., 'error', 'warning', 'info').
        data (str): The message content.
        pos (Pos): The position where the message occurred.
        endPos (Optional[Pos]): The end position, if available.
    """
    severity: str
    data: str
    pos: Pos
    endPos: Optional[Pos] = None

    def serialize(self) -> dict[str, Any]:
        """Serializes the Message to a dictionary."""
        return {
            "severity": self.severity,
            "pos": str(self.pos),
            "endPos": str(self.endPos) if self.endPos else None,
            "data": self.data
        }

@dataclass
class Environment:
    """Represents a Lean environment state.

    This is the primary data structure for tracking state between commands.

    Attributes:
        env (Optional[int]): The environment ID.
        sorries (List[Sorry]): List of sorries in this environment.
        messages (List[Message]): List of messages generated in this environment.
        goals (list[str]): List of unsolved goals in this environment.
    """

    env: Optional[int]
    sorries: list[Sorry] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)

    def serialize(self) -> dict[str, Any]:
        """Serializes the Environment to a dictionary."""
        return {
            "env": self.env,
            "sorries": [s.serialize() for s in self.sorries],
            "messages": [m.serialize() for m in self.messages],
            "goals": self.goals,
        }

@dataclass
class ProofState:
    """Represents the state of an active proof (a 'tactic state').

    Attributes:
        proofState (int): The proof state ID.
        goals (list[str]): List of open goals.
        sorries (list[Sorry]): List of sorries encountered.
        messages (list[Message]): List of messages.
        proofStatus (str): Status of the proof (e.g., incomplete).
    """
    proofState: int
    goals: list[str]
    sorries: list[Sorry] = field(default_factory=list) 
    messages: list[Message] = field(default_factory=list)
    proofStatus: str = ""

    def __str__(self) -> str:
        return "\n\n".join(self.goals) if self.goals else ""

    def content_eq(self, other: "ProofState") -> bool:
        """Compares the content of two proof states, ignoring the state ID.

        Args:
            other (ProofState): The other proof state to compare with.

        Returns:
            bool: True if contents (goals, sorries, errors) are equal.
        """
        if not isinstance(other, ProofState):
            return False
        
        error_messages_self = [m for m in self.messages if m.severity == "error"]
        error_messages_other = [m for m in other.messages if m.severity == "error"]

        return all([
            self.goals == other.goals,
            self.sorries == other.sorries,
            error_messages_self == error_messages_other,
        ])
    
    def serialize(self) -> dict[str, Any]:
        """Serializes the ProofState to a dictionary."""
        return {
            "proofState": self.proofState,
            "goals": self.goals,
            "sorries": [s.serialize() for s in self.sorries],
            "messages": [m.serialize() for m in self.messages],
            "proofStatus": self.proofStatus,
        }