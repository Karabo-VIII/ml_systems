"""Perft: legal-move enumerator for verifying move generation correctness.

perft(board, depth) counts the number of leaf nodes reachable in exactly
`depth` plies from the given position, using python-chess legal move
generation. Standard perft node counts for the start position are a
well-known correctness benchmark.
"""

import chess


def perft(board: chess.Board, depth: int) -> int:
    """Count leaf nodes at exactly `depth` plies from `board`."""
    if depth <= 0:
        return 1
    # Leaf optimization: at depth 1 the count is just the number of legal moves.
    if depth == 1:
        return board.legal_moves.count()
    nodes = 0
    for move in board.legal_moves:
        board.push(move)
        nodes += perft(board, depth - 1)
        board.pop()
    return nodes


def _selftest() -> None:
    expected = {1: 20, 2: 400, 3: 8902}
    board = chess.Board()
    for depth, want in expected.items():
        got = perft(board, depth)
        assert got == want, f"perft(start, {depth}) = {got}, expected {want}"
        print(f"perft(start, {depth}) = {got}  OK")
    print("All perft selftests passed.")


if __name__ == "__main__":
    _selftest()
