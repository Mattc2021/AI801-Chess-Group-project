# List of chess openings in Forsyth-Edwards Notation (FEN).
# Each FEN string represents a unique chess board position, typically an opening move or sequence.
openings = [
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",  # King's Pawn Opening
    "rnbqkbnr/pppppppp/8/8/3nP3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",  # Sicilian Defense
    "rnbqkbnr/pppppppp/8/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 1",  # Queen's Pawn Opening
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",  # Ruy Lopez
    "rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1",  # English Opening
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",  # Caro-Kann Defense
    "rnbqkbnr/ppp2ppp/4p3/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R b KQkq - 1 3",  # French Defense
    "rnbqkbnr/ppp2ppp/3pp3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 3",  # Pirc Defense
    "rnbqkbnr/ppp2ppp/4p3/8/4P3/3P4/PPP2PPP/RNBQKBNR b KQkq - 0 3",  # King's Indian Defense
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",  # Start Position
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",  # Philidor Defense
    "rnbqkbnr/ppp1pppp/8/3p4/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 2",  # Catalan Opening
    "rnbqkbnr/pp2pppp/8/2pp4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3",  # Benoni Defense
    "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 2 3",  # Italian Game
    "rnbqkbnr/pppp1ppp/8/4p3/1P6/8/P1PPPPPP/RNBQKBNR b KQkq - 0 1",   # Polish Opening
    "rnbqkb1r/ppp1pppp/5n2/3p4/3P4/4PN2/PPP2PPP/RNBQKB1R b KQkq - 0 3", # King's Indian Attack
    "rnbqk2r/ppp1ppbp/5np1/3p4/3P4/5NP1/PPP1PPBP/RNBQK2R b KQkq - 0 4", # Nimzo-Indian Defense
    "rnbqkb1r/pp2pppp/2p2n2/3p4/2P5/4PN2/PPP1BPPP/RNBQK2R b KQkq - 0 4", # Slav Defense
    "rnbqkb1r/ppp1pp1p/6p1/3n4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 4", # Grunfeld Defense
    "rnbqkb1r/pppppppp/8/8/4Pn2/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3", # Alekhine's Defense
    "rnbqkb1r/pp3ppp/2p1pn2/3p4/3P4/2N1PN2/PPP1BPPP/R1BQK2R b KQkq - 0 6", # Queens Gambit Declined
    "rnbqkb1r/ppp1pppp/5n2/3p2B1/3P4/8/PPP1PPPP/RN1QKBNR b KQkq - 0 3", # Scandinavian Defense
    "rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2", # Vienna Game
    "rnbqkb1r/pppppppp/8/8/3P1n2/8/PPP1PPPP/RNBQKBNR w KQkq - 0 3", # Dutch Defense
    "r1bqkb1r/pp2pppp/2np1n2/6B1/3NP3/2N5/PPP2PPP/R2QKB1R b KQkq - 0 6", # Sicillian Dragon
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # Four Knights Game
    "rnbqkb1r/pp2pppp/5n2/2pp4/2P5/5NP1/PPP1PP1P/RNBQKB1R w KQkq - 0 3",  # Symmetrical English
    "rnbqkb1r/pp1ppppp/5n2/2p5/2P5/5N2/PPP1PPPP/RNBQKB1R b KQkq - 1 2",  # Sicilian Defence, Closed
    "rnbqkbnr/pp1ppppp/8/2p5/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 2",  # English Opening, Symmetrical Variation
    "rnbqkbnr/ppp2ppp/8/3pp3/4P3/3P4/PPP2PPP/RNBQKBNR w KQkq - 0 3",  # Center Game
    "rnbqkbnr/pppppp1p/6p1/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",  # Modern Defense
    "rnbqkbnr/ppppppp1/8/7p/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",  # Dutch Defense, Hopton Attack
    "rnbqkbnr/ppp1pppp/8/3p4/1P6/8/P1PPPPPP/RNBQKBNR b KQkq - 0 2",  # Polish Opening
    "r1bqkbnr/ppp2ppp/2np4/4p3/2BPP3/8/PPP2PPP/RNBQK1NR b KQkq - 0 4",  # Scotch Game
    "rnbqkbnr/ppp1pppp/8/8/3p4/8/PPP2PPP/RNBQKBNR w KQkq - 0 3"         # Queen's Gambit Accepted
]

# File path where the openings will be saved.
openings_file_path = '../assets/chess_openings.txt'

# Writing the chess openings to a text file.
# Each opening is written on a new line.
with open(openings_file_path, 'w') as file:
    for opening in openings:
        file.write(opening + '\n')