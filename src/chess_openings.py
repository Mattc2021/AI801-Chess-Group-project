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
    "rnbqkbnr/pppp1ppp/8/4p3/1P6/8/P1PPPPPP/RNBQKBNR b KQkq - 0 1"   # Polish Opening
]

# Saving these openings to a text file
openings_file_path = '../assets/chess_openings.txt'

with open(openings_file_path, 'w') as file:
    for opening in openings:
        file.write(opening + '\n')
