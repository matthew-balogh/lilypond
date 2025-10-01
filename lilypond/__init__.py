"""
lilypond
"""

def __version__():
    return "0.0.1"

def describe():
    description = (
        "lilypond\n"
        "Version: {}\n"
    ).format(__version__())
    
    print(description)