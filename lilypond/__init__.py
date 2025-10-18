from .basin import Basin
from .pond import Pond
from .legacy_pond import LegacyPond

__all__ = ["Basin", "Pond", "LegacyPond"]

def __version__():
    return "0.0.1"

def describe():
    description = (
        "lilypond\n"
        "Version: {}\n"
    ).format(__version__())
    
    print(description)
