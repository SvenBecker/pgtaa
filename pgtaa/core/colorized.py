from logging import StreamHandler
from colorama import Fore, init, Style

init()

def color_bar(color: str="white"):
    # colorize tdqm bars
    if color == "blue":
        return "%s{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.BLUE, Fore.BLUE)
    elif color == "reset":
        return "%s{l_bar}%s{bar}%s{r_bar}" % (Fore.RESET, Fore.RESET, Fore.RESET)
    elif color == "red":
        return "%s{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RED, Fore.RED)
    elif color == "black":
        return "%s{l_bar}%s{bar}%s{r_bar}" % (Fore.BLACK, Fore.BLACK, Fore.BLACK)
    elif color == "cyan":
        return "%s{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.CYAN, Fore.CYAN)
    elif color == "green":
        return "%s{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.GREEN, Fore.GREEN)
    elif color == "yellow":
        return "%s{l_bar}%s{bar}%s{r_bar}" % (Fore.YELLOW, Fore.YELLOW, Fore.YELLOW)
    elif color == "magenta":
        return "%s{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.MAGENTA, Fore.MAGENTA)
    else:
        return "%s{l_bar}%s{bar}%s{r_bar}" % (Fore.WHITE, Fore.WHITE, Fore.WHITE)


class ColourHandler(StreamHandler):
    # check: https://www.programcreek.com/python/example/184/logging.StreamHandler

    """ A colorized output SteamHandler """

    # Some basic colour scheme defaults
    colours = {
        'DEBUG': Fore.RESET,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED
    }

    def emit(self, record):
        try:
            message = self.format(record)
            self.stream.write(self.colours[record.levelname] + message + Style.RESET_ALL)
            self.stream.write(getattr(self, 'terminator', '\n'))
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)
