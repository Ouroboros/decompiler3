from .types import *

class ScpParser(StrictBase):
    fs      : fileio.FileStream
    name    : str
    header  : ScpHeader

    def __init__(self, fs: fileio.FileStream, name: str = ''):
        self.fs = fs

    def parse(self):
        self.read_header()

    def read_header(self):
        fs = self.fs

        hdr = ScpHeader(fs = fs)
        self.header = hdr
