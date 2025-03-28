from .histmaker import HistMaker


class CutFlowReader(HistMaker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_cutflow_keeper = {}

    def process(self, config):

        # Doing only nominal for now
        for p in config.processes:
            self.event_cutflow_keeper[p.name] = {}
            for ifile in p.input_files:
                self.read_cutflow(ifile)

    def read_cutflow(self, ifile):
        with self.open_file(ifile) as tfile:
            return tfile
