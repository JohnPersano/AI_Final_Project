import os

import settings


class GeneralUtils:

    def __init__(self):
        self.output_directory = settings.DATA_OUT

    def clean_start(self):
        for out_file in os.listdir(self.output_directory):
            out_file_path = os.path.join(self.output_directory, out_file)
            os.unlink(out_file_path)




